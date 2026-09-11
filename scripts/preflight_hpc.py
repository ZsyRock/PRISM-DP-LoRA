#!/usr/bin/env python3
"""Fail-fast environment and asset checks before an HPC experiment.

The default mode is safe to run on a CPU login node. Use ``--require-cuda``
inside an allocated GPU job. Network access is only attempted when
``--check-hf-access`` is supplied. Hugging Face tokens are never printed.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Optional


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class DatasetSpec:
    relative_path: str
    sha256: str
    rows: int
    required_keys: frozenset[str]
    auxiliary_paths: tuple[str, ...] = ()


DATASET_SPECS = {
    "math10k": DatasetSpec(
        relative_path="LLM-Adapters/ft-training_set/math_10k.json",
        sha256="0342d0d860ad8592b579329337c90e42eefd3d9f2898043140cbd120630418b8",
        rows=9919,
        required_keys=frozenset({"instruction", "input", "output"}),
        auxiliary_paths=(
            "LLM-Adapters/dataset/gsm8k/test.json",
            "LLM-Adapters/dataset/AQuA/test.json",
            "LLM-Adapters/dataset/mawps/test.json",
            "LLM-Adapters/dataset/SVAMP/test.json",
        ),
    ),
    "glue8": DatasetSpec(
        relative_path="LLM-Adapters/ft-training_set/glue8_1250.json",
        sha256="281bad3305e8be7a7660b64b22704a9462f983b90df5d6f0c1c4abb649d6d091",
        rows=10000,
        required_keys=frozenset({"instruction", "input", "output"}),
    ),
}


# Distribution metadata alone cannot detect a broken native extension or a
# libstdc++/CUDA ABI mismatch.  Import the runtime module names actually used by
# training and evaluation so preflight fails before a long allocation starts.
CRITICAL_IMPORT_MODULES = (
    "torch",
    "transformers",
    "peft",
    "opacus",
    "datasets",
    "evaluate",
    "accelerate",
    "pandas",
    "sklearn",
    "scipy",
    "sentencepiece",
    "safetensors",
    "fire",
)


class Reporter:
    def __init__(self) -> None:
        self.failures = 0
        self.warnings = 0

    def pass_(self, message: str) -> None:
        print(f"[PASS] {message}")

    def warn(self, message: str) -> None:
        self.warnings += 1
        print(f"[WARN] {message}")

    def fail(self, message: str) -> None:
        self.failures += 1
        print(f"[FAIL] {message}")


def _redact(text: object, secrets: Iterable[Optional[str]] = ()) -> str:
    value = str(text)
    for secret in secrets:
        if secret:
            value = value.replace(secret, "<redacted>")
    value = re.sub(r"hf_[A-Za-z0-9_-]{6,}", "<redacted-hf-token>", value)
    value = re.sub(
        r"(?i)(authorization\s*[:=]\s*)([^\s,;]+)",
        r"\1<redacted>",
        value,
    )
    return value


def _iter_requirements(path: Path, seen: Optional[set[Path]] = None):
    from packaging.requirements import Requirement

    seen = set() if seen is None else seen
    path = path.resolve()
    if path in seen:
        return
    seen.add(path)
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        if line.startswith(("-r ", "--requirement ")):
            child = line.split(maxsplit=1)[1]
            yield from _iter_requirements(path.parent / child, seen)
            continue
        if line.startswith("-"):
            continue
        yield Requirement(line)


def check_python(reporter: Reporter) -> None:
    version = sys.version_info
    detail = f"Python {version.major}.{version.minor}.{version.micro} ({sys.executable})"
    if version < (3, 10):
        reporter.fail(f"{detail}; Python >=3.10 is required")
    elif version[:2] != (3, 11):
        reporter.warn(f"{detail}; the reference environment uses Python 3.11")
    else:
        reporter.pass_(detail)


def check_packages(reporter: Reporter, requirements_file: Path) -> None:
    try:
        requirements = list(_iter_requirements(requirements_file))
    except Exception as exc:
        reporter.fail(f"Could not parse {requirements_file}: {_redact(exc)}")
        return
    for requirement in requirements:
        if requirement.marker is not None and not requirement.marker.evaluate():
            continue
        try:
            version = importlib.metadata.version(requirement.name)
        except importlib.metadata.PackageNotFoundError:
            reporter.fail(f"Missing package: {requirement.name}{requirement.specifier}")
            continue
        if requirement.specifier and not requirement.specifier.contains(version, prereleases=True):
            reporter.fail(
                f"{requirement.name}=={version} does not satisfy {requirement.specifier}"
            )
        else:
            reporter.pass_(f"{requirement.name}=={version}")


def check_imports(
    reporter: Reporter,
    importer: Callable[[str], object] = importlib.import_module,
) -> None:
    """Import the real runtime stack and force the Gemma 3 lazy model import.

    ``importer`` is injectable so tests can exercise binary-import failures
    without depending on the packages installed in the test environment.
    Every module is attempted even if an earlier import fails, giving one
    preflight run a complete list of environment problems.
    """

    imported: dict[str, object] = {}
    for module_name in CRITICAL_IMPORT_MODULES:
        try:
            imported[module_name] = importer(module_name)
        except Exception as exc:
            reporter.fail(f"Runtime import failed for {module_name}: {_redact(exc)}")
        else:
            reporter.pass_(f"Runtime import succeeded: {module_name}")

    transformers_module = imported.get("transformers")
    if transformers_module is None:
        reporter.fail(
            "Gemma 3 runtime import could not be checked because transformers failed to import"
        )
        return

    try:
        # Transformers exposes these through a lazy module.  Accessing both
        # attributes imports the Gemma 3 configuration and multimodal modeling
        # implementation, surfacing optional/native dependency failures that a
        # plain ``import transformers`` can leave hidden.
        getattr(transformers_module, "Gemma3Config")
        getattr(transformers_module, "Gemma3ForConditionalGeneration")
    except Exception as exc:
        reporter.fail(f"Gemma 3 multimodal lazy import failed: {_redact(exc)}")
    else:
        reporter.pass_(
            "Gemma 3 lazy imports succeeded: Gemma3Config, "
            "Gemma3ForConditionalGeneration"
        )


def check_pip(reporter: Reporter) -> None:
    result = subprocess.run(
        [sys.executable, "-m", "pip", "check"],
        capture_output=True,
        text=True,
        check=False,
    )
    stdout = result.stdout.strip()
    stderr = result.stderr.strip()
    combined = "\n".join(x for x in (stdout, stderr) if x)
    if result.returncode == 0:
        reporter.pass_(stdout or "pip dependency graph is consistent")
        if stderr:
            reporter.warn(f"pip check diagnostic: {_redact(stderr)}")
    else:
        reporter.fail(f"pip check failed: {_redact(combined)}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_dataset(
    reporter: Reporter,
    dataset: str,
    data_path: Optional[Path],
    expected_sha256: Optional[str],
) -> None:
    spec = DATASET_SPECS[dataset]
    path = (data_path if data_path is not None else ROOT / spec.relative_path).resolve()
    if not path.is_file():
        reporter.fail(f"Training data is missing: {path}")
        return
    actual_hash = _sha256(path)
    expected_hash = expected_sha256 or (spec.sha256 if data_path is None else None)
    if expected_hash and actual_hash.lower() != expected_hash.lower():
        reporter.fail(
            f"Data SHA-256 mismatch for {path}: expected={expected_hash}, actual={actual_hash}"
        )
    else:
        reporter.pass_(f"Data SHA-256 {actual_hash}: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        reporter.fail(f"Training data is not valid JSON: {_redact(exc)}")
        return
    if not isinstance(payload, list) or not payload:
        reporter.fail(f"Training data must be a non-empty JSON list: {path}")
        return
    if data_path is None and len(payload) != spec.rows:
        reporter.fail(f"Expected {spec.rows} rows in {path}, found {len(payload)}")
    else:
        reporter.pass_(f"Training rows: {len(payload)}")
    malformed = [
        index
        for index, row in enumerate(payload)
        if not isinstance(row, dict) or not spec.required_keys.issubset(row)
    ]
    if malformed:
        preview = ", ".join(str(index) for index in malformed[:5])
        reporter.fail(f"Rows missing required keys {sorted(spec.required_keys)}: {preview}")
    else:
        reporter.pass_(f"All rows contain keys {sorted(spec.required_keys)}")
    if data_path is None:
        for relative in spec.auxiliary_paths:
            auxiliary = ROOT / relative
            if auxiliary.is_file():
                reporter.pass_(f"Evaluation asset present: {relative}")
            else:
                reporter.fail(f"Evaluation asset missing: {relative}")


def _positive_process_count(name: str) -> Optional[int]:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return None
    try:
        return int(raw)
    except ValueError:
        return -1


def check_single_process(reporter: Reporter) -> None:
    variables = ("WORLD_SIZE", "LOCAL_WORLD_SIZE", "SLURM_NTASKS", "PMI_SIZE", "OMPI_COMM_WORLD_SIZE")
    offenders = []
    invalid = []
    observed = []
    for name in variables:
        value = _positive_process_count(name)
        if value is None:
            continue
        observed.append(f"{name}={value}")
        if value < 1:
            invalid.append(f"{name}={os.environ.get(name)!r}")
        elif value > 1:
            offenders.append(f"{name}={value}")
    if invalid:
        reporter.fail(f"Invalid process-count variables: {', '.join(invalid)}")
    elif offenders:
        reporter.fail(
            "Distributed execution is unsupported; request one task/process only: "
            + ", ".join(offenders)
        )
    else:
        reporter.pass_("Single-process execution confirmed" + (f" ({', '.join(observed)})" if observed else ""))


def check_torch(reporter: Reporter, require_cuda: bool) -> None:
    try:
        import torch
    except Exception as exc:
        reporter.fail(f"PyTorch import failed: {_redact(exc)}")
        return
    reporter.pass_(f"torch import succeeded; version={torch.__version__}")
    available = bool(torch.cuda.is_available())
    if not available:
        message = "CUDA is unavailable to PyTorch"
        if require_cuda:
            reporter.fail(message)
        else:
            reporter.warn(message + "; acceptable for login-node preflight only")
        return
    count = int(torch.cuda.device_count())
    if count < 1:
        reporter.fail("torch.cuda.is_available() is true but no CUDA device is visible")
        return
    for index in range(count):
        props = torch.cuda.get_device_properties(index)
        memory_gib = props.total_memory / (1024**3)
        reporter.pass_(
            f"CUDA device {index}: {props.name}; capability={props.major}.{props.minor}; "
            f"memory={memory_gib:.1f} GiB"
        )
    bf16 = bool(getattr(torch.cuda, "is_bf16_supported", lambda: False)())
    if bf16:
        reporter.pass_("CUDA bfloat16 is supported")
    else:
        reporter.warn("CUDA bfloat16 is unavailable; training will fall back to float16")


def check_output_root(reporter: Reporter, output_root: Path, min_free_gb: float) -> None:
    try:
        output_root.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(prefix=".prism-write-test-", dir=output_root):
            pass
        free_gb = shutil.disk_usage(output_root).free / (1024**3)
    except Exception as exc:
        reporter.fail(f"Output root is not writable ({output_root}): {_redact(exc)}")
        return
    if free_gb < min_free_gb:
        reporter.fail(
            f"Output root has {free_gb:.1f} GiB free, below --min-free-gb={min_free_gb:g}: {output_root}"
        )
    else:
        reporter.pass_(f"Output root writable; free={free_gb:.1f} GiB: {output_root}")


def _hf_token() -> Optional[str]:
    try:
        from huggingface_hub import get_token

        return get_token()
    except Exception:
        return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


def check_hugging_face(
    reporter: Reporter,
    check_access: bool,
    model_id: str,
    revision: Optional[str],
) -> None:
    token = _hf_token()
    if token:
        reporter.pass_("A Hugging Face credential is available (value intentionally hidden)")
    else:
        message = "No Hugging Face credential was found; gated Gemma weights will be inaccessible"
        if check_access:
            reporter.fail(message)
        else:
            reporter.warn(message)
        return
    if not check_access:
        reporter.warn("Hugging Face network/model access was not checked; add --check-hf-access")
        return
    try:
        from huggingface_hub import HfApi

        api = HfApi()
        api.whoami(token=token)
        info = api.model_info(repo_id=model_id, revision=revision, token=token)
        resolved = getattr(info, "sha", None) or "unknown"
        reporter.pass_(f"Hugging Face access confirmed for {model_id}; resolved revision={resolved}")
    except Exception as exc:
        reporter.fail(
            "Hugging Face authentication/model access failed: "
            + _redact(exc, secrets=(token, os.environ.get("HF_TOKEN")))
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=sorted(DATASET_SPECS), default="math10k")
    parser.add_argument("--data-path", type=Path, default=None)
    parser.add_argument("--expected-data-sha256", default=None)
    parser.add_argument("--requirements", type=Path, default=ROOT / "requirements.txt")
    parser.add_argument("--output-root", type=Path, default=ROOT / "LLM-Adapters")
    parser.add_argument("--min-free-gb", type=float, default=30.0)
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--check-hf-access", action="store_true")
    parser.add_argument("--model-id", default="google/gemma-3-4b-pt")
    parser.add_argument("--model-revision", default=None)
    parser.add_argument("--skip-package-check", action="store_true")
    parser.add_argument(
        "--skip-import-check",
        action="store_true",
        help="Skip real runtime imports and the Gemma 3 lazy-import ABI check",
    )
    parser.add_argument("--skip-pip-check", action="store_true")
    parser.add_argument("--skip-data-check", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.min_free_gb < 0:
        raise SystemExit("--min-free-gb must be non-negative")
    reporter = Reporter()
    print(f"PRISM-DP-LoRA preflight root: {ROOT}")
    check_python(reporter)
    check_single_process(reporter)
    if not args.skip_package_check:
        check_packages(reporter, args.requirements)
    if not args.skip_pip_check:
        check_pip(reporter)
    if not args.skip_import_check:
        check_imports(reporter)
    if not args.skip_data_check:
        check_dataset(reporter, args.dataset, args.data_path, args.expected_data_sha256)
    check_output_root(reporter, args.output_root.resolve(), args.min_free_gb)
    check_torch(reporter, args.require_cuda)
    check_hugging_face(reporter, args.check_hf_access, args.model_id, args.model_revision)
    print(f"Preflight summary: failures={reporter.failures}, warnings={reporter.warnings}")
    return 1 if reporter.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
