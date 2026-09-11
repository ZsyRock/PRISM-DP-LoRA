#!/usr/bin/env python3
"""Build the locked Math-10K 4B fixed/SlaClip refinement registry.

The registry is deliberately explicit: selection code must consume only the
five preregistered public-validation runs for each candidate and must never
discover runs or evaluation artifacts by walking the campaign directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from itertools import product
from pathlib import Path
from typing import Any, Mapping


REGISTRY_SCHEMA_VERSION = 1
SELECTION_SEEDS = (42, 43, 44, 45, 46)
STAGE1_SEED = 42
FIXED_C_VALUES = (1.0, 2.0)
SLACLIP_C0_VALUES = (1.25, 1.5, 1.75, 2.0)
SLACLIP_RHO_VALUES = (0.985, 0.99, 0.995)
SLACLIP_ETA_VALUES = (0.05, 0.1)
SLACLIP_ANCHOR = (2.0, 0.97, 0.15)
SLACLIP_NUM_SLOTS = 15
SLACLIP_C_MIN = 0.1
SLACLIP_C_MAX = 15.0
EXPECTED_FIXED_COUNT = 2
EXPECTED_SLACLIP_COUNT = 25
EXPECTED_CANDIDATE_COUNT = EXPECTED_FIXED_COUNT + EXPECTED_SLACLIP_COUNT
SELECTION_METRIC = "public_math10k_numeric_exact_match_accuracy"
LOSS_DEFINITION = "response_only_per_record_mean_of_nonignored_next_token_losses"
FORBIDDEN_SCREEN_PARTS = {
    "test",
    "tests",
    "evaluation",
    "evaluations",
    "eval",
    "gsm8k",
    "aqua",
    "mawps",
    "svamp",
}
FULL_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class RegistryError(RuntimeError):
    """Raised when the locked registry cannot be created safely."""


def _slug(value: float) -> str:
    text = format(float(value), ".12g").replace(".", "p")
    return text.replace("-", "m")


def _encoded_json(payload: Any) -> bytes:
    try:
        return (
            json.dumps(
                payload,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
        raise RegistryError(f"registry is not finite JSON: {exc}") from exc


def _payload_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_locked_inputs(
    *, model_id: str, model_revision: str, code_sha: str
) -> tuple[str, str, str]:
    model_id = str(model_id).strip()
    model_revision = str(model_revision).strip()
    code_sha = str(code_sha).strip()
    if not model_id or any(character.isspace() for character in model_id):
        raise RegistryError("model_id must be a non-empty identifier without whitespace")
    if not FULL_SHA_RE.fullmatch(model_revision):
        raise RegistryError("model_revision must be a full lowercase 40-hex commit")
    if not FULL_SHA_RE.fullmatch(code_sha):
        raise RegistryError("code_sha must be a full lowercase 40-hex Git commit")
    return model_id, model_revision, code_sha


def _registered_runs(candidate_id: str) -> dict[str, dict[str, str]]:
    runs: dict[str, dict[str, str]] = {}
    for seed in SELECTION_SEEDS:
        relative = Path("screen") / "runs" / candidate_id / f"seed-{seed}"
        runs[str(seed)] = {
            "run_status": (
                relative / "adapter" / "run_status.json"
            ).as_posix(),
            "validation_metrics": (
                relative
                / "results"
                / "validation"
                / "validation_metrics.json"
            ).as_posix(),
            "split_manifest": (
                relative / "results" / "validation" / "split_manifest.json"
            ).as_posix(),
        }
    return runs


def _fixed_candidate(clip: float) -> dict[str, Any]:
    candidate_id = f"fixed-c{_slug(clip)}"
    return {
        "id": candidate_id,
        "family": "fixed",
        "method": "baseline",
        "params": {"dp_max_grad_norm": float(clip)},
        "runs": _registered_runs(candidate_id),
    }


def _slaclip_candidate(c0: float, rho: float, eta: float) -> dict[str, Any]:
    candidate_id = f"sla-c{_slug(c0)}-r{_slug(rho)}-e{_slug(eta)}"
    return {
        "id": candidate_id,
        "family": "slaclip",
        "method": "slaclip",
        "params": {
            "dp_max_grad_norm": float(c0),
            "slaclip_target_non_small_clip_fraction": float(rho),
            "slaclip_eta": float(eta),
            "slaclip_num_slots": SLACLIP_NUM_SLOTS,
            "slaclip_c_min": SLACLIP_C_MIN,
            "slaclip_c_max": SLACLIP_C_MAX,
        },
        "runs": _registered_runs(candidate_id),
    }


def _common_config(
    *, model_id: str, model_revision: str, code_sha: str
) -> dict[str, Any]:
    # This is the completed Math-10K 4B campaign's public-selection protocol.
    # Candidate-specific C/rho/eta/K/bounds are intentionally kept out.
    return {
        "dataset": "math10k",
        "privacy": "dp",
        "base_model": model_id,
        "model_revision": model_revision,
        "implementation_git_sha": code_sha,
        "lora_r": 16,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "total_update_steps": 300,
        "batch_size": 64,
        "micro_batch_size": 4,
        "learning_rate": 0.0003,
        "cutoff_len": 256,
        "train_on_inputs": True,
        "val_set_size": 500,
        "validation_seed": 1729,
        "validation_batch_size": 8,
        "validation_eval_interval": 50,
        "validation_generate_numeric": True,
        "validation_num_beams": 1,
        "validation_max_new_tokens": 128,
        "validation_max_input_length": 512,
        "protocol_stage": "selection",
        "validation_data_is_public": True,
        "dp_epsilon": 6.0,
        "dp_delta": 1e-5,
        "dp_grad_sample_mode": "functorch",
        "dp_accountant": "prv",
        "dp_secure_mode": False,
        "telemetry_mode": "research_raw",
        "allow_non_private_telemetry": True,
        "raw_hist_bins": 128,
        "raw_hist_max": 30.0,
        "spectral_svd_device": "cpu",
        "spectral_oversample": 8,
        "spectral_n_iter": 2,
        "prism_floor_factor": 0.5,
        "prism_floor_mode": "scalar",
        "prism_cond_max": 10000.0,
        "prism_cond_strategy": "raise_small",
        "prism_lift_fix": "both",
        "prism_debias_second_moment": False,
        "max_update_norm": 0.0,
    }


def _build_payload(
    *, model_id: str, model_revision: str, code_sha: str
) -> dict[str, Any]:
    fixed = [_fixed_candidate(value) for value in FIXED_C_VALUES]
    slaclip = [
        _slaclip_candidate(c0, rho, eta)
        for c0, rho, eta in product(
            SLACLIP_C0_VALUES, SLACLIP_RHO_VALUES, SLACLIP_ETA_VALUES
        )
    ]
    slaclip.append(_slaclip_candidate(*SLACLIP_ANCHOR))
    candidates = [*fixed, *slaclip]
    payload = {
        "schema_version": REGISTRY_SCHEMA_VERSION,
        "selection_protocol": {
            "name": "math10k_4b_c1_c2_slaclip_refinement_numeric_selection_v1",
            "protocol_stage": "selection",
            "validation_data_is_public": True,
            "selection_metric": SELECTION_METRIC,
            "loss_definition": LOSS_DEFINITION,
            "required_update_steps": 300,
            "target_epsilon": 6.0,
            "epsilon_tolerance": 0.02,
            "stage1_seed": STAGE1_SEED,
            "stage2_seeds": list(SELECTION_SEEDS),
            "common_config": _common_config(
                model_id=model_id,
                model_revision=model_revision,
                code_sha=code_sha,
            ),
        },
        "candidates": candidates,
    }
    _validate_payload(payload)
    return payload


def _validate_payload(payload: Mapping[str, Any]) -> None:
    candidates = payload.get("candidates")
    if not isinstance(candidates, list) or len(candidates) != EXPECTED_CANDIDATE_COUNT:
        raise RegistryError(
            f"registry must contain exactly {EXPECTED_CANDIDATE_COUNT} candidates"
        )
    ids = [candidate.get("id") for candidate in candidates if isinstance(candidate, dict)]
    if len(ids) != len(candidates) or len(set(ids)) != len(ids):
        raise RegistryError("candidate identifiers must be present and unique")
    if any(not isinstance(value, str) or not ID_RE.fullmatch(value) for value in ids):
        raise RegistryError("candidate identifiers are not selector-compatible")
    fixed = [candidate for candidate in candidates if candidate.get("family") == "fixed"]
    slaclip = [
        candidate for candidate in candidates if candidate.get("family") == "slaclip"
    ]
    if len(fixed) != EXPECTED_FIXED_COUNT or len(slaclip) != EXPECTED_SLACLIP_COUNT:
        raise RegistryError("registry family counts are inconsistent")
    expected_seeds = {str(seed) for seed in SELECTION_SEEDS}
    expected_run_keys = {"run_status", "validation_metrics", "split_manifest"}
    for candidate in candidates:
        runs = candidate.get("runs")
        if not isinstance(runs, dict) or set(runs) != expected_seeds:
            raise RegistryError(f"candidate {candidate['id']} must register all five seeds")
        if any(not isinstance(run, dict) or set(run) != expected_run_keys for run in runs.values()):
            raise RegistryError(f"candidate {candidate['id']} has an invalid run registry")


def _resolve_output(campaign_root: Path, output: Path) -> tuple[Path, Path]:
    try:
        campaign_root = campaign_root.resolve(strict=True)
    except OSError as exc:
        raise RegistryError(f"campaign root does not exist: {campaign_root}") from exc
    if not campaign_root.is_dir():
        raise RegistryError(f"campaign root is not a directory: {campaign_root}")
    screen_root = campaign_root / "screen"
    screen_root.mkdir(parents=True, exist_ok=True)
    screen_root = screen_root.resolve(strict=True)
    supplied = output if output.is_absolute() else campaign_root / output
    if supplied.name != "candidate_registry.json":
        raise RegistryError("output must be named candidate_registry.json")
    try:
        prospective_parent = supplied.parent.resolve(strict=False)
        relative_parent = prospective_parent.relative_to(screen_root)
    except (OSError, ValueError) as exc:
        raise RegistryError(f"output must remain inside {screen_root}: {supplied}") from exc
    forbidden = FORBIDDEN_SCREEN_PARTS.intersection(
        part.casefold() for part in relative_parent.parts
    )
    if forbidden:
        raise RegistryError(
            f"output path contains forbidden selector components: {sorted(forbidden)!r}"
        )
    supplied.parent.mkdir(parents=True, exist_ok=True)
    try:
        parent = supplied.parent.resolve(strict=True)
        parent.relative_to(screen_root)
    except (OSError, ValueError) as exc:  # pragma: no cover - race defence
        raise RegistryError(f"output escaped screen root during creation: {supplied}") from exc
    target = parent / supplied.name
    if target.is_symlink():
        raise RegistryError(f"output must not be a symlink: {target}")
    return campaign_root, target


def _assert_immutable_compatible(path: Path, encoded: bytes) -> None:
    if not path.exists():
        return
    try:
        existing = path.read_bytes()
    except OSError as exc:
        raise RegistryError(f"could not inspect existing registry {path}: {exc}") from exc
    if existing != encoded:
        raise RegistryError(f"refusing to overwrite inconsistent registry: {path}")


def _write_immutable(path: Path, encoded: bytes) -> None:
    _assert_immutable_compatible(path, encoded)
    if path.exists():
        os.chmod(path, 0o600)
        return
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o600)
        try:
            os.link(temporary, path)
        except FileExistsError:
            _assert_immutable_compatible(path, encoded)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)
    if not path.exists():  # pragma: no cover - defensive filesystem check
        raise RegistryError(f"atomic registry creation failed: {path}")
    os.chmod(path, 0o600)


def build_registry(
    *,
    campaign_root: Path,
    output: Path,
    model_id: str,
    model_revision: str,
    code_sha: str,
) -> dict[str, Any]:
    """Create or verify the byte-identical immutable candidate registry."""

    model_id, model_revision, code_sha = _validate_locked_inputs(
        model_id=model_id,
        model_revision=model_revision,
        code_sha=code_sha,
    )
    _, target = _resolve_output(campaign_root, output)
    payload = _build_payload(
        model_id=model_id,
        model_revision=model_revision,
        code_sha=code_sha,
    )
    _write_immutable(target, _encoded_json(payload))
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--code-sha", required=True)
    args = parser.parse_args()
    payload = build_registry(
        campaign_root=args.campaign_root,
        output=args.output,
        model_id=args.model_id,
        model_revision=args.model_revision,
        code_sha=args.code_sha,
    )
    print(
        f"registry_candidates={len(payload['candidates'])} "
        f"payload_sha256={_payload_sha256(payload)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
