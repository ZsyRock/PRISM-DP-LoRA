#!/usr/bin/env python3
"""Derive a locked full-SlaClip target grid from a fixed-C screen.

This is an explicitly NON_PRIVATE, development-only calibration step.  It
consumes the exact per-update clipping telemetry of eight completed fixed-C
runs, selects two initial clipping thresholds, and writes both an auditable
derivation artifact and a standard candidate registry for the formal public
validation selector.

No directory discovery is used for run evidence: candidate artifacts are read
only from paths registered in the supplied fixed-scan manifest.  The one raw
telemetry path is derived from each registered run-status path and must remain
inside ``<campaign>/screen``.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import math
import os
import statistics
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
SELECTOR_PATH = SCRIPT_DIR / "select_validation_candidates.py"
_SELECTOR_SPEC = importlib.util.spec_from_file_location(
    "_prism_validation_selector_for_target_grid", SELECTOR_PATH
)
if _SELECTOR_SPEC is None or _SELECTOR_SPEC.loader is None:  # pragma: no cover
    raise RuntimeError(f"could not load formal selector: {SELECTOR_PATH}")
selector = importlib.util.module_from_spec(_SELECTOR_SPEC)
_SELECTOR_SPEC.loader.exec_module(selector)


FIXED_C_VALUES = (0.1, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 15.0)
SOURCE_SEED = 42
TARGET_COUNT = 5
SLACLIP_ETA = 0.15
SLACLIP_NUM_SLOTS = 15
SLACLIP_C_MIN = 0.1
SLACLIP_C_MAX = 15.0
RHO_MIN = 0.5
RHO_MAX = 0.995
TRANSITION_Q = 0.10
UPPER_Q = 0.90
TRANSITION_THRESHOLD = 0.99
MIN_TARGET_WIDTH = 0.04
RAW_LOG_NAME = "NON_PRIVATE_train_log.jsonl"


class DerivationError(RuntimeError):
    """Raised when fixed-screen evidence cannot safely define a target grid."""


def _canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _payload_sha256(payload: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    except (TypeError, ValueError) as exc:
        raise DerivationError(f"artifact is not finite/canonical JSON: {exc}") from exc


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise DerivationError(f"could not read {label} {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise DerivationError(f"{label} must be a JSON object: {path}")
    return payload


def _require_exact_keys(
    payload: Mapping[str, Any], expected: set[str], *, label: str
) -> None:
    if set(payload) != expected:
        missing = sorted(expected - set(payload))
        unknown = sorted(set(payload) - expected)
        raise DerivationError(
            f"{label} must contain exactly {sorted(expected)!r}; "
            f"missing={missing!r}, unsupported={unknown!r}"
        )


def _finite_number(
    value: Any,
    *,
    label: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool):
        raise DerivationError(f"{label} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise DerivationError(f"{label} must be a finite number") from exc
    if not math.isfinite(result):
        raise DerivationError(f"{label} must be finite, got {value!r}")
    if minimum is not None and result < minimum:
        raise DerivationError(f"{label} must be >= {minimum}, got {result!r}")
    if maximum is not None and result > maximum:
        raise DerivationError(f"{label} must be <= {maximum}, got {result!r}")
    return result


def _resolve_input(
    campaign_root: Path,
    supplied: Path,
    *,
    required_root: str,
    label: str,
) -> Path:
    path = supplied if supplied.is_absolute() else campaign_root / supplied
    try:
        resolved = path.resolve(strict=True)
        root = (campaign_root / required_root).resolve(strict=True)
        relative = resolved.relative_to(root)
    except OSError as exc:
        raise DerivationError(f"missing {label}: {path}") from exc
    except ValueError as exc:
        raise DerivationError(
            f"{label} must remain inside {campaign_root / required_root}: {resolved}"
        ) from exc
    forbidden = selector.FORBIDDEN_PATH_PARTS.intersection(
        part.casefold() for part in relative.parts
    )
    if forbidden:
        raise DerivationError(
            f"{label} path contains forbidden test/evaluation components: "
            f"{sorted(forbidden)!r}"
        )
    if not resolved.is_file():
        raise DerivationError(f"{label} is not a regular file: {resolved}")
    return resolved


def _resolve_output(
    campaign_root: Path,
    supplied: Path,
    *,
    required_root: str,
    label: str,
    required_name: str | None = None,
) -> Path:
    path = supplied if supplied.is_absolute() else campaign_root / supplied
    if path.suffix.casefold() != ".json":
        raise DerivationError(f"{label} must be a .json file: {path}")
    if required_name is not None and path.name != required_name:
        raise DerivationError(f"{label} must be named {required_name}: {path}")
    root = campaign_root / required_root
    root.mkdir(parents=True, exist_ok=True)
    try:
        resolved_root = root.resolve(strict=True)
        # ``strict=False`` resolves every existing symlink without creating an
        # attacker-controlled/out-of-campaign parent as a validation side
        # effect.  Recheck after mkdir to close the creation-time gap.
        prospective_parent = path.parent.resolve(strict=False)
        prospective_parent.relative_to(resolved_root)
    except (OSError, ValueError) as exc:
        raise DerivationError(
            f"{label} must remain inside {resolved_root if 'resolved_root' in locals() else root}: {path}"
        ) from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        resolved_parent = path.parent.resolve(strict=True)
        resolved_parent.relative_to(resolved_root)
    except (OSError, ValueError) as exc:
        raise DerivationError(
            f"{label} must remain inside {resolved_root}: {path}"
        ) from exc
    return resolved_parent / path.name


def _assert_immutable_compatible(path: Path, encoded: bytes, *, label: str) -> None:
    if not path.exists():
        return
    try:
        existing = path.read_bytes()
    except OSError as exc:
        raise DerivationError(f"could not inspect existing {label} {path}: {exc}") from exc
    if existing != encoded:
        raise DerivationError(f"refusing to overwrite inconsistent {label}: {path}")


def _write_immutable(path: Path, encoded: bytes, *, label: str) -> None:
    _assert_immutable_compatible(path, encoded, label=label)
    if path.exists():
        return
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o600)
        try:
            os.link(temporary, path)
        except FileExistsError:
            _assert_immutable_compatible(path, encoded, label=label)
        try:
            directory_fd = os.open(path.parent, os.O_RDONLY)
        except OSError:
            directory_fd = -1
        if directory_fd >= 0:
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)
    if not path.exists():  # pragma: no cover - defensive filesystem check
        raise DerivationError(f"atomic creation of {label} failed: {path}")


def _quantile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise DerivationError("cannot compute a quantile of an empty trajectory")
    if not 0.0 <= probability <= 1.0:
        raise DerivationError(f"invalid quantile probability: {probability}")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * probability
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _slug(value: float) -> str:
    text = format(float(value), ".12g").replace(".", "p")
    return text.replace("-", "m")


def _validate_candidate_shape(candidate: Any, *, index: int) -> dict[str, Any]:
    if not isinstance(candidate, dict):
        raise DerivationError(f"fixed_candidates[{index}] must be a JSON object")
    _require_exact_keys(
        candidate,
        {"id", "family", "method", "params", "runs"},
        label=f"fixed_candidates[{index}]",
    )
    candidate_id = candidate.get("id")
    if not isinstance(candidate_id, str) or not selector.ID_RE.fullmatch(candidate_id):
        raise DerivationError(f"fixed candidate id is invalid: {candidate_id!r}")
    if candidate.get("family") != "fixed" or candidate.get("method") != "baseline":
        raise DerivationError(f"candidate {candidate_id} must be fixed/baseline")
    params = candidate.get("params")
    if not isinstance(params, dict) or set(params) != {"dp_max_grad_norm"}:
        raise DerivationError(
            f"candidate {candidate_id} params must contain only dp_max_grad_norm"
        )
    _finite_number(
        params["dp_max_grad_norm"],
        label=f"candidate {candidate_id} dp_max_grad_norm",
        minimum=1e-300,
    )
    runs = candidate.get("runs")
    if not isinstance(runs, dict) or set(runs) != {"42", "43", "44"}:
        raise DerivationError(
            f"candidate {candidate_id} runs must be registered exactly for seeds 42,43,44"
        )
    for seed in (42, 43, 44):
        try:
            selector._registered_run(candidate, seed)
        except selector.SelectionError as exc:
            raise DerivationError(str(exc)) from exc
    return copy.deepcopy(candidate)


def _validate_registered_future_paths(
    campaign_root: Path, candidate: Mapping[str, Any]
) -> None:
    screen_root = (campaign_root / "screen").resolve(strict=True)
    expected_names = {
        "run_status": "run_status.json",
        "validation_metrics": "validation_metrics.json",
        "split_manifest": "split_manifest.json",
    }
    for seed in (42, 43, 44):
        run = candidate["runs"][str(seed)]
        for key, expected_name in expected_names.items():
            value = run[key]
            if not isinstance(value, str) or not value.strip():
                raise DerivationError(
                    f"candidate {candidate['id']} seed {seed} {key} must be a path string"
                )
            supplied = Path(value)
            path = supplied if supplied.is_absolute() else campaign_root / supplied
            resolved = path.resolve(strict=False)
            try:
                relative = resolved.relative_to(screen_root)
            except ValueError as exc:
                raise DerivationError(
                    f"candidate {candidate['id']} seed {seed} {key} must remain "
                    f"inside {screen_root}: {resolved}"
                ) from exc
            if resolved.name != expected_name:
                raise DerivationError(
                    f"candidate {candidate['id']} seed {seed} {key} must be named "
                    f"{expected_name}: {resolved}"
                )
            forbidden = selector.FORBIDDEN_PATH_PARTS.intersection(
                part.casefold() for part in relative.parts
            )
            if forbidden:
                raise DerivationError(
                    f"candidate {candidate['id']} seed {seed} {key} contains "
                    f"forbidden path components: {sorted(forbidden)!r}"
                )


def _load_raw_trajectory(
    campaign_root: Path,
    candidate: Mapping[str, Any],
    *,
    config_fingerprint: Any,
) -> dict[str, Any]:
    run_status_value = candidate["runs"][str(SOURCE_SEED)]["run_status"]
    try:
        status_path = selector._resolve_screen_input(
            campaign_root,
            run_status_value,
            filename="run_status.json",
            label="run status",
        )
    except selector.SelectionError as exc:
        raise DerivationError(str(exc)) from exc
    if status_path.parent.name != "adapter":
        raise DerivationError(
            f"candidate {candidate['id']} run status must be under an adapter directory"
        )
    arm_root = status_path.parent.parent
    raw_path = arm_root / "results" / "research_raw" / RAW_LOG_NAME
    raw_path = _resolve_input(
        campaign_root,
        raw_path,
        required_root="screen",
        label=f"candidate {candidate['id']} NON_PRIVATE raw telemetry",
    )
    by_step: dict[int, dict[str, float]] = {}
    try:
        with raw_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise DerivationError(
                        f"invalid raw telemetry JSON {raw_path}:{line_number}: {exc}"
                    ) from exc
                if not isinstance(row, dict):
                    raise DerivationError(
                        f"raw telemetry row must be an object: {raw_path}:{line_number}"
                    )
                if row.get("NON_PRIVATE_TELEMETRY") is not True:
                    raise DerivationError(
                        f"raw telemetry lacks NON_PRIVATE acknowledgement: "
                        f"{raw_path}:{line_number}"
                    )
                step_value = row.get("step")
                if isinstance(step_value, bool) or not isinstance(step_value, int):
                    raise DerivationError(
                        f"raw telemetry step must be an integer: {raw_path}:{line_number}"
                    )
                step = int(step_value)
                if step in by_step:
                    raise DerivationError(
                        f"duplicate raw telemetry step {step}: {raw_path}:{line_number}"
                    )
                if row.get("config_fingerprint") != config_fingerprint:
                    raise DerivationError(
                        f"raw telemetry fingerprint mismatch: {raw_path}:{line_number}"
                    )
                clip = _finite_number(
                    row.get("raw_clip_fraction"),
                    label=f"raw_clip_fraction at {raw_path}:{line_number}",
                    minimum=0.0,
                    maximum=1.0,
                )
                small = _finite_number(
                    row.get("raw_reference_small_gradient_proxy"),
                    label=(
                        "raw_reference_small_gradient_proxy at "
                        f"{raw_path}:{line_number}"
                    ),
                )
                remaining = _finite_number(
                    row.get("raw_reference_remaining_mass_proxy"),
                    label=(
                        "raw_reference_remaining_mass_proxy at "
                        f"{raw_path}:{line_number}"
                    ),
                )
                valid = row.get("raw_reference_conditional_clip_fraction_valid")
                expected_valid = remaining > 1e-12
                if not isinstance(valid, bool) or valid is not expected_valid:
                    raise DerivationError(
                        "raw_reference_conditional_clip_fraction_valid is inconsistent "
                        f"with the remaining-mass proxy at {raw_path}:{line_number}"
                    )
                if not math.isclose(
                    small + remaining, 1.0, rel_tol=0.0, abs_tol=1e-9
                ):
                    raise DerivationError(
                        f"raw reference small/remaining mass does not sum to one: "
                        f"{raw_path}:{line_number}"
                    )
                by_step[step] = {
                    "clip": clip,
                    "small": small,
                    "remaining": remaining,
                }
    except OSError as exc:
        raise DerivationError(f"could not read raw telemetry {raw_path}: {exc}") from exc
    expected_steps = list(range(1, 301))
    if sorted(by_step) != expected_steps:
        raise DerivationError(
            f"raw telemetry must cover exactly steps 1..300: {raw_path}"
        )
    rows = [by_step[step] for step in expected_steps]
    clips = [row["clip"] for row in rows]
    small = [row["small"] for row in rows]
    remaining = [row["remaining"] for row in rows]
    return {
        "path": raw_path,
        "sha256": _file_sha256(raw_path),
        "clip": clips,
        "small": small,
        "remaining": remaining,
        "clip_q10": _quantile(clips, TRANSITION_Q),
        "clip_median": statistics.median(clips),
        "clip_q90": _quantile(clips, UPPER_Q),
        "clip_min": min(clips),
        "clip_max": max(clips),
    }


def _candidate_evidence_hashes(
    campaign_root: Path, candidate: Mapping[str, Any]
) -> dict[str, str]:
    result: dict[str, str] = {}
    for key, filename in (
        ("run_status", "run_status.json"),
        ("validation_metrics", "validation_metrics.json"),
        ("split_manifest", "split_manifest.json"),
    ):
        try:
            path = selector._resolve_screen_input(
                campaign_root,
                candidate["runs"][str(SOURCE_SEED)][key],
                filename=filename,
                label=key.replace("_", " "),
            )
        except selector.SelectionError as exc:
            raise DerivationError(str(exc)) from exc
        result[key] = _file_sha256(path)
    return result


def _build_slaclip_candidates(
    c0_values: Sequence[float],
    target_points: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    candidates: list[dict[str, Any]] = []
    mapping: list[dict[str, Any]] = []
    for c0 in c0_values:
        for point in target_points:
            rho = float(point["rho"])
            candidate_id = (
                f"sla-c{_slug(c0)}-r{_slug(rho)}-e{_slug(SLACLIP_ETA)}"
            )
            relative = Path("screen") / "runs" / candidate_id
            runs = {
                str(seed): {
                    "run_status": str(
                        relative / f"seed-{seed}" / "adapter" / "run_status.json"
                    ),
                    "validation_metrics": str(
                        relative
                        / f"seed-{seed}"
                        / "results"
                        / "validation"
                        / "validation_metrics.json"
                    ),
                    "split_manifest": str(
                        relative
                        / f"seed-{seed}"
                        / "results"
                        / "validation"
                        / "split_manifest.json"
                    ),
                }
                for seed in (42, 43, 44)
            }
            params = {
                "dp_max_grad_norm": float(c0),
                "slaclip_target_non_small_clip_fraction": rho,
                "slaclip_eta": SLACLIP_ETA,
                "slaclip_num_slots": SLACLIP_NUM_SLOTS,
                "slaclip_c_min": SLACLIP_C_MIN,
                "slaclip_c_max": SLACLIP_C_MAX,
            }
            candidates.append(
                {
                    "id": candidate_id,
                    "family": "slaclip",
                    "method": "slaclip",
                    "params": params,
                    "runs": runs,
                }
            )
            mapping.append(
                {
                    "candidate_id": candidate_id,
                    "initial_clip_threshold": float(c0),
                    "target_point_index": int(point["index"]),
                    "whole_batch_clip_proxy_target": float(point["p"]),
                    "reference_small_gradient_proxy": float(
                        point["reference_small_gradient_proxy"]
                    ),
                    "target_non_small_clip_fraction": rho,
                    "slaclip_eta": SLACLIP_ETA,
                    "slaclip_num_slots": SLACLIP_NUM_SLOTS,
                    "slaclip_c_min": SLACLIP_C_MIN,
                    "slaclip_c_max": SLACLIP_C_MAX,
                }
            )
    if len(candidates) != 10 or len({item["id"] for item in candidates}) != 10:
        raise DerivationError("derived SlaClip grid must contain ten unique candidates")
    return candidates, mapping


def derive_target_grid(
    *,
    campaign_root: Path,
    fixed_manifest: Path,
    output: Path,
    registry_out: Path,
) -> dict[str, Any]:
    try:
        campaign_root = campaign_root.resolve(strict=True)
    except OSError as exc:
        raise DerivationError(f"campaign root does not exist: {campaign_root}") from exc
    if not campaign_root.is_dir():
        raise DerivationError(f"campaign root is not a directory: {campaign_root}")
    fixed_manifest_path = _resolve_input(
        campaign_root,
        fixed_manifest,
        required_root="screen",
        label="fixed manifest",
    )
    output_path = _resolve_output(
        campaign_root,
        output,
        required_root="selection",
        label="target-grid output",
    )
    registry_path = _resolve_output(
        campaign_root,
        registry_out,
        required_root="screen",
        label="candidate registry output",
        required_name="candidate_registry.json",
    )
    if fixed_manifest_path in {output_path, registry_path} or output_path == registry_path:
        raise DerivationError("fixed manifest, derivation output, and registry must be distinct")

    manifest = _read_json(fixed_manifest_path, label="fixed manifest")
    _require_exact_keys(
        manifest,
        {"schema_version", "selection_protocol", "fixed_candidates"},
        label="fixed manifest",
    )
    if manifest.get("schema_version") != 1:
        raise DerivationError("fixed manifest schema_version must be 1")
    protocol_container = {
        "schema_version": selector.REGISTRY_SCHEMA_VERSION,
        "selection_protocol": manifest.get("selection_protocol"),
        "candidates": [],
    }
    try:
        protocol = selector._validate_protocol(protocol_container)
    except selector.SelectionError as exc:
        raise DerivationError(str(exc)) from exc
    if protocol.get("selection_metric") != selector.NUMERIC_EXACT_METRIC:
        raise DerivationError(
            "fixed target calibration requires public numeric exact accuracy with loss tie-break"
        )
    if int(protocol.get("stage1_seed")) != SOURCE_SEED:
        raise DerivationError("fixed target calibration requires stage1_seed=42")

    raw_candidates = manifest.get("fixed_candidates")
    if not isinstance(raw_candidates, list) or len(raw_candidates) != 8:
        raise DerivationError("fixed manifest must contain exactly eight fixed candidates")
    fixed_candidates = [
        _validate_candidate_shape(item, index=index)
        for index, item in enumerate(raw_candidates)
    ]
    if len({item["id"] for item in fixed_candidates}) != 8:
        raise DerivationError("fixed candidate ids must be unique")
    observed_c = sorted(
        float(item["params"]["dp_max_grad_norm"]) for item in fixed_candidates
    )
    if len(set(observed_c)) != 8 or any(
        not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12)
        for actual, expected in zip(observed_c, FIXED_C_VALUES)
    ):
        raise DerivationError(
            f"fixed C scan must be exactly {list(FIXED_C_VALUES)!r}, got {observed_c!r}"
        )

    evidence: list[dict[str, Any]] = []
    for candidate in fixed_candidates:
        _validate_registered_future_paths(campaign_root, candidate)
        try:
            validated = selector._validate_run(
                campaign_root, candidate, SOURCE_SEED, protocol
            )
        except selector.SelectionError as exc:
            raise DerivationError(str(exc)) from exc
        status_path = selector._resolve_screen_input(
            campaign_root,
            candidate["runs"][str(SOURCE_SEED)]["run_status"],
            filename="run_status.json",
            label="run status",
        )
        status = _read_json(status_path, label="run status")
        config = status.get("config")
        if not isinstance(config, dict):  # also checked by the formal validator
            raise DerivationError(f"candidate {candidate['id']} has no run config")
        if (
            config.get("telemetry_mode") != "research_raw"
            or config.get("allow_non_private_telemetry") is not True
        ):
            raise DerivationError(
                f"candidate {candidate['id']} must explicitly enable NON_PRIVATE research_raw telemetry"
            )
        trajectory = _load_raw_trajectory(
            campaign_root,
            candidate,
            config_fingerprint=status.get("config_fingerprint"),
        )
        evidence.append(
            {
                "candidate": candidate,
                "candidate_id": candidate["id"],
                "C": float(candidate["params"]["dp_max_grad_norm"]),
                "accuracy": float(validated["accuracy"]),
                "loss": float(validated["loss"]),
                "split_sha": str(validated["split_manifest_sha256"]),
                "config_fingerprint": status.get("config_fingerprint"),
                "trajectory": trajectory,
                "evidence_hashes": _candidate_evidence_hashes(
                    campaign_root, candidate
                ),
            }
        )
    split_hashes = {item["split_sha"] for item in evidence}
    if len(split_hashes) != 1:
        raise DerivationError(
            f"all fixed runs must share one public validation split, got {sorted(split_hashes)!r}"
        )
    split_sha = next(iter(split_hashes))

    validation_ranking = sorted(
        evidence,
        key=lambda item: (
            -item["accuracy"],
            item["loss"],
            item["C"],
            item["candidate_id"],
        ),
    )
    best = validation_ranking[0]
    by_c = sorted(evidence, key=lambda item: (item["C"], item["candidate_id"]))
    transition = next(
        (
            item
            for item in by_c
            if item["trajectory"]["clip_q10"] < TRANSITION_THRESHOLD
        ),
        None,
    )
    transition_fallback = False
    if transition is None:
        all_fully_clipped = all(
            math.isclose(value, 1.0, rel_tol=0.0, abs_tol=1e-12)
            for item in by_c
            for value in item["trajectory"]["clip"]
        )
        if all_fully_clipped:
            raise DerivationError(
                "all fixed-C trajectories through C=15 are 100% clipped; "
                "the scan does not identify a target transition"
            )
        transition = by_c[-1]
        transition_fallback = True

    c0_evidence = [best, transition]
    if math.isclose(best["C"], transition["C"], rel_tol=0.0, abs_tol=1e-12):
        replacement = next(
            (
                item
                for item in validation_ranking
                if not math.isclose(
                    item["C"], best["C"], rel_tol=0.0, abs_tol=1e-12
                )
            ),
            None,
        )
        if replacement is None:  # impossible after the exact eight-C check
            raise DerivationError("could not select two distinct initial C values")
        c0_evidence[1] = replacement
    c0_values = [float(item["C"]) for item in c0_evidence]
    if len(set(c0_values)) != 2:
        raise DerivationError("derived initial clipping thresholds must be distinct")

    transition_clips = transition["trajectory"]["clip"]
    lower = max(RHO_MIN, _quantile(transition_clips, TRANSITION_Q))
    upper = min(0.99, _quantile(transition_clips, UPPER_Q))
    expanded_lower = False
    if upper - lower < MIN_TARGET_WIDTH:
        lower = max(RHO_MIN, upper - MIN_TARGET_WIDTH)
        expanded_lower = True
    if not upper > lower:
        raise DerivationError(
            f"transition trajectory cannot define a non-empty target interval: L={lower}, U={upper}"
        )
    z_reference = float(statistics.median(transition["trajectory"]["small"]))
    if not 0.0 <= z_reference < 1.0:
        raise DerivationError(
            f"reference small-gradient proxy must be in [0,1), got {z_reference}"
        )
    denominator = 1.0 - z_reference
    target_points: list[dict[str, Any]] = []
    for index in range(TARGET_COUNT):
        p_value = lower + (upper - lower) * index / (TARGET_COUNT - 1)
        unprojected = p_value / denominator
        rho = max(RHO_MIN, min(RHO_MAX, unprojected))
        target_points.append(
            {
                "index": index,
                "p": float(p_value),
                "reference_small_gradient_proxy": z_reference,
                "unprojected_target_non_small_clip_fraction": float(unprojected),
                "rho": float(rho),
                "hit_lower_bound": bool(unprojected < RHO_MIN),
                "hit_upper_bound": bool(unprojected > RHO_MAX),
            }
        )
    rhos = [float(item["rho"]) for item in target_points]
    if any(not RHO_MIN <= rho <= RHO_MAX for rho in rhos):
        raise DerivationError(f"derived rho is outside [{RHO_MIN},{RHO_MAX}]: {rhos!r}")
    if any(
        not rhos[index + 1] > rhos[index] + 1e-12
        for index in range(len(rhos) - 1)
    ):
        raise DerivationError(
            f"derived rho values must be five strictly unique increasing values: {rhos!r}"
        )

    slaclip_candidates, candidate_mapping = _build_slaclip_candidates(
        c0_values, target_points
    )
    sorted_fixed = [item["candidate"] for item in by_c]
    registry = {
        "schema_version": selector.REGISTRY_SCHEMA_VERSION,
        "selection_protocol": copy.deepcopy(protocol),
        "candidates": [*copy.deepcopy(sorted_fixed), *slaclip_candidates],
    }
    try:
        selector._validate_protocol(registry)
        selector._candidate_map(registry)
    except selector.SelectionError as exc:
        raise DerivationError(f"derived registry violates formal selector schema: {exc}") from exc
    if len(registry["candidates"]) != 18:
        raise DerivationError("derived registry must contain eight fixed and ten SlaClip candidates")
    registry_encoded = _encoded_json(registry)
    registry_file_sha = hashlib.sha256(registry_encoded).hexdigest()

    ranking_rows = []
    for rank, item in enumerate(validation_ranking, 1):
        trajectory = item["trajectory"]
        ranking_rows.append(
            {
                "rank": rank,
                "candidate_id": item["candidate_id"],
                "dp_max_grad_norm": item["C"],
                "public_validation_numeric_exact_accuracy": item["accuracy"],
                "public_validation_response_only_loss": item["loss"],
                "config_fingerprint": item["config_fingerprint"],
                "split_manifest_sha256": item["split_sha"],
                "raw_clip_fraction": {
                    "q10": trajectory["clip_q10"],
                    "median": trajectory["clip_median"],
                    "q90": trajectory["clip_q90"],
                    "min": trajectory["clip_min"],
                    "max": trajectory["clip_max"],
                },
                "raw_telemetry": {
                    "path": str(trajectory["path"].relative_to(campaign_root)),
                    "sha256": trajectory["sha256"],
                    "steps": 300,
                    "NON_PRIVATE": True,
                },
                "registered_evidence_sha256": item["evidence_hashes"],
            }
        )
    artifact: dict[str, Any] = {
        "schema_version": 1,
        "derivation_protocol": {
            "name": "fixed_scan_conditional_full_slaclip_target_grid_v1",
            "source_seed": SOURCE_SEED,
            "fixed_c_values": list(FIXED_C_VALUES),
            "best_C_rule": (
                "descending_public_validation_numeric_exact_accuracy_then_"
                "ascending_response_only_loss_then_C_then_candidate_id"
            ),
            "transition_C_rule": (
                "first_ascending_C_with_q10_raw_clip_fraction_below_0.99;_"
                "otherwise_C15_if_not_fully_censored"
            ),
            "target_interval_rule": (
                "L=max(0.50,q10);_U=min(0.99,q90);_if_width_below_0.04_"
                "set_L=max(0.50,U-0.04)"
            ),
            "target_conversion": "rho=clip(p/(1-median_z_reference),0.5,0.995)",
            "target_conversion_interpretation": (
                "exploratory_fixed_trajectory_calibration_using_one_median_"
                "reference_proxy;_not_stepwise_exact_inverse_replay"
            ),
            "target_count": TARGET_COUNT,
            "slaclip_eta": SLACLIP_ETA,
            "slaclip_num_slots": SLACLIP_NUM_SLOTS,
            "slaclip_c_min": SLACLIP_C_MIN,
            "slaclip_c_max": SLACLIP_C_MAX,
            "raw_telemetry_is_non_private": True,
            "selection_is_data_dependent": True,
        },
        "provenance": {
            "campaign_root": str(campaign_root),
            "fixed_manifest_path": str(fixed_manifest_path.relative_to(campaign_root)),
            "fixed_manifest_file_sha256": _file_sha256(fixed_manifest_path),
            "fixed_manifest_payload_sha256": _payload_sha256(manifest),
            "selection_protocol_sha256": _payload_sha256(protocol),
            "common_split_manifest_sha256": split_sha,
            "candidate_registry_path": str(registry_path.relative_to(campaign_root)),
            "candidate_registry_payload_sha256": _payload_sha256(registry),
            "candidate_registry_file_sha256": registry_file_sha,
        },
        "fixed_validation_ranking": ranking_rows,
        "best_fixed": {
            "candidate_id": best["candidate_id"],
            "dp_max_grad_norm": best["C"],
        },
        "transition_fixed": {
            "candidate_id": transition["candidate_id"],
            "dp_max_grad_norm": transition["C"],
            "q10_raw_clip_fraction": transition["trajectory"]["clip_q10"],
            "q90_raw_clip_fraction": transition["trajectory"]["clip_q90"],
            "used_largest_C_fallback": transition_fallback,
        },
        "selected_initial_clip_thresholds": [
            {
                "role": "best_fixed_public_validation",
                "candidate_id": c0_evidence[0]["candidate_id"],
                "value": c0_values[0],
            },
            {
                "role": (
                    "transition_fixed"
                    if c0_evidence[1]["candidate_id"] == transition["candidate_id"]
                    else "highest_public_validation_rank_distinct_from_best"
                ),
                "candidate_id": c0_evidence[1]["candidate_id"],
                "value": c0_values[1],
            },
        ],
        "target_interval": {
            "lower_whole_batch_clip_proxy": lower,
            "upper_whole_batch_clip_proxy": upper,
            "minimum_width": MIN_TARGET_WIDTH,
            "lower_was_expanded": expanded_lower,
            "reference_small_gradient_proxy_median": z_reference,
            "reference_remaining_mass_proxy": denominator,
        },
        "target_points": target_points,
        "candidate_mapping": candidate_mapping,
        "candidate_counts": {"fixed": 8, "slaclip": 10, "total": 18},
        "NON_PRIVATE_CALIBRATION": True,
    }
    artifact["manifest_sha256"] = _payload_sha256(artifact)
    artifact_encoded = _encoded_json(artifact)

    # Check both targets before creating either so an ordinary immutable-file
    # conflict cannot leave a newly written counterpart behind.
    _assert_immutable_compatible(
        registry_path, registry_encoded, label="candidate registry"
    )
    _assert_immutable_compatible(
        output_path, artifact_encoded, label="target-grid artifact"
    )
    _write_immutable(registry_path, registry_encoded, label="candidate registry")
    _write_immutable(output_path, artifact_encoded, label="target-grid artifact")
    return artifact


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--fixed-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--registry-out", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        artifact = derive_target_grid(
            campaign_root=args.campaign_root,
            fixed_manifest=args.fixed_manifest,
            output=args.output,
            registry_out=args.registry_out,
        )
    except DerivationError as exc:
        parser.error(str(exc))
    print(
        json.dumps(
            {
                "output": str(args.output),
                "registry": str(args.registry_out),
                "manifest_sha256": artifact["manifest_sha256"],
                "rho": [item["rho"] for item in artifact["target_points"]],
            },
            sort_keys=True,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
