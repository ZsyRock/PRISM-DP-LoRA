#!/usr/bin/env python3
"""Pre-register and lock the dense fixed-C follow-up to the 4B refinement.

The helper has two deliberately separate phases.  ``prepare`` validates the
completed source campaign and creates the immutable 15-arm public-validation
plan.  ``lock`` consumes exactly those registered validation runs, ranks the
five pre-declared fixed thresholds, audits the selected SlaClip trajectories,
and creates a five-seed fresh-evaluation plan only when the pre-declared gate
passes.  It never searches for runs and never reads source final results.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SOURCE_EXPERIMENT_SHA = "feba5968285dc1651bf7726327932b3f625ace22"
SELECTION_SEEDS = (42, 43, 44, 45, 46)
FRESH_SEEDS = (191, 223, 257, 293, 331)
DENSE_C_VALUES = (1.5, 1.75, 1.25)
ALL_C_VALUES = (1.0, 1.25, 1.5, 1.75, 2.0)
MIN_MEAN_ACCURACY_DELTA = 0.005
MIN_PAIRED_WINS = 3
EXPECTED_STEPS = 300
EXPECTED_SLOTS = 15
EXPECTED_METRIC = "public_math10k_numeric_exact_match_accuracy"
EXPECTED_LOSS = "response_only_per_record_mean_of_nonignored_next_token_losses"
FULL_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
TOL = 1e-10


class FollowupError(RuntimeError):
    """Raised when the pre-registered follow-up cannot be built safely."""


def _finite(value: Any, *, label: str) -> float:
    if isinstance(value, bool):
        raise FollowupError(f"{label} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise FollowupError(f"{label} must be a finite number") from exc
    if not math.isfinite(result):
        raise FollowupError(f"{label} must be a finite number")
    return result


def _same(left: Any, right: Any, *, tolerance: float = TOL) -> bool:
    try:
        return abs(float(left) - float(right)) <= tolerance
    except (TypeError, ValueError):
        return left == right


def _json_bytes(payload: Any) -> bytes:
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
        raise FollowupError(f"artifact is not finite JSON: {exc}") from exc


def _canonical_json_bytes(payload: Any) -> bytes:
    try:
        return json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise FollowupError(f"artifact is not finite JSON: {exc}") from exc


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _file_sha256(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _payload_sha256(payload: Any) -> str:
    return _sha256_bytes(_canonical_json_bytes(payload))


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FollowupError(f"cannot read {label}: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise FollowupError(f"{label} must be a JSON object: {path}")
    return payload


def _write_immutable(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if not path.is_file() or path.read_bytes() != data:
            raise FollowupError(f"refusing to overwrite immutable artifact: {path}")
        return
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if not path.is_file() or path.read_bytes() != data:
                raise FollowupError(
                    f"concurrent writer created inconsistent artifact: {path}"
                )
    finally:
        temporary.unlink(missing_ok=True)
    os.chmod(path, 0o600)


def _write_with_sha256(path: Path, data: bytes) -> None:
    _write_immutable(path, data)
    digest = _sha256_bytes(data)
    sidecar = path.with_name(path.name + ".sha256")
    _write_immutable(sidecar, f"{digest}  {path.name}\n".encode("ascii"))


def _inside(root: Path, value: Path, *, label: str, must_exist: bool = True) -> Path:
    candidate = value if value.is_absolute() else root / value
    try:
        resolved = candidate.resolve(strict=must_exist)
        resolved.relative_to(root)
    except (OSError, ValueError) as exc:
        raise FollowupError(f"{label} must remain inside {root}: {value}") from exc
    return resolved


def _slug(value: float) -> str:
    return format(float(value), ".12g").replace("-", "m").replace(".", "p")


def _candidate_id(value: float) -> str:
    return f"fixed-c{_slug(value)}"


def _parse_plan(path: Path, *, root: Path, label: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise FollowupError(f"cannot read {label}: {path}: {exc}") from exc
    for line_number, line in enumerate(lines, 1):
        if not line:
            continue
        fields = line.split("|")
        if len(fields) != 11:
            raise FollowupError(f"{label}:{line_number} must have exactly 11 fields")
        if any("\n" in item or "\r" in item for item in fields):
            raise FollowupError(f"{label}:{line_number} has an invalid delimiter")
        try:
            seed = int(fields[4])
        except ValueError as exc:
            raise FollowupError(f"{label}:{line_number} has an invalid seed") from exc
        records.append(
            {
                "kind": fields[0],
                "phase": fields[1],
                "candidate_id": fields[2],
                "role": fields[3],
                "seed": seed,
                "method": fields[5],
                "clip": _finite(fields[6], label=f"{label}:{line_number}:clip"),
                "rho": fields[7],
                "eta": fields[8],
                "schedule": fields[9],
                "arm_root": _inside(
                    root,
                    Path(fields[10]),
                    label=f"{label}:{line_number}:arm_root",
                    must_exist=False,
                ),
            }
        )
    return records


def _candidate_map(registry: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    values = registry.get("candidates")
    if not isinstance(values, list) or not values:
        raise FollowupError("source candidate registry has no candidates")
    result: dict[str, dict[str, Any]] = {}
    for candidate in values:
        if not isinstance(candidate, dict) or not isinstance(candidate.get("id"), str):
            raise FollowupError("source candidate registry has an invalid candidate")
        identifier = candidate["id"]
        if identifier in result:
            raise FollowupError(f"duplicate source candidate: {identifier}")
        result[identifier] = candidate
    return result


def _candidate_clip(candidate: Mapping[str, Any]) -> float:
    params = candidate.get("params")
    if not isinstance(params, dict):
        raise FollowupError(f"candidate {candidate.get('id')} has no params")
    return _finite(params.get("dp_max_grad_norm"), label="candidate clip threshold")


def _validate_candidate_plan_fields(
    record: Mapping[str, Any], candidate: Mapping[str, Any], *, label: str
) -> None:
    if record["kind"] != "train" or record["candidate_id"] != candidate["id"]:
        raise FollowupError(f"{label} does not identify the registered training candidate")
    if record["method"] != candidate.get("method") or not _same(
        record["clip"], _candidate_clip(candidate)
    ):
        raise FollowupError(f"{label} method/clip differs from the candidate registry")
    params = candidate["params"]
    expected_rho = params.get(
        "slaclip_target_non_small_clip_fraction", params.get("slaclip_beta", "NA")
    )
    expected_eta = params.get("slaclip_eta", "NA")
    for supplied, expected, name in (
        (record["rho"], expected_rho, "rho"),
        (record["eta"], expected_eta, "eta"),
    ):
        if expected == "NA":
            if supplied != "NA":
                raise FollowupError(f"{label} has unexpected {name}")
        elif not _same(supplied, expected):
            raise FollowupError(f"{label} has the wrong {name}")
    if record["schedule"] != "NA":
        raise FollowupError(f"{label} unexpectedly uses a replay schedule")


def _load_source(
    *, source_root: Path, campaign_root: Path, experiment_code_sha: str
) -> dict[str, Any]:
    if experiment_code_sha != SOURCE_EXPERIMENT_SHA or not FULL_SHA_RE.fullmatch(
        experiment_code_sha
    ):
        raise FollowupError(
            "experiment_code_sha must equal the frozen source mechanism commit "
            f"{SOURCE_EXPERIMENT_SHA}"
        )
    source_root = source_root.resolve(strict=True)
    campaign_root = campaign_root.resolve(strict=False)
    if source_root == campaign_root:
        raise FollowupError("follow-up campaign root must be independent of source root")
    try:
        campaign_root.relative_to(source_root)
    except ValueError:
        pass
    else:
        raise FollowupError("follow-up campaign root cannot be nested in source root")

    paths = {
        "registry": source_root / "screen" / "candidate_registry.json",
        "selection": source_root / "selection" / "selection.json",
        "stage2_plan": source_root / "plans" / "stage2.tsv",
        "final_plan": source_root / "plans" / "final.tsv",
    }
    for label, path in paths.items():
        _inside(source_root, path, label=f"source {label}")
    registry = _read_json(paths["registry"], label="source candidate registry")
    selection = _read_json(paths["selection"], label="source locked selection")
    protocol = registry.get("selection_protocol")
    if not isinstance(protocol, dict):
        raise FollowupError("source registry has no selection protocol")
    common = protocol.get("common_config")
    if not isinstance(common, dict):
        raise FollowupError("source registry has no common configuration")
    if common.get("implementation_git_sha") != SOURCE_EXPERIMENT_SHA:
        raise FollowupError("source registry does not use the frozen experiment SHA")
    if protocol.get("stage2_seeds") != list(SELECTION_SEEDS):
        raise FollowupError("source registry selection seeds are not 42..46")
    if protocol.get("stage1_seed") != SELECTION_SEEDS[0]:
        raise FollowupError("source registry stage1 seed is not 42")
    if protocol.get("selection_metric") != EXPECTED_METRIC:
        raise FollowupError("source registry uses the wrong selection metric")
    if protocol.get("loss_definition") != EXPECTED_LOSS:
        raise FollowupError("source registry uses the wrong validation loss")
    if protocol.get("required_update_steps") != EXPECTED_STEPS:
        raise FollowupError("source registry does not require 300 update steps")

    candidates = _candidate_map(registry)
    fixed_by_c: dict[float, dict[str, Any]] = {}
    for candidate in candidates.values():
        if candidate.get("family") == "fixed" and candidate.get("method") == "baseline":
            clip = _candidate_clip(candidate)
            if any(_same(clip, wanted) for wanted in (1.0, 2.0)):
                if clip in fixed_by_c:
                    raise FollowupError(f"duplicate source fixed C={clip:g} candidate")
                fixed_by_c[clip] = candidate
    if set(fixed_by_c) != {1.0, 2.0}:
        raise FollowupError("source registry must contain unique fixed C=1 and C=2")

    if selection.get("stage") != "stage2":
        raise FollowupError("source selection is not a locked stage2 selection")
    if selection.get("required_seeds") != list(SELECTION_SEEDS):
        raise FollowupError("source locked selection seeds are not 42..46")
    if selection.get("registry_sha256") != _payload_sha256(registry):
        raise FollowupError("source selection does not match its candidate registry")
    if selection.get("selection_protocol") != protocol:
        raise FollowupError("source locked selection protocol differs from registry")
    if selection.get("selection_protocol_sha256") != _payload_sha256(protocol):
        raise FollowupError("source locked selection protocol hash is invalid")
    ranking = selection.get("ranking")
    if not isinstance(ranking, list) or len(ranking) != 5:
        raise FollowupError("source locked selection must rank exactly five candidates")
    ranking_ids = [item.get("candidate_id") for item in ranking if isinstance(item, dict)]
    if len(ranking_ids) != 5 or len(set(ranking_ids)) != 5:
        raise FollowupError("source locked selection ranking identities are invalid")
    if any(identifier not in candidates for identifier in ranking_ids):
        raise FollowupError("source locked selection references an unknown candidate")
    selected_summary = selection.get("selected_slaclip")
    if not isinstance(selected_summary, dict):
        raise FollowupError("source selection has no selected SlaClip candidate")
    selected_id = selected_summary.get("candidate_id")
    selected = candidates.get(str(selected_id))
    if (
        selected is None
        or selected.get("family") != "slaclip"
        or selected.get("method") != "slaclip"
        or selected_id not in ranking_ids
    ):
        raise FollowupError("source selected SlaClip identity is invalid")

    stage2 = _parse_plan(paths["stage2_plan"], root=source_root, label="source stage2 plan")
    expected_stage2 = {(identifier, seed) for seed in SELECTION_SEEDS[1:] for identifier in ranking_ids}
    if len(stage2) != 20 or {(row["candidate_id"], row["seed"]) for row in stage2} != expected_stage2:
        raise FollowupError("source stage2 plan is not the locked 5-candidate x 4-seed plan")
    for row in stage2:
        candidate = candidates[row["candidate_id"]]
        _validate_candidate_plan_fields(row, candidate, label="source stage2 record")
        if row["phase"] != "stage2":
            raise FollowupError("source stage2 plan has the wrong phase")
        expected_root = source_root / "screen" / "runs" / row["candidate_id"] / f"seed-{row['seed']}"
        if row["arm_root"] != expected_root.resolve(strict=False):
            raise FollowupError("source stage2 arm path is inconsistent")

    final = _parse_plan(paths["final_plan"], root=source_root, label="source final plan")
    fixed_c2 = fixed_by_c[2.0]
    expected_final = {(str(selected_id), seed) for seed in FRESH_SEEDS} | {
        (str(fixed_c2["id"]), seed) for seed in FRESH_SEEDS
    }
    if len(final) != 10 or {(row["candidate_id"], row["seed"]) for row in final} != expected_final:
        raise FollowupError("source final plan is not the locked paired five-seed plan")
    for row in final:
        candidate = candidates[row["candidate_id"]]
        _validate_candidate_plan_fields(row, candidate, label="source final record")
        if row["phase"] != "final":
            raise FollowupError("source final plan has the wrong phase")

    return {
        "source_root": source_root,
        "campaign_root": campaign_root,
        "paths": paths,
        "registry": registry,
        "selection": selection,
        "protocol": protocol,
        "common": common,
        "candidates": candidates,
        "fixed_by_c": fixed_by_c,
        "selected": selected,
        "selected_summary": selected_summary,
        "ranking_ids": ranking_ids,
        "source_final": final,
    }


def _plan_line(
    *, phase: str, candidate_id: str, role: str, seed: int, clip: float, arm_root: Path
) -> str:
    values = (
        "train",
        phase,
        candidate_id,
        role,
        str(seed),
        "baseline",
        format(clip, ".12g"),
        "NA",
        "NA",
        "NA",
        str(arm_root),
    )
    if any("|" in value or "\n" in value or "\r" in value for value in values):
        raise FollowupError("plan field contains a forbidden delimiter")
    return "|".join(values)


def _dense_plan(context: Mapping[str, Any]) -> tuple[bytes, list[dict[str, Any]]]:
    root = context["campaign_root"]
    lines: list[str] = []
    runs: list[dict[str, Any]] = []
    for clip in DENSE_C_VALUES:
        identifier = _candidate_id(clip)
        for seed in SELECTION_SEEDS:
            arm_root = root / "selection" / "runs" / identifier / f"seed-{seed}"
            lines.append(
                _plan_line(
                    phase="dense-selection",
                    candidate_id=identifier,
                    role="dense-fixed-selection",
                    seed=seed,
                    clip=clip,
                    arm_root=arm_root,
                )
            )
            runs.append(
                {
                    "candidate_id": identifier,
                    "clip_threshold": clip,
                    "seed": seed,
                    "arm_root": str(arm_root.relative_to(root)),
                    "run_status": str(
                        (arm_root / "adapter" / "run_status.json").relative_to(root)
                    ),
                    "validation_metrics": str(
                        (
                            arm_root
                            / "results"
                            / "validation"
                            / "validation_metrics.json"
                        ).relative_to(root)
                    ),
                    "split_manifest": str(
                        (
                            arm_root / "results" / "validation" / "split_manifest.json"
                        ).relative_to(root)
                    ),
                    "train_log": str(
                        (arm_root / "adapter" / "train_log.jsonl").relative_to(root)
                    ),
                }
            )
    return ("\n".join(lines) + "\n").encode("utf-8"), runs


def _prepare_payload(context: Mapping[str, Any], plan_bytes: bytes, runs: Sequence[dict[str, Any]]) -> dict[str, Any]:
    source_root: Path = context["source_root"]
    paths: Mapping[str, Path] = context["paths"]
    selected: Mapping[str, Any] = context["selected"]
    return {
        "schema_version": 1,
        "phase": "prepared",
        "protocol_name": "math10k_4b_dense_fixed_selection_and_gated_fresh_control_v1",
        "experiment_code_sha": SOURCE_EXPERIMENT_SHA,
        "source_campaign": {
            "root": str(source_root),
            "candidate_registry": {
                "path": str(paths["registry"].relative_to(source_root)),
                "file_sha256": _file_sha256(paths["registry"]),
                "payload_sha256": _payload_sha256(context["registry"]),
            },
            "locked_selection": {
                "path": str(paths["selection"].relative_to(source_root)),
                "file_sha256": _file_sha256(paths["selection"]),
                "payload_sha256": _payload_sha256(context["selection"]),
            },
            "stage2_plan": {
                "path": str(paths["stage2_plan"].relative_to(source_root)),
                "file_sha256": _file_sha256(paths["stage2_plan"]),
            },
            "final_plan": {
                "path": str(paths["final_plan"].relative_to(source_root)),
                "file_sha256": _file_sha256(paths["final_plan"]),
                "provenance_only": True,
                "final_result_files_read_for_selection": False,
            },
        },
        "selection_design": {
            "validation_data_is_public": True,
            "metric": EXPECTED_METRIC,
            "loss_definition": EXPECTED_LOSS,
            "selection_seeds": list(SELECTION_SEEDS),
            "dense_clip_threshold_order": list(DENSE_C_VALUES),
            "complete_fixed_clip_grid": list(ALL_C_VALUES),
            "dense_run_count": 15,
            "ranking_rule": (
                "descending_arithmetic_mean_public_validation_numeric_exact_accuracy_then_"
                "ascending_arithmetic_mean_response_only_validation_loss_then_ascending_C"
            ),
            "stability_rule": (
                "all candidates use the same five paired seeds; variance is descriptive only; "
                "ascending C is the deterministic conservative final tie-break"
            ),
            "fresh_seeds": list(FRESH_SEEDS),
            "selected_slaclip_candidate_id": selected["id"],
            "gate": {
                "minimum_selected_slaclip_minus_dense_best_mean_accuracy": MIN_MEAN_ACCURACY_DELTA,
                "minimum_strict_paired_seed_wins": MIN_PAIRED_WINS,
                "require_selected_slaclip_trajectory_and_log_compliance": True,
                "on_pass": "run dense-best fixed threshold on all five fresh seeds",
                "on_fail": "write an empty dense-final plan",
            },
            "data_access_policy": (
                "lock reads only registered public-validation artifacts and selected-SlaClip "
                "training telemetry; source and follow-up final evaluation results are forbidden"
            ),
        },
        "dense_selection_plan": {
            "path": "plans/dense-selection.tsv",
            "file_sha256": _sha256_bytes(plan_bytes),
            "records": list(runs),
        },
        "privacy_and_release": {
            "per_training_run_target": {"epsilon": 6.0, "delta": 1e-5},
            "new_selection_runs": 15,
            "new_selection_basic_composition_upper_bound": {
                "epsilon": 90.0,
                "delta": 1.5e-4,
                "release_count": 15,
            },
            "maximum_conditional_new_final_runs": 5,
            "maximum_followup_basic_composition_upper_bound": {
                "epsilon": 120.0,
                "delta": 2e-4,
                "release_count": 20,
            },
            "selection_is_conditioned_on_prior_source_campaign_outputs": True,
            "research_raw_telemetry": "NON_PRIVATE",
            "development_bundle_is_non_private": True,
        },
    }


def prepare(*, source_root: Path, campaign_root: Path, experiment_code_sha: str) -> dict[str, Any]:
    context = _load_source(
        source_root=source_root,
        campaign_root=campaign_root,
        experiment_code_sha=experiment_code_sha,
    )
    context["campaign_root"].mkdir(parents=True, exist_ok=True)
    plan_bytes, runs = _dense_plan(context)
    payload = _prepare_payload(context, plan_bytes, runs)
    _write_with_sha256(context["campaign_root"] / "plans" / "dense-selection.tsv", plan_bytes)
    _write_with_sha256(
        context["campaign_root"] / "selection" / "dense-fixed-protocol.json",
        _json_bytes(payload),
    )
    return payload


def _candidate_run_paths(
    *, context: Mapping[str, Any], candidate: Mapping[str, Any], seed: int
) -> dict[str, Path]:
    runs = candidate.get("runs")
    run = runs.get(str(seed)) if isinstance(runs, dict) else None
    if not isinstance(run, dict):
        raise FollowupError(f"source candidate {candidate.get('id')} lacks seed {seed}")
    root: Path = context["source_root"]
    result: dict[str, Path] = {}
    for key in ("run_status", "validation_metrics", "split_manifest"):
        supplied = run.get(key)
        if not isinstance(supplied, str):
            raise FollowupError(f"source candidate {candidate.get('id')} lacks {key}")
        result[key] = _inside(root, Path(supplied), label=f"source candidate {key}")
    result["train_log"] = result["run_status"].parent / "train_log.jsonl"
    return result


def _dense_run_paths(root: Path, clip: float, seed: int) -> dict[str, Path]:
    arm = root / "selection" / "runs" / _candidate_id(clip) / f"seed-{seed}"
    return {
        "run_status": arm / "adapter" / "run_status.json",
        "validation_metrics": arm / "results" / "validation" / "validation_metrics.json",
        "split_manifest": arm / "results" / "validation" / "split_manifest.json",
        "train_log": arm / "adapter" / "train_log.jsonl",
    }


def _validate_common_config(config: Mapping[str, Any], common: Mapping[str, Any], *, label: str) -> None:
    for key, expected in common.items():
        if key not in config or not _same(config[key], expected):
            raise FollowupError(f"{label} config differs from source common_config at {key}")


def _read_log(path: Path, *, label: str) -> tuple[list[dict[str, Any]], str]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise FollowupError(f"cannot read {label}: {path}: {exc}") from exc
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(raw.splitlines(), 1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise FollowupError(f"{label}:{line_number} is invalid JSON") from exc
        if not isinstance(row, dict):
            raise FollowupError(f"{label}:{line_number} is not an object")
        rows.append(row)
    if len(rows) != EXPECTED_STEPS:
        raise FollowupError(f"{label} must contain exactly {EXPECTED_STEPS} rows")
    return rows, _sha256_bytes(raw)


def _validate_run(
    *,
    paths: Mapping[str, Path],
    common: Mapping[str, Any],
    method: str,
    clip: float,
    seed: int,
    params: Mapping[str, Any],
    expected_split_sha: str,
) -> dict[str, Any]:
    status = _read_json(paths["run_status"], label="registered run status")
    metrics = _read_json(paths["validation_metrics"], label="registered validation metrics")
    split = _read_json(paths["split_manifest"], label="registered split manifest")
    if status.get("state") != "completed" or status.get("update_steps") != EXPECTED_STEPS:
        raise FollowupError(f"registered run seed={seed} is not completed at 300 steps")
    config = status.get("config")
    if not isinstance(config, dict):
        raise FollowupError(f"registered run seed={seed} has no config")
    _validate_common_config(config, common, label=f"run seed={seed}")
    if config.get("implementation_git_sha") != SOURCE_EXPERIMENT_SHA:
        raise FollowupError(f"registered run seed={seed} uses the wrong experiment SHA")
    if config.get("method") != method or status.get("method") != method:
        raise FollowupError(f"registered run seed={seed} uses the wrong method")
    if config.get("seed") != seed or not _same(config.get("dp_max_grad_norm"), clip):
        raise FollowupError(f"registered run seed={seed} has wrong seed/clip")
    for key, expected in params.items():
        if key not in config or not _same(config[key], expected):
            raise FollowupError(f"registered run seed={seed} differs at {key}")
    accounting = status.get("privacy_accounting")
    if not isinstance(accounting, dict) or accounting.get("completed_update_steps") != EXPECTED_STEPS:
        raise FollowupError(f"registered run seed={seed} has invalid privacy accounting")
    epsilon = _finite(accounting.get("epsilon_spent"), label="epsilon_spent")
    if abs(epsilon - float(common["dp_epsilon"])) > 0.02:
        raise FollowupError(f"registered run seed={seed} epsilon is outside tolerance")
    if metrics.get("selection_metric") != EXPECTED_METRIC:
        raise FollowupError(f"registered run seed={seed} uses the wrong selection metric")
    if metrics.get("loss_definition") != EXPECTED_LOSS:
        raise FollowupError(f"registered run seed={seed} uses the wrong validation loss")
    if metrics.get("validation_data_is_public") is not True or metrics.get("PUBLIC_VALIDATION_DATA") is not True:
        raise FollowupError(f"registered run seed={seed} is not marked public validation")
    if metrics.get("protocol_stage") != "selection" or metrics.get("records") != 500:
        raise FollowupError(f"registered run seed={seed} has the wrong validation protocol")
    accuracy = _finite(metrics.get("numeric_exact_accuracy"), label="validation accuracy")
    loss = _finite(metrics.get("loss_mean"), label="validation loss")
    if not 0.0 <= accuracy <= 1.0 or loss < 0.0:
        raise FollowupError(f"registered run seed={seed} has invalid validation metrics")
    if metrics.get("manifest_sha256") != expected_split_sha:
        raise FollowupError(f"registered run seed={seed} uses a different validation split")
    if _payload_sha256(split) != expected_split_sha:
        raise FollowupError(f"registered run seed={seed} split manifest hash is invalid")
    embedded = status.get("validation")
    if not isinstance(embedded, dict) or not _same(embedded.get("numeric_exact_accuracy"), accuracy) or not _same(
        embedded.get("loss_mean"), loss
    ):
        raise FollowupError(f"registered run seed={seed} status/metrics disagree")
    return {
        "seed": seed,
        "accuracy": accuracy,
        "loss": loss,
        "run_status_sha256": _file_sha256(paths["run_status"]),
        "validation_metrics_sha256": _file_sha256(paths["validation_metrics"]),
        "split_manifest_sha256": _file_sha256(paths["split_manifest"]),
        "config_fingerprint": status.get("config_fingerprint"),
        "run_id": status.get("run_id"),
        "train_log_path": paths["train_log"],
    }


def _validate_fixed_log(run: Mapping[str, Any], *, clip: float, seed: int) -> dict[str, Any]:
    rows, digest = _read_log(run["train_log_path"], label=f"fixed train log seed={seed}")
    fingerprint = run["config_fingerprint"]
    for expected_step, row in enumerate(rows, 1):
        if row.get("step") != expected_step or row.get("method") != "baseline":
            raise FollowupError(f"fixed train log seed={seed} has invalid step/method")
        if row.get("config_fingerprint") != fingerprint:
            raise FollowupError(f"fixed train log seed={seed} fingerprint mismatch")
        for key in ("dp_clip_threshold", "dp_next_clip_threshold"):
            if not _same(row.get(key), clip):
                raise FollowupError(f"fixed train log seed={seed} changed {key}")
    return {"train_log_sha256": digest, "rows": len(rows), "constant_clip_threshold": clip}


def _audit_slaclip_logs(
    *, runs: Sequence[Mapping[str, Any]], params: Mapping[str, Any]
) -> dict[str, Any]:
    c0 = _finite(params.get("dp_max_grad_norm"), label="selected SlaClip C0")
    c_min = _finite(params.get("slaclip_c_min"), label="selected SlaClip C_min")
    c_max = _finite(params.get("slaclip_c_max"), label="selected SlaClip C_max")
    target = _finite(
        params.get("slaclip_target_non_small_clip_fraction"), label="selected SlaClip target"
    )
    eta = _finite(params.get("slaclip_eta"), label="selected SlaClip eta")
    slots = params.get("slaclip_num_slots")
    if slots != EXPECTED_SLOTS or not (0.0 < c_min <= c0 <= c_max):
        raise FollowupError("selected SlaClip parameters/bounds are invalid")
    audits: list[dict[str, Any]] = []
    reasons: list[str] = []
    for run in runs:
        seed = int(run["seed"])
        try:
            rows, digest = _read_log(
                run["train_log_path"], label=f"selected SlaClip train log seed={seed}"
            )
            previous_next: float | None = None
            min_seen = math.inf
            max_seen = -math.inf
            for expected_step, row in enumerate(rows, 1):
                prefix = f"selected SlaClip train log seed={seed} step={expected_step}"
                if row.get("step") != expected_step or row.get("method") != "slaclip":
                    raise FollowupError(f"{prefix} has invalid step/method")
                if row.get("config_fingerprint") != run["config_fingerprint"]:
                    raise FollowupError(f"{prefix} fingerprint mismatch")
                if row.get("slaclip_num_slots") != slots:
                    raise FollowupError(f"{prefix} K mismatch")
                for key, expected in (
                    ("slaclip_c_min", c_min),
                    ("slaclip_c_max", c_max),
                    ("slaclip_eta", eta),
                    ("slaclip_target_non_small_clip_fraction", target),
                ):
                    if not _same(row.get(key), expected):
                        raise FollowupError(f"{prefix} {key} mismatch")
                current = _finite(row.get("dp_clip_threshold"), label=f"{prefix} C_t")
                next_clip = _finite(row.get("dp_next_clip_threshold"), label=f"{prefix} C_next")
                _finite(row.get("slaclip_controller_error"), label=f"{prefix} controller error")
                indicators = row.get("slack_indicator")
                if not isinstance(indicators, list) or len(indicators) != slots:
                    raise FollowupError(f"{prefix} does not contain K slack indicators")
                for index, value in enumerate(indicators):
                    _finite(value, label=f"{prefix} slack_indicator[{index}]")
                if current < c_min - TOL or current > c_max + TOL:
                    raise FollowupError(f"{prefix} C_t is outside [C_min,C_max]")
                if next_clip < c_min - TOL or next_clip > c_max + TOL:
                    raise FollowupError(f"{prefix} C_next is outside [C_min,C_max]")
                if expected_step == 1 and not _same(current, c0):
                    raise FollowupError(f"{prefix} does not start at selected C0")
                if previous_next is not None and not _same(current, previous_next):
                    raise FollowupError(f"{prefix} breaks C_t trajectory continuity")
                previous_next = next_clip
                min_seen = min(min_seen, current, next_clip)
                max_seen = max(max_seen, current, next_clip)
            audits.append(
                {
                    "seed": seed,
                    "compliant": True,
                    "rows": len(rows),
                    "train_log_sha256": digest,
                    "minimum_clip_threshold": min_seen,
                    "maximum_clip_threshold": max_seen,
                    "bounds": [c_min, c_max],
                }
            )
        except FollowupError as exc:
            reasons.append(str(exc))
            audits.append({"seed": seed, "compliant": False, "reason": str(exc)})
    return {
        "compliant": not reasons and len(audits) == len(SELECTION_SEEDS),
        "required_rows_per_seed": EXPECTED_STEPS,
        "required_seeds": list(SELECTION_SEEDS),
        "checks": (
            "step/fingerprint/controller fields; K=15; finite slack indicators; "
            "C0 start; per-step C continuity; every C_t and C_next inside selected bounds"
        ),
        "reasons": reasons,
        "seed_audits": audits,
    }


def _summary(candidate_id: str, clip: float, origin: str, runs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    accuracies = [_finite(run["accuracy"], label="accuracy") for run in runs]
    losses = [_finite(run["loss"], label="loss") for run in runs]
    mean_accuracy = math.fsum(accuracies) / len(accuracies)
    mean_loss = math.fsum(losses) / len(losses)
    mean_square = math.fsum((value - mean_accuracy) ** 2 for value in accuracies) / len(accuracies)
    return {
        "candidate_id": candidate_id,
        "method": "baseline",
        "clip_threshold": clip,
        "origin": origin,
        "seeds": list(SELECTION_SEEDS),
        "mean_validation_accuracy": mean_accuracy,
        "mean_validation_loss": mean_loss,
        "validation_accuracy_population_std": math.sqrt(mean_square),
        "per_seed": [
            {
                key: value
                for key, value in run.items()
                if key != "train_log_path"
            }
            for run in runs
        ],
    }


def _expected_split_sha(context: Mapping[str, Any]) -> str:
    value = context["selection"].get("split_manifest_sha256")
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise FollowupError("source selection has an invalid split manifest hash")
    return value


def _validate_prepare_lock(context: Mapping[str, Any]) -> tuple[dict[str, Any], bytes]:
    root: Path = context["campaign_root"]
    plan_bytes, runs = _dense_plan(context)
    expected = _prepare_payload(context, plan_bytes, runs)
    protocol_path = root / "selection" / "dense-fixed-protocol.json"
    plan_path = root / "plans" / "dense-selection.tsv"
    protocol = _read_json(protocol_path, label="dense fixed protocol")
    if protocol != expected:
        raise FollowupError("dense fixed protocol differs from pre-registered design")
    if plan_path.read_bytes() != plan_bytes:
        raise FollowupError("dense selection plan differs from pre-registered design")
    for path in (protocol_path, plan_path):
        sidecar = path.with_name(path.name + ".sha256")
        expected_sidecar = f"{_file_sha256(path)}  {path.name}\n"
        if sidecar.read_text(encoding="ascii") != expected_sidecar:
            raise FollowupError(f"invalid SHA256 sidecar: {sidecar}")
    return protocol, plan_bytes


def lock(*, source_root: Path, campaign_root: Path, experiment_code_sha: str) -> dict[str, Any]:
    context = _load_source(
        source_root=source_root,
        campaign_root=campaign_root,
        experiment_code_sha=experiment_code_sha,
    )
    protocol, dense_plan_bytes = _validate_prepare_lock(context)
    common: Mapping[str, Any] = context["common"]
    expected_split = _expected_split_sha(context)
    summaries: list[dict[str, Any]] = []
    fixed_runs_by_c: dict[float, list[dict[str, Any]]] = {}

    for clip in (1.0, 2.0):
        candidate = context["fixed_by_c"][clip]
        runs: list[dict[str, Any]] = []
        for seed in SELECTION_SEEDS:
            run = _validate_run(
                paths=_candidate_run_paths(context=context, candidate=candidate, seed=seed),
                common=common,
                method="baseline",
                clip=clip,
                seed=seed,
                params={"dp_max_grad_norm": clip},
                expected_split_sha=expected_split,
            )
            run.update(_validate_fixed_log(run, clip=clip, seed=seed))
            runs.append(run)
        fixed_runs_by_c[clip] = runs
        summaries.append(_summary(candidate["id"], clip, "source-campaign", runs))

    for clip in DENSE_C_VALUES:
        runs = []
        for seed in SELECTION_SEEDS:
            run = _validate_run(
                paths=_dense_run_paths(context["campaign_root"], clip, seed),
                common=common,
                method="baseline",
                clip=clip,
                seed=seed,
                params={"dp_max_grad_norm": clip},
                expected_split_sha=expected_split,
            )
            run.update(_validate_fixed_log(run, clip=clip, seed=seed))
            runs.append(run)
        fixed_runs_by_c[clip] = runs
        summaries.append(_summary(_candidate_id(clip), clip, "dense-followup", runs))

    ranking = sorted(
        summaries,
        key=lambda item: (
            -item["mean_validation_accuracy"],
            item["mean_validation_loss"],
            item["clip_threshold"],
        ),
    )
    for rank, item in enumerate(ranking, 1):
        item["rank"] = rank
    best = ranking[0]

    selected = context["selected"]
    selected_params = selected["params"]
    selected_runs: list[dict[str, Any]] = []
    for seed in SELECTION_SEEDS:
        selected_runs.append(
            _validate_run(
                paths=_candidate_run_paths(context=context, candidate=selected, seed=seed),
                common=common,
                method="slaclip",
                clip=_candidate_clip(selected),
                seed=seed,
                params=selected_params,
                expected_split_sha=expected_split,
            )
        )
    trajectory = _audit_slaclip_logs(runs=selected_runs, params=selected_params)
    best_runs = {int(run["seed"]): run for run in fixed_runs_by_c[best["clip_threshold"]]}
    selected_by_seed = {int(run["seed"]): run for run in selected_runs}
    pairings = []
    for seed in SELECTION_SEEDS:
        sla = selected_by_seed[seed]
        fixed = best_runs[seed]
        delta = sla["accuracy"] - fixed["accuracy"]
        pairings.append(
            {
                "seed": seed,
                "selected_slaclip_accuracy": sla["accuracy"],
                "dense_best_fixed_accuracy": fixed["accuracy"],
                "accuracy_delta_slaclip_minus_fixed": delta,
                "strict_slaclip_win": delta > TOL,
                "selected_slaclip_validation_metrics_sha256": sla["validation_metrics_sha256"],
                "dense_best_fixed_validation_metrics_sha256": fixed["validation_metrics_sha256"],
            }
        )
    mean_delta = math.fsum(item["accuracy_delta_slaclip_minus_fixed"] for item in pairings) / len(pairings)
    wins = sum(bool(item["strict_slaclip_win"]) for item in pairings)
    reasons: list[str] = []
    if mean_delta + TOL < MIN_MEAN_ACCURACY_DELTA:
        reasons.append(
            f"mean accuracy delta {mean_delta:.12g} is below {MIN_MEAN_ACCURACY_DELTA:.12g}"
        )
    if wins < MIN_PAIRED_WINS:
        reasons.append(f"strict paired wins {wins}/5 is below {MIN_PAIRED_WINS}/5")
    if not trajectory["compliant"]:
        reasons.append("selected SlaClip trajectory/log compliance audit failed")
    passed = not reasons

    model_id = str(common["base_model"])
    model_slug = model_id.rstrip("/").rsplit("/", 1)[-1]
    reuse_source_fixed_c2 = passed and _same(best["clip_threshold"], 2.0)
    final_lines: list[str] = []
    if passed and not reuse_source_fixed_c2:
        for seed in FRESH_SEEDS:
            arm = context["campaign_root"] / "final" / model_slug / f"seed-{seed}" / "dense-best-fixed"
            final_lines.append(
                _plan_line(
                    phase="dense-final",
                    candidate_id=best["candidate_id"],
                    role="dense-best-fixed",
                    seed=seed,
                    clip=best["clip_threshold"],
                    arm_root=arm,
                )
            )
    final_bytes = (("\n".join(final_lines) + "\n") if final_lines else "").encode("utf-8")
    payload = {
        "schema_version": 1,
        "phase": "locked",
        "protocol_name": protocol["protocol_name"],
        "experiment_code_sha": SOURCE_EXPERIMENT_SHA,
        "selection_evidence_class": "public_validation_only",
        "source_locked_selection_sha256": _file_sha256(context["paths"]["selection"]),
        "dense_fixed_protocol_sha256": _file_sha256(
            context["campaign_root"] / "selection" / "dense-fixed-protocol.json"
        ),
        "dense_selection_plan_sha256": _sha256_bytes(dense_plan_bytes),
        "ranking_rule": protocol["selection_design"]["ranking_rule"],
        "stability_rule": protocol["selection_design"]["stability_rule"],
        "required_seeds": list(SELECTION_SEEDS),
        "fixed_grid_ranking": ranking,
        "dense_best_fixed": best,
        "selected_slaclip": {
            "candidate_id": selected["id"],
            "params": selected_params,
            "mean_validation_accuracy": math.fsum(run["accuracy"] for run in selected_runs) / len(selected_runs),
            "mean_validation_loss": math.fsum(run["loss"] for run in selected_runs) / len(selected_runs),
            "per_seed": [
                {key: value for key, value in run.items() if key != "train_log_path"}
                for run in selected_runs
            ],
        },
        "trajectory_log_compliance": trajectory,
        "gate": {
            "passed": passed,
            "minimum_mean_accuracy_delta": MIN_MEAN_ACCURACY_DELTA,
            "observed_mean_accuracy_delta": mean_delta,
            "minimum_strict_paired_wins": MIN_PAIRED_WINS,
            "observed_strict_paired_wins": wins,
            "trajectory_log_compliant": trajectory["compliant"],
            "reasons": reasons if reasons else ["all pre-declared gate conditions passed"],
            "per_seed_pairing": pairings,
        },
        "fresh_final_plan": {
            "path": "plans/dense-final.tsv",
            "file_sha256": _sha256_bytes(final_bytes),
            "record_count": len(final_lines),
            "fresh_seeds": list(FRESH_SEEDS),
            "status": (
                "enabled"
                if passed and not reuse_source_fixed_c2
                else "empty-reuse-source-fixed-c2"
                if reuse_source_fixed_c2
                else "empty-gate-failed"
            ),
            "source_fixed_c2_reuse": {
                "enabled": reuse_source_fixed_c2,
                "reason": (
                    "validation-selected dense best is C=2; the source locked final plan "
                    "already registered the identical fixed-C=2 control on all fresh seeds"
                    if reuse_source_fixed_c2
                    else None
                ),
                "candidate_id": context["fixed_by_c"][2.0]["id"],
                "source_campaign_root": str(context["source_root"]),
                "source_final_plan": str(
                    context["paths"]["final_plan"].relative_to(context["source_root"])
                ),
                "registered_arms": [
                    {
                        "seed": row["seed"],
                        "arm_root": str(
                            row["arm_root"].relative_to(context["source_root"])
                        ),
                    }
                    for row in context["source_final"]
                    if row["candidate_id"] == context["fixed_by_c"][2.0]["id"]
                ],
            },
        },
        "data_access_audit": {
            "source_final_plan_read_for_provenance_validation": True,
            "source_final_result_files_read": False,
            "followup_final_result_files_read": False,
            "selection_used_only_registered_public_validation_metrics": True,
        },
    }
    selection_path = context["campaign_root"] / "selection" / "dense-best-fixed.json"
    final_path = context["campaign_root"] / "plans" / "dense-final.tsv"
    _write_with_sha256(selection_path, _json_bytes(payload))
    _write_with_sha256(final_path, final_bytes)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="phase", required=True)
    for phase in ("prepare", "lock"):
        child = subparsers.add_parser(phase)
        child.add_argument("--source-root", type=Path, required=True)
        child.add_argument("--campaign-root", type=Path, required=True)
        child.add_argument("--experiment-code-sha", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    function = prepare if args.phase == "prepare" else lock
    payload = function(
        source_root=args.source_root,
        campaign_root=args.campaign_root,
        experiment_code_sha=args.experiment_code_sha,
    )
    if args.phase == "prepare":
        print(
            f"dense_selection_plan={args.campaign_root / 'plans' / 'dense-selection.tsv'} "
            f"arms={payload['selection_design']['dense_run_count']}"
        )
    else:
        print(
            f"dense_best_fixed={payload['dense_best_fixed']['candidate_id']} "
            f"C={payload['dense_best_fixed']['clip_threshold']:.12g} "
            f"gate_passed={str(payload['gate']['passed']).lower()} "
            f"fresh_arms={payload['fresh_final_plan']['record_count']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
