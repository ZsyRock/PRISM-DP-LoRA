#!/usr/bin/env python3
"""Audit and summarize the locked Math-10K dense-fixed follow-up campaign.

The public 500-example validation split is the only source used to choose a
dense fixed-C control.  Fresh final results are reported only after that lock;
the script never changes the lock or chooses from final-test measurements.
Research telemetry is intentionally NON_PRIVATE and is summarized as such.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import statistics
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SELECTION_SEEDS = (42, 43, 44, 45, 46)
FINAL_SEEDS = (191, 223, 257, 293, 331)
EXPECTED_STEPS = 300
EXPECTED_FIXED_CS = (1.0, 1.25, 1.5, 1.75, 2.0)
T95_DF4 = 2.7764451051977987
PAIR_TOL = 1e-12
LOCK_TOL = 1e-10
MECHANISM_METRICS = (
    "loss_mean",
    "raw_clip_fraction",
    "dp_clip_threshold",
    "raw_clipping_bias_norm",
    "raw_realized_noise_norm",
    "raw_signal_to_noise_ratio",
    "raw_signal_retention_ratio",
    "raw_clipping_bias_to_noise_ratio",
    "raw_bias_noise_squared_error_proxy",
    "raw_unclipped_clipped_cosine",
    "raw_clipped_noisy_cosine",
    "raw_reference_small_gradient_proxy",
)


class AnalysisError(RuntimeError):
    """Raised when a supposedly locked artifact is incomplete or inconsistent."""


@dataclass(frozen=True)
class Spec:
    phase: str
    candidate_id: str
    role: str
    seed: int
    method: str
    clip: float
    rho: float | None
    eta: float | None
    arm_root: Path


def _read_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size == 0:
        raise AnalysisError(f"missing {label}: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AnalysisError(f"invalid {label}: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise AnalysisError(f"{label} must be a JSON object: {path}")
    return value


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AnalysisError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise AnalysisError(f"{label} must be finite")
    return result


def _same(left: Any, right: Any, tolerance: float = 1e-12) -> bool:
    try:
        return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=tolerance)
    except (TypeError, ValueError):
        return left == right


def _read_plan(path: Path) -> list[Spec]:
    if not path.is_file():
        raise AnalysisError(f"missing plan: {path}")
    result: list[Spec] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        fields = line.split("|")
        if len(fields) != 11 or fields[0] != "train":
            raise AnalysisError(f"invalid plan record {path}:{line_number}")
        try:
            seed = int(fields[4])
            clip = float(fields[6])
            rho = None if fields[7] == "NA" else float(fields[7])
            eta = None if fields[8] == "NA" else float(fields[8])
        except ValueError as exc:
            raise AnalysisError(f"invalid numeric plan value {path}:{line_number}") from exc
        result.append(
            Spec(
                phase=fields[1],
                candidate_id=fields[2],
                role=fields[3],
                seed=seed,
                method=fields[5],
                clip=clip,
                rho=rho,
                eta=eta,
                arm_root=Path(fields[10]).resolve(),
            )
        )
    return result


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    os.chmod(path, 0o600)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    _atomic_write(
        path,
        (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(),
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(fields), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    _atomic_write(path, buffer.getvalue().encode())


def _candidate(selection: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = selection.get(key)
    if not isinstance(value, dict):
        raise AnalysisError(f"source selection lacks {key}")
    return value


def _validate_status(spec: Spec, experiment_sha: str, final: bool) -> dict[str, Any]:
    path = spec.arm_root / "adapter" / "run_status.json"
    status = _read_json(path, "adapter status")
    if status.get("state") != "completed" or int(status.get("update_steps", -1)) != EXPECTED_STEPS:
        raise AnalysisError(f"incomplete 300-step run: {path}")
    config = status.get("config")
    if not isinstance(config, dict):
        raise AnalysisError(f"run status lacks config: {path}")
    expected: dict[str, Any] = {
        "implementation_git_sha": experiment_sha,
        "method": spec.method,
        "seed": spec.seed,
        "total_update_steps": EXPECTED_STEPS,
        "dp_max_grad_norm": spec.clip,
        "protocol_stage": "final" if final else "selection",
        "val_set_size": 0 if final else 500,
        "run_eval": final,
        "validation_data_is_public": not final,
        "batch_size": 64,
        "micro_batch_size": 4,
        "learning_rate": 0.0003,
        "dp_epsilon": 6.0,
        "dp_delta": 1e-5,
    }
    for key, expected_value in expected.items():
        if not _same(config.get(key), expected_value):
            raise AnalysisError(
                f"run config mismatch {key}={config.get(key)!r}, expected={expected_value!r}: {path}"
            )
    if spec.method == "slaclip":
        if not _same(config.get("slaclip_target_non_small_clip_fraction"), spec.rho):
            raise AnalysisError(f"SlaClip target differs from plan: {path}")
        if not _same(config.get("slaclip_eta"), spec.eta):
            raise AnalysisError(f"SlaClip eta differs from plan: {path}")
    privacy = status.get("privacy_accounting") or {}
    epsilon = _finite(privacy.get("epsilon_spent"), f"{path}:epsilon_spent")
    if abs(epsilon - 6.0) > 0.02:
        raise AnalysisError(f"unexpected epsilon={epsilon}: {path}")
    return status


def _telemetry(
    spec: Spec, status: Mapping[str, Any], *, strict: bool
) -> tuple[dict[str, float], list[dict[str, str]]]:
    summary_path = spec.arm_root / "results" / "research_raw" / "telemetry_summary.json"
    steps_path = spec.arm_root / "results" / "research_raw" / "telemetry_steps.csv"
    summary = _read_json(summary_path, "NON_PRIVATE telemetry summary")
    if summary.get("NON_PRIVATE_TELEMETRY") is not True:
        raise AnalysisError(f"telemetry is not explicitly marked NON_PRIVATE: {summary_path}")
    step_summary = summary.get("steps") or {}
    if (
        int(step_summary.get("count", -1)) != EXPECTED_STEPS
        or int(step_summary.get("first", -1)) != 1
        or int(step_summary.get("last", -1)) != EXPECTED_STEPS
        or int(step_summary.get("missing_count", -1)) != 0
    ):
        raise AnalysisError(f"telemetry must contain exactly 300 records: {summary_path}")
    metrics = summary.get("metrics")
    if not isinstance(metrics, dict):
        raise AnalysisError(f"telemetry summary lacks metrics: {summary_path}")
    fingerprint = status.get("config_fingerprint")
    identity = summary.get("run_identity") or {}
    expected_identity = {
        "method": spec.method,
        "config_fingerprint": fingerprint,
        "dataset": "math10k",
        "privacy": "dp",
    }
    for key, expected in expected_identity.items():
        if identity.get(key) != expected:
            raise AnalysisError(
                f"telemetry run identity mismatch {key}={identity.get(key)!r}, "
                f"expected={expected!r}: {summary_path}"
            )
    source = summary.get("source") or {}
    for key in ("raw_physical_records", "raw_unique_steps", "safe_physical_records", "safe_unique_steps"):
        if int(source.get(key, -1)) != EXPECTED_STEPS:
            raise AnalysisError(f"telemetry source {key} must equal 300: {summary_path}")
    if int(source.get("raw_duplicate_records", -1)) != 0 or int(source.get("safe_duplicate_records", -1)) != 0:
        raise AnalysisError(f"telemetry source contains duplicate records: {summary_path}")
    means: dict[str, float] = {}
    for name in MECHANISM_METRICS:
        item = metrics.get(name)
        if not isinstance(item, dict) or item.get("mean") is None:
            raise AnalysisError(f"telemetry lacks {name}.mean: {summary_path}")
        if int(item.get("count", -1)) != EXPECTED_STEPS or int(item.get("missing", -1)) != 0:
            raise AnalysisError(f"telemetry {name} must contain 300 nonmissing values: {summary_path}")
        means[name] = _finite(item["mean"], f"{summary_path}:{name}.mean")
    if not steps_path.is_file():
        raise AnalysisError(f"missing telemetry trajectory: {steps_path}")
    with steps_path.open("r", encoding="utf-8", newline="") as handle:
        steps = list(csv.DictReader(handle))
    if [int(row["step"]) for row in steps] != list(range(1, EXPECTED_STEPS + 1)):
        raise AnalysisError(f"telemetry steps must be exactly 1..300: {steps_path}")
    if strict:
        for row in steps:
            if (
                row.get("NON_PRIVATE_TELEMETRY") != "True"
                or row.get("method") != spec.method
                or row.get("config_fingerprint") != fingerprint
                or row.get("dataset") != "math10k"
                or row.get("privacy") != "dp"
            ):
                raise AnalysisError(f"telemetry CSV identity mismatch: {steps_path}")
        for name in MECHANISM_METRICS:
            values = [_finite(float(row[name]), f"{steps_path}:{name}") for row in steps]
            observed_mean = math.fsum(values) / len(values)
            if not math.isclose(observed_mean, means[name], rel_tol=0.0, abs_tol=1e-12):
                raise AnalysisError(f"telemetry CSV/summary mean mismatch for {name}: {steps_path}")
    return means, steps


def _validation(spec: Spec) -> tuple[float, float, str]:
    path = spec.arm_root / "results" / "validation" / "validation_metrics.json"
    payload = _read_json(path, "public validation metrics")
    if (
        payload.get("protocol_stage") != "selection"
        or payload.get("validation_data_is_public") is not True
        or payload.get("selection_metric") != "public_math10k_numeric_exact_match_accuracy"
        or int(payload.get("records", -1)) != 500
    ):
        raise AnalysisError(f"invalid public selection metrics: {path}")
    return (
        _finite(payload.get("numeric_exact_accuracy"), f"{path}:accuracy"),
        _finite(payload.get("loss_mean"), f"{path}:loss"),
        str(payload.get("manifest_sha256")),
    )


def _final_metrics(spec: Spec, expected_selection_sha: str) -> dict[str, float]:
    path = spec.arm_root / "results" / "decontaminated_metrics.json"
    value = _read_json(path, "decontaminated final metrics")
    if value.get("primary_metric") != "clean_three_task_macro_accuracy":
        raise AnalysisError(f"unexpected final primary metric: {path}")
    if value.get("selection_sha256") != expected_selection_sha:
        raise AnalysisError(f"final metrics do not match the locked source selection: {path}")
    raw = value.get("raw_task_accuracy") or {}
    clean = value.get("decontaminated_task_accuracy") or {}
    if set(raw) != {"gsm8k", "AQuA", "mawps", "SVAMP"} or set(clean) != set(raw):
        raise AnalysisError(f"final task metric maps are incomplete: {path}")
    raw_values = {key: _finite(item, f"{path}:raw:{key}") for key, item in raw.items()}
    clean_values = {key: _finite(item, f"{path}:clean:{key}") for key, item in clean.items()}
    recomputed_clean_three = math.fsum(raw_values[key] for key in ("gsm8k", "AQuA", "SVAMP")) / 3.0
    recomputed_decont_four = math.fsum(
        (raw_values["gsm8k"], raw_values["AQuA"], clean_values["mawps"], raw_values["SVAMP"])
    ) / 4.0
    recomputed_raw_four = math.fsum(raw_values.values()) / 4.0
    result = {
        "clean_three_task_macro_accuracy": _finite(
            value.get("clean_three_task_macro_accuracy"), f"{path}:clean-three"
        ),
        "decontaminated_four_task_macro_accuracy": _finite(
            value.get("decontaminated_four_task_macro_accuracy"), f"{path}:decont-four"
        ),
        "paper_raw_four_task_macro_accuracy": _finite(
            value.get("paper_raw_four_task_macro_accuracy"), f"{path}:raw-four"
        ),
    }
    expected_values = {
        "clean_three_task_macro_accuracy": recomputed_clean_three,
        "decontaminated_four_task_macro_accuracy": recomputed_decont_four,
        "paper_raw_four_task_macro_accuracy": recomputed_raw_four,
    }
    for key, expected in expected_values.items():
        if not math.isclose(result[key], expected, rel_tol=0.0, abs_tol=1e-12):
            raise AnalysisError(f"final {key} is inconsistent with task metrics: {path}")
    if int(value.get("clean_mawps_records", -1)) != 185:
        raise AnalysisError(f"final clean MAWPS count must be 185: {path}")
    return result


def _paired_summary(
    slaclip_values: Sequence[float], fixed_values: Sequence[float] | None = None
) -> dict[str, Any]:
    if fixed_values is None:
        differences = list(slaclip_values)
        slaclip_mean: float | str = ""
        fixed_mean: float | str = ""
    else:
        if len(slaclip_values) != len(fixed_values):
            raise AnalysisError("paired vectors must have identical lengths")
        differences = [left - right for left, right in zip(slaclip_values, fixed_values)]
        slaclip_mean = statistics.fmean(slaclip_values)
        fixed_mean = statistics.fmean(fixed_values)
    if len(differences) != 5:
        raise AnalysisError("paired fresh-seed inference requires exactly five independent seeds")
    mean = statistics.fmean(differences)
    standard_error = statistics.stdev(differences) / math.sqrt(len(differences))
    return {
        "n_independent_seeds": len(differences),
        "slaclip_mean": slaclip_mean,
        "fixed_mean": fixed_mean,
        "mean_difference": mean,
        "sample_difference_std": statistics.stdev(differences),
        "standard_error": standard_error,
        "ci95_low": mean - T95_DF4 * standard_error,
        "ci95_high": mean + T95_DF4 * standard_error,
        "wins": sum(value > PAIR_TOL for value in differences),
        "ties": sum(abs(value) <= PAIR_TOL for value in differences),
        "losses": sum(value < -PAIR_TOL for value in differences),
    }


def _find_specs(specs: Iterable[Spec], candidate_id: str, seeds: Sequence[int]) -> list[Spec]:
    by_seed = {spec.seed: spec for spec in specs if spec.candidate_id == candidate_id}
    if set(by_seed) != set(seeds):
        raise AnalysisError(
            f"candidate={candidate_id} seed coverage={sorted(by_seed)}, expected={list(seeds)}"
        )
    return [by_seed[seed] for seed in seeds]


def analyze(source_root: Path, campaign_root: Path, experiment_sha: str, strict: bool) -> dict[str, Any]:
    source_root = source_root.resolve(strict=True)
    campaign_root = campaign_root.resolve(strict=True)
    output = campaign_root / "artifacts"
    selection_path = source_root / "selection" / "selection.json"
    dense_lock_path = campaign_root / "selection" / "dense-best-fixed.json"
    selection = _read_json(selection_path, "source locked selection")
    dense_lock = _read_json(dense_lock_path, "dense fixed lock")
    if (
        dense_lock.get("schema_version") != 1
        or dense_lock.get("phase") != "locked"
        or dense_lock.get("experiment_code_sha") != experiment_sha
        or dense_lock.get("selection_evidence_class") != "public_validation_only"
        or dense_lock.get("required_seeds") != list(SELECTION_SEEDS)
        or dense_lock.get("source_locked_selection_sha256") != _sha(selection_path)
    ):
        raise AnalysisError("dense fixed lock has inconsistent provenance or protocol identity")
    selected = _candidate(selection, "selected_slaclip")
    selected_id = str(selected.get("candidate_id"))
    if not selected_id or selected.get("method") != "slaclip":
        raise AnalysisError("invalid source selected SlaClip identity")

    source_selection_specs = _read_plan(source_root / "plans" / "stage1.tsv")
    source_selection_specs += _read_plan(source_root / "plans" / "stage2.tsv")
    source_final_specs = _read_plan(source_root / "plans" / "final.tsv")
    dense_specs = _read_plan(campaign_root / "plans" / "dense-selection.tsv")
    dense_final_specs = _read_plan(campaign_root / "plans" / "dense-final.tsv")
    if len(dense_specs) != 15:
        raise AnalysisError(f"dense selection plan must contain 15 arms, found {len(dense_specs)}")
    observed_grid = {(spec.clip, spec.seed) for spec in dense_specs}
    expected_grid = {(clip, seed) for clip in (1.25, 1.5, 1.75) for seed in SELECTION_SEEDS}
    if observed_grid != expected_grid:
        raise AnalysisError("dense selection plan is not C={1.25,1.5,1.75} x five seeds")

    selected_specs = _find_specs(source_selection_specs, selected_id, SELECTION_SEEDS)
    fixed_by_c: dict[float, list[Spec]] = {}
    for clip in (1.0, 2.0):
        candidates = sorted(
            {
                spec.candidate_id
                for spec in source_selection_specs
                if spec.method == "baseline" and _same(spec.clip, clip)
            }
        )
        if len(candidates) != 1:
            raise AnalysisError(f"source must contain one fixed C={clip} candidate")
        fixed_by_c[clip] = _find_specs(source_selection_specs, candidates[0], SELECTION_SEEDS)
    for clip in (1.25, 1.5, 1.75):
        fixed_by_c[clip] = sorted(
            [spec for spec in dense_specs if _same(spec.clip, clip)], key=lambda spec: spec.seed
        )

    selection_rows: list[dict[str, Any]] = []
    trajectory_rows: list[dict[str, Any]] = []
    split_hashes: set[str] = set()
    for family, specs in [("selected_slaclip", selected_specs)] + [
        ("fixed", fixed_by_c[clip]) for clip in EXPECTED_FIXED_CS
    ]:
        for spec in specs:
            status = _validate_status(spec, experiment_sha, final=False)
            accuracy, loss, split_hash = _validation(spec)
            split_hashes.add(split_hash)
            means, steps = _telemetry(spec, status, strict=strict)
            selection_rows.append(
                {
                    "NON_PRIVATE_TELEMETRY": True,
                    "family": family,
                    "candidate_id": spec.candidate_id,
                    "seed": spec.seed,
                    "method": spec.method,
                    "initial_or_fixed_C": spec.clip,
                    "validation_accuracy": accuracy,
                    "validation_loss": loss,
                    **{f"{name}_mean": means[name] for name in MECHANISM_METRICS},
                    "arm_root": str(spec.arm_root),
                }
            )
            for row in steps:
                trajectory_rows.append(
                    {
                        "NON_PRIVATE_TELEMETRY": True,
                        "family": family,
                        "candidate_id": spec.candidate_id,
                        "initial_or_fixed_C": spec.clip,
                        "seed": spec.seed,
                        "step": int(row["step"]),
                        **{name: row.get(name, "") for name in MECHANISM_METRICS},
                    }
                )
    if len(split_hashes) != 1:
        raise AnalysisError(f"selection arms do not share one public split: {split_hashes}")

    response_rows: list[dict[str, Any]] = []
    for clip in EXPECTED_FIXED_CS:
        rows = [row for row in selection_rows if row["family"] == "fixed" and _same(row["initial_or_fixed_C"], clip)]
        if len(rows) != 5:
            raise AnalysisError(f"fixed C={clip} does not have five audited validation runs")
        response_rows.append(
            {
                "fixed_C": clip,
                "n_seeds": 5,
                "mean_validation_accuracy": statistics.fmean(float(row["validation_accuracy"]) for row in rows),
                "mean_validation_loss": statistics.fmean(float(row["validation_loss"]) for row in rows),
                **{
                    f"mean_{name}": statistics.fmean(float(row[f"{name}_mean"]) for row in rows)
                    for name in MECHANISM_METRICS
                },
            }
        )
    ranked_response = sorted(
        response_rows,
        key=lambda row: (-float(row["mean_validation_accuracy"]), float(row["mean_validation_loss"]), float(row["fixed_C"])),
    )
    computed_best_c = float(ranked_response[0]["fixed_C"])
    locked_best = dense_lock.get("dense_best_fixed") or dense_lock.get("best_fixed")
    if not isinstance(locked_best, dict):
        raise AnalysisError("dense lock lacks dense_best_fixed")
    locked_best_c = _finite(
        locked_best.get(
            "fixed_C",
            locked_best.get(
                "clip_threshold",
                locked_best.get("clip", (locked_best.get("params") or {}).get("dp_max_grad_norm")),
            ),
        ),
        "locked dense best C",
    )
    if not _same(computed_best_c, locked_best_c):
        raise AnalysisError(
            f"locked dense best C={locked_best_c} differs from validation-only result={computed_best_c}"
        )

    selected_accuracy = {
        int(row["seed"]): float(row["validation_accuracy"])
        for row in selection_rows
        if row["family"] == "selected_slaclip"
    }
    best_accuracy = {
        int(row["seed"]): float(row["validation_accuracy"])
        for row in selection_rows
        if row["family"] == "fixed" and _same(row["initial_or_fixed_C"], computed_best_c)
    }
    validation_diffs = [selected_accuracy[seed] - best_accuracy[seed] for seed in SELECTION_SEEDS]
    validation_mean_delta = statistics.fmean(validation_diffs)
    validation_wins = sum(value > LOCK_TOL for value in validation_diffs)
    performance_gate = validation_mean_delta + LOCK_TOL >= 0.005 and validation_wins >= 3
    locked_gate_value = dense_lock.get("gate_passed")
    gate_record = dense_lock.get("gate")
    if locked_gate_value is None and isinstance(gate_record, dict):
        locked_gate_value = gate_record.get("passed")
    if not isinstance(locked_gate_value, bool):
        raise AnalysisError("dense lock lacks boolean gate_passed")
    if not isinstance(gate_record, dict):
        raise AnalysisError("dense lock lacks its gate audit")
    trajectory = dense_lock.get("trajectory_log_compliance") or {}
    trajectory_compliant = trajectory.get("compliant")
    if not isinstance(trajectory_compliant, bool):
        raise AnalysisError("dense lock lacks a boolean trajectory compliance audit")
    if not _same(gate_record.get("observed_mean_accuracy_delta"), validation_mean_delta):
        raise AnalysisError("dense lock mean validation delta differs from audited artifacts")
    if gate_record.get("observed_strict_paired_wins") != validation_wins:
        raise AnalysisError("dense lock paired wins differ from audited artifacts")
    if locked_gate_value != (performance_gate and trajectory_compliant):
        raise AnalysisError("dense lock gate decision differs from its predeclared evidence")
    final_plan_record = dense_lock.get("fresh_final_plan")
    if not isinstance(final_plan_record, dict):
        raise AnalysisError("dense lock lacks fresh_final_plan provenance")
    dense_final_path = campaign_root / "plans" / "dense-final.tsv"
    if (
        final_plan_record.get("path") != "plans/dense-final.tsv"
        or final_plan_record.get("file_sha256") != _sha(dense_final_path)
        or int(final_plan_record.get("record_count", -1)) != len(dense_final_specs)
        or final_plan_record.get("fresh_seeds") != list(FINAL_SEEDS)
    ):
        raise AnalysisError("dense final plan differs from its locked hash/count/seeds")
    reuse_source_fixed_c2 = (
        locked_gate_value and _same(computed_best_c, 2.0) and len(dense_final_specs) == 0
    )
    if locked_gate_value and len(dense_final_specs) != 5 and not reuse_source_fixed_c2:
        raise AnalysisError(
            "a passed gate requires five dense-final arms unless source fixed C=2 is reused"
        )
    if not locked_gate_value and dense_final_specs:
        raise AnalysisError("a failed gate must not create final-test arms")
    expected_final_status = (
        "empty-gate-failed"
        if not locked_gate_value
        else "empty-reuse-source-fixed-c2"
        if reuse_source_fixed_c2
        else "enabled"
    )
    if final_plan_record.get("status") != expected_final_status:
        raise AnalysisError("dense final plan status differs from the locked gate branch")
    reuse_record = final_plan_record.get("source_fixed_c2_reuse")
    if not isinstance(reuse_record, dict) or reuse_record.get("enabled") is not reuse_source_fixed_c2:
        raise AnalysisError("dense final source-fixed-C2 reuse metadata is inconsistent")
    if reuse_source_fixed_c2:
        registered = reuse_record.get("registered_arms")
        if (
            reuse_record.get("candidate_id") != "fixed-c2"
            or reuse_record.get("source_campaign_root") != str(source_root)
            or not isinstance(registered, list)
            or [item.get("seed") for item in registered if isinstance(item, dict)] != list(FINAL_SEEDS)
        ):
            raise AnalysisError("source fixed-C2 reuse registration is incomplete")

    final_seed_rows: list[dict[str, Any]] = []
    final_summary_rows: list[dict[str, Any]] = []
    if locked_gate_value:
        if dense_final_specs and {spec.seed for spec in dense_final_specs} != set(FINAL_SEEDS):
            raise AnalysisError("dense final plan does not cover the five fresh seeds")
        selected_final = [spec for spec in source_final_specs if spec.method == "slaclip"]
        fixed_c2_final = [spec for spec in source_final_specs if spec.method == "baseline"]
        selected_final = sorted(selected_final, key=lambda spec: spec.seed)
        fixed_c2_final = sorted(fixed_c2_final, key=lambda spec: spec.seed)
        dense_final = (
            fixed_c2_final
            if reuse_source_fixed_c2
            else sorted(dense_final_specs, key=lambda spec: spec.seed)
        )
        for group in (selected_final, fixed_c2_final, dense_final):
            if tuple(spec.seed for spec in group) != FINAL_SEEDS:
                raise AnalysisError("fresh final comparison lacks exact paired seed coverage")
        metric_by_group: dict[str, dict[int, dict[str, float]]] = {}
        for label, specs in (
            ("selected_slaclip", selected_final),
            ("fixed_C2", fixed_c2_final),
            ("dense_best_fixed", dense_final),
        ):
            metric_by_group[label] = {}
            for spec in specs:
                status = _validate_status(spec, experiment_sha, final=True)
                _telemetry(spec, status, strict=strict)
                metric_by_group[label][spec.seed] = _final_metrics(
                    spec, _sha(selection_path)
                )
        comparators = (
            ("dense_best_fixed",)
            if reuse_source_fixed_c2
            else ("fixed_C2", "dense_best_fixed")
        )
        for comparator in comparators:
            for metric in (
                "clean_three_task_macro_accuracy",
                "decontaminated_four_task_macro_accuracy",
                "paper_raw_four_task_macro_accuracy",
            ):
                differences: list[float] = []
                slaclip_values: list[float] = []
                fixed_values: list[float] = []
                for seed in FINAL_SEEDS:
                    sla_value = metric_by_group["selected_slaclip"][seed][metric]
                    fixed_value = metric_by_group[comparator][seed][metric]
                    difference = sla_value - fixed_value
                    slaclip_values.append(sla_value)
                    fixed_values.append(fixed_value)
                    differences.append(difference)
                    final_seed_rows.append(
                        {
                            "seed": seed,
                            "metric": metric,
                            "comparator": comparator,
                            "slaclip_value": sla_value,
                            "fixed_value": fixed_value,
                            "slaclip_minus_fixed": difference,
                        }
                    )
                final_summary_rows.append(
                    {
                        "metric": metric,
                        "comparator": comparator,
                        **_paired_summary(slaclip_values, fixed_values),
                    }
                )

    selection_fields = list(selection_rows[0])
    trajectory_fields = list(trajectory_rows[0])
    response_fields = list(response_rows[0])
    _write_csv(output / "dense_fixed_selection_runs.csv", selection_rows, selection_fields)
    _write_csv(output / "fixed_c_response_curve.csv", response_rows, response_fields)
    _write_csv(output / "step_trajectories.csv", trajectory_rows, trajectory_fields)
    _write_csv(
        output / "fresh_final_seed_differences.csv",
        final_seed_rows,
        ("seed", "metric", "comparator", "slaclip_value", "fixed_value", "slaclip_minus_fixed"),
    )
    _write_csv(
        output / "fresh_final_paired_summary.csv",
        final_summary_rows,
        (
            "metric",
            "comparator",
            "n_independent_seeds",
            "slaclip_mean",
            "fixed_mean",
            "mean_difference",
            "sample_difference_std",
            "standard_error",
            "ci95_low",
            "ci95_high",
            "wins",
            "ties",
            "losses",
        ),
    )
    inputs = [
        selection_path,
        dense_lock_path,
        source_root / "plans" / "stage1.tsv",
        source_root / "plans" / "stage2.tsv",
        source_root / "plans" / "final.tsv",
        campaign_root / "plans" / "dense-selection.tsv",
        campaign_root / "plans" / "dense-final.tsv",
    ]
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "analysis": "math10k_4b_dense_fixed_followup",
        "NON_PRIVATE_TELEMETRY": True,
        "experiment_code_sha": experiment_sha,
        "source_campaign_root": str(source_root),
        "campaign_root": str(campaign_root),
        "selection_seeds": list(SELECTION_SEEDS),
        "fresh_final_seeds": list(FINAL_SEEDS),
        "fixed_C_grid": list(EXPECTED_FIXED_CS),
        "dense_best_fixed_C": computed_best_c,
        "dense_best_final_source": (
            "skipped_gate_failed"
            if not locked_gate_value
            else (
                "source_campaign_fixed_C2"
                if reuse_source_fixed_c2
                else "followup_dense_final"
            )
        ),
        "public_validation_split_manifest_sha256": next(iter(split_hashes)),
        "selection_slaclip_minus_dense_best": {
            "per_seed": {str(seed): validation_diffs[index] for index, seed in enumerate(SELECTION_SEEDS)},
            "mean": statistics.fmean(validation_diffs),
            "wins": sum(value > 0.0 for value in validation_diffs),
        },
        "gate_passed": locked_gate_value,
        "gate_interpretation": (
            "validation-only exploratory gate; final-test metrics never select C or alter the gate"
        ),
        "privacy_and_release": {
            "per_run_target": {"epsilon": 6.0, "delta": 1e-5},
            "new_selection_runs": 15,
            "new_final_runs": len(dense_final_specs),
            "new_runs_basic_composition_upper_bound": {
                "epsilon": 6.0 * (15 + len(dense_final_specs)),
                "delta": 1e-5 * (15 + len(dense_final_specs)),
                "release_count": 15 + len(dense_final_specs),
            },
            "research_raw_telemetry": "NON_PRIVATE",
            "development_bundle_is_non_private": True,
        },
        "strict": strict,
        "inputs": {str(path): _sha(path) for path in inputs},
        "outputs": {},
    }
    for path in (
        output / "dense_fixed_selection_runs.csv",
        output / "fixed_c_response_curve.csv",
        output / "step_trajectories.csv",
        output / "fresh_final_seed_differences.csv",
        output / "fresh_final_paired_summary.csv",
    ):
        manifest["outputs"][path.name] = {"sha256": _sha(path), "bytes": path.stat().st_size}
    manifest_path = output / "dense_fixed_followup_manifest.json"
    _write_json(manifest_path, manifest)
    _atomic_write(
        manifest_path.with_name(manifest_path.name + ".sha256"),
        f"{_sha(manifest_path)}  {manifest_path.name}\n".encode(),
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--experiment-code-sha", required=True)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    if not (len(args.experiment_code_sha) == 40 and all(c in "0123456789abcdef" for c in args.experiment_code_sha)):
        parser.error("--experiment-code-sha must be a lowercase 40-character Git SHA")
    manifest = analyze(args.source_root, args.campaign_root, args.experiment_code_sha, args.strict)
    print(
        f"dense_fixed_analysis={Path(manifest['campaign_root']) / 'artifacts'} "
        f"best_C={manifest['dense_best_fixed_C']} gate_passed={manifest['gate_passed']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
