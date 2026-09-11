#!/usr/bin/env python3
"""Analyze NON-PRIVATE PRISM/SlaClip mechanism telemetry.

This analyzer is intentionally separate from the formal accuracy analyzer.  It
consumes exact ``research_raw`` telemetry for internal mechanism diagnostics,
joins it to the final public-benchmark accuracy artifacts, and writes only
outputs that remain explicitly marked NON-PRIVATE.

For legacy logs that do not contain the exact unnoised Slack Indicator, the
script derives rigorous per-slot lower/upper bounds from the fixed-edge gradient
norm histogram.  It also creates a deterministic replay schedule by averaging
the selected 4B SlaClip arm's released ``dp_clip_threshold`` trajectory across
seeds.  Exact raw telemetry is never used to construct that schedule, but the
schedule remains data-dependent DP post-processing of its source runs and is
not a free/public clipping schedule.
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
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence


ANALYSIS_SCHEMA_VERSION = 2
REPLAY_SCHEMA_VERSION = 2
WARNING = (
    "NON_PRIVATE: contains exact training-example statistics and must not be "
    "treated as a differentially private release"
)
TASK_COLUMN_ALIASES = {
    "gsm8k": "gsm8k",
    "aqua": "AQuA",
    "mawps": "mawps",
    "svamp": "SVAMP",
}
PAIR_CONFIG_EXCLUSIONS = {
    "config_fingerprint",
    # The clipping threshold is part of the controller under comparison: a
    # tuned SlaClip C0 need not equal the fixed-C reference threshold.
    "dp_max_grad_norm",
    "force_eval",
    "force_train",
    "method",
    "output_dir",
    "repeat_id",
    "result_dir",
    "root",
    "run_id",
    "run_name",
}
DP_SAFE_FORBIDDEN_FIELDS = {
    "NON_PRIVATE_TELEMETRY",
    "batch_n",
    "loss_mean",
    "tokens",
}
T95 = {
    1: 12.706204736432095,
    2: 4.302652729911275,
    3: 3.182446305284263,
    4: 2.7764451051977987,
    5: 2.570581835636305,
    6: 2.4469118487916806,
    7: 2.3646242510102993,
    8: 2.306004135204166,
    9: 2.2621571628540993,
    10: 2.2281388519649385,
    11: 2.200985160082949,
    12: 2.1788128296634177,
    13: 2.1603686564610127,
    14: 2.1447866879169273,
    15: 2.131449545559323,
    16: 2.1199052992210112,
    17: 2.1098155778331806,
    18: 2.10092204024096,
    19: 2.093024054408263,
    20: 2.0859634472658364,
    24: 2.0638985616280205,
    29: 2.045229642132703,
}


class AnalysisError(RuntimeError):
    """The campaign is incomplete, inconsistent, or unsafe to analyze."""


@dataclass
class Run:
    model_slug: str
    seed: int
    arm: str
    candidate_id: str
    run_root: Path
    telemetry_path: Path
    safe_log_path: Path
    status_path: Path
    summary_path: Path
    config_fingerprint: str
    run_id: str
    base_model: str
    model_revision: str
    method: str
    config: dict[str, Any]
    rows: list[dict[str, Any]]
    accuracy: dict[str, float]
    prediction_paths: dict[str, Path]

    @property
    def key(self) -> tuple[str, int, str]:
        return (self.model_slug, self.seed, self.arm)


def _canonical_json(payload: Any) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path, *, label: str) -> Any:
    if not path.is_file():
        raise AnalysisError(f"missing {label}: {path}")
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise AnalysisError(f"could not read {label} {path}: {exc}") from exc


def _finite(
    value: Any,
    *,
    label: str,
    minimum: Optional[float] = None,
    maximum: Optional[float] = None,
) -> float:
    if isinstance(value, bool):
        raise AnalysisError(f"{label} must be numeric, not boolean")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise AnalysisError(f"{label} must be numeric, got {value!r}") from exc
    if not math.isfinite(number):
        raise AnalysisError(f"{label} must be finite, got {value!r}")
    if minimum is not None and number < minimum:
        raise AnalysisError(f"{label} must be >= {minimum}, got {number}")
    if maximum is not None and number > maximum:
        raise AnalysisError(f"{label} must be <= {maximum}, got {number}")
    return number


def _optional_float(value: Any) -> Optional[float]:
    if value in (None, "") or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _integer(value: Any, *, label: str, minimum: Optional[int] = None) -> int:
    if isinstance(value, bool):
        raise AnalysisError(f"{label} must be an integer")
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise AnalysisError(f"{label} must be an integer, got {value!r}") from exc
    if str(number) != str(value).strip() and not (
        isinstance(value, float) and value.is_integer()
    ):
        raise AnalysisError(f"{label} must be an integer, got {value!r}")
    if minimum is not None and number < minimum:
        raise AnalysisError(f"{label} must be >= {minimum}, got {number}")
    return number


def _parse_json_cell(value: Any, *, label: str) -> Any:
    if isinstance(value, (list, dict)):
        return value
    if not isinstance(value, str) or not value:
        raise AnalysisError(f"{label} must contain JSON")
    try:
        return json.loads(value)
    except json.JSONDecodeError as exc:
        raise AnalysisError(f"{label} contains invalid JSON: {exc}") from exc


def _parse_key_value_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.is_file():
        return values
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise AnalysisError(f"{path}:{line_number}: expected key=value")
        key, value = line.split("=", 1)
        if not key or key in values:
            raise AnalysisError(f"{path}:{line_number}: duplicate/empty key")
        values[key] = value
    return values


def _read_csv_rows(path: Path, *, label: str) -> list[dict[str, str]]:
    if not path.is_file():
        raise AnalysisError(f"missing {label}: {path}")
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames:
                raise AnalysisError(f"{label} has no CSV header: {path}")
            return list(reader)
    except OSError as exc:
        raise AnalysisError(f"could not read {label} {path}: {exc}") from exc


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _csv_bytes(rows: Sequence[Mapping[str, Any]], preferred: Sequence[str]) -> bytes:
    all_fields = set().union(*(row.keys() for row in rows)) if rows else set(preferred)
    fields = [field for field in preferred if field in all_fields]
    fields.extend(sorted(all_fields - set(fields)))
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n", extrasaction="raise")
    writer.writeheader()
    for row in rows:
        writer.writerow({field: "" if row.get(field) is None else row.get(field) for field in fields})
    return buffer.getvalue().encode("utf-8")


def _load_expected_steps(campaign_root: Path, strict: bool, inputs: dict[str, str]) -> Optional[int]:
    manifest_path = campaign_root / "final" / "manifest.json"
    if not manifest_path.is_file():
        if strict:
            raise AnalysisError(f"strict mode requires final manifest: {manifest_path}")
        return None
    payload = _read_json(manifest_path, label="final manifest")
    if not isinstance(payload, dict):
        raise AnalysisError("final manifest must be a JSON object")
    inputs[str(manifest_path.relative_to(campaign_root))] = _sha256_file(manifest_path)
    value = payload.get("expected_update_steps")
    return _integer(value, label="final expected_update_steps", minimum=1)


def _read_accuracy(path: Path) -> dict[str, float]:
    rows = _read_csv_rows(path, label="accuracy summary")
    if len(rows) != 1:
        raise AnalysisError(f"accuracy summary must contain exactly one row: {path}")
    row = rows[0]
    task_columns = [key for key in row if key != "Average"]
    if not task_columns or "Average" not in row:
        raise AnalysisError(f"accuracy summary lacks task columns/Average: {path}")
    values = {
        key: _finite(row[key], label=f"{path}:{key}", minimum=0.0, maximum=1.0)
        for key in task_columns
    }
    macro = _finite(row["Average"], label=f"{path}:Average", minimum=0.0, maximum=1.0)
    calculated = math.fsum(values.values()) / len(values)
    if not math.isclose(macro, calculated, rel_tol=0.0, abs_tol=1e-10):
        raise AnalysisError(
            f"accuracy Average is not the equal-task macro ({macro} != {calculated}): {path}"
        )
    return {**values, "macro": macro}


def _prediction_path(result_dir: Path, task: str) -> Optional[Path]:
    direct = result_dir / f"{task}.json"
    if direct.is_file():
        return direct
    canonical = TASK_COLUMN_ALIASES.get(task.lower())
    if canonical is not None and (result_dir / f"{canonical}.json").is_file():
        return result_dir / f"{canonical}.json"
    matches = [path for path in result_dir.glob("*.json") if path.stem.lower() == task.lower()]
    return matches[0] if len(matches) == 1 else None


def _read_telemetry(
    path: Path,
    *,
    expected_steps: Optional[int],
    strict: bool,
) -> tuple[list[dict[str, Any]], str, str]:
    rows = _read_csv_rows(path, label="step telemetry")
    if not rows:
        raise AnalysisError(f"step telemetry is empty: {path}")
    normalized: list[dict[str, Any]] = []
    seen_steps: set[int] = set()
    fingerprints: set[str] = set()
    run_ids: set[str] = set()
    for physical_index, row in enumerate(rows, start=2):
        if str(row.get("NON_PRIVATE_TELEMETRY", "")).lower() != "true":
            raise AnalysisError(
                f"{path}:{physical_index}: missing NON_PRIVATE_TELEMETRY=true"
            )
        step = _integer(row.get("step"), label=f"{path}:{physical_index}:step", minimum=1)
        if step in seen_steps:
            raise AnalysisError(f"{path}:{physical_index}: duplicate step {step}")
        seen_steps.add(step)
        fingerprint = str(row.get("config_fingerprint", ""))
        run_id = str(row.get("run_id", ""))
        if not fingerprint or not run_id:
            raise AnalysisError(f"{path}:{physical_index}: missing run identity")
        fingerprints.add(fingerprint)
        run_ids.add(run_id)
        copied: dict[str, Any] = dict(row)
        copied["step"] = step
        normalized.append(copied)
    normalized.sort(key=lambda item: int(item["step"]))
    if len(fingerprints) != 1 or len(run_ids) != 1:
        raise AnalysisError(f"telemetry mixes run identities: {path}")
    observed = [int(row["step"]) for row in normalized]
    target_steps = expected_steps if expected_steps is not None else max(observed)
    if strict and observed != list(range(1, int(target_steps) + 1)):
        raise AnalysisError(
            f"strict telemetry coverage must be 1..{target_steps}: {path}; "
            f"observed first={observed[0]} last={observed[-1]} count={len(observed)}"
        )
    return normalized, next(iter(fingerprints)), next(iter(run_ids))


def _discover_runs(
    campaign_root: Path,
    *,
    expected_steps: Optional[int],
    strict: bool,
    inputs: dict[str, str],
) -> list[Run]:
    final_root = campaign_root / "final"
    telemetry_paths = sorted(
        final_root.glob("*/seed-*/*/results/research_raw/telemetry_steps.csv")
    )
    if not telemetry_paths:
        raise AnalysisError(f"no formal research telemetry found below {final_root}")
    runs: list[Run] = []
    seen_keys: set[tuple[str, int, str]] = set()
    for telemetry_path in telemetry_paths:
        run_root = telemetry_path.parents[2]
        arm = run_root.name
        seed_name = run_root.parent.name
        model_slug = run_root.parent.parent.name
        if not seed_name.startswith("seed-"):
            raise AnalysisError(f"invalid run path (expected seed-*): {run_root}")
        seed = _integer(seed_name[5:], label=f"seed in {run_root}", minimum=0)
        key = (model_slug, seed, arm)
        if key in seen_keys:
            raise AnalysisError(f"duplicate run identity discovered: {key}")
        seen_keys.add(key)
        status_path = run_root / "adapter" / "run_status.json"
        summary_path = run_root / "results" / "summary.csv"
        safe_log_path = run_root / "adapter" / "train_log.jsonl"
        orchestration_path = run_root / "orchestration-status.txt"
        status = _read_json(status_path, label="run status")
        if not isinstance(status, dict):
            raise AnalysisError(f"run status must be an object: {status_path}")
        if strict and status.get("state") != "completed":
            raise AnalysisError(f"run is not completed: {status_path}")
        config = status.get("config")
        if not isinstance(config, dict):
            raise AnalysisError(f"run status has no config object: {status_path}")
        rows, fingerprint, run_id = _read_telemetry(
            telemetry_path,
            expected_steps=expected_steps,
            strict=strict,
        )
        if status.get("config_fingerprint") != fingerprint:
            raise AnalysisError(f"status/telemetry fingerprint mismatch: {run_root}")
        if status.get("run_id") != run_id:
            raise AnalysisError(f"status/telemetry run_id mismatch: {run_root}")
        method = str(status.get("method", ""))
        if method != str(config.get("method", "")):
            raise AnalysisError(f"top-level/config method mismatch: {status_path}")
        if strict and arm in {"baseline", "slaclip"} and method != arm:
            raise AnalysisError(f"artifact arm/method mismatch: {run_root}")
        if expected_steps is not None and strict:
            if _integer(status.get("update_steps"), label=f"{status_path}:update_steps") != expected_steps:
                raise AnalysisError(f"status update count mismatch: {status_path}")
        accuracy = _read_accuracy(summary_path)
        prediction_paths: dict[str, Path] = {}
        for task in sorted(key for key in accuracy if key != "macro"):
            candidate = _prediction_path(run_root / "results", task)
            if candidate is None:
                if strict:
                    raise AnalysisError(f"missing prediction JSON for {task}: {run_root}")
                continue
            prediction_paths[task] = candidate
            inputs[str(candidate.relative_to(campaign_root))] = _sha256_file(candidate)
        orchestration = _parse_key_value_file(orchestration_path)
        candidate_id = orchestration.get("candidate_id") or (
            "unknown-slaclip" if method == "slaclip" else "unknown-baseline"
        )
        for path in (telemetry_path, status_path, summary_path):
            inputs[str(path.relative_to(campaign_root))] = _sha256_file(path)
        if orchestration_path.is_file():
            inputs[str(orchestration_path.relative_to(campaign_root))] = _sha256_file(
                orchestration_path
            )
        runs.append(
            Run(
                model_slug=model_slug,
                seed=seed,
                arm=arm,
                candidate_id=candidate_id,
                run_root=run_root,
                telemetry_path=telemetry_path,
                safe_log_path=safe_log_path,
                status_path=status_path,
                summary_path=summary_path,
                config_fingerprint=fingerprint,
                run_id=run_id,
                base_model=str(status.get("base_model", "")),
                model_revision=str(status.get("model_revision", "")),
                method=method,
                config=dict(config),
                rows=rows,
                accuracy=accuracy,
                prediction_paths=prediction_paths,
            )
        )
    runs.sort(key=lambda run: run.key)
    return runs


def _pair_signature(config: Mapping[str, Any]) -> bytes:
    retained = {
        key: value
        for key, value in config.items()
        if key not in PAIR_CONFIG_EXCLUSIONS and not key.startswith("slaclip_")
    }
    return _canonical_json(retained)


def _validate_pairs(runs: Sequence[Run], strict: bool) -> None:
    if not strict:
        return
    indexed = {run.key: run for run in runs}
    for run in runs:
        if run.arm != "baseline":
            continue
        candidate = indexed.get((run.model_slug, run.seed, "slaclip"))
        if candidate is None:
            raise AnalysisError(
                f"strict mode requires paired SlaClip run: model={run.model_slug} seed={run.seed}"
            )
        if _pair_signature(run.config) != _pair_signature(candidate.config):
            raise AnalysisError(
                f"paired baseline/SlaClip configurations differ beyond controller fields: "
                f"model={run.model_slug} seed={run.seed}"
            )


def _safe_ratio(numerator: Optional[float], denominator: Optional[float]) -> Optional[float]:
    if numerator is None or denominator is None or denominator == 0.0:
        return None
    result = numerator / denominator
    return result if math.isfinite(result) else None


def _cosine_from_norms(
    left: Optional[float],
    right: Optional[float],
    difference: Optional[float],
) -> Optional[float]:
    if left is None or right is None or difference is None or left <= 0.0 or right <= 0.0:
        return None
    value = (left * left + right * right - difference * difference) / (2.0 * left * right)
    return max(-1.0, min(1.0, value)) if math.isfinite(value) else None


def _signal_noise_cosine(
    signal: Optional[float],
    noise: Optional[float],
    noisy: Optional[float],
) -> Optional[float]:
    if signal is None or noise is None or noisy is None or signal <= 0.0 or noise <= 0.0:
        return None
    value = (noisy * noisy - signal * signal - noise * noise) / (2.0 * signal * noise)
    return max(-1.0, min(1.0, value)) if math.isfinite(value) else None


def _mean(values: Iterable[Optional[float]]) -> Optional[float]:
    materialized = [value for value in values if value is not None and math.isfinite(value)]
    return math.fsum(materialized) / len(materialized) if materialized else None


def _sample_sd(values: Iterable[Optional[float]]) -> Optional[float]:
    materialized = [value for value in values if value is not None and math.isfinite(value)]
    return statistics.stdev(materialized) if len(materialized) >= 2 else None


def _ols_slope(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    x_mean = math.fsum(xs) / len(xs)
    y_mean = math.fsum(ys) / len(ys)
    denominator = math.fsum((value - x_mean) ** 2 for value in xs)
    if denominator == 0.0:
        return None
    return math.fsum(
        (x - x_mean) * (y - y_mean) for x, y in zip(xs, ys)
    ) / denominator


def _trapezoid_auc(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    width = xs[-1] - xs[0]
    if width <= 0.0:
        return None
    area = math.fsum(
        (xs[index + 1] - xs[index]) * (ys[index] + ys[index + 1]) / 2.0
        for index in range(len(xs) - 1)
    )
    return area / width


STEP_NUMERIC_FIELDS = (
    "loss_mean",
    "dp_clip_threshold",
    "dp_next_clip_threshold",
    "raw_clip_fraction",
    "raw_clip_coefficient_mean",
    "raw_global_norm_q50",
    "raw_global_norm_q95",
    "raw_global_norm_q99",
    "raw_unclipped_signal_norm",
    "raw_clipped_signal_norm",
    "raw_clipping_bias_norm",
    "raw_realized_noise_norm",
    "raw_signal_to_noise_ratio",
    "dp_noisy_tangent_gradient_norm",
    "dp_factor_product_update_norm",
    "dp_std_per_factor",
    "eps_spent",
    "slack_unclipped_proxy",
    "slaclip_target_unclipped_proxy",
    "slaclip_controller_error",
    "raw_slack_indicator_noise_residual_l2",
    "raw_slack_indicator_noise_residual_rmse",
    "raw_slack_indicator_noise_residual_first_coordinate",
    "raw_unclipped_clipped_cosine",
    "raw_clipped_noisy_cosine",
    "raw_clipping_bias_to_noise_ratio",
    "raw_bias_noise_squared_error_proxy",
)


def _build_step_rows(runs: Sequence[Run], smooth_window: int) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for run in runs:
        loss_history: list[float] = []
        for source in run.rows:
            numeric = {field: _optional_float(source.get(field)) for field in STEP_NUMERIC_FIELDS}
            loss = numeric["loss_mean"]
            if loss is not None:
                loss_history.append(loss)
            rolling = (
                math.fsum(loss_history[-smooth_window:]) / len(loss_history[-smooth_window:])
                if loss_history
                else None
            )
            unclipped = numeric["raw_unclipped_signal_norm"]
            clipped = numeric["raw_clipped_signal_norm"]
            bias = numeric["raw_clipping_bias_norm"]
            noise = numeric["raw_realized_noise_norm"]
            noisy = numeric["dp_noisy_tangent_gradient_norm"]
            row: dict[str, Any] = {
                "NON_PRIVATE_TELEMETRY": True,
                "warning": WARNING,
                "model_slug": run.model_slug,
                "base_model": run.base_model,
                "model_revision": run.model_revision,
                "seed": run.seed,
                "arm": run.arm,
                "candidate_id": run.candidate_id,
                "config_fingerprint": run.config_fingerprint,
                "run_id": run.run_id,
                "step": int(source["step"]),
                **numeric,
                "loss_rolling_mean": rolling,
                "raw_signal_retention_ratio_recomputed": _safe_ratio(clipped, unclipped),
                "raw_clipping_bias_ratio_recomputed": _safe_ratio(bias, unclipped),
                "raw_bias_to_noise_ratio": _safe_ratio(bias, noise),
                "raw_noise_to_unclipped_ratio": _safe_ratio(noise, unclipped),
                "raw_bias_noise_squared_proxy": (
                    bias * bias + noise * noise
                    if bias is not None and noise is not None
                    else None
                ),
                "raw_unclipped_clipped_cosine": _cosine_from_norms(
                    unclipped, clipped, bias
                ),
                "raw_clipped_noise_cosine": _signal_noise_cosine(clipped, noise, noisy),
            }
            output.append(row)
    return output


SUMMARY_METRICS = (
    "loss_mean",
    "dp_clip_threshold",
    "raw_clip_fraction",
    "raw_global_norm_q50",
    "raw_global_norm_q95",
    "raw_signal_retention_ratio_recomputed",
    "raw_clipping_bias_ratio_recomputed",
    "raw_realized_noise_norm",
    "raw_signal_to_noise_ratio",
    "raw_bias_to_noise_ratio",
    "raw_noise_to_unclipped_ratio",
    "raw_bias_noise_squared_proxy",
    "raw_unclipped_clipped_cosine",
    "raw_clipped_noise_cosine",
    "raw_clipped_noisy_cosine",
    "raw_slack_indicator_noise_residual_rmse",
    "dp_factor_product_update_norm",
    "slack_unclipped_proxy",
    "slaclip_target_unclipped_proxy",
    "slaclip_controller_error",
)


def _build_run_summaries(
    runs: Sequence[Run],
    step_rows: Sequence[Mapping[str, Any]],
    window: int,
) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, int, str], list[Mapping[str, Any]]] = {}
    for row in step_rows:
        key = (str(row["model_slug"]), int(row["seed"]), str(row["arm"]))
        by_key.setdefault(key, []).append(row)
    summaries: list[dict[str, Any]] = []
    for run in runs:
        rows = sorted(by_key[run.key], key=lambda item: int(item["step"]))
        first = rows[: min(window, len(rows))]
        last = rows[-min(window, len(rows)) :]
        summary: dict[str, Any] = {
            "NON_PRIVATE_TELEMETRY": True,
            "warning": WARNING,
            "model_slug": run.model_slug,
            "base_model": run.base_model,
            "model_revision": run.model_revision,
            "seed": run.seed,
            "arm": run.arm,
            "candidate_id": run.candidate_id,
            "config_fingerprint": run.config_fingerprint,
            "run_id": run.run_id,
            "steps": len(rows),
            "window": min(window, len(rows)),
            "accuracy_macro": run.accuracy["macro"],
        }
        loss_pairs = [
            (float(row["step"]), float(row["loss_mean"]))
            for row in rows
            if row.get("loss_mean") is not None
        ]
        summary["loss_auc"] = _trapezoid_auc(
            [pair[0] for pair in loss_pairs],
            [pair[1] for pair in loss_pairs],
        )
        summary["loss_ols_slope"] = _ols_slope(
            [pair[0] for pair in loss_pairs],
            [pair[1] for pair in loss_pairs],
        )
        for metric in SUMMARY_METRICS:
            summary[f"{metric}__all_mean"] = _mean(
                _optional_float(row.get(metric)) for row in rows
            )
            summary[f"{metric}__first_window_mean"] = _mean(
                _optional_float(row.get(metric)) for row in first
            )
            summary[f"{metric}__last_window_mean"] = _mean(
                _optional_float(row.get(metric)) for row in last
            )
            summary[f"{metric}__last_window_sd"] = _sample_sd(
                _optional_float(row.get(metric)) for row in last
            )
        summaries.append(summary)
    return summaries


PAIR_METRICS = {
    "accuracy_macro": "accuracy_macro",
    "loss_auc": "loss_auc",
    "loss_last_window": "loss_mean__last_window_mean",
    "clip_threshold_last_window": "dp_clip_threshold__last_window_mean",
    "clip_fraction_last_window": "raw_clip_fraction__last_window_mean",
    "signal_retention_last_window": "raw_signal_retention_ratio_recomputed__last_window_mean",
    "clipping_bias_ratio_last_window": "raw_clipping_bias_ratio_recomputed__last_window_mean",
    "signal_to_noise_last_window": "raw_signal_to_noise_ratio__last_window_mean",
    "bias_noise_proxy_last_window": "raw_bias_noise_squared_proxy__last_window_mean",
}


def _t95(df: int) -> float:
    if df in T95:
        return T95[df]
    available = sorted(T95)
    lower = max((value for value in available if value <= df), default=available[0])
    if df >= 30:
        return 1.959963984540054
    return T95[lower]


def _build_paired_rows(
    runs: Sequence[Run],
    summaries: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    indexed = {
        (str(row["model_slug"]), int(row["seed"]), str(row["arm"])): row
        for row in summaries
    }
    models = sorted({run.model_slug for run in runs})
    output: list[dict[str, Any]] = []
    for model in models:
        paired_seeds = sorted(
            run.seed
            for run in runs
            if run.model_slug == model
            and run.arm == "baseline"
            and (model, run.seed, "slaclip") in indexed
        )
        for metric, column in PAIR_METRICS.items():
            deltas: list[float] = []
            candidate_values: list[float] = []
            reference_values: list[float] = []
            used_seeds: list[int] = []
            for seed in paired_seeds:
                baseline = _optional_float(indexed[(model, seed, "baseline")].get(column))
                slaclip = _optional_float(indexed[(model, seed, "slaclip")].get(column))
                if baseline is None or slaclip is None:
                    continue
                delta = slaclip - baseline
                deltas.append(delta)
                candidate_values.append(slaclip)
                reference_values.append(baseline)
                used_seeds.append(seed)
                output.append(
                    {
                        "NON_PRIVATE_TELEMETRY": True,
                        "warning": WARNING,
                        "scope": "seed",
                        "model_slug": model,
                        "comparison": "slaclip_vs_baseline",
                        "metric": metric,
                        "seed": seed,
                        "n": 1,
                        "candidate_value": slaclip,
                        "reference_value": baseline,
                        "paired_mean_delta": delta,
                        "sample_sd": None,
                        "t95_ci_low": None,
                        "t95_ci_high": None,
                        "wins": int(delta > 0.0),
                        "ties": int(delta == 0.0),
                        "losses": int(delta < 0.0),
                        "seeds": str(seed),
                    }
                )
            if not deltas:
                continue
            n = len(deltas)
            mean_delta = math.fsum(deltas) / n
            sd = statistics.stdev(deltas) if n >= 2 else None
            half_width = _t95(n - 1) * sd / math.sqrt(n) if sd is not None else None
            output.append(
                {
                    "NON_PRIVATE_TELEMETRY": True,
                    "warning": WARNING,
                    "scope": "aggregate",
                    "model_slug": model,
                    "comparison": "slaclip_vs_baseline",
                    "metric": metric,
                    "seed": None,
                    "n": n,
                    "candidate_value": math.fsum(candidate_values) / n,
                    "reference_value": math.fsum(reference_values) / n,
                    "paired_mean_delta": mean_delta,
                    "sample_sd": sd,
                    "t95_ci_low": (
                        mean_delta - half_width if half_width is not None else None
                    ),
                    "t95_ci_high": (
                        mean_delta + half_width if half_width is not None else None
                    ),
                    "wins": sum(delta > 0.0 for delta in deltas),
                    "ties": sum(delta == 0.0 for delta in deltas),
                    "losses": sum(delta < 0.0 for delta in deltas),
                    "seeds": ",".join(str(seed) for seed in used_seeds),
                }
            )
    return output


def _slot_basis(norm: float, clip: float, slots: int, slot: int) -> float:
    value = float(slots) * (1.0 - float(norm) / float(clip)) - float(slot)
    return max(0.0, min(1.0, value))


def _cdf_bounds_from_histogram(
    *,
    counts: Sequence[Any],
    edges: Sequence[Any],
    overflow: int,
    clip: float,
    slots: int,
    slot: int,
    denominator: float,
    label: str,
) -> tuple[float, float]:
    if len(edges) != len(counts) + 1 or not counts:
        raise AnalysisError(f"{label}: histogram requires len(edges)=len(counts)+1")
    numeric_edges = [
        _finite(value, label=f"{label}:edge", minimum=0.0) for value in edges
    ]
    if any(right <= left for left, right in zip(numeric_edges, numeric_edges[1:])):
        raise AnalysisError(f"{label}: histogram edges must be strictly increasing")
    lower_sum = 0.0
    upper_sum = 0.0
    for index, raw_count in enumerate(counts):
        count = _integer(raw_count, label=f"{label}:count[{index}]", minimum=0)
        left, right = numeric_edges[index], numeric_edges[index + 1]
        lower_sum += count * _slot_basis(right, clip, slots, slot)
        upper_sum += count * _slot_basis(left, clip, slots, slot)
    if overflow:
        # Overflow norms lie in [last_edge, infinity).  The basis is decreasing
        # and eventually zero, so this is a rigorous interval even if C exceeds
        # the declared histogram maximum.
        upper_sum += overflow * _slot_basis(
            numeric_edges[-1], clip, slots, slot
        )
    return lower_sum / denominator, upper_sum / denominator


def _build_cdf_rows(runs: Sequence[Run], strict: bool) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for run in runs:
        if run.method != "slaclip":
            continue
        for source in run.rows:
            label = f"{run.telemetry_path}:step={source['step']}"
            indicator = _parse_json_cell(
                source.get("slack_indicator_json"), label=f"{label}:slack_indicator"
            )
            counts = _parse_json_cell(
                source.get("raw_global_norm_hist_counts_json"),
                label=f"{label}:hist_counts",
            )
            edges = _parse_json_cell(
                source.get("raw_global_norm_hist_edges_json"),
                label=f"{label}:hist_edges",
            )
            if not isinstance(indicator, list) or not isinstance(counts, list) or not isinstance(edges, list):
                raise AnalysisError(f"{label}: indicator/histogram cells must decode to lists")
            slots = _integer(
                source.get("slaclip_num_slots"), label=f"{label}:K", minimum=1
            )
            if len(indicator) != slots:
                raise AnalysisError(
                    f"{label}: indicator length {len(indicator)} does not match K={slots}"
                )
            clip = _finite(
                source.get("dp_clip_threshold"), label=f"{label}:C", minimum=1e-300
            )
            denominator = _finite(
                source.get("dp_expected_batch_size"),
                label=f"{label}:expected_batch_size",
                minimum=1e-300,
            )
            tau = _finite(
                source.get("slack_indicator_noise_std"),
                label=f"{label}:noise_std",
                minimum=1e-300,
            )
            overflow = _integer(
                source.get("raw_global_norm_hist_overflow", 0),
                label=f"{label}:hist_overflow",
                minimum=0,
            )
            realized = _integer(
                source.get("raw_realized_batch_size"),
                label=f"{label}:realized_batch",
                minimum=1,
            )
            if strict:
                hist_total = math.fsum(
                    _integer(value, label=f"{label}:hist_count", minimum=0)
                    for value in counts
                ) + overflow
                if int(hist_total) != realized:
                    raise AnalysisError(
                        f"{label}: histogram count {hist_total} != realized batch {realized}"
                    )
            exact_raw = source.get("raw_slack_indicator_json")
            if exact_raw in (None, ""):
                exact_raw = source.get("raw_slack_indicator")
            exact_values: Optional[list[Any]] = None
            if exact_raw not in (None, ""):
                parsed = _parse_json_cell(exact_raw, label=f"{label}:raw_slack_indicator")
                if not isinstance(parsed, list) or len(parsed) != slots:
                    raise AnalysisError(f"{label}: raw Slack Indicator length mismatch")
                exact_values = parsed
            raw_clip = _optional_float(source.get("raw_clip_fraction"))
            for slot in range(slots):
                lower, upper = _cdf_bounds_from_histogram(
                    counts=counts,
                    edges=edges,
                    overflow=overflow,
                    clip=clip,
                    slots=slots,
                    slot=slot,
                    denominator=denominator,
                    label=label,
                )
                noised = _finite(
                    indicator[slot], label=f"{label}:indicator[{slot}]"
                )
                exact = (
                    _finite(exact_values[slot], label=f"{label}:raw_indicator[{slot}]")
                    if exact_values is not None
                    else None
                )
                if exact is not None and strict and not (
                    lower - 1e-6 <= exact <= upper + 1e-6
                ):
                    raise AnalysisError(
                        f"{label}: exact raw slot {slot}={exact} lies outside "
                        f"histogram bounds [{lower}, {upper}]"
                    )
                if noised < lower:
                    z_distance = (lower - noised) / tau
                elif noised > upper:
                    z_distance = (noised - upper) / tau
                else:
                    z_distance = 0.0
                gaussian_95_multiplier = 1.959963984540054
                legacy_histogram_intersection = bool(
                    noised + gaussian_95_multiplier * tau >= lower
                    and noised - gaussian_95_multiplier * tau <= upper
                )
                exact_noise_residual = (
                    noised - exact if exact is not None else None
                )
                exact_standardized_noise_residual = (
                    exact_noise_residual / tau
                    if exact_noise_residual is not None
                    else None
                )
                exact_within_95pct = (
                    abs(exact_standardized_noise_residual)
                    <= gaussian_95_multiplier
                    if exact_standardized_noise_residual is not None
                    else None
                )
                output.append(
                    {
                        "NON_PRIVATE_TELEMETRY": True,
                        "warning": WARNING,
                        "model_slug": run.model_slug,
                        "seed": run.seed,
                        "arm": run.arm,
                        "candidate_id": run.candidate_id,
                        "config_fingerprint": run.config_fingerprint,
                        "step": int(source["step"]),
                        "slot": slot,
                        "num_slots": slots,
                        "dp_clip_threshold": clip,
                        "dp_expected_batch_size": denominator,
                        "raw_realized_batch_size": realized,
                        "noised_slot": noised,
                        "oracle_source": (
                            "exact_raw_slack_indicator"
                            if exact is not None
                            else "fixed_histogram_bounds"
                        ),
                        "oracle_exact": exact,
                        "oracle_lower": lower,
                        "oracle_upper": upper,
                        "oracle_midpoint": (lower + upper) / 2.0,
                        "oracle_interval_width": upper - lower,
                        "noise_std": tau,
                        "z_distance_to_oracle_interval": z_distance,
                        # Backward-compatible field: this is a conservative
                        # intersection with histogram bounds, not exact Gaussian
                        # coverage.  The explicit alias below prevents it from
                        # being mistaken for ``exact_within_95pct``.
                        "noise_95_band_intersects_oracle": (
                            legacy_histogram_intersection
                        ),
                        "legacy_histogram_interval_intersects_noised_95pct_band": (
                            legacy_histogram_intersection
                        ),
                        "exact_noise_residual": exact_noise_residual,
                        "exact_standardized_noise_residual": (
                            exact_standardized_noise_residual
                        ),
                        "exact_within_95pct": exact_within_95pct,
                        "raw_unclipped_fraction": (
                            1.0 - raw_clip if raw_clip is not None else None
                        ),
                        "dynamic_target_unclipped_proxy": (
                            _optional_float(source.get("slaclip_target_unclipped_proxy"))
                            if slot == 0
                            else None
                        ),
                    }
                )
    if strict and not output:
        raise AnalysisError("strict analysis found no SlaClip CDF telemetry")
    return output


def _average_ranks(values: Sequence[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: (item[1], item[0]))
    ranks = [0.0] * len(values)
    cursor = 0
    while cursor < len(indexed):
        end = cursor + 1
        while end < len(indexed) and indexed[end][1] == indexed[cursor][1]:
            end += 1
        rank = (cursor + 1 + end) / 2.0
        for original_index, _ in indexed[cursor:end]:
            ranks[original_index] = rank
        cursor = end
    return ranks


def _pearson(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    x_mean = math.fsum(xs) / len(xs)
    y_mean = math.fsum(ys) / len(ys)
    x_ss = math.fsum((value - x_mean) ** 2 for value in xs)
    y_ss = math.fsum((value - y_mean) ** 2 for value in ys)
    if x_ss <= 0.0 or y_ss <= 0.0:
        return None
    return math.fsum(
        (x - x_mean) * (y - y_mean) for x, y in zip(xs, ys)
    ) / math.sqrt(x_ss * y_ss)


ASSOCIATION_PREDICTORS = (
    "loss_auc",
    "loss_mean__last_window_mean",
    "dp_clip_threshold__last_window_mean",
    "raw_clip_fraction__last_window_mean",
    "raw_signal_retention_ratio_recomputed__last_window_mean",
    "raw_clipping_bias_ratio_recomputed__last_window_mean",
    "raw_signal_to_noise_ratio__last_window_mean",
    "raw_bias_noise_squared_proxy__last_window_mean",
)


def _association_row(
    *,
    level: str,
    model: str,
    arm: str,
    predictor: str,
    xs: Sequence[float],
    ys: Sequence[float],
) -> dict[str, Any]:
    return {
        "NON_PRIVATE_TELEMETRY": True,
        "warning": WARNING,
        "analysis_level": level,
        "model_slug": model,
        "arm": arm,
        "predictor": predictor,
        "outcome": "accuracy_macro",
        "n": len(xs),
        "pearson_r": _pearson(xs, ys),
        "spearman_rho": _pearson(_average_ranks(xs), _average_ranks(ys)),
        "exploratory_small_n": bool(len(xs) < 10),
    }


def _build_associations(
    runs: Sequence[Run],
    summaries: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    models = sorted({run.model_slug for run in runs})
    for model in models:
        for arm in ("baseline", "slaclip", "ALL"):
            selected = [
                row
                for row in summaries
                if row["model_slug"] == model and (arm == "ALL" or row["arm"] == arm)
            ]
            for predictor in ASSOCIATION_PREDICTORS:
                pairs = [
                    (_optional_float(row.get(predictor)), _optional_float(row.get("accuracy_macro")))
                    for row in selected
                ]
                finite_pairs = [
                    (x, y) for x, y in pairs if x is not None and y is not None
                ]
                if len(finite_pairs) >= 2:
                    output.append(
                        _association_row(
                            level="run",
                            model=model,
                            arm=arm,
                            predictor=predictor,
                            xs=[pair[0] for pair in finite_pairs],
                            ys=[pair[1] for pair in finite_pairs],
                        )
                    )
    indexed = {
        (str(row["model_slug"]), int(row["seed"]), str(row["arm"])): row
        for row in summaries
    }
    for model in models:
        seeds = sorted(
            seed
            for candidate_model, seed, arm in indexed
            if candidate_model == model
            and arm == "baseline"
            and (model, seed, "slaclip") in indexed
        )
        for predictor in ASSOCIATION_PREDICTORS:
            xs: list[float] = []
            ys: list[float] = []
            for seed in seeds:
                baseline = indexed[(model, seed, "baseline")]
                slaclip = indexed[(model, seed, "slaclip")]
                bx = _optional_float(baseline.get(predictor))
                sx = _optional_float(slaclip.get(predictor))
                by = _optional_float(baseline.get("accuracy_macro"))
                sy = _optional_float(slaclip.get("accuracy_macro"))
                if None not in (bx, sx, by, sy):
                    xs.append(float(sx) - float(bx))
                    ys.append(float(sy) - float(by))
            if len(xs) >= 2:
                output.append(
                    _association_row(
                        level="paired_delta",
                        model=model,
                        arm="slaclip_vs_baseline",
                        predictor=predictor,
                        xs=xs,
                        ys=ys,
                    )
                )
    return output


def _parse_failed(
    record: Mapping[str, Any],
    *,
    task: Optional[str] = None,
) -> bool:
    """Return whether the evaluator failed to produce a parseable prediction.

    The repository's numeric evaluator encodes "no number found" as positive
    infinity, which survives ``json.dump``/``json.load`` as a non-finite float.
    Math numeric tasks therefore require a finite float rather than merely a
    non-empty value.  AQuA is the one multiple-choice task and legitimately
    stores an A--E letter instead.
    """

    prediction = record.get("pred")
    if prediction is None:
        return True
    if isinstance(prediction, str):
        prediction = prediction.strip()
        if not prediction:
            return True
    if str(task or "").casefold() == "aqua":
        return not (
            isinstance(prediction, str)
            and len(prediction) == 1
            and prediction.upper() in {"A", "B", "C", "D", "E"}
        )
    try:
        numeric_prediction = float(prediction)
    except (TypeError, ValueError, OverflowError):
        return True
    return not math.isfinite(numeric_prediction)


def _mcnemar_exact_p(discordant_a: int, discordant_b: int) -> float:
    total = int(discordant_a) + int(discordant_b)
    if total <= 0:
        return 1.0
    tail = math.fsum(
        math.comb(total, value) * (0.5 ** total)
        for value in range(0, min(discordant_a, discordant_b) + 1)
    )
    return min(1.0, 2.0 * tail)


def _prediction_identity(record: Mapping[str, Any]) -> tuple[str, str, str]:
    return (
        str(record.get("instruction", "")),
        str(record.get("input", "")),
        str(record.get("answer", "")),
    )


def _build_prediction_transitions(runs: Sequence[Run], strict: bool) -> list[dict[str, Any]]:
    indexed = {run.key: run for run in runs}
    output: list[dict[str, Any]] = []
    for baseline in runs:
        if baseline.arm != "baseline":
            continue
        slaclip = indexed.get((baseline.model_slug, baseline.seed, "slaclip"))
        if slaclip is None:
            continue
        common_tasks = sorted(set(baseline.prediction_paths) & set(slaclip.prediction_paths))
        if strict and set(common_tasks) != set(baseline.prediction_paths):
            raise AnalysisError(
                f"paired prediction task mismatch: model={baseline.model_slug} seed={baseline.seed}"
            )
        for task in common_tasks:
            base_payload = _read_json(
                baseline.prediction_paths[task], label="baseline prediction JSON"
            )
            sla_payload = _read_json(
                slaclip.prediction_paths[task], label="SlaClip prediction JSON"
            )
            if not isinstance(base_payload, list) or not isinstance(sla_payload, list):
                raise AnalysisError(f"prediction artifacts must contain JSON arrays: {task}")
            if len(base_payload) != len(sla_payload):
                raise AnalysisError(
                    f"paired prediction lengths differ: model={baseline.model_slug} "
                    f"seed={baseline.seed} task={task}"
                )
            counts = {"00": 0, "01": 0, "10": 0, "11": 0}
            baseline_parse_fail = 0
            slaclip_parse_fail = 0
            for index, (base_record, sla_record) in enumerate(
                zip(base_payload, sla_payload)
            ):
                if not isinstance(base_record, dict) or not isinstance(sla_record, dict):
                    raise AnalysisError(f"prediction row {index} is not an object: {task}")
                if strict and _prediction_identity(base_record) != _prediction_identity(sla_record):
                    raise AnalysisError(
                        f"paired prediction identity differs at row {index}: "
                        f"model={baseline.model_slug} seed={baseline.seed} task={task}"
                    )
                base_correct = bool(base_record.get("flag"))
                sla_correct = bool(sla_record.get("flag"))
                counts[f"{int(base_correct)}{int(sla_correct)}"] += 1
                baseline_parse_fail += int(
                    _parse_failed(base_record, task=task)
                )
                slaclip_parse_fail += int(
                    _parse_failed(sla_record, task=task)
                )
            total = len(base_payload)
            baseline_correct_n = counts["10"] + counts["11"]
            slaclip_correct_n = counts["01"] + counts["11"]
            output.append(
                {
                    "NON_PRIVATE_TELEMETRY": True,
                    "warning": WARNING,
                    "model_slug": baseline.model_slug,
                    "seed": baseline.seed,
                    "comparison": "slaclip_vs_baseline",
                    "task": task,
                    "n": total,
                    "both_wrong_00": counts["00"],
                    "baseline_wrong_slaclip_right_01": counts["01"],
                    "baseline_right_slaclip_wrong_10": counts["10"],
                    "both_right_11": counts["11"],
                    "baseline_accuracy": baseline_correct_n / max(1, total),
                    "slaclip_accuracy": slaclip_correct_n / max(1, total),
                    "accuracy_delta": (
                        slaclip_correct_n - baseline_correct_n
                    ) / max(1, total),
                    "baseline_parse_failures": baseline_parse_fail,
                    "slaclip_parse_failures": slaclip_parse_fail,
                    "mcnemar_exact_two_sided_p": _mcnemar_exact_p(
                        counts["01"], counts["10"]
                    ),
                }
            )
    return output


def _read_safe_log(
    run: Run,
    *,
    expected_steps: int,
    strict: bool,
    campaign_root: Path,
    inputs: dict[str, str],
) -> tuple[list[dict[str, Any]], str]:
    path = run.safe_log_path
    if not path.is_file():
        raise AnalysisError(f"missing DP-safe train log for replay schedule: {path}")
    records: list[dict[str, Any]] = []
    seen: set[int] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            try:
                record = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise AnalysisError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(record, dict):
                raise AnalysisError(f"{path}:{line_number}: record must be an object")
            if strict:
                forbidden = sorted(
                    key
                    for key in record
                    if key in DP_SAFE_FORBIDDEN_FIELDS or key.startswith("raw_")
                )
                if forbidden:
                    raise AnalysisError(
                        f"DP-safe replay source contains non-private fields {forbidden}: "
                        f"{path}:{line_number}"
                    )
            step = _integer(
                record.get("step"), label=f"{path}:{line_number}:step", minimum=1
            )
            if step in seen:
                raise AnalysisError(f"{path}:{line_number}: duplicate step {step}")
            seen.add(step)
            if record.get("config_fingerprint") != run.config_fingerprint:
                raise AnalysisError(f"{path}:{line_number}: fingerprint mismatch")
            threshold = _finite(
                record.get("dp_clip_threshold"),
                label=f"{path}:{line_number}:dp_clip_threshold",
                minimum=1e-300,
            )
            records.append({"step": step, "dp_clip_threshold": threshold})
    records.sort(key=lambda record: int(record["step"]))
    if [record["step"] for record in records] != list(range(1, expected_steps + 1)):
        raise AnalysisError(f"DP-safe replay source does not cover 1..{expected_steps}: {path}")
    if strict:
        raw_thresholds = {
            int(row["step"]): _finite(
                row.get("dp_clip_threshold"),
                label=f"{run.telemetry_path}:dp_clip_threshold",
                minimum=1e-300,
            )
            for row in run.rows
        }
        for record in records:
            if not math.isclose(
                record["dp_clip_threshold"],
                raw_thresholds[record["step"]],
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise AnalysisError(f"DP-safe/raw threshold mismatch: {path} step {record['step']}")
    digest = _sha256_file(path)
    inputs[str(path.relative_to(campaign_root))] = digest
    return records, digest


def _selected_candidate(
    campaign_root: Path, inputs: dict[str, str]
) -> tuple[Optional[str], Optional[str]]:
    path = campaign_root / "selection" / "selection.json"
    if not path.is_file():
        return None, None
    payload = _read_json(path, label="selection lock")
    if not isinstance(payload, dict):
        raise AnalysisError("selection lock must be an object")
    inputs[str(path.relative_to(campaign_root))] = _sha256_file(path)
    selected = payload.get("selected_slaclip")
    if not isinstance(selected, dict):
        return None, None
    candidate_id = selected.get("candidate_id")
    protocol = payload.get("selection_protocol")
    base_model = None
    if isinstance(protocol, dict) and isinstance(protocol.get("common_config"), dict):
        base_model = protocol["common_config"].get("base_model")
    return (
        str(candidate_id) if candidate_id else None,
        str(base_model) if base_model else None,
    )


def _build_replay_schedule(
    *,
    campaign_root: Path,
    runs: Sequence[Run],
    expected_steps: int,
    strict: bool,
    inputs: dict[str, str],
) -> dict[str, Any]:
    selected_id, selected_base_model = _selected_candidate(campaign_root, inputs)
    slaclip_runs = [run for run in runs if run.arm == "slaclip"]
    four_b = [
        run
        for run in slaclip_runs
        if "4b" in run.model_slug.lower()
        or "4b" in run.base_model.lower()
        or (selected_base_model is not None and run.base_model == selected_base_model)
    ]
    if selected_id is not None:
        matching = [run for run in four_b if run.candidate_id == selected_id]
        if matching:
            four_b = matching
    model_slugs = sorted({run.model_slug for run in four_b})
    if not four_b or len(model_slugs) != 1:
        raise AnalysisError(
            "replay schedule requires exactly one selected 4B SlaClip model; "
            f"found models={model_slugs}"
        )
    trajectories: list[list[dict[str, Any]]] = []
    sources: list[dict[str, Any]] = []
    for run in sorted(four_b, key=lambda item: item.seed):
        records, digest = _read_safe_log(
            run,
            expected_steps=expected_steps,
            strict=strict,
            campaign_root=campaign_root,
            inputs=inputs,
        )
        trajectories.append(records)
        sources.append(
            {
                "seed": run.seed,
                "candidate_id": run.candidate_id,
                "config_fingerprint": run.config_fingerprint,
                "path": str(run.safe_log_path.relative_to(campaign_root)),
                "sha256": digest,
            }
        )
    source_epsilons = {
        _finite(
            run.config.get("dp_epsilon"),
            label=f"{run.status_path}:config.dp_epsilon",
            minimum=0.0,
        )
        for run in four_b
    }
    source_deltas = {
        _finite(
            run.config.get("dp_delta"),
            label=f"{run.status_path}:config.dp_delta",
            minimum=0.0,
            maximum=1.0,
        )
        for run in four_b
    }
    if len(source_epsilons) != 1 or len(source_deltas) != 1:
        raise AnalysisError(
            "replay schedule source runs must share one declared DP budget"
        )
    source_epsilon = next(iter(source_epsilons))
    source_delta = next(iter(source_deltas))
    source_count = len(four_b)
    schedule_composed_epsilon = source_count * source_epsilon
    schedule_composed_delta = source_count * source_delta
    schedule_plus_replay_epsilon = (source_count + 1) * source_epsilon
    schedule_plus_replay_delta = (source_count + 1) * source_delta
    thresholds: list[float] = []
    per_step: list[dict[str, Any]] = []
    for index in range(expected_steps):
        values = [trajectory[index]["dp_clip_threshold"] for trajectory in trajectories]
        mean_value = math.fsum(values) / len(values)
        if not math.isfinite(mean_value) or mean_value <= 0.0:
            raise AnalysisError(f"replay threshold at step {index + 1} is not positive/finite")
        thresholds.append(mean_value)
        per_step.append(
            {
                "step": index + 1,
                "clip_threshold": mean_value,
                "source_seed_count": len(values),
                "source_min": min(values),
                "source_max": max(values),
                "source_sample_sd": statistics.stdev(values) if len(values) >= 2 else 0.0,
            }
        )
    core: dict[str, Any] = {
        "schema_version": REPLAY_SCHEMA_VERSION,
        "schedule_id": (
            f"selected-slaclip-dp-derived-mean-{model_slugs[0]}-{expected_steps}steps"
        ),
        "description": (
            "Deterministic replay of the cross-seed mean C_t trajectory from "
            "released selected-SlaClip train logs. No research_raw field is "
            "used to build the schedule, but it remains data-dependent DP "
            "post-processing and is only a mechanistic control."
        ),
        "schedule_privacy_class": (
            f"DP_DERIVED_FROM_{source_count}_SLACLIP_RUNS"
        ),
        "control_interpretation": "DATA_DEPENDENT_MECHANISTIC_CONTROL",
        "privacy_accounting": {
            "accounting_rule": "basic_composition_upper_bound",
            "conditional_replay_given_fixed_schedule": {
                "epsilon": source_epsilon,
                "delta": source_delta,
                "interpretation": (
                    "Only the replay gradient mechanism, conditional on treating "
                    "the already-derived schedule as fixed."
                ),
            },
            "schedule_source_basic_composition": {
                "source_run_count": source_count,
                "per_source_epsilon": source_epsilon,
                "per_source_delta": source_delta,
                "epsilon_upper_bound": schedule_composed_epsilon,
                "delta_upper_bound": schedule_composed_delta,
            },
            "schedule_plus_one_replay_basic_composition": {
                "release_count": source_count + 1,
                "epsilon_upper_bound": schedule_plus_replay_epsilon,
                "delta_upper_bound": schedule_plus_replay_delta,
            },
            "research_raw_bundle_privacy_class": "NON_PRIVATE",
            "warning": (
                "The analyzer bundle includes exact research_raw telemetry and "
                "must never be presented as a DP release."
            ),
        },
        "model_slug": model_slugs[0],
        "base_model": four_b[0].base_model,
        "selected_candidate_id": selected_id or four_b[0].candidate_id,
        "num_steps": expected_steps,
        "source_seed_count": len(four_b),
        "source_seeds": [run.seed for run in sorted(four_b, key=lambda item: item.seed)],
        "source_logs": sources,
        "clip_thresholds": thresholds,
        "statistics": per_step,
    }
    core["schedule_content_sha256"] = _sha256_bytes(_canonical_json(core))
    return core


def analyze(
    *,
    campaign_root: Path,
    output_dir: Path,
    window: int,
    smooth_window: int,
    strict: bool,
) -> dict[str, Any]:
    campaign_root = campaign_root.resolve(strict=True)
    output_dir = (
        output_dir if output_dir.is_absolute() else campaign_root / output_dir
    ).resolve()
    try:
        output_dir.relative_to(campaign_root)
    except ValueError as exc:
        raise AnalysisError("output_dir must remain inside campaign_root") from exc
    if window <= 0 or smooth_window <= 0:
        raise AnalysisError("window and smooth_window must be positive")
    inputs: dict[str, str] = {}
    expected_steps = _load_expected_steps(campaign_root, strict, inputs)
    runs = _discover_runs(
        campaign_root,
        expected_steps=expected_steps,
        strict=strict,
        inputs=inputs,
    )
    _validate_pairs(runs, strict)
    if expected_steps is None:
        step_counts = {len(run.rows) for run in runs}
        if len(step_counts) != 1:
            raise AnalysisError(f"runs have inconsistent step counts: {sorted(step_counts)}")
        expected_steps = next(iter(step_counts))
    step_rows = _build_step_rows(runs, smooth_window)
    summaries = _build_run_summaries(runs, step_rows, window)
    paired = _build_paired_rows(runs, summaries)
    cdf = _build_cdf_rows(runs, strict)
    associations = _build_associations(runs, summaries)
    transitions = _build_prediction_transitions(runs, strict)
    replay = _build_replay_schedule(
        campaign_root=campaign_root,
        runs=runs,
        expected_steps=expected_steps,
        strict=strict,
        inputs=inputs,
    )
    payloads = {
        "step_metrics.csv": _csv_bytes(
            step_rows,
            (
                "NON_PRIVATE_TELEMETRY",
                "warning",
                "model_slug",
                "seed",
                "arm",
                "candidate_id",
                "step",
                "loss_mean",
                "loss_rolling_mean",
                "dp_clip_threshold",
                "raw_clip_fraction",
                "raw_signal_retention_ratio_recomputed",
                "raw_clipping_bias_ratio_recomputed",
                "raw_signal_to_noise_ratio",
                "raw_bias_noise_squared_proxy",
            ),
        ),
        "run_summaries.csv": _csv_bytes(
            summaries,
            (
                "NON_PRIVATE_TELEMETRY",
                "warning",
                "model_slug",
                "seed",
                "arm",
                "candidate_id",
                "steps",
                "accuracy_macro",
                "loss_auc",
                "loss_ols_slope",
            ),
        ),
        "paired_mechanism.csv": _csv_bytes(
            paired,
            (
                "NON_PRIVATE_TELEMETRY",
                "warning",
                "scope",
                "model_slug",
                "comparison",
                "metric",
                "seed",
                "n",
                "candidate_value",
                "reference_value",
                "paired_mean_delta",
                "sample_sd",
                "t95_ci_low",
                "t95_ci_high",
                "wins",
                "ties",
                "losses",
                "seeds",
            ),
        ),
        "cdf_diagnostics.csv": _csv_bytes(
            cdf,
            (
                "NON_PRIVATE_TELEMETRY",
                "warning",
                "model_slug",
                "seed",
                "step",
                "slot",
                "noised_slot",
                "oracle_source",
                "oracle_exact",
                "oracle_lower",
                "oracle_upper",
                "noise_std",
                "z_distance_to_oracle_interval",
                "noise_95_band_intersects_oracle",
                "legacy_histogram_interval_intersects_noised_95pct_band",
                "exact_noise_residual",
                "exact_standardized_noise_residual",
                "exact_within_95pct",
            ),
        ),
        "loss_accuracy_association.csv": _csv_bytes(
            associations,
            (
                "NON_PRIVATE_TELEMETRY",
                "warning",
                "analysis_level",
                "model_slug",
                "arm",
                "predictor",
                "outcome",
                "n",
                "pearson_r",
                "spearman_rho",
                "exploratory_small_n",
            ),
        ),
        "prediction_transitions.csv": _csv_bytes(
            transitions,
            (
                "NON_PRIVATE_TELEMETRY",
                "warning",
                "model_slug",
                "seed",
                "comparison",
                "task",
                "n",
                "both_wrong_00",
                "baseline_wrong_slaclip_right_01",
                "baseline_right_slaclip_wrong_10",
                "both_right_11",
                "accuracy_delta",
                "mcnemar_exact_two_sided_p",
            ),
        ),
        "replay_schedule.json": (
            json.dumps(replay, indent=2, sort_keys=True, allow_nan=False) + "\n"
        ).encode("utf-8"),
    }
    script_path = Path(__file__).resolve()
    manifest: dict[str, Any] = {
        "analysis_schema_version": ANALYSIS_SCHEMA_VERSION,
        "analysis_status": "COMPLETE",
        "NON_PRIVATE_TELEMETRY": True,
        "warning": WARNING,
        "strict": bool(strict),
        "campaign_root": str(campaign_root),
        "window": window,
        "smooth_window": smooth_window,
        "expected_steps": expected_steps,
        "run_count": len(runs),
        "models": sorted({run.model_slug for run in runs}),
        "seeds": sorted({run.seed for run in runs}),
        "arms": sorted({run.arm for run in runs}),
        "row_counts": {
            "step_metrics.csv": len(step_rows),
            "run_summaries.csv": len(summaries),
            "paired_mechanism.csv": len(paired),
            "cdf_diagnostics.csv": len(cdf),
            "loss_accuracy_association.csv": len(associations),
            "prediction_transitions.csv": len(transitions),
            "replay_schedule.json": len(replay["clip_thresholds"]),
        },
        "analyzer": {
            "path": str(script_path),
            "sha256": _sha256_file(script_path),
        },
        "inputs": dict(sorted(inputs.items())),
        "outputs": {
            name: _sha256_bytes(content) for name, content in sorted(payloads.items())
        },
        "replay_schedule_content_sha256": replay["schedule_content_sha256"],
    }
    for name, content in payloads.items():
        _atomic_write(output_dir / name, content)
    manifest_bytes = (
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    _atomic_write(output_dir / "manifest.json", manifest_bytes)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("deep-analysis-v1"),
        help="Output directory, absolute or relative to campaign root.",
    )
    parser.add_argument("--window", type=int, default=50)
    parser.add_argument("--smooth-window", type=int, default=25)
    parser.add_argument(
        "--strict",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fail closed on incomplete, mismatched, or unsafe artifacts (default).",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        manifest = analyze(
            campaign_root=args.campaign_root,
            output_dir=args.output_dir,
            window=args.window,
            smooth_window=args.smooth_window,
            strict=args.strict,
        )
        output_dir = (
            args.output_dir
            if args.output_dir.is_absolute()
            else args.campaign_root / args.output_dir
        )
        print(f"WARNING: {WARNING}")
        print(
            f"Analyzed {manifest['run_count']} runs and "
            f"{manifest['row_counts']['step_metrics.csv']} steps"
        )
        print(f"Output: {output_dir.resolve()}")
        return 0
    except AnalysisError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
