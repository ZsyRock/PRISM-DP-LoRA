#!/usr/bin/env python3
"""Summarize PRISM research telemetry with Python's standard library.

The raw JSONL is authoritative and every non-empty record must contain
``"NON_PRIVATE_TELEMETRY": true``. An optional DP-safe train log can fill fields
in older raw-log formats; raw values always win when the same key exists.

Typical use::

    python scripts/summarize_telemetry.py \
      LLM-Adapters/experiment/RUN/research_raw/NON_PRIVATE_train_log.jsonl \
      --safe-log LLM-Adapters/trained_models/RUN/train_log.jsonl \
      --format both

By default this writes ``telemetry_steps.csv`` and ``telemetry_summary.json``
next to the raw log. The CSV contains one row per update, including flattened
gradient-norm quantiles. The JSON contains source hashes, run identity, duplicate
and missing-step diagnostics, field coverage, and numeric aggregate statistics.

These outputs remain NON-PRIVATE research artifacts and must not be published as
part of a differentially private release.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence


SUMMARY_SCHEMA_VERSION = 4
NON_PRIVATE_WARNING = (
    "Contains exact statistics derived from training examples; this summary is "
    "NON-PRIVATE and must not be treated as a DP release."
)

IDENTITY_FIELDS = (
    "run_id",
    "config_fingerprint",
    "method",
    "privacy",
    "dataset",
    "base_model",
    "model_revision",
    "resolved_model_revision",
    "loss_definition",
)

QUANTILES = ("0.1", "0.25", "0.5", "0.75", "0.9", "0.95", "0.99")
QUANTILE_COLUMNS = {
    "0.1": "raw_global_norm_q10",
    "0.25": "raw_global_norm_q25",
    "0.5": "raw_global_norm_q50",
    "0.75": "raw_global_norm_q75",
    "0.9": "raw_global_norm_q90",
    "0.95": "raw_global_norm_q95",
    "0.99": "raw_global_norm_q99",
}

FIELD_ALIASES = {
    # Early development logs used these shorter names.
    "dp_noise_norm": "raw_realized_noise_norm",
    "dp_signal_norm": "raw_clipped_signal_norm",
    "dp_clip_frac": "raw_clip_fraction",
    "dp_noisy_gradient_norm": "dp_noisy_tangent_gradient_norm",
    "dp_update_norm": "dp_factor_product_update_norm",
    "epsilon": "eps_spent",
    # ``beta`` was the historical name for full SlaClip's configured target
    # clipped fraction within the non-small-gradient mass.
    "slaclip_beta": "slaclip_target_non_small_clip_fraction",
}

PREFERRED_COLUMNS = (
    "NON_PRIVATE_TELEMETRY",
    "telemetry_schema_version",
    "run_id",
    "config_fingerprint",
    "method",
    "privacy",
    "dataset",
    "base_model",
    "model_revision",
    "resolved_model_revision",
    "loss_definition",
    "step",
    "loss_mean",
    "loss_delta",
    "tokens",
    "batch_n",
    "raw_realized_batch_size",
    "dp_expected_batch_size",
    "dp_clip_threshold",
    "dp_next_clip_threshold",
    "clip_threshold_delta",
    "clip_threshold_ratio",
    "replay_schedule_index",
    "replay_clip_schedule_sha256",
    "slaclip_gamma_t",
    "slaclip_eta",
    "slaclip_controller",
    "slaclip_target_non_small_clip_fraction",
    "slaclip_beta",
    "slaclip_small_gradient_proxy_noisy",
    "slaclip_remaining_mass_proxy_noisy",
    "slaclip_target_unclipped_proxy_preprojection",
    "slaclip_target_clip_fraction",
    "slaclip_observed_unclipped_proxy",
    "slaclip_target_unclipped_proxy",
    "slaclip_target_clipped_proxy",
    "slaclip_controller_error",
    "slaclip_c_next_unbounded",
    "slaclip_c_min",
    "slaclip_c_max",
    "slaclip_c_hit_min",
    "slaclip_c_hit_max",
    "slaclip_num_slots",
    "slack_indicator_noise_std",
    "raw_global_norm_mean",
    "raw_global_norm_std",
    "raw_global_norm_min",
    "raw_global_norm_max",
    *QUANTILE_COLUMNS.values(),
    "raw_global_norm_hist_counts_json",
    "raw_global_norm_hist_edges_json",
    "raw_global_norm_hist_overflow",
    "raw_clip_fraction",
    "raw_clip_fraction_reference_target",
    "raw_clip_fraction_error_target_kind",
    "raw_clip_fraction_error",
    "raw_clip_coefficient_mean",
    "raw_clip_coefficient_min",
    "raw_unclipped_signal_norm",
    "raw_clipped_signal_norm",
    "raw_signal_retention_ratio",
    "raw_clipping_bias_norm",
    "raw_clipping_bias_ratio",
    "raw_realized_noise_norm",
    "raw_signal_to_noise_ratio",
    "dp_noisy_tangent_gradient_norm",
    "dp_factor_product_update_norm",
    "dp_update_clip_coef_min",
    "dp_noise_multiplier",
    "dp_std_per_factor",
    "slack_unclipped_proxy",
    "slack_clipped_proxy",
    "slack_indicator_json",
    "raw_slack_indicator_json",
    "raw_slack_indicator_noise_residual_json",
    "raw_slack_indicator_noise_residual_l2",
    "raw_slack_indicator_noise_residual_rmse",
    "raw_slack_indicator_noise_residual_first_coordinate",
    "raw_unclipped_clipped_cosine",
    "raw_clipped_noisy_cosine",
    "raw_clipping_bias_to_noise_ratio",
    "raw_bias_noise_squared_error_proxy",
    "eps_spent",
    "epsilon_increment",
)

BOOLEAN_METRIC_COLUMNS = (
    "slaclip_c_hit_min",
    "slaclip_c_hit_max",
)

AGGREGATE_EXCLUSIONS = {
    "NON_PRIVATE_TELEMETRY",
    "telemetry_schema_version",
    "step",
    "lora_r",
}


class TelemetryError(ValueError):
    """Raised when a telemetry input is malformed or unsafe to summarize."""


@dataclass
class LoadedLog:
    path: Path
    records: Dict[int, Dict[str, Any]]
    physical_records: int
    duplicate_records: int
    sha256: str


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _coerce_step(value: Any, *, path: Path, line_number: int) -> int:
    if isinstance(value, bool):
        raise TelemetryError(f"{path}:{line_number}: step must be an integer, not boolean")
    if isinstance(value, int):
        step = value
    elif isinstance(value, float) and value.is_integer():
        step = int(value)
    elif isinstance(value, str) and value.strip().isdigit():
        step = int(value.strip())
    else:
        raise TelemetryError(f"{path}:{line_number}: invalid step value {value!r}")
    if step <= 0:
        raise TelemetryError(f"{path}:{line_number}: step must be positive, got {step}")
    return step


def load_jsonl(
    path: Path,
    *,
    require_non_private_marker: bool,
    duplicate_policy: str,
) -> LoadedLog:
    path = path.resolve()
    if not path.is_file():
        raise TelemetryError(f"JSONL file does not exist: {path}")
    records: Dict[int, Dict[str, Any]] = {}
    physical_records = 0
    duplicate_records = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                continue
            physical_records += 1
            try:
                payload = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise TelemetryError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(payload, dict):
                raise TelemetryError(f"{path}:{line_number}: each JSONL record must be an object")
            if require_non_private_marker and payload.get("NON_PRIVATE_TELEMETRY") is not True:
                raise TelemetryError(
                    f"{path}:{line_number}: raw record lacks NON_PRIVATE_TELEMETRY=true; "
                    "refusing to misclassify the input"
                )
            if "step" not in payload:
                raise TelemetryError(f"{path}:{line_number}: record has no step")
            step = _coerce_step(payload["step"], path=path, line_number=line_number)
            payload = dict(payload)
            payload["step"] = step
            if step in records:
                duplicate_records += 1
                if duplicate_policy == "error":
                    raise TelemetryError(f"{path}:{line_number}: duplicate step {step}")
                if duplicate_policy == "first":
                    continue
            records[step] = payload
    if not records:
        raise TelemetryError(f"JSONL file contains no records: {path}")
    return LoadedLog(
        path=path,
        records=records,
        physical_records=physical_records,
        duplicate_records=duplicate_records,
        sha256=file_sha256(path),
    )


def _consistent_identity(records: Iterable[Mapping[str, Any]], source: str) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    materialized = list(records)
    for field in IDENTITY_FIELDS:
        values = []
        for record in materialized:
            value = record.get(field)
            if value is None or value == "":
                continue
            if value not in values:
                values.append(value)
        if len(values) > 1:
            raise TelemetryError(f"{source} mixes multiple {field} values: {values!r}")
        if values:
            result[field] = values[0]
    return result


def _merge_identity(raw_identity: Mapping[str, Any], safe_identity: Mapping[str, Any]) -> Dict[str, Any]:
    merged = dict(safe_identity)
    for field, value in raw_identity.items():
        if field in merged and merged[field] != value:
            raise TelemetryError(
                f"Raw and DP-safe logs disagree on {field}: raw={value!r}, safe={merged[field]!r}"
            )
        merged[field] = value
    return merged


def _json_cell(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def flatten_record(record: Mapping[str, Any]) -> Dict[str, Any]:
    flattened: Dict[str, Any] = {}
    for key, value in record.items():
        if key == "raw_global_norm_quantiles" and isinstance(value, Mapping):
            for quantile in QUANTILES:
                if quantile in value:
                    flattened[QUANTILE_COLUMNS[quantile]] = value[quantile]
            for quantile, quantile_value in value.items():
                text = str(quantile)
                if text not in QUANTILE_COLUMNS:
                    safe = text.replace("-", "m").replace(".", "p")
                    flattened[f"raw_global_norm_quantile_{safe}"] = quantile_value
        elif isinstance(value, (list, dict)):
            flattened[f"{key}_json"] = _json_cell(value)
        else:
            flattened[key] = value
    for old_name, canonical_name in FIELD_ALIASES.items():
        if canonical_name not in flattened and old_name in flattened:
            flattened[canonical_name] = flattened[old_name]
    return flattened


def _finite_number(value: Any) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _safe_ratio(numerator: Any, denominator: Any) -> Optional[float]:
    num = _finite_number(numerator)
    den = _finite_number(denominator)
    if num is None or den is None or den == 0:
        return None
    value = num / den
    return value if math.isfinite(value) else None


def add_derived_fields(rows: Sequence[Dict[str, Any]]) -> None:
    previous_loss: Optional[float] = None
    previous_epsilon: Optional[float] = None
    for row in rows:
        current_c = _finite_number(row.get("dp_clip_threshold"))
        next_c = _finite_number(row.get("dp_next_clip_threshold"))
        if current_c is not None and next_c is not None:
            row["clip_threshold_delta"] = next_c - current_c
            ratio = _safe_ratio(next_c, current_c)
            if ratio is not None:
                row["clip_threshold_ratio"] = ratio
        retention = _safe_ratio(
            row.get("raw_clipped_signal_norm"), row.get("raw_unclipped_signal_norm")
        )
        if retention is not None:
            row["raw_signal_retention_ratio"] = retention
        bias_ratio = _safe_ratio(
            row.get("raw_clipping_bias_norm"), row.get("raw_unclipped_signal_norm")
        )
        if bias_ratio is not None:
            row["raw_clipping_bias_ratio"] = bias_ratio
        proxy = _finite_number(row.get("slaclip_observed_unclipped_proxy"))
        if proxy is None:
            proxy = _finite_number(row.get("slack_unclipped_proxy"))
        target_proxy = _finite_number(row.get("slaclip_target_unclipped_proxy"))
        if proxy is not None and target_proxy is not None:
            row["slaclip_controller_error"] = target_proxy - proxy
        raw_clip = _finite_number(row.get("raw_clip_fraction"))
        controller = str(
            row.get("slaclip_controller") or row.get("method") or ""
        ).casefold()
        reference_clip: Optional[float] = None
        reference_kind: Optional[str] = None
        if controller == "slaclip":
            # Full SlaClip's configured rho applies only to the mass remaining
            # after its near-zero/small-gradient proxy. Consequently the
            # per-step comparison target is dynamic, not rho itself.
            reference_clip = _finite_number(
                row.get("slaclip_target_clipped_proxy")
            )
            if reference_clip is None:
                legacy_gamma = _finite_number(row.get("slaclip_gamma_t"))
                if legacy_gamma is not None:
                    reference_clip = 1.0 - legacy_gamma
            reference_kind = "full_dynamic_clipped_proxy"
        elif controller == "slaclip_q":
            # SlaClip-Q deliberately ignores the near-zero endpoint and tracks
            # one fixed requested global clipped fraction.
            reference_clip = _finite_number(
                row.get("slaclip_target_clip_fraction")
            )
            reference_kind = "slaclip_q_fixed_clipped_fraction"
        if raw_clip is not None and reference_clip is not None:
            row["raw_clip_fraction_reference_target"] = reference_clip
            row["raw_clip_fraction_error_target_kind"] = reference_kind
            row["raw_clip_fraction_error"] = raw_clip - reference_clip
        loss = _finite_number(row.get("loss_mean"))
        if loss is not None:
            if previous_loss is not None:
                row["loss_delta"] = loss - previous_loss
            previous_loss = loss
        epsilon = _finite_number(row.get("eps_spent"))
        if epsilon is not None:
            if previous_epsilon is not None:
                row["epsilon_increment"] = epsilon - previous_epsilon
            previous_epsilon = epsilon


def merge_rows(raw_log: LoadedLog, safe_log: Optional[LoadedLog]) -> tuple[list[Dict[str, Any]], int]:
    rows = []
    safe_records = {} if safe_log is None else safe_log.records
    for step in sorted(raw_log.records):
        merged = dict(safe_records.get(step, {}))
        merged.update(raw_log.records[step])
        rows.append(flatten_record(merged))
    add_derived_fields(rows)
    unmatched_safe = 0 if safe_log is None else len(set(safe_records) - set(raw_log.records))
    return rows, unmatched_safe


def ordered_columns(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    discovered = {key for row in rows for key in row}
    preferred = [key for key in PREFERRED_COLUMNS if key in discovered]
    remaining = sorted(discovered - set(preferred))
    return preferred + remaining


def numeric_summaries(rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> Dict[str, Any]:
    summaries: Dict[str, Any] = {}
    total = len(rows)
    for column in columns:
        if column in AGGREGATE_EXCLUSIONS or column.endswith("_json"):
            continue
        values = []
        for row in rows:
            number = _finite_number(row.get(column))
            if number is not None:
                values.append(number)
        if not values:
            continue
        summaries[column] = {
            "count": len(values),
            "missing": total - len(values),
            "first": values[0],
            "last": values[-1],
            "min": min(values),
            "max": max(values),
            "mean": math.fsum(values) / len(values),
            "change": values[-1] - values[0],
        }
    return summaries


def boolean_summaries(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Summarize declared boolean diagnostics without coercing numbers."""
    summaries: Dict[str, Any] = {}
    total = len(rows)
    for column in BOOLEAN_METRIC_COLUMNS:
        values = [row.get(column) for row in rows if isinstance(row.get(column), bool)]
        if not values:
            continue
        true_count = sum(values)
        summaries[column] = {
            "count": len(values),
            "missing": total - len(values),
            "true_count": true_count,
            "false_count": len(values) - true_count,
            "true_rate": true_count / len(values),
        }
    return summaries


def build_summary(
    *,
    raw_log: LoadedLog,
    safe_log: Optional[LoadedLog],
    rows: Sequence[Mapping[str, Any]],
    columns: Sequence[str],
    identity: Mapping[str, Any],
    duplicate_policy: str,
    unmatched_safe_steps: int,
    csv_output: Optional[Path],
) -> Dict[str, Any]:
    steps = [int(row["step"]) for row in rows]
    missing_steps = sorted(set(range(min(steps), max(steps) + 1)) - set(steps))
    source: Dict[str, Any] = {
        "raw_log": str(raw_log.path),
        "raw_sha256": raw_log.sha256,
        "raw_physical_records": raw_log.physical_records,
        "raw_unique_steps": len(raw_log.records),
        "raw_duplicate_records": raw_log.duplicate_records,
        "duplicate_policy": duplicate_policy,
    }
    if safe_log is not None:
        source.update(
            {
                "safe_log": str(safe_log.path),
                "safe_sha256": safe_log.sha256,
                "safe_physical_records": safe_log.physical_records,
                "safe_unique_steps": len(safe_log.records),
                "safe_duplicate_records": safe_log.duplicate_records,
                "safe_steps_without_raw_record": unmatched_safe_steps,
            }
        )
    coverage = {
        column: {
            "present": sum(1 for row in rows if row.get(column) not in (None, "")),
            "missing": sum(1 for row in rows if row.get(column) in (None, "")),
        }
        for column in PREFERRED_COLUMNS
    }
    result: Dict[str, Any] = {
        "summary_schema_version": SUMMARY_SCHEMA_VERSION,
        "NON_PRIVATE_TELEMETRY": True,
        "warning": NON_PRIVATE_WARNING,
        "source": source,
        "run_identity": dict(identity),
        "steps": {
            "count": len(steps),
            "first": min(steps),
            "last": max(steps),
            "missing_count": len(missing_steps),
            "missing": missing_steps,
        },
        "columns": list(columns),
        "field_coverage": coverage,
        "metrics": numeric_summaries(rows, columns),
        "boolean_metrics": boolean_summaries(rows),
    }
    if csv_output is not None:
        result["csv_output"] = str(csv_output.resolve())
    return result


def _atomic_write(path: Path, text: str) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> None:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(columns), extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({key: "" if row.get(key) is None else row.get(key) for key in columns})
    _atomic_write(path, buffer.getvalue())


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    text = json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
    _atomic_write(path, text)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("raw_log", type=Path, help="NON_PRIVATE research_raw JSONL file")
    parser.add_argument("--safe-log", type=Path, default=None, help="Optional DP-safe train_log.jsonl")
    parser.add_argument("--format", choices=("both", "csv", "json"), default="both")
    parser.add_argument("--csv-out", type=Path, default=None)
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument(
        "--duplicate-policy",
        choices=("last", "first", "error"),
        default="last",
        help="How to handle repeated steps, commonly left by a resumed legacy run",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        raw_log = load_jsonl(
            args.raw_log,
            require_non_private_marker=True,
            duplicate_policy=args.duplicate_policy,
        )
        safe_log = None
        if args.safe_log is not None:
            safe_log = load_jsonl(
                args.safe_log,
                require_non_private_marker=False,
                duplicate_policy=args.duplicate_policy,
            )
        raw_identity = _consistent_identity(raw_log.records.values(), "raw log")
        safe_identity = (
            {} if safe_log is None else _consistent_identity(safe_log.records.values(), "DP-safe log")
        )
        identity = _merge_identity(raw_identity, safe_identity)
        rows, unmatched_safe = merge_rows(raw_log, safe_log)
        columns = ordered_columns(rows)

        default_directory = raw_log.path.parent
        csv_output = args.csv_out or default_directory / "telemetry_steps.csv"
        json_output = args.json_out or default_directory / "telemetry_summary.json"
        outputs = []
        if args.format in {"both", "csv"}:
            outputs.append(csv_output.resolve())
        if args.format in {"both", "json"}:
            outputs.append(json_output.resolve())
        inputs = {raw_log.path.resolve()}
        if safe_log is not None:
            inputs.add(safe_log.path.resolve())
        if len(set(outputs)) != len(outputs):
            raise TelemetryError("CSV and JSON outputs must be different files")
        if any(output in inputs for output in outputs):
            raise TelemetryError("Refusing to overwrite an input JSONL file")

        written_csv: Optional[Path] = None
        if args.format in {"both", "csv"}:
            write_csv(csv_output, rows, columns)
            written_csv = csv_output
        summary = build_summary(
            raw_log=raw_log,
            safe_log=safe_log,
            rows=rows,
            columns=columns,
            identity=identity,
            duplicate_policy=args.duplicate_policy,
            unmatched_safe_steps=unmatched_safe,
            csv_output=written_csv,
        )
        if args.format in {"both", "json"}:
            write_json(json_output, summary)

        print(f"WARNING: {NON_PRIVATE_WARNING}")
        if written_csv is not None:
            print(f"CSV: {written_csv.resolve()}")
        if args.format in {"both", "json"}:
            print(f"JSON: {json_output.resolve()}")
        return 0
    except TelemetryError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
