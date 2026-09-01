#!/usr/bin/env python3
"""Build a strict, NON-PRIVATE landscape of registered fixed-C baselines.

This tool is deliberately conservative.  It reads baseline arms only from each
campaign's ``plans/manifest.json`` and binds the adapter status, result status,
telemetry summary, and raw telemetry by run identity, configuration
fingerprint, and SHA-256.  A directory which merely looks like a run is never
discovered or included.

The resulting calibration statistics contain exact training-data-derived
telemetry.  They are NON_PRIVATE research artifacts and must not be published
as a differentially private release.
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
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


WARNING = (
    "NON_PRIVATE calibration: exact clipping/CDF/slack telemetry is derived "
    "from training examples and must not be treated as a DP release."
)
QUANTILES = (("q10", 0.10), ("q25", 0.25), ("median", 0.50),
             ("q75", 0.75), ("q90", 0.90))
LANDSCAPE_FIELDS = (
    "NON_PRIVATE_CALIBRATION",
    "campaign_root",
    "campaign_profile",
    "campaign_code_sha",
    "manifest_sha256",
    "arm_id",
    "setting_id",
    "run_id",
    "config_fingerprint",
    "raw_telemetry_sha256",
    "summary_sha256",
    "dataset",
    "model",
    "revision",
    "epsilon",
    "rank",
    "C",
    "seed",
    "steps",
    "burn_in_steps",
    "analysis_steps",
    "official_utility_metric",
    "official_utility",
    "clip_mean",
    "clip_q10",
    "clip_q25",
    "clip_median",
    "clip_q75",
    "clip_q90",
    "clip_IQR",
    "clip_range",
    "clip_first_half_mean",
    "clip_last_half_mean",
    "clip_last_minus_first_half_delta",
    "clip_OLS_slope_per_step",
    "clip_rolling_25_mean_range",
    "small_gradient_proxy_median",
    "remaining_proxy_median",
    "conditional_proxy_valid_records",
    "conditional_proxy_valid_fraction",
    "conditional_proxy_normalization",
    "conditional_proxy_raw_q10",
    "conditional_proxy_raw_q25",
    "conditional_proxy_raw_median",
    "conditional_proxy_raw_q75",
    "conditional_proxy_raw_q90",
    "projected_rho_q10",
    "projected_rho_q25",
    "projected_rho_median",
    "projected_rho_q75",
    "projected_rho_q90",
    "projected_rho_unique_count",
    "cdf_slack_noise_std_estimate_median",
    "small_proxy_to_noise_ratio",
    "slack_score",
    "variation_score",
    "eligible_clip_below_threshold",
    "eligible_clip_variation",
    "eligible_small_proxy_noise_ratio",
    "eligible_conditional_proxy_coverage",
    "eligible_five_unique_projected_rhos",
    "screen_eligible",
    "eligibility_reasons",
)
COVERAGE_FIELDS = (
    "campaign_root",
    "campaign_profile",
    "manifest_sha256",
    "arm_id",
    "setting_id",
    "dataset",
    "model",
    "epsilon",
    "rank",
    "C",
    "seed",
    "planned_steps",
    "status_state",
    "classification",
    "included_in_landscape",
    "reason",
)


class LandscapeError(RuntimeError):
    """Raised for a malformed campaign or unsafe output operation."""


class ArmArtifactError(RuntimeError):
    """Raised when a completed arm fails a strict artifact binding check."""


def _read_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size == 0:
        raise ArmArtifactError(f"missing {label}: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArmArtifactError(f"invalid {label}: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ArmArtifactError(f"{label} must be a JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise ArmArtifactError(f"cannot hash artifact {path}: {exc}") from exc
    return digest.hexdigest()


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ArmArtifactError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ArmArtifactError(f"{label} must be finite")
    return result


def _integer(value: Any, label: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ArmArtifactError(f"{label} must be an integer")
    if minimum is not None and value < minimum:
        raise ArmArtifactError(f"{label} must be >= {minimum}")
    return value


def _same_number(left: Any, right: Any, *, tolerance: float = 1e-10) -> bool:
    try:
        return math.isclose(
            float(left), float(right), rel_tol=0.0, abs_tol=tolerance
        )
    except (TypeError, ValueError):
        return False


def _require_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise ArmArtifactError(
            f"identity mismatch for {label}: expected {expected!r}, got {actual!r}"
        )


def _require_number(actual: Any, expected: Any, label: str) -> None:
    if not _same_number(actual, expected):
        raise ArmArtifactError(
            f"identity mismatch for {label}: expected {expected!r}, got {actual!r}"
        )


def _inside(root: Path, relative: Any, label: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise ArmArtifactError(f"{label} must be a non-empty relative path")
    candidate = Path(relative)
    if candidate.is_absolute():
        raise ArmArtifactError(f"{label} must be relative to the campaign")
    root_resolved = root.resolve()
    resolved = (root / candidate).resolve()
    try:
        resolved.relative_to(root_resolved)
    except ValueError as exc:
        raise ArmArtifactError(f"{label} escapes the campaign root: {relative}") from exc
    return resolved


def _quantile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise ArmArtifactError("cannot compute a quantile of an empty series")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _ols_slope(steps: Sequence[int], values: Sequence[float]) -> float:
    if len(values) <= 1:
        return 0.0
    x_mean = statistics.fmean(steps)
    y_mean = statistics.fmean(values)
    denominator = math.fsum((step - x_mean) ** 2 for step in steps)
    if denominator == 0.0:
        return 0.0
    numerator = math.fsum(
        (step - x_mean) * (value - y_mean)
        for step, value in zip(steps, values, strict=True)
    )
    return numerator / denominator


def _unique_float_count(values: Iterable[float], tolerance: float = 1e-10) -> int:
    unique: list[float] = []
    for value in sorted(values):
        if not unique or not math.isclose(
            value, unique[-1], rel_tol=0.0, abs_tol=tolerance
        ):
            unique.append(value)
    return len(unique)


def _manifest_identity(arm: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "arm_id": str,
        "relative_root": str,
        "dataset": str,
        "model_id": str,
        "model_revision": str,
        "epsilon": (int, float),
        "lora_r": int,
        "initial_c": (int, float),
        "seed": int,
        "steps": int,
    }
    for key, expected_type in required.items():
        value = arm.get(key)
        if isinstance(value, bool) or not isinstance(value, expected_type):
            raise ArmArtifactError(f"manifest arm {key} has the wrong type")
    steps = _integer(arm["steps"], "manifest.steps", minimum=1)
    rank = _integer(arm["lora_r"], "manifest.lora_r", minimum=1)
    return {
        "arm_id": arm["arm_id"],
        "setting_id": arm.get("setting_id", arm["arm_id"]),
        "dataset": arm["dataset"],
        "model": arm["model_id"],
        "revision": arm["model_revision"],
        "epsilon": _finite(arm["epsilon"], "manifest.epsilon"),
        "rank": rank,
        "C": _finite(arm["initial_c"], "manifest.initial_c"),
        "seed": arm["seed"],
        "steps": steps,
    }


def _coverage_base(
    root: Path,
    manifest: Mapping[str, Any],
    manifest_sha: str,
    identity: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "campaign_root": str(root),
        "campaign_profile": manifest.get("profile", "unknown"),
        "manifest_sha256": manifest_sha,
        "arm_id": identity["arm_id"],
        "setting_id": identity["setting_id"],
        "dataset": identity["dataset"],
        "model": identity["model"],
        "epsilon": identity["epsilon"],
        "rank": identity["rank"],
        "C": identity["C"],
        "seed": identity["seed"],
        "planned_steps": identity["steps"],
    }


def _validate_status(
    status: Mapping[str, Any],
    arm: Mapping[str, Any],
    identity: Mapping[str, Any],
    code_sha: Any,
    label: str,
) -> None:
    _require_equal(status.get("state"), "completed", f"{label}.state")
    _require_equal(status.get("method"), "baseline", f"{label}.method")
    _require_equal(status.get("dataset"), identity["dataset"], f"{label}.dataset")
    _require_equal(status.get("base_model"), identity["model"], f"{label}.base_model")
    _require_equal(
        status.get("resolved_model_revision"),
        identity["revision"],
        f"{label}.resolved_model_revision",
    )
    _require_equal(status.get("privacy"), "dp", f"{label}.privacy")
    _require_equal(
        status.get("non_private_telemetry"), True, f"{label}.non_private_telemetry"
    )
    _require_number(status.get("update_steps"), identity["steps"], f"{label}.update_steps")
    _require_number(
        status.get("training_lora_r"), identity["rank"], f"{label}.training_lora_r"
    )
    run_id = status.get("run_id")
    fingerprint = status.get("config_fingerprint")
    if not isinstance(run_id, str) or not run_id:
        raise ArmArtifactError(f"{label}.run_id must be non-empty")
    if not isinstance(fingerprint, str) or len(fingerprint) != 64:
        raise ArmArtifactError(f"{label}.config_fingerprint must be a SHA-256 string")
    config = status.get("config")
    if not isinstance(config, dict):
        raise ArmArtifactError(f"{label}.config must be an object")
    _require_equal(config.get("config_fingerprint"), fingerprint,
                   f"{label}.config.config_fingerprint")
    # Schema 7 writes run_id into the serialized config, while earlier locked
    # campaigns bind it only at the status/raw/summary level.  Validate it when
    # present without rejecting the older, still fingerprinted schema.
    if "run_id" in config:
        _require_equal(config.get("run_id"), run_id, f"{label}.config.run_id")
    _require_equal(config.get("method"), "baseline", f"{label}.config.method")
    _require_equal(config.get("dataset"), identity["dataset"],
                   f"{label}.config.dataset")
    _require_equal(config.get("base_model"), identity["model"],
                   f"{label}.config.base_model")
    _require_equal(config.get("resolved_model_revision"), identity["revision"],
                   f"{label}.config.resolved_model_revision")
    _require_number(config.get("dp_epsilon"), identity["epsilon"],
                    f"{label}.config.dp_epsilon")
    _require_number(config.get("lora_r"), identity["rank"],
                    f"{label}.config.lora_r")
    _require_number(config.get("dp_max_grad_norm"), identity["C"],
                    f"{label}.config.dp_max_grad_norm")
    _require_number(config.get("seed"), identity["seed"], f"{label}.config.seed")
    _require_number(config.get("total_update_steps"), identity["steps"],
                    f"{label}.config.total_update_steps")
    if code_sha is not None:
        _require_equal(config.get("implementation_git_sha"), code_sha,
                       f"{label}.config.implementation_git_sha")


def _read_and_validate_raw(
    raw_path: Path,
    status: Mapping[str, Any],
    identity: Mapping[str, Any],
) -> list[dict[str, Any]]:
    if not raw_path.is_file() or raw_path.stat().st_size == 0:
        raise ArmArtifactError(f"missing raw telemetry: {raw_path}")
    records: list[dict[str, Any]] = []
    try:
        with raw_path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    raise ArmArtifactError(
                        f"blank raw telemetry record at line {line_number}"
                    )
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ArmArtifactError(
                        f"raw telemetry line {line_number} is not an object"
                    )
                records.append(value)
    except (OSError, json.JSONDecodeError) as exc:
        raise ArmArtifactError(f"invalid raw telemetry {raw_path}: {exc}") from exc
    expected_steps = list(range(1, identity["steps"] + 1))
    actual_steps = [record.get("step") for record in records]
    if actual_steps != expected_steps:
        raise ArmArtifactError(
            "raw telemetry steps must be exactly 1..planned_steps in order; "
            f"expected {expected_steps[:3]}...{expected_steps[-3:]}, "
            f"got {actual_steps[:3]}...{actual_steps[-3:]}"
        )
    for step, record in enumerate(records, start=1):
        prefix = f"raw step {step}"
        _require_equal(record.get("NON_PRIVATE_TELEMETRY"), True,
                       f"{prefix}.NON_PRIVATE_TELEMETRY")
        _require_equal(record.get("run_id"), status["run_id"], f"{prefix}.run_id")
        _require_equal(record.get("config_fingerprint"),
                       status["config_fingerprint"], f"{prefix}.config_fingerprint")
        _require_equal(record.get("method"), "baseline", f"{prefix}.method")
        _require_equal(record.get("privacy"), "dp", f"{prefix}.privacy")
        _require_equal(record.get("dataset"), identity["dataset"], f"{prefix}.dataset")
        _require_equal(record.get("base_model"), identity["model"],
                       f"{prefix}.base_model")
        _require_equal(record.get("resolved_model_revision"), identity["revision"],
                       f"{prefix}.resolved_model_revision")
        _require_number(record.get("dp_clip_threshold"), identity["C"],
                        f"{prefix}.dp_clip_threshold")
        _require_number(record.get("dp_next_clip_threshold"), identity["C"],
                        f"{prefix}.dp_next_clip_threshold")
    return records


def _validate_summary(
    summary: Mapping[str, Any],
    status: Mapping[str, Any],
    identity: Mapping[str, Any],
    raw_path: Path,
    raw_sha: str,
) -> None:
    _require_equal(summary.get("NON_PRIVATE_TELEMETRY"), True,
                   "telemetry_summary.NON_PRIVATE_TELEMETRY")
    run_identity = summary.get("run_identity")
    if not isinstance(run_identity, dict):
        raise ArmArtifactError("telemetry_summary.run_identity must be an object")
    for key, expected in {
        "run_id": status["run_id"],
        "config_fingerprint": status["config_fingerprint"],
        "method": "baseline",
        "privacy": "dp",
        "dataset": identity["dataset"],
        "base_model": identity["model"],
        "resolved_model_revision": identity["revision"],
    }.items():
        _require_equal(run_identity.get(key), expected,
                       f"telemetry_summary.run_identity.{key}")
    source = summary.get("source")
    if not isinstance(source, dict):
        raise ArmArtifactError("telemetry_summary.source must be an object")
    _require_equal(source.get("raw_sha256"), raw_sha,
                   "telemetry_summary.source.raw_sha256")
    for key in ("raw_physical_records", "raw_unique_steps"):
        _require_number(source.get(key), identity["steps"],
                        f"telemetry_summary.source.{key}")
    _require_number(source.get("raw_duplicate_records"), 0,
                    "telemetry_summary.source.raw_duplicate_records")
    steps = summary.get("steps")
    if not isinstance(steps, dict):
        raise ArmArtifactError("telemetry_summary.steps must be an object")
    expected = identity["steps"]
    _require_number(steps.get("count"), expected, "telemetry_summary.steps.count")
    _require_number(steps.get("first"), 1, "telemetry_summary.steps.first")
    _require_number(steps.get("last"), expected, "telemetry_summary.steps.last")
    _require_equal(steps.get("missing"), [], "telemetry_summary.steps.missing")
    _require_number(steps.get("missing_count"), 0,
                    "telemetry_summary.steps.missing_count")


def _analyze_records(
    records: Sequence[Mapping[str, Any]],
    identity: Mapping[str, Any],
    *,
    burn_in_fraction: float,
    minimum_burn_in_steps: int,
    rho_min: float,
    rho_max: float,
    clip_median_max: float,
    clip_iqr_min: float,
    clip_half_delta_min: float,
    small_proxy_noise_ratio_min: float,
    conditional_valid_fraction_min: float,
) -> dict[str, Any]:
    burn_in_steps = max(
        minimum_burn_in_steps,
        math.ceil(identity["steps"] * burn_in_fraction),
    )
    post = records[burn_in_steps:]
    if not post:
        raise ArmArtifactError("burn-in removed every telemetry record")
    clip: list[float] = []
    small: list[float] = []
    remaining: list[float] = []
    conditional: list[float] = []
    noise_std: list[float] = []
    conditional_normalizations: set[str] = set()
    post_steps: list[int] = []
    for record in post:
        step = _integer(record.get("step"), "raw.step", minimum=1)
        clip_value = _finite(record.get("raw_clip_fraction"),
                             f"raw step {step}.raw_clip_fraction")
        if not 0.0 <= clip_value <= 1.0:
            raise ArmArtifactError(f"raw step {step}.raw_clip_fraction is outside [0,1]")
        small_value = _finite(record.get("raw_reference_small_gradient_proxy"),
                              f"raw step {step}.small_gradient_proxy")
        remaining_value = _finite(record.get("raw_reference_remaining_mass_proxy"),
                                  f"raw step {step}.remaining_mass_proxy")
        if small_value < 0.0 or remaining_value < 0.0:
            raise ArmArtifactError(f"raw step {step} has a negative CDF mass proxy")
        clip.append(clip_value)
        small.append(small_value)
        remaining.append(remaining_value)
        post_steps.append(step)
        multiplier = _finite(record.get("dp_noise_multiplier"),
                             f"raw step {step}.dp_noise_multiplier")
        expected_batch = _finite(record.get("dp_expected_batch_size"),
                                 f"raw step {step}.dp_expected_batch_size")
        slots_value = _finite(record.get("raw_reference_slaclip_num_slots"),
                              f"raw step {step}.slaclip_num_slots")
        slots = round(slots_value)
        if multiplier <= 0.0 or expected_batch <= 0.0 or slots <= 0 or not math.isclose(
            slots_value, slots, rel_tol=0.0, abs_tol=1e-10
        ):
            raise ArmArtifactError(
                f"raw step {step} has invalid noise multiplier, expected batch, or K"
            )
        normalization = record.get(
            "raw_reference_expected_batch_size_normalization"
        )
        if normalization is not None:
            _require_number(normalization, expected_batch,
                            f"raw step {step}.expected_batch_normalization")
        realized_batch = record.get("raw_realized_batch_size")
        conditional_valid = record.get(
            "raw_reference_conditional_clip_fraction_valid", True
        )
        conditional_value = record.get("raw_reference_conditional_clip_fraction")
        if conditional_valid not in (True, False):
            raise ArmArtifactError(
                f"raw step {step}.conditional validity flag must be boolean"
            )
        expected_valid = remaining_value > 1e-12
        if conditional_valid != expected_valid:
            raise ArmArtifactError(
                f"raw step {step}.conditional validity disagrees with remaining mass"
            )
        if conditional_valid and conditional_value is not None:
            logged = _finite(
                conditional_value,
                f"raw step {step}.conditional_clip_fraction",
            )
            if realized_batch is None:
                # Some early synthetic/unit fixtures did not retain realized
                # batch size. Preserve their declared legacy coordinate, but
                # label it so it cannot be confused with current calibration.
                value = logged
                conditional_normalizations.add(
                    "legacy_logged_realized_batch_fraction"
                )
            else:
                realized = _finite(
                    realized_batch, f"raw step {step}.raw_realized_batch_size"
                )
                if realized <= 0:
                    raise ArmArtifactError(
                        f"raw step {step}.realized batch must be positive"
                    )
                expected_normalized_clip_mass = (
                    clip_value * realized / expected_batch
                )
                value = expected_normalized_clip_mass / remaining_value
                conditional_normalizations.add(
                    "recomputed_expected_batch_size"
                )
                if int(record.get("telemetry_schema_version", 0)) >= 7:
                    if record.get("raw_reference_conditional_normalization") != (
                        "expected_batch_size"
                    ):
                        raise ArmArtifactError(
                            f"raw step {step}.conditional normalization marker is invalid"
                        )
                    _require_number(
                        record.get("raw_reference_expected_normalized_clip_mass"),
                        expected_normalized_clip_mass,
                        f"raw step {step}.expected-normalized clip mass",
                    )
                    _require_number(
                        logged,
                        value,
                        f"raw step {step}.conditional clip mass",
                    )
            if value < 0.0:
                raise ArmArtifactError(
                    f"raw step {step}.conditional_clip_fraction is negative"
                )
            conditional.append(value)
        elif conditional_valid:
            raise ArmArtifactError(
                f"raw step {step}.valid conditional proxy is missing"
            )
        noise_std.append(multiplier * math.sqrt(slots) / expected_batch)

    clip_quantiles = {name: _quantile(clip, probability)
                      for name, probability in QUANTILES}
    conditional_quantiles: dict[str, float | None] = {
        name: (_quantile(conditional, probability) if conditional else None)
        for name, probability in QUANTILES
    }
    projected: dict[str, float | None] = {
        name: (min(rho_max, max(rho_min, value)) if value is not None else None)
        for name, value in conditional_quantiles.items()
    }
    projected_values = [value for value in projected.values() if value is not None]
    unique_count = _unique_float_count(projected_values)
    conditional_valid_fraction = len(conditional) / len(post)
    split = len(clip) // 2
    if split == 0:
        first_half = last_half = clip
    else:
        first_half = clip[:split]
        last_half = clip[-split:]
    first_mean = statistics.fmean(first_half)
    last_mean = statistics.fmean(last_half)
    delta = last_mean - first_mean
    clip_iqr = clip_quantiles["q75"] - clip_quantiles["q25"]
    clip_range = max(clip) - min(clip)
    noise_median = _quantile(noise_std, 0.5)
    small_median = _quantile(small, 0.5)
    ratio = small_median / noise_median
    rolling_window = min(25, len(clip))
    rolling_means = [
        statistics.fmean(clip[start:start + rolling_window])
        for start in range(0, len(clip) - rolling_window + 1)
    ]
    rolling_range = max(rolling_means) - min(rolling_means)
    eligible_clip = clip_quantiles["median"] < clip_median_max
    eligible_variation = (
        clip_iqr >= clip_iqr_min or abs(delta) >= clip_half_delta_min
    )
    eligible_ratio = ratio >= small_proxy_noise_ratio_min
    eligible_conditional = conditional_valid_fraction >= conditional_valid_fraction_min
    eligible_targets = len(projected_values) == 5 and unique_count == 5
    reasons: list[str] = []
    if not eligible_clip:
        reasons.append(f"clip_median>={clip_median_max:g}")
    if not eligible_variation:
        reasons.append(
            "clip_variation_below_thresholds"
            f"(IQR<{clip_iqr_min:g} and abs_delta<{clip_half_delta_min:g})"
        )
    if not eligible_ratio:
        reasons.append(
            f"small_proxy_to_noise_ratio<{small_proxy_noise_ratio_min:g}"
        )
    if not eligible_conditional:
        reasons.append(
            "conditional_proxy_valid_fraction<"
            f"{conditional_valid_fraction_min:g}"
        )
    if not eligible_targets:
        reasons.append(
            f"five_unique_projected_rhos_unavailable(count={unique_count})"
        )
    eligible = not reasons
    return {
        "burn_in_steps": burn_in_steps,
        "analysis_steps": len(post),
        "clip_mean": statistics.fmean(clip),
        "clip_q10": clip_quantiles["q10"],
        "clip_q25": clip_quantiles["q25"],
        "clip_median": clip_quantiles["median"],
        "clip_q75": clip_quantiles["q75"],
        "clip_q90": clip_quantiles["q90"],
        "clip_IQR": clip_iqr,
        "clip_range": clip_range,
        "clip_first_half_mean": first_mean,
        "clip_last_half_mean": last_mean,
        "clip_last_minus_first_half_delta": delta,
        "clip_OLS_slope_per_step": _ols_slope(post_steps, clip),
        "clip_rolling_25_mean_range": rolling_range,
        "small_gradient_proxy_median": small_median,
        "remaining_proxy_median": _quantile(remaining, 0.5),
        "conditional_proxy_valid_records": len(conditional),
        "conditional_proxy_valid_fraction": conditional_valid_fraction,
        "conditional_proxy_normalization": "+".join(
            sorted(conditional_normalizations)
        ),
        **{f"conditional_proxy_raw_{name}": value
           for name, value in conditional_quantiles.items()},
        **{f"projected_rho_{name}": value for name, value in projected.items()},
        "projected_rho_unique_count": unique_count,
        "cdf_slack_noise_std_estimate_median": noise_median,
        "small_proxy_to_noise_ratio": ratio,
        "slack_score": 1.0 - clip_quantiles["median"],
        "variation_score": max(clip_iqr, abs(delta), rolling_range),
        "eligible_clip_below_threshold": eligible_clip,
        "eligible_clip_variation": eligible_variation,
        "eligible_small_proxy_noise_ratio": eligible_ratio,
        "eligible_conditional_proxy_coverage": eligible_conditional,
        "eligible_five_unique_projected_rhos": eligible_targets,
        "screen_eligible": eligible,
        "eligibility_reasons": "eligible" if eligible else ";".join(reasons),
    }


def _inspect_completed_arm(
    root: Path,
    manifest: Mapping[str, Any],
    manifest_sha: str,
    arm: Mapping[str, Any],
    identity: Mapping[str, Any],
    *,
    burn_in_fraction: float,
    minimum_burn_in_steps: int,
    rho_min: float,
    rho_max: float,
    clip_median_max: float,
    clip_iqr_min: float,
    clip_half_delta_min: float,
    small_proxy_noise_ratio_min: float,
    conditional_valid_fraction_min: float,
) -> dict[str, Any]:
    arm_root = _inside(root, arm.get("relative_root"), "manifest.relative_root")
    adapter_status_path = arm_root / "adapter" / "run_status.json"
    result_status_path = arm_root / "results" / "run_status.json"
    raw_path = arm_root / "results" / "research_raw" / "NON_PRIVATE_train_log.jsonl"
    summary_path = arm_root / "results" / "research_raw" / "telemetry_summary.json"
    utility_summary_path = arm_root / "results" / "summary.csv"
    adapter_status = _read_json(adapter_status_path, "adapter run status")
    result_status = _read_json(result_status_path, "result run status")
    code_sha = manifest.get("code_sha")
    _validate_status(adapter_status, arm, identity, code_sha, "adapter_status")
    _validate_status(result_status, arm, identity, code_sha, "result_status")
    for key in (
        "run_id", "config_fingerprint", "data_content_sha256", "method",
        "dataset", "base_model", "resolved_model_revision", "update_steps",
    ):
        _require_equal(result_status.get(key), adapter_status.get(key),
                       f"adapter/result status {key}")
    records = _read_and_validate_raw(raw_path, adapter_status, identity)
    raw_sha = _sha256(raw_path)
    summary = _read_json(summary_path, "telemetry summary")
    _validate_summary(summary, adapter_status, identity, raw_path, raw_sha)
    official_utility_metric: str | None = None
    official_utility: float | None = None
    utility_summary_sha: str | None = None
    profile = str(manifest.get("profile", ""))
    requires_official_summary = (
        profile.startswith("baseline-reproduction")
        or profile in {
            "baseline-gap-fill-cached",
            "baseline-gap-fill-all-cached",
            "baseline-gap-fill-math-only-cached",
        }
    )
    if utility_summary_path.is_file():
        try:
            with utility_summary_path.open(encoding="utf-8", newline="") as handle:
                utility_rows = list(csv.DictReader(handle))
        except (OSError, csv.Error) as exc:
            raise ArmArtifactError(
                f"invalid official utility summary: {utility_summary_path}: {exc}"
            ) from exc
        if len(utility_rows) != 1:
            raise ArmArtifactError(
                "official utility summary must contain exactly one data row"
            )
        utility_row = utility_rows[0]
        populated = [
            metric for metric in ("Average", "GLUE8_Avg", "Math10K_Avg")
            if utility_row.get(metric) not in (None, "")
        ]
        if len(populated) != 1:
            raise ArmArtifactError(
                "official utility summary must contain exactly one of "
                "Average/GLUE8_Avg/Math10K_Avg"
            )
        official_utility_metric = populated[0]
        try:
            official_utility = float(utility_row[official_utility_metric])
        except (TypeError, ValueError) as exc:
            raise ArmArtifactError("official utility must be numeric") from exc
        if not math.isfinite(official_utility):
            raise ArmArtifactError("official utility must be finite")
        utility_summary_sha = _sha256(utility_summary_path)
    elif requires_official_summary:
        raise ArmArtifactError(
            f"missing official utility summary: {utility_summary_path}"
        )
    statistics_row = _analyze_records(
        records,
        identity,
        burn_in_fraction=burn_in_fraction,
        minimum_burn_in_steps=minimum_burn_in_steps,
        rho_min=rho_min,
        rho_max=rho_max,
        clip_median_max=clip_median_max,
        clip_iqr_min=clip_iqr_min,
        clip_half_delta_min=clip_half_delta_min,
        small_proxy_noise_ratio_min=small_proxy_noise_ratio_min,
        conditional_valid_fraction_min=conditional_valid_fraction_min,
    )
    return {
        "NON_PRIVATE_CALIBRATION": True,
        "campaign_root": str(root),
        "campaign_profile": manifest.get("profile", "unknown"),
        "campaign_code_sha": code_sha,
        "manifest_sha256": manifest_sha,
        "arm_id": identity["arm_id"],
        "setting_id": identity["setting_id"],
        "run_id": adapter_status["run_id"],
        "config_fingerprint": adapter_status["config_fingerprint"],
        "raw_telemetry_sha256": raw_sha,
        "summary_sha256": utility_summary_sha,
        "dataset": identity["dataset"],
        "model": identity["model"],
        "revision": identity["revision"],
        "epsilon": identity["epsilon"],
        "rank": identity["rank"],
        "C": identity["C"],
        "seed": identity["seed"],
        "steps": identity["steps"],
        "official_utility_metric": official_utility_metric,
        "official_utility": official_utility,
        **statistics_row,
    }


def analyze_campaigns(
    campaign_roots: Sequence[Path],
    *,
    burn_in_fraction: float = 0.10,
    minimum_burn_in_steps: int = 50,
    rho_min: float = 0.05,
    rho_max: float = 0.95,
    clip_median_max: float = 0.90,
    clip_iqr_min: float = 0.05,
    clip_half_delta_min: float = 0.05,
    small_proxy_noise_ratio_min: float = 2.0,
    conditional_valid_fraction_min: float = 0.99,
) -> dict[str, Any]:
    if not 0.0 <= burn_in_fraction < 1.0:
        raise LandscapeError("burn-in fraction must be in [0,1)")
    if isinstance(minimum_burn_in_steps, bool) or minimum_burn_in_steps < 0:
        raise LandscapeError("minimum burn-in steps must be a non-negative integer")
    if not 0.0 <= rho_min < rho_max <= 1.0:
        raise LandscapeError("rho bounds must satisfy 0 <= min < max <= 1")
    if not 0.0 < clip_median_max <= 1.0:
        raise LandscapeError("clip median threshold must be in (0,1]")
    for label, value in (
        ("clip IQR threshold", clip_iqr_min),
        ("clip half-delta threshold", clip_half_delta_min),
        ("small-proxy/noise threshold", small_proxy_noise_ratio_min),
    ):
        if not math.isfinite(value) or value < 0.0:
            raise LandscapeError(f"{label} must be finite and non-negative")
    if not 0.0 <= conditional_valid_fraction_min <= 1.0:
        raise LandscapeError("conditional valid-fraction threshold must be in [0,1]")
    roots: list[Path] = []
    seen_roots: set[Path] = set()
    for supplied in campaign_roots:
        resolved = supplied.resolve()
        if resolved not in seen_roots:
            roots.append(resolved)
            seen_roots.add(resolved)
    if not roots:
        raise LandscapeError("at least one campaign root is required")
    landscape: list[dict[str, Any]] = []
    coverage: list[dict[str, Any]] = []
    invalid_count = 0
    incomplete_count = 0
    for root in roots:
        manifest_path = root / "plans" / "manifest.json"
        try:
            manifest = _read_json(manifest_path, "campaign manifest")
            manifest_sha = _sha256(manifest_path)
        except ArmArtifactError as exc:
            raise LandscapeError(str(exc)) from exc
        arms = manifest.get("arms")
        if not isinstance(arms, list):
            raise LandscapeError(f"manifest arms must be a list: {manifest_path}")
        baseline_arms = [arm for arm in arms
                         if isinstance(arm, dict) and arm.get("method") == "baseline"]
        arm_ids = [arm.get("arm_id") for arm in baseline_arms]
        if len(set(arm_ids)) != len(arm_ids):
            raise LandscapeError(f"duplicate registered baseline arm_id: {manifest_path}")
        for arm in baseline_arms:
            try:
                identity = _manifest_identity(arm)
            except ArmArtifactError as exc:
                raise LandscapeError(
                    f"malformed registered baseline arm in {manifest_path}: {exc}"
                ) from exc
            base = _coverage_base(root, manifest, manifest_sha, identity)
            try:
                arm_root = _inside(root, arm.get("relative_root"),
                                   "manifest.relative_root")
            except ArmArtifactError as exc:
                coverage.append({
                    **base, "status_state": "unknown", "classification": "invalid",
                    "included_in_landscape": False, "reason": str(exc),
                })
                invalid_count += 1
                continue
            adapter_status_path = arm_root / "adapter" / "run_status.json"
            if not adapter_status_path.is_file():
                coverage.append({
                    **base, "status_state": "missing", "classification": "incomplete",
                    "included_in_landscape": False,
                    "reason": "missing adapter/run_status.json",
                })
                incomplete_count += 1
                continue
            try:
                adapter_status = _read_json(adapter_status_path, "adapter run status")
            except ArmArtifactError as exc:
                coverage.append({
                    **base, "status_state": "invalid", "classification": "invalid",
                    "included_in_landscape": False, "reason": str(exc),
                })
                invalid_count += 1
                continue
            state = adapter_status.get("state")
            if state != "completed":
                coverage.append({
                    **base, "status_state": state if isinstance(state, str) else "unknown",
                    "classification": "incomplete", "included_in_landscape": False,
                    "reason": f"adapter state is {state!r}, not 'completed'",
                })
                incomplete_count += 1
                continue
            try:
                row = _inspect_completed_arm(
                    root, manifest, manifest_sha, arm, identity,
                    burn_in_fraction=burn_in_fraction,
                    minimum_burn_in_steps=minimum_burn_in_steps,
                    rho_min=rho_min,
                    rho_max=rho_max,
                    clip_median_max=clip_median_max,
                    clip_iqr_min=clip_iqr_min,
                    clip_half_delta_min=clip_half_delta_min,
                    small_proxy_noise_ratio_min=small_proxy_noise_ratio_min,
                    conditional_valid_fraction_min=conditional_valid_fraction_min,
                )
            except ArmArtifactError as exc:
                coverage.append({
                    **base, "status_state": "completed", "classification": "invalid",
                    "included_in_landscape": False, "reason": str(exc),
                })
                invalid_count += 1
                continue
            landscape.append(row)
            coverage.append({
                **base, "status_state": "completed", "classification": "included",
                "included_in_landscape": True, "reason": "validated",
            })
    landscape.sort(key=lambda row: (
        not row["screen_eligible"],
        -row["slack_score"],
        -row["variation_score"],
        -row["small_proxy_to_noise_ratio"],
        row["dataset"], row["model"], row["epsilon"], row["rank"],
        row["C"], row["seed"], row["arm_id"],
    ))
    coverage.sort(key=lambda row: (
        {"invalid": 0, "incomplete": 1, "included": 2}.get(
            row["classification"], 3
        ),
        row["campaign_root"], row["arm_id"],
    ))
    recommended = [dict(row) for row in landscape if row["screen_eligible"]]
    return {
        "schema_version": 1,
        "NON_PRIVATE_CALIBRATION": True,
        "warning": WARNING,
        "campaign_roots": [str(root) for root in roots],
        "thresholds": {
            "burn_in_fraction": burn_in_fraction,
            "minimum_burn_in_steps": minimum_burn_in_steps,
            "rho_projection_min": rho_min,
            "rho_projection_max": rho_max,
            "clip_median_strictly_below": clip_median_max,
            "clip_IQR_at_least": clip_iqr_min,
            "absolute_first_last_half_delta_at_least": clip_half_delta_min,
            "variation_rule": "IQR OR absolute first/last-half delta",
            "small_proxy_to_noise_ratio_at_least": small_proxy_noise_ratio_min,
            "conditional_proxy_valid_fraction_at_least":
                conditional_valid_fraction_min,
            "required_unique_projected_rhos": 5,
            "cdf_slack_noise_std_formula":
                "dp_noise_multiplier * sqrt(K) / dp_expected_batch_size",
        },
        "counts": {
            "registered_baseline_arms": len(coverage),
            "included_completed_fixed_arms": len(landscape),
            "recommended_settings": len(recommended),
            "incomplete_arms": incomplete_count,
            "invalid_arms": invalid_count,
        },
        "coverage": coverage,
        "landscape": landscape,
        "recommended_settings": recommended,
    }


def _csv_bytes(rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(fields), extrasaction="ignore",
                            lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False)
            + "\n").encode("utf-8")


def _atomic_write(path: Path, payload: bytes) -> None:
    if path.exists() and path.is_symlink():
        raise LandscapeError(f"refusing to replace symlink output: {path}")
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix=f".{path.name}.", suffix=".tmp",
            dir=path.parent, delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except OSError as exc:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise LandscapeError(f"cannot atomically write {path}: {exc}") from exc


def write_outputs(output_dir: Path, report: Mapping[str, Any]) -> None:
    if output_dir.exists() and output_dir.is_symlink():
        raise LandscapeError(f"output directory must not be a symlink: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    if not output_dir.is_dir():
        raise LandscapeError(f"output path is not a directory: {output_dir}")
    landscape = report["landscape"]
    coverage = report["coverage"]
    recommended = report["recommended_settings"]
    payloads = {
        "baseline_landscape.csv": _csv_bytes(landscape, LANDSCAPE_FIELDS),
        "baseline_landscape.json": _json_bytes(report),
        "baseline_coverage.csv": _csv_bytes(coverage, COVERAGE_FIELDS),
        "baseline_coverage.json": _json_bytes({
            "NON_PRIVATE_CALIBRATION": True,
            "warning": WARNING,
            "counts": report["counts"],
            "coverage": coverage,
        }),
        "recommended_settings.csv": _csv_bytes(recommended, LANDSCAPE_FIELDS),
    }
    for name, payload in payloads.items():
        _atomic_write(output_dir / name, payload)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--campaign-root", action="append", nargs="+", type=Path, required=True,
        help="One or more campaign roots; repeat the option if desired.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--burn-in-fraction", type=float, default=0.10)
    parser.add_argument("--minimum-burn-in-steps", type=int, default=50)
    parser.add_argument("--rho-min", type=float, default=0.05)
    parser.add_argument("--rho-max", type=float, default=0.95)
    parser.add_argument("--clip-median-max", type=float, default=0.90)
    parser.add_argument("--clip-iqr-min", type=float, default=0.05)
    parser.add_argument("--clip-half-delta-min", type=float, default=0.05)
    parser.add_argument("--small-proxy-noise-ratio-min", type=float, default=2.0)
    parser.add_argument("--conditional-valid-fraction-min", type=float, default=0.99)
    parser.add_argument(
        "--allow-incomplete", action="store_true",
        help=("Return success when registered arms are merely unfinished. "
              "Corrupt completed artifacts always remain an error."),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    roots = [root for group in args.campaign_root for root in group]
    try:
        report = analyze_campaigns(
            roots,
            burn_in_fraction=args.burn_in_fraction,
            minimum_burn_in_steps=args.minimum_burn_in_steps,
            rho_min=args.rho_min,
            rho_max=args.rho_max,
            clip_median_max=args.clip_median_max,
            clip_iqr_min=args.clip_iqr_min,
            clip_half_delta_min=args.clip_half_delta_min,
            small_proxy_noise_ratio_min=args.small_proxy_noise_ratio_min,
            conditional_valid_fraction_min=args.conditional_valid_fraction_min,
        )
        write_outputs(args.output_dir, report)
    except LandscapeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    counts = report["counts"]
    print(
        "baseline landscape: "
        f"included={counts['included_completed_fixed_arms']} "
        f"recommended={counts['recommended_settings']} "
        f"incomplete={counts['incomplete_arms']} "
        f"invalid={counts['invalid_arms']}"
    )
    print(WARNING)
    if counts["invalid_arms"]:
        return 2
    if counts["incomplete_arms"] and not args.allow_incomplete:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
