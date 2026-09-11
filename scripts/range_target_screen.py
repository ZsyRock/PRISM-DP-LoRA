"""Full SlaClip targets at positions within baseline clipping-rate ranges.

For an observed hard-clipping trajectory p[1:N], the candidate at position q
is min(p) + q * (max(p) - min(p)). This is neither a quantile of gradient
norms nor an empirical quantile of the observed clipping-rate distribution.
The existing empirical-quantile profile is intentionally left unchanged.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import importlib.util
import math
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable


def _empirical_module():
    spec = importlib.util.spec_from_file_location(
        "range_shared_quantile_helpers", Path(__file__).with_name("quantile_target_screen.py")
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_EMPIRICAL = _empirical_module()
DEFAULT_POSITIONS = (0.25, 0.5, 0.75)
RANGE_METHOD = "linear_min_max_clipping_range_all_steps"
TARGET_FORMULA = "rho_q=min(raw_clip_fraction_t)+q*(max(raw_clip_fraction_t)-min(raw_clip_fraction_t))"
PROFILE = "glue-range-target-screen"
SCREEN_SEED = 50
SCREEN_STEPS = 150
ETA = 0.05
C_MAX = 100.0
SETTING_ORDER = _EMPIRICAL.SETTING_ORDER
SELECTED_FIXED_C = dict(_EMPIRICAL.SELECTED_FIXED_C)
PINNED_DEFAULT_RANGES = {
    "glue8-4b-eps6-r8": (0.3235294117647059, 1.0),
    "glue8-4b-eps6-r16": (0.27586206896551724, 1.0),
    "glue8-4b-eps6-r32": (0.31343283582089554, 1.0),
    "glue8-4b-eps3-r16": (0.3275862068965517, 1.0),
}
PINNED_DEFAULT_TARGETS = {
    sid: tuple(lower + q * (upper - lower) for q in DEFAULT_POSITIONS)
    for sid, (lower, upper) in PINNED_DEFAULT_RANGES.items()
}


def _positions(positions: Iterable[float]) -> tuple[float, ...]:
    try:
        result = tuple(float(q) for q in positions)
    except (TypeError, ValueError) as exc:
        raise ValueError("range positions must be finite probabilities in [0,1]") from exc
    if not result or any(not math.isfinite(q) or not 0 <= q <= 1 for q in result):
        raise ValueError("range positions must be finite probabilities in [0,1]")
    if len(set(result)) != len(result):
        raise ValueError("range positions must be distinct")
    return result


def _summary(lower: float, upper: float, mean: float, steps: int,
             positions: tuple[float, ...]) -> dict[str, Any]:
    targets = []
    unique_targets = []
    first_by_rho = {}
    for q in positions:
        rho = lower + q * (upper - lower)
        prior = first_by_rho.get(rho)
        target = {
            "range_position": q,
            "rho": rho,
            "degenerate_target": rho in (0.0, 1.0),
            "saturated_target": rho == 1.0,
            "zero_target": rho == 0.0,
            "duplicate_of_range_position": None if prior is None else prior["range_positions"][0],
        }
        targets.append(target)
        if prior is None:
            unique = {key: value for key, value in target.items()
                      if key not in ("range_position", "duplicate_of_range_position")}
            unique["range_positions"] = [q]
            unique_targets.append(unique)
            first_by_rho[rho] = unique
        else:
            prior["range_positions"].append(q)
    flags = []
    if len(unique_targets) < len(targets):
        flags.append("duplicate_range_targets_deduplicated")
    if any(target["saturated_target"] for target in targets):
        flags.append("saturated_range_target_at_one")
    if any(target["zero_target"] for target in targets):
        flags.append("zero_range_target")
    if lower == upper:
        flags.append("constant_clipping_trajectory")
    return {
        "targets": targets,
        "unique_targets": unique_targets,
        "requested_range_positions": list(positions),
        "source_steps": steps,
        "source_step_range": [1, steps],
        "source_window": "all_steps_no_burn_in",
        "realized_clip_min": lower,
        "realized_clip_max": upper,
        "realized_clip_mean": mean,
        "method": RANGE_METHOD,
        "formula": TARGET_FORMULA,
        "empirical_clipping_percentiles": False,
        "gradient_norm_quantiles": False,
        "inverse_small_mass_adjustment": False,
        "controller_global_target": "p_star_t=rho_q*(1-z_t)",
        "duplicate_targets": len(unique_targets) < len(targets),
        "saturated_source": lower >= 0.99,
        "all_steps_clipped": lower == 1.0,
        "all_steps_unclipped": upper == 0.0,
        "flags": flags,
    }


def derive_targets(records: dict[int, dict[str, Any]],
                   positions: Iterable[float] = DEFAULT_POSITIONS) -> dict[str, Any]:
    """Derive interval-position targets from all validated consecutive steps.

    Input validation shares the existing empirical profile's strict trajectory
    checks. No noisy CDF, norm quantile, or inverse small-mass correction is
    used when selecting rho. Boundary and duplicate values remain auditable.
    """
    probabilities = _positions(positions)
    values = _EMPIRICAL._trajectory(records)
    return _summary(min(values), max(values), fmean(values), len(values), probabilities)


def _seal(entries: list[dict[str, Any]], positions: tuple[float, ...]) -> dict[str, Any]:
    content = {
        "schema_version": "prism_full_slaclip_clipping_range_targets_v1",
        "method": RANGE_METHOD,
        "requested_range_positions": list(positions),
        "empirical_clipping_percentiles": False,
        "gradient_norm_quantiles": False,
        "inverse_small_mass_adjustment": False,
        "NON_PRIVATE_CALIBRATION": True,
        "end_to_end_dp_claim": False,
        "settings": entries,
    }
    return {**content, "content_sha256": hashlib.sha256(_EMPIRICAL._canonical_bytes(content)).hexdigest()}


def build_target_manifest(sources: dict[str, dict[str, Any]],
                          positions: Iterable[float] = DEFAULT_POSITIONS) -> dict[str, Any]:
    """Seal targets and caller-verified provenance without retaining raw rows."""
    probabilities = _positions(positions)
    if not sources or any(not isinstance(sid, str) or not sid for sid in sources):
        raise ValueError("sources need at least one nonempty string setting ID")
    entries = []
    for sid in sorted(sources):
        source = sources[sid]
        if "records" not in source or not source.get("provenance"):
            raise ValueError(f"source {sid} needs records and nonempty provenance")
        entries.append({
            "setting_id": sid,
            "source_provenance": deepcopy(source["provenance"]),
            "source_metadata": deepcopy({key: value for key, value in source.items()
                                         if key not in ("records", "provenance")}),
            "target_summary": derive_targets(source["records"], probabilities),
        })
    return _seal(entries, probabilities)


def verify_target_manifest(manifest: dict[str, Any]) -> None:
    content = {key: value for key, value in manifest.items() if key != "content_sha256"}
    if (not isinstance(manifest.get("content_sha256"), str)
            or hashlib.sha256(_EMPIRICAL._canonical_bytes(content)).hexdigest() != manifest["content_sha256"]):
        raise ValueError("range target manifest content hash mismatch")


def build_manifest(builder: Any, code_sha: str, model4revision: str,
                   model9revision: str, model12revision: str | None = None) -> dict[str, Any]:
    """Build 20 matched fresh-seed arms, with default C0=1 for every SlaClip."""
    weighted = builder._weighted_target_module()
    manifest = weighted.build_manifest(builder, code_sha, model4revision,
                                       model9revision, model12revision)
    settings = {setting["id"]: {**setting, "model_revision": model4revision}
                for setting in builder.REGIME_SETTINGS}
    arms = []
    for sid in SETTING_ORDER:
        c = SELECTED_FIXED_C[sid]
        for candidate, initial_c, rho, role, position in (
            ("range-default-fixed-c1", 1.0, None, "range_default_fixed", None),
            (f"range-tuned-fixed-c{c:g}", c, None, "fresh_fixed_comparator", None),
            *((f"range-default-q{int(q * 100)}", 1.0, rho, "range_default_target", q)
              for q, rho in zip(DEFAULT_POSITIONS, PINNED_DEFAULT_TARGETS[sid])),
        ):
            arm = weighted._arm(settings[sid], candidate, initial_c, SCREEN_SEED,
                                SCREEN_STEPS, role, rho)
            arm.update({"range_position": position, "stage": 2})
            arms.append(arm)
    manifest.update({
        "profile": PROFILE,
        "protocol": "prism_glue_default_clipping_range_screen_v1",
        "seed": SCREEN_SEED,
        "arms": arms,
        "inference_class": "fresh_seed_exploratory_default_range_screen_requires_full_length_multi_seed_confirmation",
    })
    manifest["weighted_target_screen"] = {"enabled": False}
    manifest["quantile_target_screen"] = {"enabled": False}
    manifest["full_slaclip"].update({"C_max": C_MAX, "eta": ETA})
    manifest["regime_map"].update({
        "exploratory": True,
        "screen_steps": SCREEN_STEPS,
        "fixed_C_grid": {sid: [1.0, SELECTED_FIXED_C[sid]] for sid in SETTING_ORDER},
        "conditional_rho_grid": "paper_default_C1_all_step_minmax_range_positions_25_50_75",
    })
    manifest["range_target_screen"] = {
        "enabled": True,
        "settings": list(SETTING_ORDER),
        "total_planned_arms": len(arms),
        "fixed_arms": 8,
        "slaclip_arms": 12,
        "screen_steps": SCREEN_STEPS,
        "screen_seed": SCREEN_SEED,
        "target_source": "paper_default_C1",
        "target_source_seed": 42,
        "source_steps": 500,
        "source_window": "all_steps_no_burn_in",
        "range_method": RANGE_METHOD,
        "range_positions": list(DEFAULT_POSITIONS),
        "source_ranges": {sid: list(PINNED_DEFAULT_RANGES[sid]) for sid in SETTING_ORDER},
        "targets": {sid: list(PINNED_DEFAULT_TARGETS[sid]) for sid in SETTING_ORDER},
        "target_formula": TARGET_FORMULA,
        "empirical_clipping_percentiles": False,
        "gradient_norm_quantiles": False,
        "inverse_small_mass_adjustment": False,
        "controller_global_target": "p_star_t=rho_q*(1-z_t)",
        "controller_update": "gamma_t=clip(1-rho_q*(1-z_t),0,1); C_next=clip(C_t*exp(eta*(gamma_t-u_t)),C_min,C_max)",
        "selected_fixed_C": dict(SELECTED_FIXED_C),
        "C0": 1.0,
        "eta": ETA,
        "K": 15,
        "C_bounds": [0.1, C_MAX],
        "recipe": ["fresh_default_fixed", "fresh_tuned_fixed", "full_default_range25",
                   "full_default_range50", "full_default_range75"],
        "primary_comparator": "fresh_default_fixed_same_C0",
        "strong_comparator": "fresh_tuned_fixed",
        "selected_fixed_C_not_claimed_global_optimum": True,
        "selection_metric": "minimum_final_public_validation_loss_among_three_range_targets",
        "selection_claim": "exploratory_selected_validation_result_not_independent_confirmation",
        "deferred_math": {
            "default_configurations": 6,
            "constant_one_configurations": 5,
            "constant_one_targets": [1.0, 1.0, 1.0],
            "nonconstant_setting": "math10k-4b-eps6-r32",
            "nonconstant_source_range": [0.9594594594594594, 1.0],
            "nonconstant_range_targets": [0.9695945945945945, 0.9797297297297297, 0.9898648648648649],
            "reason": "five constant saturated trajectories have duplicate targets; rank32 has distinct near-saturated targets and is deferred to prioritize four non-saturated GLUE settings within one 24h allocation",
            "silently_replaced_with_interior_targets": False,
            "future_requirement": "separate saturated or near-saturated regime study",
        },
        "journal_confirmation": False,
        "official_task_evaluation": False,
        "end_to_end_dp_claim": False,
        "NON_PRIVATE_CALIBRATION": True,
        "default_source_specs": weighted.source_specs()["default"],
        "previous_fixed_selection": {
            "campaign_id": _EMPIRICAL.PREVIOUS_CAMPAIGN,
            "code_sha": _EMPIRICAL.PREVIOUS_CODE,
            "manifest_sha256": _EMPIRICAL.PREVIOUS_MANIFEST_SHA256,
            "lock_sha256": _EMPIRICAL.PREVIOUS_LOCK_SHA256,
        },
        "public_holdout": {
            "rows": builder.GLUE_SLACLIP_VALIDATION_ROWS,
            "seed": builder.GLUE_SLACLIP_VALIDATION_SEED,
            "indices_sha256": builder.GLUE_SLACLIP_VALIDATION_INDICES_SHA256,
            "records_sha256": builder.GLUE_SLACLIP_VALIDATION_RECORDS_SHA256,
        },
        "resources": {"gpus": 1, "gpu_type": "a100", "memory": "80G", "time": "24:00:00", "lanes": 1},
    }
    return manifest


def verify_sources(builder: Any, root: Path) -> dict[str, Any]:
    """Reuse independently hash-pinned raw-source audits, then derive ranges.

    The empirical verifier rereads all baseline telemetry and validates source
    code, manifests, status, trajectory hashes, and fixed-comparator selection.
    Its sealed min/max summaries are therefore sufficient to calculate interval
    positions without re-reading the same large telemetry files a second time.
    No old campaign, empirical target, or source artifact is rewritten.
    """
    empirical = builder._quantile_target_module()
    verified = empirical.verify_sources(builder, root)
    empirical.verify_target_manifest(verified["targets"])
    source_entries = verified["targets"]["settings"]
    if (len(source_entries) != len(SETTING_ORDER)
            or {entry["setting_id"] for entry in source_entries} != set(SETTING_ORDER)):
        raise builder.CampaignError("range source settings mismatch")
    entries = []
    for source in sorted(source_entries, key=lambda entry: entry["setting_id"]):
        sid = source["setting_id"]
        summary = source["target_summary"]
        metadata = source["source_metadata"]
        actual_range = (summary["realized_clip_min"], summary["realized_clip_max"])
        if (summary["source_steps"] != 500
                or summary["source_step_range"] != [1, 500]
                or summary["source_window"] != "all_steps_no_burn_in"
                or actual_range != PINNED_DEFAULT_RANGES[sid]
                or metadata["source_initial_c"] != 1.0
                or metadata["source_seed"] != 42
                or metadata["source_kind"] != "paper_default_C1"):
            raise builder.CampaignError(f"range source baseline range or provenance changed: {sid}")
        targets = _summary(*actual_range, summary["realized_clip_mean"], 500, DEFAULT_POSITIONS)
        if tuple(entry["rho"] for entry in targets["targets"]) != PINNED_DEFAULT_TARGETS[sid]:
            raise builder.CampaignError(f"range source pinned targets changed: {sid}")
        entries.append({
            "setting_id": sid,
            "source_provenance": deepcopy(source["source_provenance"]),
            "source_metadata": {**deepcopy(metadata), "screen_initial_c": 1.0},
            "target_summary": targets,
        })
    return {
        "profile": PROFILE,
        "targets": _seal(entries, DEFAULT_POSITIONS),
        "previous_selection": deepcopy(verified["previous_selection"]),
        "NON_PRIVATE_CALIBRATION": True,
        "end_to_end_dp_claim": False,
    }
