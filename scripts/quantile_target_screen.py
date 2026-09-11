"""Pure empirical clipping-quantile targets for Full SlaClip screens.

Targets are conditional controller ``rho`` values taken directly from the
observed *hard* clipping trajectory.  They are not inverted through an
estimated small-gradient mass.  File validation, source selection, privacy
budgeting and experiment launch remain responsibilities of the caller.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable


DEFAULT_QUANTILES = (0.25, 0.5, 0.75)
MIN_STEPS = 50
QUANTILE_METHOD = "linear_empirical_quantile_all_steps"
PROFILE = "glue-quantile-target-screen"
SCREEN_SEED = 49
SCREEN_STEPS = 150
ETA = 0.05
C_MAX = 100.0
SETTING_ORDER = (
    "glue8-4b-eps6-r32", "glue8-4b-eps6-r8",
    "glue8-4b-eps6-r16", "glue8-4b-eps3-r16",
)
SELECTED_FIXED_C = {
    "glue8-4b-eps6-r8": 40.0,
    "glue8-4b-eps6-r16": 30.0,
    "glue8-4b-eps6-r32": 15.0,
    "glue8-4b-eps3-r16": 30.0,
}
PINNED_DEFAULT_TARGETS = {
    "glue8-4b-eps6-r8": (0.4444444444444444, 0.5, 0.5488402678144428),
    "glue8-4b-eps6-r16": (0.4633017163504969, 0.5096189419163892, 0.5689655172413793),
    "glue8-4b-eps6-r32": (0.4909090909090909, 0.5410800385728062, 0.6068788171006108),
    "glue8-4b-eps3-r16": (0.4741902834008097, 0.5250069463739928, 0.5797101449275363),
}
PREVIOUS_CAMPAIGN = "paper-coverage-7199f4eb4002-glue-weighted-target-screen-v2"
PREVIOUS_CODE = "7199f4eb4002e606f9af62302e338492de3b311b"
PREVIOUS_MANIFEST_SHA256 = "6f3ec20fd6dfd48f77bcad3fb4cf5a493511b938be6ed59a3de6e9541ba721a4"
PREVIOUS_LOCK_SHA256 = "b421f7417202ddb5855c70b8eb51016122bdeb6e426e8d367931cca8f01fc970"


def _quantiles(quantiles: Iterable[float]) -> tuple[float, ...]:
    try:
        result = tuple(float(q) for q in quantiles)
    except (TypeError, ValueError) as exc:
        raise ValueError("quantiles must be finite probabilities in [0,1]") from exc
    if not result or any(not math.isfinite(q) or not 0.0 <= q <= 1.0 for q in result):
        raise ValueError("quantiles must be finite probabilities in [0,1]")
    if len(set(result)) != len(result):
        raise ValueError("quantile probabilities must be distinct")
    return result


def _trajectory(records: dict[int, dict[str, Any]]) -> list[float]:
    # An already-collapsed dictionary cannot reveal overwritten duplicate
    # rows.  Callers parsing JSONL must reject duplicate steps before building
    # it.  Missing/shifted steps and mismatching embedded steps are rejected.
    if (
        len(records) < MIN_STEPS
        or any(type(step) is not int for step in records)
        or set(records) != set(range(1, len(records) + 1))
    ):
        raise ValueError("quantile targets need at least 50 consecutive steps starting at 1")
    values = []
    for step in range(1, len(records) + 1):
        row = records[step]
        if "step" in row and (type(row["step"]) is not int or row["step"] != step):
            raise ValueError("embedded telemetry step must match its unique trajectory key")
        try:
            value = float(row["raw_clip_fraction"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("raw_clip_fraction must be finite and in [0,1] at every step") from exc
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError("raw_clip_fraction must be finite and in [0,1] at every step")
        values.append(value)
    return values


def _linear_quantile(sorted_values: list[float], probability: float) -> float:
    position = (len(sorted_values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    fraction = position - lower
    return sorted_values[lower] + fraction * (sorted_values[upper] - sorted_values[lower])


def derive_targets(
    records: dict[int, dict[str, Any]],
    quantiles: Iterable[float] = DEFAULT_QUANTILES,
) -> dict[str, Any]:
    """Return direct conditional rho candidates and auditable source statistics.

    Linear empirical quantiles use index ``q * (N - 1)`` in the sorted values
    from steps 1..N.  This is not interpolation between only the minimum and
    maximum clipping fractions.  Repeated percentile values are retained in
    ``targets`` for audit and collapsed in ``unique_targets`` for scheduling.
    Neither zero nor one is silently replaced by a convenient interior target.
    """
    probabilities = _quantiles(quantiles)
    values = _trajectory(records)
    sorted_values = sorted(values)
    targets: list[dict[str, Any]] = []
    unique_targets: list[dict[str, Any]] = []
    first_by_rho: dict[float, dict[str, Any]] = {}
    for q in probabilities:
        rho = _linear_quantile(sorted_values, q)
        prior = first_by_rho.get(rho)
        target = {
            "quantile": q,
            "rho": rho,
            "degenerate_target": rho in (0.0, 1.0),
            "saturated_target": rho == 1.0,
            "zero_target": rho == 0.0,
            "duplicate_of_quantile": None if prior is None else prior["quantiles"][0],
        }
        targets.append(target)
        if prior is None:
            unique = {key: value for key, value in target.items()
                      if key not in ("quantile", "duplicate_of_quantile")}
            unique["quantiles"] = [q]
            unique_targets.append(unique)
            first_by_rho[rho] = unique
        else:
            prior["quantiles"].append(q)
    flags = []
    if len(unique_targets) < len(targets):
        flags.append("duplicate_percentile_targets_deduplicated")
    if any(target["saturated_target"] for target in targets):
        flags.append("saturated_percentile_at_one")
    if any(target["zero_target"] for target in targets):
        flags.append("zero_percentile_target")
    if min(values) == max(values):
        flags.append("constant_clipping_trajectory")
    return {
        "targets": targets,
        "unique_targets": unique_targets,
        "requested_quantiles": list(probabilities),
        "source_steps": len(values),
        "source_step_range": [1, len(values)],
        "source_window": "all_steps_no_burn_in",
        "realized_clip_min": min(values),
        "realized_clip_max": max(values),
        "realized_clip_mean": fmean(values),
        "method": QUANTILE_METHOD,
        "formula": "rho_q=linear_quantile({raw_clip_fraction_t:t=1..N},q)",
        "inverse_small_mass_adjustment": False,
        "controller_global_target": "p_star_t=rho_q*(1-z_t)",
        "duplicate_targets": len(unique_targets) < len(targets),
        "saturated_source": min(values) >= 0.99,
        "all_steps_clipped": min(values) == 1.0,
        "all_steps_unclipped": max(values) == 0.0,
        "flags": flags,
    }


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def build_target_manifest(
    sources: dict[str, dict[str, Any]],
    quantiles: Iterable[float] = DEFAULT_QUANTILES,
) -> dict[str, Any]:
    """Seal pure target entries for any selected default or tuned sources.

    ``sources`` maps each setting ID to ``{"records": ..., "provenance": ...}``.
    All remaining JSON-serializable source fields (such as ``initial_c`` or
    ``source_kind``) are copied into ``source_metadata``.  The provenance must
    be nonempty; the caller must verify its hashes against the original files.

    The returned content digest detects later changes via
    :func:`verify_target_manifest`; it is not a substitute for write-once file
    handling or an external source-trust anchor.  No input objects are mutated.
    """
    probabilities = _quantiles(quantiles)
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
    content = {
        "schema_version": "prism_full_slaclip_empirical_quantile_targets_v1",
        "method": QUANTILE_METHOD,
        "requested_quantiles": list(probabilities),
        "inverse_small_mass_adjustment": False,
        "NON_PRIVATE_CALIBRATION": True,
        "end_to_end_dp_claim": False,
        "settings": entries,
    }
    return {**content, "content_sha256": hashlib.sha256(_canonical_bytes(content)).hexdigest()}


def verify_target_manifest(manifest: dict[str, Any]) -> None:
    """Raise if any sealed manifest field changed after construction."""
    content = {key: value for key, value in manifest.items() if key != "content_sha256"}
    expected = manifest.get("content_sha256")
    if not isinstance(expected, str) or hashlib.sha256(_canonical_bytes(content)).hexdigest() != expected:
        raise ValueError("quantile target manifest content hash mismatch")


def build_manifest(
    builder: Any,
    code_sha: str,
    model4revision: str,
    model9revision: str,
    model12revision: str | None = None,
) -> dict[str, Any]:
    """Create twenty matched arms from pinned default-C1 quantile candidates."""
    weighted = builder._weighted_target_module()
    manifest = weighted.build_manifest(builder, code_sha, model4revision,
                                       model9revision, model12revision)
    settings = {setting["id"]: {**setting, "model_revision": model4revision}
                for setting in builder.REGIME_SETTINGS}
    arms = []
    for sid in SETTING_ORDER:
        c = SELECTED_FIXED_C[sid]
        for candidate, initial_c, rho, role, probability in (
            ("quantile-default-fixed-c1", 1.0, None, "quantile_default_fixed", None),
            (f"quantile-tuned-fixed-c{c:g}", c, None, "fresh_fixed_comparator", None),
            *((f"quantile-default-q{int(q * 100)}", c, rho,
               "quantile_default_target", q)
              for q, rho in zip(DEFAULT_QUANTILES, PINNED_DEFAULT_TARGETS[sid])),
        ):
            arm = weighted._arm(settings[sid], candidate, initial_c, SCREEN_SEED,
                                SCREEN_STEPS, role, rho)
            arm.update({"quantile": probability, "stage": 2})
            arms.append(arm)
    manifest.update({
        "profile": PROFILE,
        "protocol": "prism_glue_default_empirical_quantile_screen_v1",
        "seed": SCREEN_SEED,
        "arms": arms,
        "inference_class": "fresh_seed_exploratory_default_quantile_screen_requires_full_length_multi_seed_confirmation",
    })
    manifest["weighted_target_screen"] = {"enabled": False}
    manifest["full_slaclip"].update({"C_max": C_MAX, "eta": ETA})
    manifest["regime_map"].update({
        "exploratory": True,
        "screen_steps": SCREEN_STEPS,
        "fixed_C_grid": {sid: [1.0, SELECTED_FIXED_C[sid]] for sid in SETTING_ORDER},
        "conditional_rho_grid": "paper_default_C1_all_step_empirical_q25_q50_q75",
    })
    manifest["quantile_target_screen"] = {
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
        "quantile_method": QUANTILE_METHOD,
        "quantiles": list(DEFAULT_QUANTILES),
        "targets": {sid: list(PINNED_DEFAULT_TARGETS[sid]) for sid in SETTING_ORDER},
        "target_formula": "rho_q=linear_quantile({raw_clip_fraction_t:t=1..500},q)",
        "inverse_small_mass_adjustment": False,
        "controller_global_target": "p_star_t=rho_q*(1-z_t)",
        "selected_fixed_C": dict(SELECTED_FIXED_C),
        "C0": "selected_tuned_fixed_C",
        "eta": ETA,
        "K": 15,
        "C_bounds": [0.1, C_MAX],
        "recipe": ["fresh_default_fixed", "fresh_tuned_fixed", "full_default_q25",
                   "full_default_q50", "full_default_q75"],
        "primary_comparator": "fresh_tuned_fixed",
        "secondary_comparator": "fresh_default_fixed",
        "selected_fixed_C_not_claimed_global_optimum": True,
        "deferred_math": {
            "default_configurations": 6,
            "reason": "all three empirical clipping percentiles equal one; duplicate saturated targets are not informative",
            "requested_percentiles": [1.0, 1.0, 1.0],
            "silently_replaced_with_interior_targets": False,
            "future_requirement": "a separately preregistered interior-target or fixed-C calibration study",
        },
        "journal_confirmation": False,
        "official_task_evaluation": False,
        "end_to_end_dp_claim": False,
        "NON_PRIVATE_CALIBRATION": True,
        "default_source_specs": weighted.source_specs()["default"],
        "previous_fixed_selection": {
            "campaign_id": PREVIOUS_CAMPAIGN,
            "code_sha": PREVIOUS_CODE,
            "manifest_sha256": PREVIOUS_MANIFEST_SHA256,
            "lock_sha256": PREVIOUS_LOCK_SHA256,
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
    """Read-only source revalidation and serializable target/selection audit.

    Besides the original paper-default raw hashes, validate the complete
    six-arm fixed calibration that supplied the two newly selected fixed C
    values.  The other two selected values are validated using the existing
    weighted source pins.  No previous campaign file is rewritten.
    """
    weighted = builder._weighted_target_module()
    sources = weighted.verify_sources(builder, root)
    previous_root = root.parent / PREVIOUS_CAMPAIGN
    previous_manifest_path = previous_root / "plans/manifest.json"
    previous_lock_path = previous_root / "selection/weighted-targets.lock.json"
    weighted._check_hash(builder, previous_manifest_path, PREVIOUS_MANIFEST_SHA256)
    weighted._check_hash(builder, previous_lock_path, PREVIOUS_LOCK_SHA256)
    previous_manifest = weighted._read(builder, previous_manifest_path)
    previous_lock = weighted._read(builder, previous_lock_path)
    expected = weighted.build_manifest(builder, PREVIOUS_CODE, weighted.MODEL_REVISION, "0" * 40)
    if (previous_manifest.get("code_sha") != PREVIOUS_CODE
            or previous_manifest.get("arms") != expected["arms"]
            or previous_manifest.get("weighted_target_screen") != expected["weighted_target_screen"]
            or previous_lock.get("code_sha") != PREVIOUS_CODE
            or previous_lock.get("manifest_sha256") != PREVIOUS_MANIFEST_SHA256):
        raise builder.CampaignError("quantile source previous fixed-selection plan changed")
    selected = {entry["setting_id"]: entry for entry in previous_lock["settings"]}
    if set(selected) != set(SETTING_ORDER) or len(previous_lock["settings"]) != len(SETTING_ORDER):
        raise builder.CampaignError("quantile source previous selection settings mismatch")
    candidates: dict[str, list[dict[str, Any]]] = {}
    artifacts = {}
    for arm in previous_manifest["arms"]:
        status, records, _, hashes = builder._load_focused_arm(previous_root, arm, PREVIOUS_CODE)
        loss = float(status["validation"]["loss_mean"])
        if not math.isfinite(loss):
            raise builder.CampaignError("quantile source fixed calibration loss is not finite")
        artifacts[arm["arm_id"]] = hashes
        candidates.setdefault(arm["setting_id"], []).append({
            "arm": arm, "records": records, "hashes": hashes, "loss": loss,
        })
    if artifacts != previous_lock.get("stage1_artifacts"):
        raise builder.CampaignError("quantile source fixed calibration artifacts changed")
    manifest_sources = {}
    selection_audit = {}
    for sid in SETTING_ORDER:
        default = sources["default"][sid]
        target_summary = derive_targets(default["records"])
        rho_values = tuple(entry["rho"] for entry in target_summary["targets"])
        if (target_summary["source_steps"] != 500
                or any(abs(actual - pinned) > 1e-15
                       for actual, pinned in zip(rho_values, PINNED_DEFAULT_TARGETS[sid]))):
            raise builder.CampaignError(f"quantile source empirical target changed: {sid}")
        entry = selected[sid]
        if (entry["selected_C"] != SELECTED_FIXED_C[sid]
                or entry["default_source"] != default["provenance"]
                or entry["default_target"] != default["target"]):
            raise builder.CampaignError(f"quantile source selected fixed/default provenance mismatch: {sid}")
        if sid in candidates:
            ranking = sorted(candidates[sid], key=lambda item: (
                item["loss"], item["arm"]["initial_c"], item["arm"]["candidate_id"],
            ))
            winner = ranking[0]
            arm = winner["arm"]
            expected_ranking = [{"C": item["arm"]["initial_c"],
                                 "validation_loss": item["loss"],
                                 "arm_id": item["arm"]["arm_id"]} for item in ranking]
            provenance = {"campaign_id": PREVIOUS_CAMPAIGN, "relative_root": arm["relative_root"],
                          "code_sha": PREVIOUS_CODE, "hashes": winner["hashes"]}
            tuned_target = weighted.derive_target(winner["records"])
            if arm["initial_c"] != SELECTED_FIXED_C[sid] or entry["fixed_ranking"] != expected_ranking:
                raise builder.CampaignError(f"quantile source selected C is no longer calibration winner: {sid}")
        else:
            tuned = sources["tuned"][sid]
            provenance = tuned["provenance"]
            tuned_target = tuned["target"]
            if tuned["arm"]["initial_c"] != SELECTED_FIXED_C[sid]:
                raise builder.CampaignError(f"quantile source prior fixed C mismatch: {sid}")
        if entry["tuned_source"] != provenance or entry["tuned_target"] != tuned_target:
            raise builder.CampaignError(f"quantile source tuned selection provenance mismatch: {sid}")
        manifest_sources[sid] = {
            "records": default["records"],
            "provenance": default["provenance"],
            "source_kind": "paper_default_C1",
            "source_initial_c": 1.0,
            "source_seed": 42,
            "screen_initial_c": SELECTED_FIXED_C[sid],
        }
        selection_audit[sid] = {
            "selected_C": SELECTED_FIXED_C[sid],
            "fixed_ranking": entry["fixed_ranking"],
            "tuned_source": provenance,
            "boundary_winner": entry["boundary_winner"],
        }
    return {
        "profile": PROFILE,
        "targets": build_target_manifest(manifest_sources),
        "previous_selection": {
            "campaign_id": PREVIOUS_CAMPAIGN,
            "code_sha": PREVIOUS_CODE,
            "manifest_sha256": PREVIOUS_MANIFEST_SHA256,
            "lock_sha256": PREVIOUS_LOCK_SHA256,
            "stage1_artifacts": artifacts,
            "settings": selection_audit,
        },
        "NON_PRIVATE_CALIBRATION": True,
        "end_to_end_dp_claim": False,
    }
