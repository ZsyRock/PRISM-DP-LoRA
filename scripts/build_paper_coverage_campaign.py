#!/usr/bin/env python3
"""Build and summarize PRISM paper-coverage and clipping-regime screens.

Both profiles are one-seed exploratory screens rather than confirmatory
experiments.  ``paper-breadth`` preserves the historical 15-arm plan;
``regime-map`` crosses the paper's available dataset, privacy, rank, and 4B/9B
axes with fixed-C and conditional-rho grids.  All arms run inside one immutable
two-lane Slurm allocation, and measured clipping strata are reported without
pretending they are pre-established SlaClip failure thresholds.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 2
SEED = 42
MODEL_4B = "google/gemma-3-4b-pt"
MODEL_9B = "google/gemma-2-9b"
MODEL_12B = "google/gemma-3-12b-pt"
FULL_SHA_LENGTH = 40

BREADTH_SETTINGS = (
    {
        "id": "glue8-4b-eps6-r16",
        "lane": 0,
        "paper_reference": "Table 2 / Table 7",
        "dataset": "glue8",
        "model_slug": "gemma-3-4b-pt",
        "model_id": MODEL_4B,
        "epsilon": 6.0,
        "lora_r": 16,
        "steps": 500,
        "learning_rate": 0.0002,
        "cutoff_len": 384,
        "train_on_inputs": False,
    },
    {
        "id": "glue8-4b-eps3-r16",
        "lane": 1,
        "paper_reference": "Table 2 / Table 7",
        "dataset": "glue8",
        "model_slug": "gemma-3-4b-pt",
        "model_id": MODEL_4B,
        "epsilon": 3.0,
        "lora_r": 16,
        "steps": 500,
        "learning_rate": 0.0002,
        "cutoff_len": 384,
        "train_on_inputs": False,
    },
    {
        "id": "math10k-9b-eps6-r16",
        "lane": 0,
        "paper_reference": "Table 3",
        "dataset": "math10k",
        "model_slug": "gemma-2-9b",
        "model_id": MODEL_9B,
        "epsilon": 6.0,
        "lora_r": 16,
        "steps": 300,
        "learning_rate": 0.0003,
        "cutoff_len": 256,
        "train_on_inputs": True,
    },
    {
        "id": "math10k-4b-eps6-r8",
        "lane": 1,
        "paper_reference": "Table 4",
        "dataset": "math10k",
        "model_slug": "gemma-3-4b-pt",
        "model_id": MODEL_4B,
        "epsilon": 6.0,
        "lora_r": 8,
        "steps": 300,
        "learning_rate": 0.0003,
        "cutoff_len": 256,
        "train_on_inputs": True,
    },
    {
        "id": "math10k-4b-eps6-r32",
        "lane": 1,
        "paper_reference": "Table 4",
        "dataset": "math10k",
        "model_slug": "gemma-3-4b-pt",
        "model_id": MODEL_4B,
        "epsilon": 6.0,
        "lora_r": 32,
        "steps": 300,
        "learning_rate": 0.0003,
        "cutoff_len": 256,
        "train_on_inputs": True,
    },
)

BREADTH_CANDIDATES = (
    {
        "id": "fixed-c1",
        "method": "baseline",
        "initial_c": 1.0,
        "rho": None,
        "eta": None,
    },
    {
        "id": "full-sla-rho090",
        "method": "slaclip",
        "initial_c": 1.0,
        "rho": 0.90,
        "eta": 0.05,
    },
    {
        "id": "full-sla-rho098",
        "method": "slaclip",
        "initial_c": 1.0,
        "rho": 0.98,
        "eta": 0.05,
    },
)

# This crosses every 4B dataset/privacy/rank axis reported by the paper plus
# the paper's 9B Math setting.  The 12B row remains optional because it needs a
# separately staged gated checkpoint; it must not silently fall back to 4B.
REGIME_SETTINGS = (
    *BREADTH_SETTINGS,
    {
        "id": "math10k-4b-eps6-r16",
        "lane": 0,
        "paper_reference": "Table 2 / Table 3 / Table 4",
        "dataset": "math10k",
        "model_slug": "gemma-3-4b-pt",
        "model_id": MODEL_4B,
        "epsilon": 6.0,
        "lora_r": 16,
        "steps": 300,
        "learning_rate": 0.0003,
        "cutoff_len": 256,
        "train_on_inputs": True,
    },
    {
        "id": "math10k-4b-eps3-r16",
        "lane": 1,
        "paper_reference": "Table 2 epsilon axis",
        "dataset": "math10k",
        "model_slug": "gemma-3-4b-pt",
        "model_id": MODEL_4B,
        "epsilon": 3.0,
        "lora_r": 16,
        "steps": 300,
        "learning_rate": 0.0003,
        "cutoff_len": 256,
        "train_on_inputs": True,
    },
    {
        "id": "glue8-4b-eps6-r8",
        "lane": 0,
        "paper_reference": "Table 4",
        "dataset": "glue8",
        "model_slug": "gemma-3-4b-pt",
        "model_id": MODEL_4B,
        "epsilon": 6.0,
        "lora_r": 8,
        "steps": 500,
        "learning_rate": 0.0002,
        "cutoff_len": 384,
        "train_on_inputs": False,
    },
    {
        "id": "glue8-4b-eps6-r32",
        "lane": 1,
        "paper_reference": "Table 4",
        "dataset": "glue8",
        "model_slug": "gemma-3-4b-pt",
        "model_id": MODEL_4B,
        "epsilon": 6.0,
        "lora_r": 32,
        "steps": 500,
        "learning_rate": 0.0002,
        "cutoff_len": 384,
        "train_on_inputs": False,
    },
)

BASELINE_12B_SETTING = {
    "id": "math10k-12b-eps6-r16",
    "lane": 1,
    "paper_reference": "Table 3",
    "dataset": "math10k",
    "model_slug": "gemma-3-12b-pt",
    "model_id": MODEL_12B,
    "epsilon": 6.0,
    "lora_r": 16,
    "steps": 300,
    "learning_rate": 0.0003,
    "cutoff_len": 256,
    "train_on_inputs": True,
}

REGIME_CANDIDATES = tuple(
    {
        "id": f"fixed-c{str(value).replace('.', 'p')}",
        "method": "baseline",
        "initial_c": value,
        "rho": None,
        "eta": None,
    }
    for value in (0.5, 1.0, 2.0, 3.0, 5.0)
) + tuple(
    {
        "id": f"full-sla-rho{int(value * 100):03d}",
        "method": "slaclip",
        "initial_c": 1.0,
        "rho": value,
        "eta": 0.05,
    }
    for value in (0.50, 0.70, 0.80, 0.90, 0.98)
) + (
    {
        "id": "full-sla-c2-rho090",
        "method": "slaclip",
        "initial_c": 2.0,
        "rho": 0.90,
        "eta": 0.05,
    },
)

BASELINE_CANDIDATES = (
    {
        "id": "fixed-c1-paper-default",
        "method": "baseline",
        "initial_c": 1.0,
        "rho": None,
        "eta": None,
    },
)


class CampaignError(RuntimeError):
    """The campaign plan or output is incomplete or inconsistent."""


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _write_immutable(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != data:
            raise CampaignError(f"refusing to overwrite immutable artifact: {path}")
        return
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        os.chmod(path, 0o600)
    except FileExistsError:
        if path.read_bytes() != data:
            raise CampaignError(f"concurrent inconsistent writer: {path}")
    finally:
        temporary.unlink(missing_ok=True)


def _with_sha(path: Path, data: bytes) -> None:
    _write_immutable(path, data)
    digest = hashlib.sha256(data).hexdigest()
    _write_immutable(path.with_name(path.name + ".sha256"), f"{digest}  {path.name}\n".encode())


def build_manifest(
    code_sha: str,
    model_4b_revision: str,
    model_9b_revision: str,
    profile: str = "paper-breadth",
    model_12b_revision: str | None = None,
) -> dict[str, Any]:
    for label, value in (
        ("code_sha", code_sha),
        ("model_4b_revision", model_4b_revision),
        ("model_9b_revision", model_9b_revision),
    ):
        if len(value) != FULL_SHA_LENGTH or any(ch not in "0123456789abcdef" for ch in value):
            raise CampaignError(f"{label} must be a full lowercase commit SHA")
    revisions = {MODEL_4B: model_4b_revision, MODEL_9B: model_9b_revision}
    if profile == "paper-breadth":
        settings = BREADTH_SETTINGS
        candidates = BREADTH_CANDIDATES
        screen_steps = None
        eval_limit = 0
    elif profile == "regime-map":
        settings = REGIME_SETTINGS
        candidates = REGIME_CANDIDATES
        screen_steps = 150
        eval_limit = 512
    elif profile in {"baseline-reproduction", "baseline-reproduction-cached"}:
        settings = REGIME_SETTINGS
        if profile == "baseline-reproduction":
            if model_12b_revision is None:
                raise CampaignError("baseline-reproduction requires a pinned 12B revision")
            if len(model_12b_revision) != FULL_SHA_LENGTH or any(
                ch not in "0123456789abcdef" for ch in model_12b_revision
            ):
                raise CampaignError("model_12b_revision must be a full lowercase commit SHA")
            revisions[MODEL_12B] = model_12b_revision
            settings = (*settings, BASELINE_12B_SETTING)
        candidates = BASELINE_CANDIDATES
        screen_steps = None
        eval_limit = 0
    else:
        raise CampaignError(f"unknown campaign profile: {profile}")
    arms = []
    for setting in settings:
        for candidate in candidates:
            arm_id = f"{setting['id']}--{candidate['id']}--seed{SEED}"
            arms.append(
                {
                    **setting,
                    **candidate,
                    "setting_id": setting["id"],
                    "candidate_id": candidate["id"],
                    "arm_id": arm_id,
                    "seed": SEED,
                    "model_revision": revisions[setting["model_id"]],
                    "steps": screen_steps or setting["steps"],
                    "eval_limit": eval_limit,
                    "relative_root": f"runs/{setting['id']}/{candidate['id']}/seed-{SEED}",
                }
            )
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol": f"prism_paper_coverage_{profile.replace('-', '_')}_v2",
        "profile": profile,
        "inference_class": "single_seed_exploratory_breadth_screen_requires_fresh_seed_confirmation",
        "code_sha": code_sha,
        "seed": SEED,
        "privacy": {"delta": 1e-5, "accountant": "prv", "secure_mode": False},
        "full_slaclip": {
            "K": 15,
            "C_min": 0.1,
            "C_max": 15.0,
            "target_semantics": "p_star_t=rho*(1-z_t); rho is conditional on residual non-small mass",
        },
        "selection_warning": (
            "Task-test metrics are descriptive only. Any promising setting must be "
            "repeated on fresh seeds with a locked tuned-fixed comparator."
        ),
        "baseline_reproduction": {
            "paper_default_fixed_C": 1.0,
            "full_length": profile.startswith("baseline-reproduction"),
            "covered_settings": len(settings),
            "paper_total_settings": 10,
            "excluded_setting": (
                "Math-10K/Gemma-3-12B-pt/epsilon=6/rank=16: gated checkpoint not staged"
                if profile == "baseline-reproduction-cached" else None
            ),
            "purpose": "estimate clipping trajectories and predeclare later SlaClip target grids",
        },
        "regime_map": {
            "exploratory": profile == "regime-map",
            "screen_steps": screen_steps,
            "per_task_eval_limit": eval_limit,
            "fixed_C_grid": [0.5, 1.0, 2.0, 3.0, 5.0] if profile == "regime-map" else [1.0],
            "conditional_rho_grid": [0.5, 0.7, 0.8, 0.9, 0.98] if profile == "regime-map" else [0.9, 0.98],
            "initial_C_sensitivity_control": (
                {"C_0": 2.0, "rho": 0.9, "eta": 0.05}
                if profile == "regime-map" else None
            ),
            "interpretation": (
                "descriptive one-seed screen; clipping-rate bins are measured outcomes, "
                "not predeclared failure thresholds or confirmatory evidence"
            ),
        },
        "arms": arms,
    }


PLAN_FIELDS = (
    "lane", "arm_id", "setting_id", "dataset", "model_slug", "model_id",
    "model_revision", "epsilon", "lora_r", "method", "initial_c", "rho",
    "eta", "steps", "learning_rate", "cutoff_len", "train_on_inputs",
    "seed", "eval_limit", "relative_root",
)

SEQUENTIAL_SETTING_ORDER = (
    "glue8-4b-eps6-r16",
    "math10k-4b-eps6-r16",
    "math10k-9b-eps6-r16",
    "math10k-12b-eps6-r16",
    "glue8-4b-eps3-r16",
    "math10k-4b-eps3-r16",
    "glue8-4b-eps6-r8",
    "math10k-4b-eps6-r8",
    "glue8-4b-eps6-r32",
    "math10k-4b-eps6-r32",
)


def _plan_bytes(manifest: dict[str, Any], lane: int, include_all: bool = False) -> bytes:
    rows = []
    arms = manifest["arms"]
    if include_all:
        priority = {setting: index for index, setting in enumerate(SEQUENTIAL_SETTING_ORDER)}
        arms = sorted(
            arms,
            key=lambda arm: (priority.get(arm["setting_id"], len(priority)), arm["arm_id"]),
        )
    for arm in arms:
        if not include_all and arm["lane"] != lane:
            continue
        values = {
            "lane": 0 if include_all else lane,
            "arm_id": arm["arm_id"],
            "setting_id": arm["setting_id"],
            "dataset": arm["dataset"],
            "model_slug": arm["model_slug"],
            "model_id": arm["model_id"],
            "model_revision": arm["model_revision"],
            "epsilon": arm["epsilon"],
            "lora_r": arm["lora_r"],
            "method": arm["method"],
            "initial_c": arm["initial_c"],
            "rho": "NA" if arm["rho"] is None else arm["rho"],
            "eta": "NA" if arm["eta"] is None else arm["eta"],
            "steps": arm["steps"],
            "learning_rate": arm["learning_rate"],
            "cutoff_len": arm["cutoff_len"],
            "train_on_inputs": str(arm["train_on_inputs"]).lower(),
            "seed": arm["seed"],
            "eval_limit": arm["eval_limit"],
            "relative_root": arm["relative_root"],
        }
        rows.append("|".join(str(values[field]) for field in PLAN_FIELDS))
    return ("\n".join(rows) + "\n").encode()


def prepare(root: Path, code_sha: str, model_4b_revision: str, model_9b_revision: str, profile: str, model_12b_revision: str | None = None) -> None:
    manifest = build_manifest(
        code_sha,
        model_4b_revision,
        model_9b_revision,
        profile=profile,
        model_12b_revision=model_12b_revision,
    )
    _with_sha(root / "plans" / "manifest.json", _json_bytes(manifest))
    _with_sha(root / "plans" / "lane-0.tsv", _plan_bytes(manifest, 0))
    _with_sha(root / "plans" / "lane-1.tsv", _plan_bytes(manifest, 1))
    _with_sha(root / "plans" / "sequential.tsv", _plan_bytes(manifest, 0, include_all=True))
    lane0 = sum(arm['lane'] == 0 for arm in manifest['arms'])
    lane1 = sum(arm['lane'] == 1 for arm in manifest['arms'])
    print(f"prepared_arms={len(manifest['arms'])} lane0={lane0} lane1={lane1} profile={profile}")


def _finite(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise CampaignError(f"{label} is not numeric") from exc
    if not math.isfinite(result):
        raise CampaignError(f"{label} is not finite")
    return result


def _task_average(summary_path: Path) -> float:
    try:
        rows = list(csv.DictReader(summary_path.open(encoding="utf-8", newline="")))
    except OSError as exc:
        raise CampaignError(f"cannot read evaluation summary: {summary_path}") from exc
    if len(rows) != 1:
        raise CampaignError(f"invalid evaluation summary: {summary_path}")
    for key in ("Average", "GLUE8_Avg", "Math10K_Avg"):
        if key in rows[0] and rows[0][key] not in (None, ""):
            return _finite(rows[0][key], f"{summary_path}:{key}")
    raise CampaignError(f"evaluation average column is missing: {summary_path}")


def _quantile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise CampaignError("cannot calculate a quantile of an empty series")
    position = (len(ordered) - 1) * fraction
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def _raw_series(path: Path, field: str) -> list[float]:
    values = []
    try:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                record = json.loads(line)
                if field in record and record[field] is not None:
                    values.append(_finite(record[field], f"{path}:{line_number}:{field}"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CampaignError(f"cannot read raw telemetry: {path}") from exc
    if not values:
        raise CampaignError(f"raw telemetry lacks {field}: {path}")
    return values


def _clip_bin(value: float) -> str:
    if value < 0.70:
        return "lt_70pct"
    if value < 0.90:
        return "70_to_lt_90pct"
    if value < 0.98:
        return "90_to_lt_98pct"
    return "ge_98pct"


def analyze(root: Path) -> None:
    manifest_path = root / "plans" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    results = []
    trajectory_rows = []
    for arm in manifest.get("arms", []):
        arm_root = root / arm["relative_root"]
        status_path = arm_root / "adapter" / "run_status.json"
        telemetry_path = arm_root / "results" / "research_raw" / "telemetry_summary.json"
        summary_path = arm_root / "results" / "summary.csv"
        try:
            status = json.loads(status_path.read_text(encoding="utf-8"))
            telemetry = json.loads(telemetry_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise CampaignError(f"incomplete arm {arm['arm_id']}: {exc}") from exc
        if status.get("state") != "completed" or status.get("config", {}).get("implementation_git_sha") != manifest["code_sha"]:
            raise CampaignError(f"arm is not completed at the locked SHA: {arm['arm_id']}")
        numeric = telemetry.get("metrics", telemetry.get("numeric_metrics", {}))
        metric = lambda name, key="mean": _finite(numeric.get(name, {}).get(key), f"{arm['arm_id']}:{name}.{key}")
        raw_path = arm_root / "results" / "research_raw" / "NON_PRIVATE_train_log.jsonl"
        clip_values = _raw_series(raw_path, "raw_clip_fraction")
        small_proxy_values = _raw_series(raw_path, "raw_reference_small_gradient_proxy")
        clip_median = _quantile(clip_values, 0.5)
        scalar_fields = (
            "step", "loss_mean", "eps_spent", "dp_clip_threshold",
            "dp_next_clip_threshold", "dp_noise_multiplier",
            "raw_realized_batch_size", "raw_clip_fraction",
            "raw_clip_coefficient_mean", "raw_clip_coefficient_min",
            "raw_global_norm_mean", "raw_global_norm_std",
            "raw_global_norm_min", "raw_global_norm_max",
            "raw_clipped_signal_norm", "raw_unclipped_signal_norm",
            "raw_clipping_bias_norm", "raw_realized_noise_norm",
            "raw_signal_to_noise_ratio", "raw_clipping_bias_to_noise_ratio",
            "raw_bias_noise_squared_error_proxy",
            "raw_unclipped_clipped_cosine", "raw_clipped_noisy_cosine",
            "raw_reference_small_gradient_proxy",
            "raw_reference_remaining_mass_proxy",
            "raw_reference_conditional_clip_fraction",
        )
        with raw_path.open(encoding="utf-8") as raw_handle:
            for raw_line in raw_handle:
                raw_record = json.loads(raw_line)
                trajectory_rows.append({
                    "setting_id": arm["setting_id"],
                    "dataset": arm["dataset"],
                    "model": arm["model_id"],
                    "epsilon": arm["epsilon"],
                    "lora_r": arm["lora_r"],
                    "fixed_C": arm["initial_c"],
                    "seed": arm["seed"],
                    **{field: raw_record.get(field) for field in scalar_fields},
                })
        results.append(
            {
                "setting_id": arm["setting_id"],
                "paper_reference": arm["paper_reference"],
                "dataset": arm["dataset"],
                "model": arm["model_id"],
                "epsilon": arm["epsilon"],
                "lora_r": arm["lora_r"],
                "candidate": arm["candidate_id"],
                "method": arm["method"],
                "rho": arm["rho"],
                "task_average": _task_average(summary_path),
                "loss_last": metric("loss_mean", "last"),
                "clip_fraction_mean": metric("raw_clip_fraction"),
                "clip_fraction_last": metric("raw_clip_fraction", "last"),
                "clip_fraction_p10": _quantile(clip_values, 0.1),
                "clip_fraction_median": clip_median,
                "clip_fraction_p90": _quantile(clip_values, 0.9),
                "clip_regime_bin": _clip_bin(clip_median),
                "small_gradient_proxy_mean": sum(small_proxy_values) / len(small_proxy_values),
                "small_gradient_proxy_median": _quantile(small_proxy_values, 0.5),
                "clip_threshold_mean": metric("dp_clip_threshold"),
                "clip_threshold_last": metric("dp_clip_threshold", "last"),
                "signal_to_noise_mean": metric("raw_signal_to_noise_ratio"),
                "bias_to_noise_mean": metric("raw_clipping_bias_to_noise_ratio"),
            }
        )
    out = root / "artifacts"
    out.mkdir(parents=True, exist_ok=True)
    fields = list(results[0])
    buffer = []
    buffer.append(",".join(fields))
    for row in results:
        buffer.append(",".join("" if row[k] is None else str(row[k]) for k in fields))
    _with_sha(out / "paper_coverage_summary.csv", ("\n".join(buffer) + "\n").encode())
    if trajectory_rows:
        trajectory_fields = list(trajectory_rows[0])
        trajectory_buffer = [",".join(trajectory_fields)]
        for row in trajectory_rows:
            trajectory_buffer.append(",".join(
                "" if row[field] is None else str(row[field])
                for field in trajectory_fields
            ))
        _with_sha(
            out / "baseline_telemetry_steps.csv",
            ("\n".join(trajectory_buffer) + "\n").encode(),
        )
    best_fixed = {}
    for row in results:
        if row["method"] == "baseline":
            best_fixed[row["setting_id"]] = max(
                best_fixed.get(row["setting_id"], float("-inf")), row["task_average"]
            )
    regime_groups: dict[str, list[dict[str, Any]]] = {}
    for row in results:
        if row["method"] != "slaclip":
            continue
        row["delta_vs_setting_best_fixed"] = row["task_average"] - best_fixed[row["setting_id"]]
        regime_groups.setdefault(row["clip_regime_bin"], []).append(row)
    regime_rows = []
    for label in ("lt_70pct", "70_to_lt_90pct", "90_to_lt_98pct", "ge_98pct"):
        members = regime_groups.get(label, [])
        deltas = [row["delta_vs_setting_best_fixed"] for row in members]
        regime_rows.append({
            "clip_regime_bin": label,
            "slaclip_arms": len(deltas),
            "mean_delta_vs_setting_best_fixed": sum(deltas) / len(deltas) if deltas else None,
            "win_rate_vs_setting_best_fixed": sum(value > 0 for value in deltas) / len(deltas) if deltas else None,
            "mean_small_gradient_proxy": (
                sum(row["small_gradient_proxy_mean"] for row in members) / len(members)
                if members else None
            ),
        })
    _with_sha(out / "paper_coverage_summary.json", _json_bytes({"schema_version": 2, "rows": results}))
    regime_fields = list(regime_rows[0])
    regime_csv = [",".join(regime_fields)]
    for row in regime_rows:
        regime_csv.append(",".join("" if row[key] is None else str(row[key]) for key in regime_fields))
    _with_sha(out / "clipping_regime_summary.csv", ("\n".join(regime_csv) + "\n").encode())
    _with_sha(out / "clipping_regime_summary.json", _json_bytes({
        "schema_version": 1,
        "inference": "exploratory_one_seed_descriptive_only",
        "rows": regime_rows,
    }))
    print(f"analyzed_arms={len(results)}")


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    sub = value.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--campaign-root", required=True, type=Path)
    prep.add_argument("--code-sha", required=True)
    prep.add_argument("--model-4b-revision", required=True)
    prep.add_argument("--model-9b-revision", required=True)
    prep.add_argument(
        "--profile",
        choices=(
            "paper-breadth", "regime-map", "baseline-reproduction",
            "baseline-reproduction-cached",
        ),
        default="paper-breadth",
    )
    prep.add_argument("--model-12b-revision")
    report = sub.add_parser("analyze")
    report.add_argument("--campaign-root", required=True, type=Path)
    return value


def main() -> None:
    args = parser().parse_args()
    if args.command == "prepare":
        prepare(
            args.campaign_root,
            args.code_sha,
            args.model_4b_revision,
            args.model_9b_revision,
            args.profile,
            args.model_12b_revision,
        )
    else:
        analyze(args.campaign_root)


if __name__ == "__main__":
    main()
