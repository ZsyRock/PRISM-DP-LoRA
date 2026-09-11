#!/usr/bin/env python3
"""Build and lock the one-A100 Math-10K low-target Full-SlaClip campaign.

The campaign has three phases. ``prepare`` writes an immutable screen registry
and plan. ``lock`` consumes only the registered public-validation artifacts,
selects one Full-SlaClip configuration for each conditional target, and writes
the immutable seed-matched confirmation plan. ``analyze`` consumes that locked
plan after all final evaluations complete and emits paired confidence
intervals. Exact clipping telemetry is used only for the explicitly
``NON_PRIVATE`` tracking analysis; task-test results are never read by
``prepare`` or ``lock``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import re
import statistics
import tempfile
from itertools import product
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = 1
STAGE1_SEED = 42
CONFIRMATION_SEEDS = (401, 433, 467, 503, 547)
MATH10K_DATA_SHA256 = "0342d0d860ad8592b579329337c90e42eefd3d9f2898043140cbd120630418b8"
EXPECTED_TEST_ASSETS = {
    "gsm8k": (1319, "2cc616a1e0b23ea1df29370476365cde71c4e0aa823fb06cc194c5a8a9381abe"),
    "AQuA": (254, "de677f5f0139340009eb01f17c0db781b4161912cd1f4efa77292f2fcf3478ee"),
    "mawps": (238, "c1a708979cbf2b3df9ac4020baacf3e8e73fcf089e1de7dd1b305b810bfaa15f"),
    "SVAMP": (1000, "ffed015784f738f317c5603177861d044c4fc94e09211d78b25e96d46c656e2c"),
}
RHO_VALUES = (0.8, 0.9)
C0_VALUES = (2.0, 3.0)
ETA_VALUES = (0.1,)
K = 15
C_MIN = 0.1
C_MAX = 15.0
STAGE1_STEPS = 150
STAGE1_BURN_IN_STEP = 50
FINAL_STEPS = 300
FINAL_BURN_IN_STEP = 100
TRACKING_GATE_MAE = 0.05
BOUND_HIT_RATE_MAX = 0.01
EPSILON_ACCOUNTING_TOL = 1e-6
CDF_INFORMATIVE_RATE_MIN = 0.05
MODEL_SLUG = "gemma-3-4b-pt"
FULL_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
TOL = 1e-9
EXPECTED_CANDIDATES = 4
EXPECTED_FINAL_CORE_ARMS = 15
T_95_DF4 = 2.7764451051977987


class CampaignError(RuntimeError):
    """Raised when a campaign artifact is missing or inconsistent."""


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise CampaignError(f"{label} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise CampaignError(f"{label} must be a finite number") from exc
    if not math.isfinite(result):
        raise CampaignError(f"{label} must be a finite number")
    return result


def _slug(value: float) -> str:
    return format(float(value), ".12g").replace("-", "m").replace(".", "p")


def _fixed_id(value: float) -> str:
    return f"fixed-c{_slug(value)}"


def _sla_id(c0: float, rho: float, eta: float) -> str:
    return f"sla-c{_slug(c0)}-r{_slug(rho)}-e{_slug(eta)}"


def _json_bytes(payload: Any) -> bytes:
    try:
        return (
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise CampaignError(f"artifact is not finite JSON: {exc}") from exc


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_immutable(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if not path.is_file() or path.read_bytes() != data:
            raise CampaignError(f"refusing to overwrite immutable artifact: {path}")
        return
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if not path.is_file() or path.read_bytes() != data:
                raise CampaignError(f"concurrent inconsistent writer: {path}")
    finally:
        temporary.unlink(missing_ok=True)
    os.chmod(path, 0o600)


def _write_with_sha(path: Path, data: bytes) -> None:
    _write_immutable(path, data)
    _write_immutable(
        path.with_name(path.name + ".sha256"),
        f"{_sha256(data)}  {path.name}\n".encode("ascii"),
    )


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CampaignError(f"cannot read {label}: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise CampaignError(f"{label} must be a JSON object: {path}")
    return payload


def _inside(root: Path, relative: str, label: str) -> Path:
    value = Path(relative)
    if value.is_absolute():
        raise CampaignError(f"{label} must be relative to the campaign root")
    try:
        resolved = (root / value).resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, ValueError) as exc:
        raise CampaignError(f"{label} escapes or is missing from {root}: {relative}") from exc
    return resolved


def _verify_sha_sidecar(path: Path) -> None:
    sidecar = path.with_name(path.name + ".sha256")
    try:
        fields = sidecar.read_text(encoding="ascii").strip().split()
    except OSError as exc:
        raise CampaignError(f"missing SHA-256 sidecar: {sidecar}") from exc
    if fields != [_sha256(path.read_bytes()), path.name]:
        raise CampaignError(f"invalid SHA-256 sidecar: {sidecar}")


def _candidate_run(candidate_id: str) -> dict[str, str]:
    relative = Path("screen") / "runs" / candidate_id / f"seed-{STAGE1_SEED}"
    return {
        "arm_root": relative.as_posix(),
        "run_status": (relative / "adapter" / "run_status.json").as_posix(),
        "validation_metrics": (
            relative / "results" / "validation" / "validation_metrics.json"
        ).as_posix(),
        "split_manifest": (
            relative / "results" / "validation" / "split_manifest.json"
        ).as_posix(),
        "raw_log": (
            relative
            / "results"
            / "research_raw"
            / "NON_PRIVATE_train_log.jsonl"
        ).as_posix(),
    }


def build_registry(
    *,
    code_sha: str,
    model_id: str,
    model_revision: str,
) -> dict[str, Any]:
    if not FULL_SHA_RE.fullmatch(code_sha) or not FULL_SHA_RE.fullmatch(model_revision):
        raise CampaignError("code_sha and model_revision must be full lowercase SHA values")
    model_id = str(model_id).strip()
    if not model_id or any(character.isspace() for character in model_id):
        raise CampaignError("model_id must be a non-empty identifier without whitespace")
    candidates: list[dict[str, Any]] = []
    for c0, rho, eta in product(C0_VALUES, RHO_VALUES, ETA_VALUES):
        identifier = _sla_id(c0, rho, eta)
        candidates.append(
            {
                "id": identifier,
                "family": "slaclip",
                "method": "slaclip",
                "params": {
                    "dp_max_grad_norm": c0,
                    "slaclip_target_non_small_clip_fraction": rho,
                    "slaclip_eta": eta,
                    "slaclip_num_slots": K,
                    "slaclip_c_min": C_MIN,
                    "slaclip_c_max": C_MAX,
                },
                "stage1_run": _candidate_run(identifier),
            }
        )
    if (
        len(candidates) != EXPECTED_CANDIDATES
        or len({item["id"] for item in candidates}) != EXPECTED_CANDIDATES
    ):
        raise CampaignError(
            f"low-target registry must contain {EXPECTED_CANDIDATES} unique candidates"
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol": "math10k_4b_full_slaclip_low_conditional_target_v1",
        "code_sha": code_sha,
        "model_id": model_id,
        "model_revision": model_revision,
        "stage1_seed": STAGE1_SEED,
        "confirmation_seeds": list(CONFIRMATION_SEEDS),
        "historical_test_results_available_before_protocol": True,
        "inference_class": (
            "locked_internal_paired_confirmation_not_untouched_external_replication"
        ),
        "predeclared_strongest_fixed_c2": {
            "id": "fixed-c2",
            "family": "fixed",
            "method": "baseline",
            "params": {"dp_max_grad_norm": 2.0},
            "selection_basis": (
                "predeclared before this campaign from the completed historical "
                "fixed-C grid; rerun on the same A100 allocation and the same "
                "confirmation seeds as both SlaClip arms"
            ),
            "claim_scope": (
                "established fixed C=2 control and strongest in the completed "
                "pre-campaign grid; not a global optimum over every possible C"
            ),
        },
        "full_slaclip_target_semantics": {
            "rho": "conditional clipped fraction of residual/non-small proxy mass",
            "dynamic_global_target": "p_star_t=rho*(1-z_t)",
            "z_t": "noisy last Slack-Indicator coordinate divided by C_t",
            "warning": "rho is not a fixed whole-batch clipping fraction",
        },
        "selection": {
            "public_holdout_rows": 500,
            "public_holdout_seed": 1729,
            "primary_metric": "public_math10k_numeric_exact_match_accuracy",
            "tie_break": "response_only_public_validation_loss",
            "common_config": {
                "dataset": "math10k",
                "privacy": "dp",
                "base_model": model_id,
                "model_revision": model_revision,
                "implementation_git_sha": code_sha,
                "total_update_steps": STAGE1_STEPS,
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
                "run_eval": False,
                "dp_epsilon": 6.0,
                "dp_delta": 1e-5,
                "dp_accountant": "prv",
                "dp_secure_mode": False,
                "telemetry_mode": "research_raw",
                "allow_non_private_telemetry": True,
                "raw_hist_bins": 128,
                "raw_hist_max": 30.0,
            },
            "tracking_audit": {
                "NON_PRIVATE": True,
                "burn_in_steps_excluded": STAGE1_BURN_IN_STEP,
                "descriptive_mae_reference": TRACKING_GATE_MAE,
                "descriptive_bound_hit_rate_reference": BOUND_HIT_RATE_MAX,
                "selection_effect": (
                    "descriptive_only; accuracy is primary and tracking metrics "
                    "are tie-break diagnostics"
                ),
            },
        },
        "final": {
            "comparisons": [
                "rho_0.9_full_slaclip_vs_established_fixed_C2",
                "rho_0.8_full_slaclip_vs_established_fixed_C2",
            ],
            "primary_locked_target": 0.9,
            "secondary_exploratory_target": 0.8,
            "primary_metric": "clean_three_task_macro_accuracy",
            "clean_tasks": ["gsm8k", "AQuA", "SVAMP"],
            "total_update_steps": FINAL_STEPS,
            "same_allocation_pairing": True,
        },
        "candidates": candidates,
    }


def _emit_spec(
    *,
    phase: str,
    candidate: Mapping[str, Any],
    role: str,
    seed: int,
    arm_root: Path,
) -> str:
    params = candidate["params"]
    fields = (
        "train",
        phase,
        candidate["id"],
        role,
        seed,
        candidate["method"],
        params["dp_max_grad_norm"],
        params.get("slaclip_target_non_small_clip_fraction", "NA"),
        params.get("slaclip_eta", "NA"),
        "NA",
        arm_root,
    )
    if any("|" in str(value) or "\n" in str(value) for value in fields):
        raise CampaignError("plan field contains a forbidden delimiter")
    return "|".join(str(value) for value in fields)


def build_stage1_plan(registry: Mapping[str, Any], campaign_root: Path) -> bytes:
    rows = []
    for candidate in registry["candidates"]:
        rows.append(
            _emit_spec(
                phase="selection",
                candidate=candidate,
                role="public-validation-screen",
                seed=STAGE1_SEED,
                arm_root=(
                    campaign_root
                    / "screen"
                    / "runs"
                    / candidate["id"]
                    / f"seed-{STAGE1_SEED}"
                ),
            )
        )
    if len(rows) != EXPECTED_CANDIDATES:
        raise CampaignError(
            f"stage1 plan must contain exactly {EXPECTED_CANDIDATES} arms"
        )
    return ("\n".join(rows) + "\n").encode("utf-8")


def prepare(
    *, campaign_root: Path, code_sha: str, model_id: str, model_revision: str
) -> dict[str, Any]:
    campaign_root.mkdir(parents=True, exist_ok=True)
    registry = build_registry(
        code_sha=code_sha,
        model_id=model_id,
        model_revision=model_revision,
    )
    registry_path = campaign_root / "screen" / "candidate_registry.json"
    plan_path = campaign_root / "plans" / "stage1.tsv"
    _write_with_sha(registry_path, _json_bytes(registry))
    _write_with_sha(plan_path, build_stage1_plan(registry, campaign_root.resolve()))
    return {
        "registry": str(registry_path),
        "stage1_plan": str(plan_path),
    }


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise CampaignError(f"invalid JSONL object: {path}:{number}")
                rows.append(value)
    except (OSError, json.JSONDecodeError) as exc:
        raise CampaignError(f"cannot read raw telemetry {path}: {exc}") from exc
    return rows


def _validate_privacy_accounting(
    status: Mapping[str, Any], *, expected_steps: int, label: str
) -> dict[str, float | str]:
    accounting = status.get("privacy_accounting")
    if not isinstance(accounting, dict):
        raise CampaignError(f"{label} has no privacy accounting")
    epsilon_spent = _finite(accounting.get("epsilon_spent"), f"{label} epsilon spent")
    noise_multiplier = _finite(
        accounting.get("noise_multiplier"), f"{label} noise multiplier"
    )
    sample_rate = _finite(accounting.get("sample_rate"), f"{label} sample rate")
    expected_batch_size = _finite(
        accounting.get("expected_batch_size"), f"{label} expected batch size"
    )
    if (
        accounting.get("accountant") != "prv"
        or accounting.get("grad_sample_mode") != "functorch"
        or accounting.get("secure_mode") is not False
        or accounting.get("scope") != "single_training_run"
        or int(accounting.get("completed_update_steps", -1)) != expected_steps
        or not math.isclose(
            _finite(accounting.get("target_epsilon"), f"{label} target epsilon"),
            6.0,
            rel_tol=0.0,
            abs_tol=TOL,
        )
        or not math.isclose(
            _finite(accounting.get("target_delta"), f"{label} target delta"),
            1e-5,
            rel_tol=0.0,
            abs_tol=TOL,
        )
        or not 5.9 <= epsilon_spent <= 6.0 + EPSILON_ACCOUNTING_TOL
        or noise_multiplier <= 0.0
        or not 0.0 < sample_rate <= 1.0
        or not 63.0 <= expected_batch_size <= 65.0
    ):
        raise CampaignError(f"{label} privacy accounting mismatch")
    return {
        "scope": "single_training_run",
        "epsilon_spent": epsilon_spent,
        "noise_multiplier": noise_multiplier,
        "sample_rate": sample_rate,
        "expected_batch_size": expected_batch_size,
    }


def _validate_run(
    campaign_root: Path,
    candidate: Mapping[str, Any],
    registry: Mapping[str, Any],
) -> dict[str, Any]:
    run = candidate["stage1_run"]
    status_path = _inside(campaign_root, run["run_status"], "run status")
    metrics_path = _inside(
        campaign_root, run["validation_metrics"], "validation metrics"
    )
    split_path = _inside(campaign_root, run["split_manifest"], "split manifest")
    raw_path = _inside(campaign_root, run["raw_log"], "raw telemetry")
    status = _read_json(status_path, "stage1 run status")
    metrics = _read_json(metrics_path, "public validation metrics")
    split = _read_json(split_path, "public validation split manifest")
    config = status.get("config")
    if not isinstance(config, dict):
        raise CampaignError(f"stage1 run has no configuration: {status_path}")
    params = candidate["params"]
    common = registry["selection"]["common_config"]
    config_fingerprint = config.get("config_fingerprint")
    if not isinstance(config_fingerprint, str) or not config_fingerprint:
        raise CampaignError(f"stage1 run has no config fingerprint: {status_path}")
    if (
        status.get("state") != "completed"
        or int(status.get("update_steps", -1)) != STAGE1_STEPS
        or status.get("method") != candidate["method"]
        or status.get("privacy") != "dp"
        or status.get("dataset") != "math10k"
        or status.get("base_model") != registry["model_id"]
        or status.get("model_revision") != registry["model_revision"]
        or status.get("resolved_model_revision") != registry["model_revision"]
        or status.get("data_content_sha256") != MATH10K_DATA_SHA256
        or status.get("config_fingerprint") != config_fingerprint
        or status.get("non_private_telemetry") is not True
    ):
        raise CampaignError(f"stage1 run is incomplete: {status_path}")
    if (
        config.get("method") != candidate["method"]
        or int(config.get("seed", -1)) != STAGE1_SEED
        or config.get("resolved_model_revision") != registry["model_revision"]
        or config.get("implementation_git_dirty") is not False
        or config.get("data_content_sha256") != MATH10K_DATA_SHA256
        or config.get("run_train") is not True
        or config.get("dp_grad_sample_mode") != "functorch"
        or int(config.get("lora_r", -1)) != 16
        or int(config.get("lora_alpha", -1)) != 16
        or config.get("prism_floor_mode") != "scalar"
        or config.get("prism_lift_fix") != "both"
        or config.get("prism_debias_second_moment") is not False
    ):
        raise CampaignError(f"stage1 method/seed mismatch: {status_path}")
    for key, expected in {
        "lora_dropout": 0.05,
        "prism_floor_factor": 0.5,
        "prism_cond_max": 10000.0,
    }.items():
        if not math.isclose(
            _finite(config.get(key), f"stage1 {key}"),
            expected,
            rel_tol=0.0,
            abs_tol=TOL,
        ):
            raise CampaignError(f"stage1 {key} mismatch: {status_path}")
    if sorted(config.get("target_modules") or []) != sorted(
        ["q_proj", "k_proj", "v_proj", "up_proj", "down_proj"]
    ):
        raise CampaignError(f"stage1 target modules mismatch: {status_path}")
    privacy_calibration = _validate_privacy_accounting(
        status, expected_steps=STAGE1_STEPS, label="stage1 run"
    )
    exact_common = (
        "dataset",
        "privacy",
        "base_model",
        "model_revision",
        "implementation_git_sha",
        "total_update_steps",
        "batch_size",
        "micro_batch_size",
        "cutoff_len",
        "train_on_inputs",
        "val_set_size",
        "validation_seed",
        "validation_batch_size",
        "validation_eval_interval",
        "validation_generate_numeric",
        "validation_num_beams",
        "validation_max_new_tokens",
        "validation_max_input_length",
        "protocol_stage",
        "validation_data_is_public",
        "run_eval",
        "dp_accountant",
        "dp_secure_mode",
        "telemetry_mode",
        "allow_non_private_telemetry",
        "raw_hist_bins",
    )
    for key in exact_common:
        if config.get(key) != common[key]:
            raise CampaignError(
                f"stage1 common config mismatch for {key}: {status_path}"
            )
    for key in ("learning_rate", "dp_epsilon", "dp_delta", "raw_hist_max"):
        if not math.isclose(
            _finite(config.get(key), f"run {key}"),
            _finite(common[key], f"registered {key}"),
            rel_tol=0.0,
            abs_tol=TOL,
        ):
            raise CampaignError(f"stage1 common config mismatch for {key}: {status_path}")
    if not math.isclose(
        _finite(config.get("dp_max_grad_norm"), "run C"),
        _finite(params["dp_max_grad_norm"], "candidate C"),
        rel_tol=0.0,
        abs_tol=TOL,
    ):
        raise CampaignError(f"stage1 clipping threshold mismatch: {status_path}")
    if candidate["family"] == "slaclip":
        for key in (
            "slaclip_target_non_small_clip_fraction",
            "slaclip_eta",
            "slaclip_c_min",
            "slaclip_c_max",
        ):
            if not math.isclose(
                _finite(config.get(key), f"run {key}"),
                _finite(params[key], f"candidate {key}"),
                rel_tol=0.0,
                abs_tol=TOL,
            ):
                raise CampaignError(f"stage1 {key} mismatch: {status_path}")
        if int(config.get("slaclip_num_slots", -1)) != K:
            raise CampaignError(f"stage1 K mismatch: {status_path}")

    if (
        split.get("validation_data_is_public") is not True
        or split.get("protocol_stage") != "selection"
        or int(split.get("seed", -1)) != 1729
        or int(split.get("validation_rows", -1)) != 500
    ):
        raise CampaignError(f"invalid public split manifest: {split_path}")
    split_sha = _sha256(_json_bytes(split))
    accuracy = _finite(metrics.get("numeric_exact_accuracy"), "validation accuracy")
    loss = _finite(metrics.get("loss_mean"), "validation loss")
    if not 0.0 <= accuracy <= 1.0:
        raise CampaignError(f"validation accuracy is outside [0,1]: {metrics_path}")
    if (
        int(metrics.get("records", -1)) != 500
        or int(metrics.get("seed", -1)) != 1729
        or metrics.get("validation_data_is_public") is not True
        or metrics.get("PUBLIC_VALIDATION_DATA") is not True
        or metrics.get("protocol_stage") != "selection"
        or metrics.get("selection_metric")
        != "public_math10k_numeric_exact_match_accuracy"
        or metrics.get("loss_definition")
        != "response_only_per_record_mean_of_nonignored_next_token_losses"
        or metrics.get("manifest_sha256") != split.get("manifest_sha256")
    ):
        raise CampaignError(f"invalid public validation metrics: {metrics_path}")

    raw = _load_jsonl(raw_path)
    if [int(row.get("step", -1)) for row in raw] != list(
        range(1, STAGE1_STEPS + 1)
    ):
        raise CampaignError(f"raw telemetry must cover steps 1..150: {raw_path}")
    if any(
        row.get("NON_PRIVATE_TELEMETRY") is not True
        or row.get("method") != candidate["method"]
        or row.get("privacy") != "dp"
        or row.get("dataset") != "math10k"
        or row.get("base_model") != registry["model_id"]
        or row.get("model_revision") != registry["model_revision"]
        or row.get("resolved_model_revision") != registry["model_revision"]
        or row.get("config_fingerprint") != config_fingerprint
        or row.get("telemetry_mode") != "research_raw"
        for row in raw
    ):
        raise CampaignError(f"raw telemetry identity is invalid: {raw_path}")
    post = [row for row in raw if int(row["step"]) > STAGE1_BURN_IN_STEP]
    clips = [_finite(row.get("raw_clip_fraction"), "raw clip fraction") for row in post]
    if any(not 0.0 <= value <= 1.0 for value in clips):
        raise CampaignError(f"raw clipping fraction is outside [0,1]: {raw_path}")
    result: dict[str, Any] = {
        "candidate_id": candidate["id"],
        "family": candidate["family"],
        "method": candidate["method"],
        "params": params,
        "validation_accuracy": accuracy,
        "validation_loss": loss,
        "public_split_sha256": split_sha,
        "validation_metrics_sha256": _sha256(metrics_path.read_bytes()),
        "raw_telemetry_sha256": _sha256(raw_path.read_bytes()),
        "privacy_calibration": privacy_calibration,
        "post_burn_in_clip_mean": statistics.fmean(clips),
        "post_burn_in_clip_sd": statistics.stdev(clips),
        "post_burn_in_clip_min": min(clips),
        "post_burn_in_clip_max": max(clips),
        "post_burn_in_steps": len(post),
    }
    if candidate["family"] == "slaclip":
        rho = _finite(params["slaclip_target_non_small_clip_fraction"], "rho")
        targets: list[float] = []
        errors: list[float] = []
        exact_z_values: list[float] = []
        exact_z_snrs: list[float] = []
        hit_count = 0
        formula_residuals: list[float] = []
        for row, clip in zip(post, clips, strict=True):
            z_t = _finite(row.get("slaclip_small_gradient_proxy_noisy"), "z_t")
            target = _finite(
                row.get("slaclip_target_clipped_proxy"), "dynamic target"
            )
            expected = max(0.0, min(1.0, rho * (1.0 - z_t)))
            formula_residuals.append(abs(target - expected))
            targets.append(target)
            errors.append(clip - target)
            hit_min = bool(row.get("slaclip_c_hit_min"))
            hit_max = bool(row.get("slaclip_c_hit_max"))
            if hit_min and hit_max:
                raise CampaignError(f"controller hit both C bounds: {raw_path}")
            hit_count += int(hit_min or hit_max)
            exact_z = _finite(
                row.get("raw_reference_small_gradient_proxy"), "exact z_t"
            )
            noise_sd = _finite(
                row.get("slack_indicator_noise_std"), "Slack noise standard deviation"
            ) / _finite(row.get("dp_clip_threshold"), "C_t")
            exact_z_values.append(exact_z)
            exact_z_snrs.append(abs(exact_z) / noise_sd if noise_sd > 0 else 0.0)
        if max(formula_residuals) > 2e-6:
            raise CampaignError(
                f"dynamic target does not equal projected rho*(1-z_t): {raw_path}"
            )
        mae = statistics.fmean(abs(value) for value in errors)
        rmse = math.sqrt(statistics.fmean(value * value for value in errors))
        hit_rate = hit_count / len(post)
        result.update(
            {
                "conditional_rho": rho,
                "dynamic_global_target_mean": statistics.fmean(targets),
                "dynamic_global_target_sd": statistics.stdev(targets),
                "dynamic_target_formula_max_abs_error": max(formula_residuals),
                "post_burn_in_tracking_mae": mae,
                "post_burn_in_tracking_rmse": rmse,
                "post_burn_in_tracking_bias": statistics.fmean(errors),
                "post_burn_in_time_within_0p05": statistics.fmean(
                    abs(value) <= TRACKING_GATE_MAE for value in errors
                ),
                "post_burn_in_controller_bound_hit_rate": hit_rate,
                "exact_small_gradient_proxy_mean": statistics.fmean(exact_z_values),
                "exact_small_gradient_proxy_max": max(exact_z_values),
                "exact_small_gradient_proxy_nonzero_rate": statistics.fmean(
                    abs(value) > TOL for value in exact_z_values
                ),
                "exact_small_gradient_proxy_snr_mean": statistics.fmean(exact_z_snrs),
                "exact_small_gradient_proxy_snr_gt_1_rate": statistics.fmean(
                    value > 1.0 for value in exact_z_snrs
                ),
                "tracking_reference_passed": (
                    mae <= TRACKING_GATE_MAE and hit_rate <= BOUND_HIT_RATE_MAX
                ),
            }
        )
    return result


def _select_slaclip(rows: Sequence[dict[str, Any]], rho: float) -> dict[str, Any]:
    group = [
        row
        for row in rows
        if row["family"] == "slaclip"
        and math.isclose(row["conditional_rho"], rho, abs_tol=TOL)
    ]
    if len(group) != len(C0_VALUES) * len(ETA_VALUES):
        raise CampaignError(
            f"rho={rho:g} must have exactly "
            f"{len(C0_VALUES) * len(ETA_VALUES)} stage1 candidates"
        )
    ranked = sorted(
        group,
        key=lambda row: (
            -row["validation_accuracy"],
            row["validation_loss"],
            row["post_burn_in_tracking_mae"],
            row["candidate_id"],
        ),
    )
    return {
        **ranked[0],
        "selection_basis": (
            "descending_public_validation_accuracy_then_ascending_loss_then_"
            "tracking_mae_then_candidate_id"
        ),
    }


def _candidate_map(registry: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    candidates = [
        *registry["candidates"],
        registry["predeclared_strongest_fixed_c2"],
    ]
    return {item["id"]: item for item in candidates}


def _build_final_plan(
    *,
    registry: Mapping[str, Any],
    selection: Mapping[str, Any],
    campaign_root: Path,
) -> bytes:
    candidates = _candidate_map(registry)
    selected = {
        rho: candidates[selection["selected_slaclip"][str(rho)]["candidate_id"]]
        for rho in ("0.8", "0.9")
    }
    strongest = candidates[
        selection["predeclared_strongest_fixed_c2"]["candidate_id"]
    ]
    rows: list[str] = []
    for seed in CONFIRMATION_SEEDS:
        for rho in ("0.8", "0.9"):
            role = f"slaclip-rho-{rho.replace('.', 'p')}"
            rows.append(
                _emit_spec(
                    phase="final",
                    candidate=selected[rho],
                    role=role,
                    seed=seed,
                    arm_root=campaign_root
                    / "final"
                    / MODEL_SLUG
                    / f"seed-{seed}"
                    / role,
                )
            )
        rows.append(
            _emit_spec(
                phase="final",
                candidate=strongest,
                role="predeclared-strongest-fixed-c2",
                seed=seed,
                arm_root=campaign_root
                / "final"
                / MODEL_SLUG
                / f"seed-{seed}"
                / "predeclared-strongest-fixed-c2",
            )
        )
    if len(rows) != EXPECTED_FINAL_CORE_ARMS:
        raise CampaignError(
            f"final plan must contain {EXPECTED_FINAL_CORE_ARMS} arms, got {len(rows)}"
        )
    return ("\n".join(rows) + "\n").encode("utf-8")


def lock(*, campaign_root: Path) -> dict[str, Any]:
    campaign_root = campaign_root.resolve(strict=True)
    registry_path = campaign_root / "screen" / "candidate_registry.json"
    _verify_sha_sidecar(registry_path)
    _verify_sha_sidecar(campaign_root / "plans" / "stage1.tsv")
    registry = _read_json(registry_path, "candidate registry")
    if registry.get("schema_version") != SCHEMA_VERSION:
        raise CampaignError("unsupported candidate registry schema")
    expected_registry = build_registry(
        code_sha=str(registry.get("code_sha", "")),
        model_id=str(registry.get("model_id", "")),
        model_revision=str(registry.get("model_revision", "")),
    )
    if registry != expected_registry:
        raise CampaignError("candidate registry differs from the predeclared protocol")
    expected_stage1 = build_stage1_plan(registry, campaign_root)
    if (campaign_root / "plans" / "stage1.tsv").read_bytes() != expected_stage1:
        raise CampaignError("stage1 plan differs from the candidate registry")
    rows = [
        _validate_run(campaign_root, item, registry)
        for item in registry["candidates"]
    ]
    split_hashes = {row["public_split_sha256"] for row in rows}
    if len(split_hashes) != 1:
        raise CampaignError("stage1 candidates did not use one identical public split")
    selected_slaclip = {
        str(rho): _select_slaclip(rows, rho) for rho in RHO_VALUES
    }
    selection = {
        "schema_version": SCHEMA_VERSION,
        "protocol": registry["protocol"],
        "candidate_registry_sha256": _sha256(registry_path.read_bytes()),
        "stage1_seed": STAGE1_SEED,
        "test_assets_accessed": False,
        "selection_uses_public_math10k_holdout_only": True,
        "historical_test_results_available_before_protocol": True,
        "inference_class": registry["inference_class"],
        "public_split_sha256": next(iter(split_hashes)),
        "predeclared_strongest_fixed_c2": {
            "candidate_id": registry["predeclared_strongest_fixed_c2"]["id"],
            "params": registry["predeclared_strongest_fixed_c2"]["params"],
            "selection_basis": registry["predeclared_strongest_fixed_c2"][
                "selection_basis"
            ],
            "claim_scope": registry["predeclared_strongest_fixed_c2"]["claim_scope"],
        },
        "selected_slaclip": selected_slaclip,
        "all_stage1_results": rows,
        "confirmation_seeds": list(CONFIRMATION_SEEDS),
        "inference_warning": (
            "rho is conditional; every claim must report dynamic p_star_t=rho*(1-z_t) "
            "and realized raw clipping separately. Exact tracking telemetry is "
            "NON_PRIVATE and descriptive, not an end-to-end DP release."
        ),
    }
    selection_path = campaign_root / "selection" / "selection.json"
    plan_path = campaign_root / "plans" / "final.tsv"
    _write_with_sha(selection_path, _json_bytes(selection))
    _write_with_sha(
        plan_path,
        _build_final_plan(
            registry=registry, selection=selection, campaign_root=campaign_root
        ),
    )
    buffer = io.StringIO()
    fields = (
        "candidate_id",
        "family",
        "validation_accuracy",
        "validation_loss",
        "post_burn_in_clip_mean",
        "post_burn_in_clip_sd",
        "conditional_rho",
        "dynamic_global_target_mean",
        "post_burn_in_tracking_mae",
        "post_burn_in_tracking_rmse",
        "post_burn_in_tracking_bias",
        "post_burn_in_time_within_0p05",
        "post_burn_in_controller_bound_hit_rate",
        "exact_small_gradient_proxy_nonzero_rate",
        "exact_small_gradient_proxy_snr_mean",
        "exact_small_gradient_proxy_snr_gt_1_rate",
        "tracking_reference_passed",
    )
    writer = csv.DictWriter(buffer, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    _write_with_sha(
        campaign_root / "selection" / "stage1_results.csv",
        buffer.getvalue().encode("utf-8"),
    )
    return {
        "selection": str(selection_path),
        "final_plan": str(plan_path),
        "selected_rho_0p8": selected_slaclip["0.8"]["candidate_id"],
        "selected_rho_0p9": selected_slaclip["0.9"]["candidate_id"],
    }


def _plan_path_inside(campaign_root: Path, value: str, label: str) -> Path:
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = campaign_root / candidate
    try:
        resolved = candidate.resolve(strict=False)
        resolved.relative_to(campaign_root)
    except (OSError, ValueError) as exc:
        raise CampaignError(f"{label} escapes campaign root: {value}") from exc
    return resolved


def _parse_plan(path: Path, campaign_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise CampaignError(f"cannot read final plan: {path}: {exc}") from exc
    for number, line in enumerate(lines, 1):
        if not line:
            continue
        fields = line.split("|")
        if len(fields) != 11:
            raise CampaignError(f"final plan row {number} must have 11 fields")
        try:
            seed = int(fields[4])
        except ValueError as exc:
            raise CampaignError(f"final plan row {number} has invalid seed") from exc
        rows.append(
            {
                "kind": fields[0],
                "phase": fields[1],
                "candidate_id": fields[2],
                "role": fields[3],
                "seed": seed,
                "method": fields[5],
                "clip": _finite(fields[6], "plan C"),
                "rho": fields[7],
                "eta": fields[8],
                "schedule": fields[9],
                "arm_root": _plan_path_inside(
                    campaign_root, fields[10], f"final plan row {number} arm"
                ),
            }
        )
    return rows


def _validate_final_result(
    *,
    record: Mapping[str, Any],
    candidate: Mapping[str, Any],
    registry: Mapping[str, Any],
) -> dict[str, Any]:
    arm_root = Path(record["arm_root"])
    status_path = arm_root / "adapter" / "run_status.json"
    summary_path = arm_root / "results" / "summary.csv"
    evaluation_config_path = arm_root / "results" / "evaluation_config.json"
    split_manifest_path = (
        arm_root / "results" / "validation" / "split_manifest.json"
    )
    raw_path = (
        arm_root
        / "results"
        / "research_raw"
        / "NON_PRIVATE_train_log.jsonl"
    )
    status = _read_json(status_path, "final run status")
    evaluation_config = _read_json(
        evaluation_config_path, "final evaluation config"
    )
    split_manifest = _read_json(split_manifest_path, "final split manifest")
    config = status.get("config")
    if not isinstance(config, dict):
        raise CampaignError(f"final run has no config: {status_path}")
    if status.get("data_split") != split_manifest or (
        split_manifest.get("schema_version") != 2
        or split_manifest.get("protocol_stage") != "final"
        or split_manifest.get("validation_data_is_public") is not False
        or int(split_manifest.get("seed", -1)) != 1729
        or int(split_manifest.get("requested_validation_rows", -1)) != 0
        or int(split_manifest.get("validation_rows", -1)) != 0
        or int(split_manifest.get("source_rows", -1)) != 9919
        or int(split_manifest.get("train_rows", -1)) != 9919
        or split_manifest.get("source_content_sha256") != MATH10K_DATA_SHA256
    ):
        raise CampaignError(f"final split manifest mismatch: {split_manifest_path}")
    if (
        status.get("state") != "completed"
        or int(status.get("update_steps", -1)) != FINAL_STEPS
        or status.get("method") != candidate["method"]
        or status.get("dataset") != "math10k"
        or status.get("privacy") != "dp"
        or status.get("base_model") != registry["model_id"]
        or status.get("model_revision") != registry["model_revision"]
        or status.get("resolved_model_revision") != registry["model_revision"]
        or status.get("data_content_sha256") != MATH10K_DATA_SHA256
        or status.get("non_private_telemetry") is not True
        or config.get("method") != candidate["method"]
        or int(config.get("seed", -1)) != int(record["seed"])
        or config.get("protocol_stage") != "final"
        or int(config.get("val_set_size", -1)) != 0
        or config.get("run_train") is not True
        or config.get("run_eval") is not True
        or config.get("base_model") != registry["model_id"]
        or config.get("model_revision") != registry["model_revision"]
        or config.get("resolved_model_revision") != registry["model_revision"]
        or config.get("implementation_git_sha") != registry["code_sha"]
        or config.get("implementation_git_dirty") is not False
        or config.get("data_content_sha256") != MATH10K_DATA_SHA256
        or int(config.get("total_update_steps", -1)) != FINAL_STEPS
        or int(config.get("batch_size", -1)) != 64
        or int(config.get("micro_batch_size", -1)) != 4
        or int(config.get("cutoff_len", -1)) != 256
        or config.get("train_on_inputs") is not True
        or config.get("privacy") != "dp"
        or config.get("dp_accountant") != "prv"
        or config.get("dp_grad_sample_mode") != "functorch"
        or config.get("dp_secure_mode") is not False
        or config.get("telemetry_mode") != "research_raw"
        or config.get("allow_non_private_telemetry") is not True
        or int(config.get("raw_hist_bins", -1)) != 128
        or int(config.get("lora_r", -1)) != 16
        or int(config.get("lora_alpha", -1)) != 16
        or config.get("prism_floor_mode") != "scalar"
        or config.get("prism_lift_fix") != "both"
        or config.get("prism_debias_second_moment") is not False
        or not math.isclose(
            _finite(config.get("dp_epsilon"), "final epsilon"),
            6.0,
            rel_tol=0.0,
            abs_tol=TOL,
        )
        or not math.isclose(
            _finite(config.get("dp_delta"), "final delta"),
            1e-5,
            rel_tol=0.0,
            abs_tol=TOL,
        )
    ):
        raise CampaignError(f"final run identity/config mismatch: {status_path}")
    for key, expected in {
        "learning_rate": 0.0003,
        "lora_dropout": 0.05,
        "raw_hist_max": 30.0,
        "prism_floor_factor": 0.5,
        "prism_cond_max": 10000.0,
    }.items():
        if not math.isclose(
            _finite(config.get(key), f"final {key}"),
            expected,
            rel_tol=0.0,
            abs_tol=TOL,
        ):
            raise CampaignError(f"final run {key} mismatch: {status_path}")
    if sorted(config.get("target_modules") or []) != sorted(
        ["q_proj", "k_proj", "v_proj", "up_proj", "down_proj"]
    ):
        raise CampaignError(f"final target modules mismatch: {status_path}")
    privacy_calibration = _validate_privacy_accounting(
        status, expected_steps=FINAL_STEPS, label="final run"
    )
    params = candidate["params"]
    if not math.isclose(
        _finite(config.get("dp_max_grad_norm"), "final run C"),
        _finite(params["dp_max_grad_norm"], "candidate C"),
        rel_tol=0.0,
        abs_tol=TOL,
    ):
        raise CampaignError(f"final run C mismatch: {status_path}")
    if candidate["family"] == "slaclip":
        for key in ("slaclip_target_non_small_clip_fraction", "slaclip_eta"):
            if not math.isclose(
                _finite(config.get(key), f"final {key}"),
                _finite(params[key], f"candidate {key}"),
                rel_tol=0.0,
                abs_tol=TOL,
            ):
                raise CampaignError(f"final run {key} mismatch: {status_path}")
        if (
            int(config.get("slaclip_num_slots", -1)) != K
            or not math.isclose(
                _finite(config.get("slaclip_c_min"), "final C min"),
                C_MIN,
                rel_tol=0.0,
                abs_tol=TOL,
            )
            or not math.isclose(
                _finite(config.get("slaclip_c_max"), "final C max"),
                C_MAX,
                rel_tol=0.0,
                abs_tol=TOL,
            )
        ):
            raise CampaignError(f"final SlaClip K/bounds mismatch: {status_path}")
    expected_evaluation = {
        "evaluation_schema_version": 1,
        "dataset": "math10k",
        "base_model": registry["model_id"],
        "requested_model_revision": registry["model_revision"],
        "resolved_model_revision": registry["model_revision"],
        "config_fingerprint": config.get("config_fingerprint"),
        "tasks": ["gsm8k", "AQuA", "mawps", "SVAMP"],
        "batch_size": 8,
        "num_beams": 4,
        "max_new_tokens": 256,
        "max_input_length": 1024,
    }
    for key, expected in expected_evaluation.items():
        if evaluation_config.get(key) != expected:
            raise CampaignError(
                f"final evaluation config mismatch for {key}: {evaluation_config_path}"
            )
    test_assets = evaluation_config.get("test_assets")
    if not isinstance(test_assets, dict) or set(test_assets) != set(
        EXPECTED_TEST_ASSETS
    ):
        raise CampaignError(
            f"final evaluation test asset set mismatch: {evaluation_config_path}"
        )
    for task, (expected_rows, expected_sha256) in EXPECTED_TEST_ASSETS.items():
        asset = test_assets.get(task)
        if (
            not isinstance(asset, dict)
            or int(asset.get("rows", -1)) != expected_rows
            or asset.get("sha256") != expected_sha256
        ):
            raise CampaignError(
                f"final evaluation test asset mismatch for {task}: "
                f"{evaluation_config_path}"
            )
    try:
        with summary_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            expected_fields = ["gsm8k", "AQuA", "mawps", "SVAMP", "Average"]
            if reader.fieldnames != expected_fields:
                raise CampaignError(
                    f"final summary columns must be {expected_fields}: {summary_path}"
                )
            summary_rows = list(reader)
    except OSError as exc:
        raise CampaignError(f"cannot read final summary: {summary_path}: {exc}") from exc
    if len(summary_rows) != 1:
        raise CampaignError(f"final summary must contain exactly one row: {summary_path}")
    task_accuracy = {
        task: _finite(summary_rows[0].get(task), f"{task} accuracy")
        for task in ("gsm8k", "AQuA", "mawps", "SVAMP")
    }
    reported_average = _finite(summary_rows[0].get("Average"), "four-task Average")
    if any(not 0.0 <= value <= 1.0 for value in task_accuracy.values()):
        raise CampaignError(f"final task accuracy outside [0,1]: {summary_path}")
    four_task = statistics.fmean(task_accuracy.values())
    if not math.isclose(
        reported_average, four_task, rel_tol=0.0, abs_tol=1e-10
    ):
        raise CampaignError(f"final summary Average is inconsistent: {summary_path}")
    clean_three = statistics.fmean(
        task_accuracy[task] for task in ("gsm8k", "AQuA", "SVAMP")
    )

    raw = _load_jsonl(raw_path)
    if [int(row.get("step", -1)) for row in raw] != list(
        range(1, FINAL_STEPS + 1)
    ):
        raise CampaignError(f"final raw telemetry must cover steps 1..300: {raw_path}")
    if any(
        row.get("NON_PRIVATE_TELEMETRY") is not True
        or row.get("method") != candidate["method"]
        or row.get("privacy") != "dp"
        or row.get("dataset") != "math10k"
        or row.get("base_model") != registry["model_id"]
        or row.get("model_revision") != registry["model_revision"]
        or row.get("resolved_model_revision") != registry["model_revision"]
        or row.get("config_fingerprint") != config.get("config_fingerprint")
        or row.get("telemetry_mode") != "research_raw"
        for row in raw
    ):
        raise CampaignError(f"final raw telemetry identity mismatch: {raw_path}")
    post = [row for row in raw if int(row["step"]) > FINAL_BURN_IN_STEP]
    clips = [_finite(row.get("raw_clip_fraction"), "final raw clip") for row in post]
    thresholds = [_finite(row.get("dp_clip_threshold"), "final C_t") for row in post]
    if any(not 0.0 <= value <= 1.0 for value in clips) or any(
        value <= 0.0 for value in thresholds
    ):
        raise CampaignError(f"invalid final clipping telemetry: {raw_path}")
    trajectory: dict[str, Any] = {
        "NON_PRIVATE": True,
        "post_burn_in_steps": len(post),
        "raw_clip_mean": statistics.fmean(clips),
        "raw_clip_sd": statistics.stdev(clips),
        "raw_clip_min": min(clips),
        "raw_clip_max": max(clips),
        "clip_threshold_mean": statistics.fmean(thresholds),
        "clip_threshold_sd": statistics.stdev(thresholds),
        "clip_threshold_total_variation": math.fsum(
            abs(right - left) for left, right in zip(thresholds, thresholds[1:])
        ),
    }
    diagnostic_fields = {
        "clipping_bias_norm": "raw_clipping_bias_norm",
        "realized_noise_norm": "raw_realized_noise_norm",
        "signal_to_noise_ratio": "raw_signal_to_noise_ratio",
        "bias_noise_mse_proxy": "raw_bias_noise_squared_error_proxy",
    }
    for label, field in diagnostic_fields.items():
        values = [_finite(row.get(field), field) for row in post]
        trajectory[f"{label}_mean"] = statistics.fmean(values)
        trajectory[f"{label}_auc"] = math.fsum(values)
    if candidate["family"] == "slaclip":
        rho = _finite(
            params["slaclip_target_non_small_clip_fraction"], "final rho"
        )
        targets: list[float] = []
        errors: list[float] = []
        exact_z: list[float] = []
        exact_z_snr: list[float] = []
        formula_error: list[float] = []
        bound_hits = 0
        for row, clip, threshold in zip(post, clips, thresholds, strict=True):
            noisy_z = _finite(
                row.get("slaclip_small_gradient_proxy_noisy"), "final noisy z_t"
            )
            target = _finite(
                row.get("slaclip_target_clipped_proxy"), "final dynamic target"
            )
            expected_target = max(0.0, min(1.0, rho * (1.0 - noisy_z)))
            formula_error.append(abs(target - expected_target))
            targets.append(target)
            errors.append(clip - target)
            hit_min = bool(row.get("slaclip_c_hit_min"))
            hit_max = bool(row.get("slaclip_c_hit_max"))
            if hit_min and hit_max:
                raise CampaignError(f"final controller hit both C bounds: {raw_path}")
            bound_hits += int(hit_min or hit_max)
            exact = _finite(
                row.get("raw_reference_small_gradient_proxy"), "final exact z_t"
            )
            z_noise_sd = _finite(
                row.get("slack_indicator_noise_std"), "final Slack noise std"
            ) / threshold
            exact_z.append(exact)
            exact_z_snr.append(abs(exact) / z_noise_sd if z_noise_sd > 0 else 0.0)
        if max(formula_error) > 2e-6:
            raise CampaignError(f"final dynamic target formula mismatch: {raw_path}")
        trajectory.update(
            {
                "conditional_rho": rho,
                "dynamic_target_mean": statistics.fmean(targets),
                "dynamic_target_sd": statistics.stdev(targets),
                "dynamic_target_formula_max_abs_error": max(formula_error),
                "tracking_mae": statistics.fmean(abs(value) for value in errors),
                "tracking_rmse": math.sqrt(
                    statistics.fmean(value * value for value in errors)
                ),
                "tracking_bias": statistics.fmean(errors),
                "tracking_time_within_0p05": statistics.fmean(
                    abs(value) <= TRACKING_GATE_MAE for value in errors
                ),
                "controller_bound_hit_rate": bound_hits / len(post),
                "exact_z_mean": statistics.fmean(exact_z),
                "exact_z_max": max(exact_z),
                "exact_z_nonzero_rate": statistics.fmean(
                    abs(value) > TOL for value in exact_z
                ),
                "exact_z_snr_mean": statistics.fmean(exact_z_snr),
                "exact_z_snr_gt_1_rate": statistics.fmean(
                    value > 1.0 for value in exact_z_snr
                ),
            }
        )
    return {
        "seed": int(record["seed"]),
        "role": record["role"],
        "candidate_id": candidate["id"],
        "method": candidate["method"],
        "clean_three_task_macro_accuracy": clean_three,
        "raw_four_task_macro_accuracy": four_task,
        "task_accuracy": task_accuracy,
        "trajectory": trajectory,
        "summary_sha256": _sha256(summary_path.read_bytes()),
        "raw_telemetry_sha256": _sha256(raw_path.read_bytes()),
        "test_assets_sha256": _sha256(_json_bytes(test_assets)),
        "privacy_calibration": privacy_calibration,
    }


def _paired_summary(deltas: Sequence[float]) -> dict[str, Any]:
    if len(deltas) != len(CONFIRMATION_SEEDS):
        raise CampaignError("paired analysis requires exactly five confirmation seeds")
    mean = statistics.fmean(deltas)
    sd = statistics.stdev(deltas)
    margin = T_95_DF4 * sd / math.sqrt(len(deltas))
    return {
        "n": len(deltas),
        "mean_delta": mean,
        "sample_sd_delta": sd,
        "paired_t_95_ci": [mean - margin, mean + margin],
        "strict_slaclip_wins": sum(value > TOL for value in deltas),
        "ties": sum(abs(value) <= TOL for value in deltas),
        "strict_fixed_wins": sum(value < -TOL for value in deltas),
    }


def _aggregate_trajectories(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if len(rows) != len(CONFIRMATION_SEEDS):
        raise CampaignError("trajectory aggregation requires five runs")
    trajectories = [row["trajectory"] for row in rows]
    numeric_keys = sorted(
        set.intersection(
            *(
                {
                    key
                    for key, value in trajectory.items()
                    if isinstance(value, (int, float)) and not isinstance(value, bool)
                }
                for trajectory in trajectories
            )
        )
    )
    result: dict[str, Any] = {}
    for key in numeric_keys:
        values = [_finite(trajectory[key], f"trajectory {key}") for trajectory in trajectories]
        result[key] = {
            "across_seed_mean": statistics.fmean(values),
            "across_seed_sample_sd": statistics.stdev(values),
        }
    return result


def analyze(
    *, campaign_root: Path, output_dir: Path = Path("artifacts")
) -> dict[str, Any]:
    campaign_root = campaign_root.resolve(strict=True)
    registry_path = campaign_root / "screen" / "candidate_registry.json"
    selection_path = campaign_root / "selection" / "selection.json"
    plan_path = campaign_root / "plans" / "final.tsv"
    for path in (registry_path, selection_path, plan_path):
        _verify_sha_sidecar(path)
    registry = _read_json(registry_path, "candidate registry")
    selection = _read_json(selection_path, "locked selection")
    if selection.get("test_assets_accessed") is not False:
        raise CampaignError("selection does not prove test assets were sealed")
    candidates = _candidate_map(registry)
    plan = _parse_plan(plan_path, campaign_root)
    core = [record for record in plan if record["phase"] == "final"]
    if len(plan) != EXPECTED_FINAL_CORE_ARMS or len(core) != EXPECTED_FINAL_CORE_ARMS:
        raise CampaignError(
            f"final plan must contain exactly {EXPECTED_FINAL_CORE_ARMS} final arms"
        )
    expected_roles = {
        "slaclip-rho-0p8": selection["selected_slaclip"]["0.8"]["candidate_id"],
        "slaclip-rho-0p9": selection["selected_slaclip"]["0.9"]["candidate_id"],
        "predeclared-strongest-fixed-c2": selection[
            "predeclared_strongest_fixed_c2"
        ]["candidate_id"],
    }
    expected_pairs = {
        (seed, role) for seed in CONFIRMATION_SEEDS for role in expected_roles
    }
    if {(record["seed"], record["role"]) for record in core} != expected_pairs:
        raise CampaignError("final plan seed/role coverage is not the locked 5x3 design")
    results: list[dict[str, Any]] = []
    for record in core:
        expected_id = expected_roles[record["role"]]
        if (
            record["kind"] != "train"
            or record["candidate_id"] != expected_id
            or record["schedule"] != "NA"
        ):
            raise CampaignError("final plan candidate identity is inconsistent")
        results.append(
            _validate_final_result(
                record=record,
                candidate=candidates[expected_id],
                registry=registry,
            )
        )
    if len({row["test_assets_sha256"] for row in results}) != 1:
        raise CampaignError("final arms did not use one identical test asset set")
    reference_calibration = results[0]["privacy_calibration"]
    for row in results[1:]:
        calibration = row["privacy_calibration"]
        if calibration.get("scope") != reference_calibration.get("scope") or any(
            not math.isclose(
                _finite(calibration.get(key), f"final {key}"),
                _finite(reference_calibration.get(key), f"reference final {key}"),
                rel_tol=0.0,
                abs_tol=1e-9,
            )
            for key in (
                "epsilon_spent",
                "noise_multiplier",
                "sample_rate",
                "expected_batch_size",
            )
        ):
            raise CampaignError("final arms used different privacy calibration")
    by_key = {(row["seed"], row["role"]): row for row in results}
    pair_rows: list[dict[str, Any]] = []
    comparisons: dict[str, Any] = {}
    fixed_values = [
        by_key[(seed, "predeclared-strongest-fixed-c2")][
            "clean_three_task_macro_accuracy"
        ]
        for seed in CONFIRMATION_SEEDS
    ]
    fixed_rows = [
        by_key[(seed, "predeclared-strongest-fixed-c2")]
        for seed in CONFIRMATION_SEEDS
    ]
    for rho, role in (("0.8", "slaclip-rho-0p8"), ("0.9", "slaclip-rho-0p9")):
        sla_rows = [by_key[(seed, role)] for seed in CONFIRMATION_SEEDS]
        sla_values = [
            row["clean_three_task_macro_accuracy"] for row in sla_rows
        ]
        deltas = [sla - fixed for sla, fixed in zip(sla_values, fixed_values, strict=True)]
        for seed, sla, fixed, delta in zip(
            CONFIRMATION_SEEDS, sla_values, fixed_values, deltas, strict=True
        ):
            pair_rows.append(
                {
                    "rho": rho,
                    "seed": seed,
                    "slaclip_accuracy": sla,
                    "fixed_c2_accuracy": fixed,
                    "delta_slaclip_minus_fixed": delta,
                }
            )
        paired = _paired_summary(deltas)
        is_primary = rho == "0.9"
        slaclip_trajectory = _aggregate_trajectories(sla_rows)
        cdf_signal_informative = (
            slaclip_trajectory["exact_z_nonzero_rate"]["across_seed_mean"]
            >= CDF_INFORMATIVE_RATE_MIN
            and slaclip_trajectory["exact_z_snr_gt_1_rate"]["across_seed_mean"]
            >= CDF_INFORMATIVE_RATE_MIN
        )
        comparisons[rho] = {
            "analysis_role": (
                "primary_locked_internal" if is_primary else "secondary_exploratory"
            ),
            "slaclip_candidate_id": expected_roles[role],
            "fixed_candidate_id": expected_roles["predeclared-strongest-fixed-c2"],
            "slaclip_mean_accuracy": statistics.fmean(sla_values),
            "slaclip_sample_sd_accuracy": statistics.stdev(sla_values),
            "fixed_c2_mean_accuracy": statistics.fmean(fixed_values),
            "fixed_c2_sample_sd_accuracy": statistics.stdev(fixed_values),
            "stability_definition": {
                "target_tracking_stability": (
                    "within-run realized clipping versus dynamic p_star_t after "
                    "the predeclared 100-step burn-in"
                ),
                "accuracy_variability": (
                    "descriptive across-seed sample standard deviation of endpoint "
                    "clean-three accuracy; it is not an inferential stability test"
                ),
            },
            "slaclip_trajectory_stability": slaclip_trajectory,
            "fixed_c2_trajectory_stability": _aggregate_trajectories(fixed_rows),
            "mechanism_attribution": {
                "cdf_signal_informative": cdf_signal_informative,
                "descriptive_rule": (
                    "both the across-seed mean exact-z nonzero rate and the "
                    "exact-z SNR>1 rate must be at least 0.05"
                ),
                "claim_guard": (
                    "If false, an accuracy result can support the adaptive target "
                    "controller comparison but cannot be attributed to informative "
                    "Slack/CDF correction. This flag does not alter the accuracy test."
                ),
            },
            **paired,
            "accuracy_superiority_supported": (
                paired["paired_t_95_ci"][0] > 0.0 if is_primary else None
            ),
            "inference_rule": (
                "For predeclared primary rho=0.9 only, superiority requires the "
                "lower bound of the two-sided paired-t 95% CI for SlaClip minus "
                "established fixed C=2 accuracy to be greater than zero. rho=0.8 "
                "is exploratory regardless of its point estimate."
            ),
        }
    payload = {
        "schema_version": SCHEMA_VERSION,
        "protocol": registry["protocol"],
        "primary_metric": "clean_three_task_macro_accuracy",
        "confirmation_seeds": list(CONFIRMATION_SEEDS),
        "selection_sha256": _sha256(selection_path.read_bytes()),
        "final_plan_sha256": _sha256(plan_path.read_bytes()),
        "common_final_privacy_calibration": reference_calibration,
        "comparisons": comparisons,
        "per_seed_pairs": pair_rows,
        "per_run_results": results,
        "interpretation_guard": (
            "A positive point estimate alone is not evidence of superiority; "
            "report the paired confidence interval and all five seed outcomes. "
            "Historical benchmark test results were available before this protocol, "
            "so even a positive primary result requires external-data validation or "
            "an independent replication for a journal-level generalization claim. "
            "CDF/Slack attribution additionally requires the reported exact-z signal "
            "informativeness flag; accuracy superiority alone is not mechanism evidence."
        ),
        "historical_test_results_available_before_protocol": True,
        "inference_class": (
            "locked_internal_paired_confirmation_not_untouched_external_replication"
        ),
    }
    output_root = _plan_path_inside(
        campaign_root, str(output_dir), "analysis output directory"
    )
    analysis_path = output_root / "final_paired_results.json"
    _write_with_sha(analysis_path, _json_bytes(payload))
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=tuple(pair_rows[0]))
    writer.writeheader()
    writer.writerows(pair_rows)
    csv_path = output_root / "final_paired_results.csv"
    _write_with_sha(csv_path, buffer.getvalue().encode("utf-8"))
    return {
        "analysis": str(analysis_path),
        "paired_csv": str(csv_path),
        "comparisons": comparisons,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare_parser = sub.add_parser("prepare")
    prepare_parser.add_argument("--campaign-root", type=Path, required=True)
    prepare_parser.add_argument("--code-sha", required=True)
    prepare_parser.add_argument("--model-id", required=True)
    prepare_parser.add_argument("--model-revision", required=True)
    lock_parser = sub.add_parser("lock")
    lock_parser.add_argument("--campaign-root", type=Path, required=True)
    analyze_parser = sub.add_parser("analyze")
    analyze_parser.add_argument("--campaign-root", type=Path, required=True)
    analyze_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts"),
        help="Output directory relative to campaign root (default: artifacts).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "prepare":
        result = prepare(
            campaign_root=args.campaign_root,
            code_sha=args.code_sha,
            model_id=args.model_id,
            model_revision=args.model_revision,
        )
    elif args.command == "lock":
        result = lock(campaign_root=args.campaign_root)
    else:
        result = analyze(
            campaign_root=args.campaign_root,
            output_dir=args.output_dir,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
