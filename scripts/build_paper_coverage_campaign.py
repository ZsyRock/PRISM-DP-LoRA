#!/usr/bin/env python3
"""Build and summarize PRISM paper-coverage and clipping-regime screens.

These profiles are exploratory screens rather than confirmatory experiments.
``paper-breadth`` preserves the historical 15-arm plan; ``regime-map`` crosses
the paper's available dataset, privacy, rank, and 4B/9B axes; and
``glue-slaclip-screen`` focuses on the first baseline setting whose fixed-C
trajectory exhibited non-saturated clipping and measurable slack.  All arms
run inside one immutable one- or two-lane Slurm allocation, and measured
clipping strata are reported without pretending they are pre-established
SlaClip failure thresholds.
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
        "role": "tuned_fixed_candidate",
    }
    for value in (0.5, 1.0, 2.0, 3.0, 5.0)
) + tuple(
    {
        "id": f"full-sla-rho{int(value * 100):03d}",
        "method": "slaclip",
        "initial_c": 1.0,
        "rho": value,
        "eta": 0.05,
        "role": "slaclip_target_candidate",
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

# Job 1402286 established that the paper-default GLUE8/4B/eps=6/r=16 fixed-C
# trajectory whose steps 51--500 have whole-batch clipping
# q10/median/q90 = 0.418818/0.500000/0.584725 and median small-gradient proxy
# z=0.213266.  The following compact conditional-rho grid was frozen before
# launching any adaptive arm: the rho grid brackets the directly observed
# conditional-clipping q10--q90 interval, while rho*(1-z) records the implied
# approximate whole-batch targets at median z. The plan retains a tuned
# fixed-C comparator and adds
# two C0 controls at the central target.  It deliberately uses seed 43 so the
# seed-42 calibration trajectory is not reused for candidate screening.
GLUE_SLACLIP_SOURCE = {
    "campaign_id": "paper-coverage-8495ac8f0c07-baseline-reproduction-cached-v2",
    "relative_arm_root": "runs/glue8-4b-eps6-r16/fixed-c1-paper-default/seed-42",
    "job_id": "1402286",
    "code_sha": "8495ac8f0c07addf910d8f3a2e7eec6a88884a92",
    "run_id": "paper-coverage-glue8-4b-eps6-r16-baseline-seed42_C1_689312340b",
    "config_fingerprint": "689312340b135904e37ecd372a71955c6e617cdd95636755ac33003d15aab524",
    "data_sha256": "281bad3305e8be7a7660b64b22704a9462f983b90df5d6f0c1c4abb649d6d091",
    "model_id": MODEL_4B,
    "model_revision": "cc012e0a6d0787b4adcc0fa2c4da74402494554d",
    "raw_telemetry_sha256": "ebe0085350f0d3dc4b7cbe90cbc18dd3a9179056cc9e6f899fe99785260312e8",
    "telemetry_summary_sha256": "51a37be6392e9085a10482c8eceae41535e6318ce8c7133b1c0dfffae2c33d36",
    "raw_records": 500,
    "burn_in_rule": "exclude steps 1 through 50; summarize steps 51 through 500",
    "post_burn_in_records": 450,
    "post_burn_in_whole_batch_clip_fraction_q10": 0.4188180718031464,
    "post_burn_in_whole_batch_clip_fraction_median": 0.5,
    "post_burn_in_whole_batch_clip_fraction_q90": 0.5847252747252748,
    "post_burn_in_small_gradient_proxy_median": 0.2132660700076269,
    "post_burn_in_conditional_clip_fraction_q10": 0.5524977719630925,
    "post_burn_in_conditional_clip_fraction_median": 0.6401230104328237,
    "post_burn_in_conditional_clip_fraction_q90": 0.7290751684816961,
    "rho_to_global_target_at_median_z": {
        "0.55": 0.43270366149580525,
        "0.60": 0.4720403579954239,
        "0.65": 0.5113770544950426,
        "0.70": 0.5507137509946611,
        "0.75": 0.5900504474942798,
    },
}
GLUE_SLACLIP_SCREEN_SEED = 43
GLUE_SLACLIP_FIXED_GRID = (0.5, 1.0, 2.0, 3.0, 5.0)
GLUE_SLACLIP_RHO_GRID = (0.55, 0.60, 0.65, 0.70, 0.75)
GLUE_SLACLIP_VALIDATION_SEED = 1729
GLUE_SLACLIP_VALIDATION_ROWS = 800
GLUE_SLACLIP_VALIDATION_INDICES_SHA256 = (
    "34a59e5cf4d98300f3d484d9c82b37172b850939bc19e002c22428be22a12805"
)
GLUE_SLACLIP_VALIDATION_RECORDS_SHA256 = (
    "43f7a3d0db422b2331a59d3611e777faf99d0bf434a405ef98a8ea7b1c582fad"
)


def _candidate_id_value(value: float) -> str:
    return str(value).replace(".", "p")


_glue_slaclip_candidates = [
    {
        "id": f"fixed-c{_candidate_id_value(value)}",
        "method": "baseline",
        "initial_c": value,
        "rho": None,
        "eta": None,
        "role": "tuned_fixed_candidate",
    }
    for value in GLUE_SLACLIP_FIXED_GRID
]
_glue_slaclip_candidates.extend(
    {
        "id": f"full-sla-c1-rho{int(value * 100):03d}",
        "method": "slaclip",
        "initial_c": 1.0,
        "rho": value,
        "eta": 0.05,
        "role": "slaclip_target_candidate",
    }
    for value in GLUE_SLACLIP_RHO_GRID
)
_glue_slaclip_candidates.extend(
    {
        "id": f"full-sla-c{_candidate_id_value(initial_c)}-rho065",
        "method": "slaclip",
        "initial_c": initial_c,
        "rho": 0.65,
        "eta": 0.05,
        "role": "initial_C_sensitivity_control",
    }
    for initial_c in (0.5, 2.0)
)
GLUE_SLACLIP_CANDIDATES = tuple(
    {**candidate, "lane": index % 2}
    for index, candidate in enumerate(_glue_slaclip_candidates)
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


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise CampaignError(f"cannot hash required artifact: {path}") from exc
    return digest.hexdigest()


def _verify_glue_slaclip_source(campaign_root: Path) -> None:
    """Fail closed unless the calibration artifact exactly matches its lock."""

    source_root = (
        campaign_root.parent
        / GLUE_SLACLIP_SOURCE["campaign_id"]
        / GLUE_SLACLIP_SOURCE["relative_arm_root"]
    )
    raw_path = source_root / "results" / "research_raw" / "NON_PRIVATE_train_log.jsonl"
    summary_path = source_root / "results" / "research_raw" / "telemetry_summary.json"
    status_path = source_root / "adapter" / "run_status.json"
    expected_hashes = {
        raw_path: GLUE_SLACLIP_SOURCE["raw_telemetry_sha256"],
        summary_path: GLUE_SLACLIP_SOURCE["telemetry_summary_sha256"],
    }
    for path, expected in expected_hashes.items():
        actual = _file_sha256(path)
        if actual != expected:
            raise CampaignError(
                f"calibration artifact hash mismatch: {path}; expected={expected}, actual={actual}"
            )
    try:
        status = json.loads(status_path.read_text(encoding="utf-8"))
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CampaignError("cannot read locked GLUE SlaClip calibration provenance") from exc
    expected_status = {
        "state": "completed",
        "run_id": GLUE_SLACLIP_SOURCE["run_id"],
        "config_fingerprint": GLUE_SLACLIP_SOURCE["config_fingerprint"],
        "data_content_sha256": GLUE_SLACLIP_SOURCE["data_sha256"],
        "base_model": GLUE_SLACLIP_SOURCE["model_id"],
        "model_revision": GLUE_SLACLIP_SOURCE["model_revision"],
        "resolved_model_revision": GLUE_SLACLIP_SOURCE["model_revision"],
        "method": "baseline",
        "privacy": "dp",
        "update_steps": GLUE_SLACLIP_SOURCE["raw_records"],
    }
    for key, expected in expected_status.items():
        if status.get(key) != expected:
            raise CampaignError(
                f"calibration status identity mismatch for {key}: "
                f"expected={expected!r}, actual={status.get(key)!r}"
            )
    expected_config = {
        "implementation_git_sha": GLUE_SLACLIP_SOURCE["code_sha"],
        "implementation_git_dirty": False,
        "dataset": "glue8",
        "method": "baseline",
        "privacy": "dp",
        "seed": 42,
        "total_update_steps": GLUE_SLACLIP_SOURCE["raw_records"],
        "dp_max_grad_norm": 1.0,
        "lora_r": 16,
        "dp_epsilon": 6.0,
        "protocol_stage": "final",
        "val_set_size": 0,
        "base_model": GLUE_SLACLIP_SOURCE["model_id"],
        "model_revision": GLUE_SLACLIP_SOURCE["model_revision"],
    }
    config = status.get("config")
    if not isinstance(config, dict):
        raise CampaignError("calibration status lacks its config identity")
    for key, expected in expected_config.items():
        if config.get(key) != expected:
            raise CampaignError(
                f"calibration config mismatch for {key}: "
                f"expected={expected!r}, actual={config.get(key)!r}"
            )
    if (
        summary.get("summary_schema_version") != 4
        or summary.get("NON_PRIVATE_TELEMETRY") is not True
        or summary.get("source", {}).get("raw_sha256")
        != GLUE_SLACLIP_SOURCE["raw_telemetry_sha256"]
    ):
        raise CampaignError("calibration telemetry summary is not bound to the locked raw log")

    records: dict[int, dict[str, Any]] = {}
    try:
        with raw_path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                record = json.loads(line)
                step = int(record.get("step", -1))
                if step in records:
                    raise CampaignError(f"duplicate calibration step {step}")
                for key, expected in (
                    ("NON_PRIVATE_TELEMETRY", True),
                    ("run_id", GLUE_SLACLIP_SOURCE["run_id"]),
                    ("config_fingerprint", GLUE_SLACLIP_SOURCE["config_fingerprint"]),
                    ("method", "baseline"),
                    ("privacy", "dp"),
                    ("dataset", "glue8"),
                    ("base_model", GLUE_SLACLIP_SOURCE["model_id"]),
                    ("model_revision", GLUE_SLACLIP_SOURCE["model_revision"]),
                ):
                    if record.get(key) != expected:
                        raise CampaignError(
                            f"calibration raw identity mismatch at line {line_number}: {key}"
                        )
                records[step] = record
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise CampaignError("cannot parse locked calibration telemetry") from exc
    expected_steps = set(range(1, int(GLUE_SLACLIP_SOURCE["raw_records"]) + 1))
    if set(records) != expected_steps:
        raise CampaignError("calibration telemetry does not contain exactly steps 1 through 500")
    post = [records[step] for step in range(51, 501)]
    if len(post) != int(GLUE_SLACLIP_SOURCE["post_burn_in_records"]):
        raise CampaignError("calibration burn-in slice has the wrong record count")
    try:
        clip_values = [float(record["raw_clip_fraction"]) for record in post]
        small_values = [
            float(record["raw_reference_small_gradient_proxy"]) for record in post
        ]
        conditional_values = [
            float(record["raw_reference_conditional_clip_fraction"])
            for record in post
        ]
    except (KeyError, TypeError, ValueError) as exc:
        raise CampaignError("calibration telemetry lacks a finite target statistic") from exc
    if not all(
        math.isfinite(value)
        for values in (clip_values, small_values, conditional_values)
        for value in values
    ):
        raise CampaignError("calibration target statistics are not finite")
    observed = {
        "post_burn_in_whole_batch_clip_fraction_q10": _quantile(clip_values, 0.1),
        "post_burn_in_whole_batch_clip_fraction_median": _quantile(clip_values, 0.5),
        "post_burn_in_whole_batch_clip_fraction_q90": _quantile(clip_values, 0.9),
        "post_burn_in_small_gradient_proxy_median": _quantile(small_values, 0.5),
        "post_burn_in_conditional_clip_fraction_q10": _quantile(
            conditional_values, 0.1
        ),
        "post_burn_in_conditional_clip_fraction_median": _quantile(
            conditional_values, 0.5
        ),
        "post_burn_in_conditional_clip_fraction_q90": _quantile(
            conditional_values, 0.9
        ),
    }
    for key, actual in observed.items():
        expected = float(GLUE_SLACLIP_SOURCE[key])
        if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12):
            raise CampaignError(
                f"calibration statistic mismatch for {key}: expected={expected}, actual={actual}"
            )
    median_z = observed["post_burn_in_small_gradient_proxy_median"]
    for rho_text, expected_target in GLUE_SLACLIP_SOURCE[
        "rho_to_global_target_at_median_z"
    ].items():
        actual_target = float(rho_text) * (1.0 - median_z)
        if not math.isclose(
            actual_target, float(expected_target), rel_tol=0.0, abs_tol=1e-12
        ):
            raise CampaignError(
                f"calibration rho mapping mismatch for rho={rho_text}: "
                f"expected={expected_target}, actual={actual_target}"
            )


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
    screen_seed = SEED
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
    elif profile == "glue-slaclip-screen":
        settings = (BREADTH_SETTINGS[0],)
        candidates = GLUE_SLACLIP_CANDIDATES
        screen_steps = 150
        eval_limit = 0
        screen_seed = GLUE_SLACLIP_SCREEN_SEED
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
            arm_id = f"{setting['id']}--{candidate['id']}--seed{screen_seed}"
            arms.append(
                {
                    **setting,
                    **candidate,
                    "setting_id": setting["id"],
                    "candidate_id": candidate["id"],
                    "arm_id": arm_id,
                    "seed": screen_seed,
                    "model_revision": revisions[setting["model_id"]],
                    "steps": screen_steps or setting["steps"],
                    "eval_limit": eval_limit,
                    "relative_root": f"runs/{setting['id']}/{candidate['id']}/seed-{screen_seed}",
                }
            )
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol": f"prism_paper_coverage_{profile.replace('-', '_')}_v2",
        "profile": profile,
        "inference_class": "single_seed_exploratory_breadth_screen_requires_fresh_seed_confirmation",
        "code_sha": code_sha,
        "seed": screen_seed,
        "privacy": {"delta": 1e-5, "accountant": "prv", "secure_mode": False},
        "full_slaclip": {
            "K": 15,
            "C_min": 0.1,
            "C_max": 15.0,
            "target_semantics": "p_star_t=rho*(1-z_t); rho is conditional on residual non-small mass",
        },
        "selection_warning": (
            "The focused GLUE screen uses only a fixed public training holdout and does "
            "not run official task evaluation. Any promising setting must be repeated "
            "at full length on fresh seeds with a locked tuned-fixed comparator."
            if profile == "glue-slaclip-screen"
            else "Task-test metrics are descriptive only. Any promising setting must be "
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
            "exploratory": profile in {"regime-map", "glue-slaclip-screen"},
            "screen_steps": screen_steps,
            "per_task_eval_limit": eval_limit,
            "fixed_C_grid": (
                list(GLUE_SLACLIP_FIXED_GRID)
                if profile == "glue-slaclip-screen"
                else [0.5, 1.0, 2.0, 3.0, 5.0]
                if profile == "regime-map"
                else [1.0]
            ),
            "conditional_rho_grid": (
                list(GLUE_SLACLIP_RHO_GRID)
                if profile == "glue-slaclip-screen"
                else [0.5, 0.7, 0.8, 0.9, 0.98]
                if profile == "regime-map"
                else [0.9, 0.98]
            ),
            "initial_C_sensitivity_control": (
                {"C_0": [0.5, 2.0], "rho": 0.65, "eta": 0.05}
                if profile == "glue-slaclip-screen"
                else {"C_0": 2.0, "rho": 0.9, "eta": 0.05}
                if profile == "regime-map"
                else None
            ),
            "interpretation": (
                "descriptive one-seed screen; clipping-rate bins are measured outcomes, "
                "not predeclared failure thresholds or confirmatory evidence"
            ),
        },
        "glue_slaclip_screen": {
            "enabled": profile == "glue-slaclip-screen",
            "baseline_source": (
                GLUE_SLACLIP_SOURCE if profile == "glue-slaclip-screen" else None
            ),
            "target_derivation": (
                "rho grid brackets the post-burn-in fixed-C1 conditional-clipping q10-q90 interval"
                if profile == "glue-slaclip-screen" else None
            ),
            "fixed_C_grid": (
                list(GLUE_SLACLIP_FIXED_GRID)
                if profile == "glue-slaclip-screen" else None
            ),
            "conditional_rho_grid": (
                list(GLUE_SLACLIP_RHO_GRID)
                if profile == "glue-slaclip-screen" else None
            ),
            "eta": 0.05 if profile == "glue-slaclip-screen" else None,
            "K": 15 if profile == "glue-slaclip-screen" else None,
            "C_bounds": [0.1, 15.0] if profile == "glue-slaclip-screen" else None,
            "initial_C_sensitivity": (
                {"rho": 0.65, "C_0": [0.5, 1.0, 2.0]}
                if profile == "glue-slaclip-screen" else None
            ),
            "selection": (
                {
                    "seed": GLUE_SLACLIP_SCREEN_SEED,
                    "public_holdout_rows": GLUE_SLACLIP_VALIDATION_ROWS,
                    "public_holdout_stratification": "100 rows per GLUE8 task",
                    "public_holdout_seed": GLUE_SLACLIP_VALIDATION_SEED,
                    "public_holdout_indices_sha256": (
                        GLUE_SLACLIP_VALIDATION_INDICES_SHA256
                    ),
                    "public_holdout_records_sha256": (
                        GLUE_SLACLIP_VALIDATION_RECORDS_SHA256
                    ),
                    "metric": "response_only_mean_per_record_causal_lm_loss",
                    "official_task_evaluation": False,
                }
                if profile == "glue-slaclip-screen" else None
            ),
            "privacy_scope": (
                "NON_PRIVATE exploratory calibration; the seed-42 source trained on all 10k rows, "
                "including rows later assigned to the public selection holdout"
                if profile == "glue-slaclip-screen" else None
            ),
            "calibration_selection_overlap": (
                True if profile == "glue-slaclip-screen" else None
            ),
            "confirmation_requirement": (
                "lock one Full-SlaClip and one tuned-fixed candidate, then rerun 500 steps on fresh seeds"
                if profile == "glue-slaclip-screen" else None
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
        declaration_order = {
            arm["arm_id"]: index for index, arm in enumerate(manifest["arms"])
        }
        arms = sorted(
            arms,
            key=lambda arm: (
                priority.get(arm["setting_id"], len(priority)),
                arm.get("role") == "initial_C_sensitivity_control",
                declaration_order[arm["arm_id"]],
            ),
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
    if profile == "glue-slaclip-screen":
        _verify_glue_slaclip_source(root)
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


ANALYSIS_REQUIRED_METRICS = (
    "loss_mean",
    "raw_clip_fraction",
    "dp_clip_threshold",
    "raw_signal_to_noise_ratio",
    "raw_clipping_bias_to_noise_ratio",
    "raw_signal_retention_ratio",
    "raw_clipping_bias_norm",
    "raw_realized_noise_norm",
    "raw_bias_noise_squared_error_proxy",
)


def _validate_arm_telemetry(
    arm: dict[str, Any],
    status: dict[str, Any],
    telemetry: dict[str, Any],
    raw_path: Path,
    expected_code_sha: str,
) -> dict[int, dict[str, Any]]:
    """Bind status, raw telemetry, and its aggregate to one manifest arm."""

    expected_steps = int(arm["steps"])
    expected_top = {
        "state": "completed",
        "update_steps": expected_steps,
        "dataset": arm["dataset"],
        "method": arm["method"],
        "privacy": "dp",
        "base_model": arm["model_id"],
        "model_revision": arm["model_revision"],
        "resolved_model_revision": arm["model_revision"],
        "data_content_sha256": GLUE_SLACLIP_SOURCE["data_sha256"],
        "telemetry_mode": "research_raw",
        "non_private_telemetry": True,
    }
    for key, expected in expected_top.items():
        if status.get(key) != expected:
            raise CampaignError(
                f"arm status mismatch for {arm['arm_id']}:{key}; "
                f"expected={expected!r}, actual={status.get(key)!r}"
            )
    run_id = status.get("run_id")
    fingerprint = status.get("config_fingerprint")
    if not isinstance(run_id, str) or not run_id:
        raise CampaignError(f"arm status lacks run_id: {arm['arm_id']}")
    if not isinstance(fingerprint, str) or len(fingerprint) != 64:
        raise CampaignError(f"arm status lacks config fingerprint: {arm['arm_id']}")
    config = status.get("config")
    if not isinstance(config, dict):
        raise CampaignError(f"arm status lacks config: {arm['arm_id']}")
    expected_config = {
        "implementation_git_sha": expected_code_sha,
        "implementation_git_dirty": False,
        "dataset": arm["dataset"],
        "method": arm["method"],
        "privacy": "dp",
        "base_model": arm["model_id"],
        "model_revision": arm["model_revision"],
        "seed": arm["seed"],
        "lora_r": arm["lora_r"],
        "total_update_steps": expected_steps,
        "batch_size": 64,
        "micro_batch_size": 4,
        "learning_rate": arm["learning_rate"],
        "cutoff_len": arm["cutoff_len"],
        "train_on_inputs": arm["train_on_inputs"],
        "val_set_size": GLUE_SLACLIP_VALIDATION_ROWS,
        "validation_seed": GLUE_SLACLIP_VALIDATION_SEED,
        "validation_eval_interval": 50,
        "protocol_stage": "selection",
        "validation_data_is_public": True,
        "dp_epsilon": arm["epsilon"],
        "dp_delta": 1e-5,
        "dp_max_grad_norm": arm["initial_c"],
        "dp_accountant": "prv",
        "telemetry_mode": "research_raw",
        "allow_non_private_telemetry": True,
        "slaclip_num_slots": 15,
        "run_train": True,
        "run_eval": False,
        "resume": True,
        "checkpoint_every": 25,
    }
    if arm["method"] == "slaclip":
        expected_config.update({
            "slaclip_target_non_small_clip_fraction": arm["rho"],
            "slaclip_eta": arm["eta"],
            "slaclip_c_min": 0.1,
            "slaclip_c_max": 15.0,
        })
    else:
        # The paper config retains its inactive rho=0.5 default for fixed-C
        # runs. It is identity metadata only; method=baseline never constructs
        # or applies the SlaClip controller.
        expected_config["slaclip_target_non_small_clip_fraction"] = 0.5
    for key, expected in expected_config.items():
        if config.get(key) != expected:
            raise CampaignError(
                f"arm config mismatch for {arm['arm_id']}:{key}; "
                f"expected={expected!r}, actual={config.get(key)!r}"
            )

    split = status.get("data_split")
    validation = status.get("validation")
    if not isinstance(split, dict) or not isinstance(validation, dict):
        raise CampaignError(f"arm lacks locked selection split: {arm['arm_id']}")
    for payload_name, payload in (("data_split", split), ("validation", validation)):
        if (
            payload.get("protocol_stage") != "selection"
            or payload.get("validation_data_is_public") is not True
            or payload.get("validation_rows") != GLUE_SLACLIP_VALIDATION_ROWS
            or payload.get("seed") != GLUE_SLACLIP_VALIDATION_SEED
            or payload.get("validation_indices_sha256")
            != GLUE_SLACLIP_VALIDATION_INDICES_SHA256
            or payload.get("validation_record_hashes_sha256")
            != GLUE_SLACLIP_VALIDATION_RECORDS_SHA256
        ):
            raise CampaignError(
                f"invalid {payload_name} identity for focused arm: {arm['arm_id']}"
            )
    if (
        validation.get("PUBLIC_VALIDATION_DATA") is not True
        or validation.get("NON_PRIVATE_SELECTION_METRIC") is not True
        or validation.get("records") != GLUE_SLACLIP_VALIDATION_ROWS
        or validation.get("selection_metric")
        != "response_only_mean_per_record_causal_lm_loss"
        or validation.get("loss_definition")
        != "response_only_per_record_mean_of_nonignored_next_token_losses"
        or validation.get("manifest_sha256") != split.get("manifest_sha256")
    ):
        raise CampaignError(f"invalid endpoint validation payload: {arm['arm_id']}")

    raw_sha = _file_sha256(raw_path)
    source = telemetry.get("source")
    steps = telemetry.get("steps")
    identity = telemetry.get("run_identity")
    if (
        telemetry.get("summary_schema_version") != 4
        or telemetry.get("NON_PRIVATE_TELEMETRY") is not True
        or not isinstance(source, dict)
        or source.get("raw_sha256") != raw_sha
        or source.get("raw_physical_records") != expected_steps
        or source.get("raw_unique_steps") != expected_steps
        or source.get("raw_duplicate_records") != 0
        or not isinstance(steps, dict)
        or steps.get("count") != expected_steps
        or steps.get("first") != 1
        or steps.get("last") != expected_steps
        or steps.get("missing_count") != 0
        or steps.get("missing") != []
    ):
        raise CampaignError(f"stale or incomplete telemetry summary: {arm['arm_id']}")
    expected_identity = {
        "run_id": run_id,
        "config_fingerprint": fingerprint,
        "method": arm["method"],
        "privacy": "dp",
        "dataset": arm["dataset"],
        "base_model": arm["model_id"],
        "model_revision": arm["model_revision"],
        "resolved_model_revision": arm["model_revision"],
    }
    if not isinstance(identity, dict):
        raise CampaignError(f"telemetry summary lacks run identity: {arm['arm_id']}")
    for key, expected in expected_identity.items():
        if identity.get(key) != expected:
            raise CampaignError(f"telemetry identity mismatch for {arm['arm_id']}:{key}")
    numeric = telemetry.get("metrics")
    if not isinstance(numeric, dict):
        raise CampaignError(f"telemetry summary lacks metrics: {arm['arm_id']}")
    for name in ANALYSIS_REQUIRED_METRICS:
        aggregate = numeric.get(name)
        if (
            not isinstance(aggregate, dict)
            or aggregate.get("count") != expected_steps
            or aggregate.get("missing") != 0
        ):
            raise CampaignError(
                f"telemetry metric is incomplete for {arm['arm_id']}:{name}"
            )
        _finite(aggregate.get("mean"), f"{arm['arm_id']}:{name}.mean")
        _finite(aggregate.get("last"), f"{arm['arm_id']}:{name}.last")

    records: dict[int, dict[str, Any]] = {}
    try:
        with raw_path.open(encoding="utf-8") as raw_handle:
            for line_number, raw_line in enumerate(raw_handle, start=1):
                record = json.loads(raw_line)
                step = int(record.get("step", -1))
                if step in records:
                    raise CampaignError(
                        f"duplicate raw step for {arm['arm_id']} at line {line_number}"
                    )
                for key, expected in (
                    ("NON_PRIVATE_TELEMETRY", True),
                    ("run_id", run_id),
                    ("config_fingerprint", fingerprint),
                    ("method", arm["method"]),
                    ("privacy", "dp"),
                    ("dataset", arm["dataset"]),
                    ("base_model", arm["model_id"]),
                    ("model_revision", arm["model_revision"]),
                ):
                    if record.get(key) != expected:
                        raise CampaignError(
                            f"raw identity mismatch for {arm['arm_id']}:{key} "
                            f"at line {line_number}"
                        )
                records[step] = record
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise CampaignError(f"cannot parse raw telemetry: {arm['arm_id']}") from exc
    if set(records) != set(range(1, expected_steps + 1)):
        raise CampaignError(f"raw telemetry has incomplete steps: {arm['arm_id']}")
    return records


def analyze(root: Path) -> None:
    manifest_path = root / "plans" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    results = []
    trajectory_rows = []
    validation_curve_rows = []
    for arm in manifest.get("arms", []):
        arm_root = root / arm["relative_root"]
        status_path = arm_root / "adapter" / "run_status.json"
        telemetry_path = arm_root / "results" / "research_raw" / "telemetry_summary.json"
        summary_path = arm_root / "results" / "summary.csv"
        raw_path = arm_root / "results" / "research_raw" / "NON_PRIVATE_train_log.jsonl"
        try:
            status = json.loads(status_path.read_text(encoding="utf-8"))
            telemetry = json.loads(telemetry_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise CampaignError(f"incomplete arm {arm['arm_id']}: {exc}") from exc
        if status.get("state") != "completed" or status.get("config", {}).get("implementation_git_sha") != manifest["code_sha"]:
            raise CampaignError(f"arm is not completed at the locked SHA: {arm['arm_id']}")
        focused_screen = manifest.get("profile") == "glue-slaclip-screen"
        raw_records = None
        if focused_screen:
            raw_records = _validate_arm_telemetry(
                arm, status, telemetry, raw_path, manifest["code_sha"]
            )
        validation_loss = None
        task_average = None
        if focused_screen:
            validation = status.get("validation")
            if not isinstance(validation, dict):
                raise CampaignError(f"focused screen lacks validation: {arm['arm_id']}")
            if (
                validation.get("PUBLIC_VALIDATION_DATA") is not True
                or validation.get("protocol_stage") != "selection"
                or validation.get("validation_data_is_public") is not True
                or validation.get("records") != GLUE_SLACLIP_VALIDATION_ROWS
                or validation.get("seed") != GLUE_SLACLIP_VALIDATION_SEED
                or validation.get("validation_indices_sha256")
                != GLUE_SLACLIP_VALIDATION_INDICES_SHA256
                or validation.get("validation_record_hashes_sha256")
                != GLUE_SLACLIP_VALIDATION_RECORDS_SHA256
            ):
                raise CampaignError(f"invalid focused-screen validation lock: {arm['arm_id']}")
            validation_loss = _finite(
                validation.get("loss_mean"), f"{arm['arm_id']}:validation.loss_mean"
            )
            selection_metric = "public_holdout_response_only_loss"
            selection_value = validation_loss
            selection_score = -validation_loss
            curve_path = arm_root / "results" / "validation" / "validation_curve.jsonl"
            curve_steps = set()
            curve_loss_by_step = {}
            try:
                with curve_path.open(encoding="utf-8") as curve_handle:
                    for line_number, curve_line in enumerate(curve_handle, start=1):
                        curve = json.loads(curve_line)
                        step = int(curve.get("step", -1))
                        if step in curve_steps:
                            raise CampaignError(
                                f"duplicate validation curve step for {arm['arm_id']}: {step}"
                            )
                        curve_steps.add(step)
                        if (
                            curve.get("PUBLIC_VALIDATION_DATA") is not True
                            or curve.get("NON_PRIVATE_SELECTION_METRIC") is not True
                            or curve.get("records") != GLUE_SLACLIP_VALIDATION_ROWS
                            or curve.get("run_id") != status["run_id"]
                            or curve.get("config_fingerprint")
                            != status["config_fingerprint"]
                            or curve.get("planned_update_steps") != arm["steps"]
                            or curve.get("manifest_sha256")
                            != validation.get("manifest_sha256")
                            or curve.get("selection_metric")
                            != "response_only_mean_per_record_causal_lm_loss"
                            or curve.get("loss_definition")
                            != "response_only_per_record_mean_of_nonignored_next_token_losses"
                            or curve.get("validation_indices_sha256")
                            != GLUE_SLACLIP_VALIDATION_INDICES_SHA256
                            or curve.get("validation_record_hashes_sha256")
                            != GLUE_SLACLIP_VALIDATION_RECORDS_SHA256
                        ):
                            raise CampaignError(
                                f"invalid validation curve lock for {arm['arm_id']} "
                                f"at line {line_number}"
                            )
                        curve_loss = _finite(
                            curve.get("loss_mean"),
                            f"{arm['arm_id']}:validation_curve:{step}:loss_mean",
                        )
                        curve_loss_by_step[step] = curve_loss
                        validation_curve_rows.append({
                            "setting_id": arm["setting_id"],
                            "candidate_id": arm["candidate_id"],
                            "candidate_role": arm.get("role", "standard"),
                            "method": arm["method"],
                            "initial_C": arm["initial_c"],
                            "conditional_rho": arm["rho"],
                            "controller_eta": arm["eta"],
                            "seed": arm["seed"],
                            "step": step,
                            "records": curve["records"],
                            "loss_mean": curve_loss,
                            "token_mean_loss": _finite(
                                curve.get("token_mean_loss"),
                                f"{arm['arm_id']}:validation_curve:{step}:token_mean_loss",
                            ),
                            "supervised_tokens": int(curve["supervised_tokens"]),
                        })
            except (OSError, json.JSONDecodeError, KeyError, ValueError) as exc:
                raise CampaignError(
                    f"cannot read focused-screen validation curve: {arm['arm_id']}"
                ) from exc
            if curve_steps != {0, 50, 100, 150}:
                raise CampaignError(
                    f"focused-screen validation curve has wrong steps for {arm['arm_id']}: "
                    f"{sorted(curve_steps)}"
                )
            if not math.isclose(
                curve_loss_by_step[150], validation_loss, rel_tol=1e-9, abs_tol=1e-8
            ):
                raise CampaignError(
                    f"endpoint validation and curve disagree: {arm['arm_id']}"
                )
        else:
            task_average = _task_average(summary_path)
            selection_metric = "task_average"
            selection_value = task_average
            selection_score = task_average
        numeric = telemetry.get("metrics", telemetry.get("numeric_metrics", {}))
        metric = lambda name, key="mean": _finite(numeric.get(name, {}).get(key), f"{arm['arm_id']}:{name}.{key}")
        if raw_records is None:
            clip_values = _raw_series(raw_path, "raw_clip_fraction")
            small_proxy_values = _raw_series(
                raw_path, "raw_reference_small_gradient_proxy"
            )
        else:
            clip_values = [
                _finite(raw_records[step].get("raw_clip_fraction"), f"{arm['arm_id']}:{step}:raw_clip_fraction")
                for step in range(1, int(arm["steps"]) + 1)
            ]
            small_proxy_values = [
                _finite(
                    raw_records[step].get("raw_reference_small_gradient_proxy"),
                    f"{arm['arm_id']}:{step}:raw_reference_small_gradient_proxy",
                )
                for step in range(1, int(arm["steps"]) + 1)
            ]
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
            "slack_unclipped_proxy", "slack_clipped_proxy",
            "slack_indicator_noise_std", "slaclip_gamma_t",
            "slaclip_target_non_small_clip_fraction",
            "slaclip_target_unclipped_proxy",
            "slaclip_target_unclipped_proxy_preprojection",
            "slaclip_target_clipped_proxy",
            "slaclip_observed_unclipped_proxy",
            "slaclip_controller_error", "slaclip_c_next_unbounded",
            "slaclip_c_hit_min", "slaclip_c_hit_max",
            "slaclip_small_gradient_proxy_noisy",
            "slaclip_remaining_mass_proxy_noisy",
        )
        if raw_records is None:
            with raw_path.open(encoding="utf-8") as raw_handle:
                ordered_raw_records = [json.loads(raw_line) for raw_line in raw_handle]
        else:
            ordered_raw_records = [
                raw_records[step] for step in range(1, int(arm["steps"]) + 1)
            ]
        for raw_record in ordered_raw_records:
            trajectory_rows.append({
                "setting_id": arm["setting_id"],
                "candidate_id": arm["candidate_id"],
                "candidate_role": arm.get("role", "standard"),
                "method": arm["method"],
                "dataset": arm["dataset"],
                "model": arm["model_id"],
                "epsilon": arm["epsilon"],
                "lora_r": arm["lora_r"],
                "fixed_C": arm["initial_c"],
                "initial_C": arm["initial_c"],
                "conditional_rho": arm["rho"],
                "controller_eta": arm["eta"],
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
                "candidate_role": arm.get("role", "standard"),
                "method": arm["method"],
                "initial_C": arm["initial_c"],
                "rho": arm["rho"],
                "eta": arm["eta"],
                "selection_metric": selection_metric,
                "selection_value": selection_value,
                "selection_score": selection_score,
                "task_average": task_average,
                "public_validation_loss": validation_loss,
                "loss_last": metric("loss_mean", "last"),
                "loss_mean": metric("loss_mean"),
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
                "signal_retention_mean": metric("raw_signal_retention_ratio"),
                "clipping_bias_mean": metric("raw_clipping_bias_norm"),
                "realized_noise_mean": metric("raw_realized_noise_norm"),
                "bias_noise_mse_proxy_mean": metric("raw_bias_noise_squared_error_proxy"),
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
    if validation_curve_rows:
        validation_fields = list(validation_curve_rows[0])
        validation_buffer = [",".join(validation_fields)]
        for row in validation_curve_rows:
            validation_buffer.append(",".join(
                "" if row[field] is None else str(row[field])
                for field in validation_fields
            ))
        _with_sha(
            out / "public_validation_curve.csv",
            ("\n".join(validation_buffer) + "\n").encode(),
        )
    best_fixed = {}
    for row in results:
        if row["method"] == "baseline":
            best_fixed[row["setting_id"]] = max(
                best_fixed.get(row["setting_id"], float("-inf")), row["selection_score"]
            )
    regime_groups: dict[str, list[dict[str, Any]]] = {}
    for row in results:
        if (
            row["method"] != "slaclip"
            or row["candidate_role"] == "initial_C_sensitivity_control"
        ):
            continue
        row["delta_vs_setting_best_fixed"] = (
            row["selection_score"] - best_fixed[row["setting_id"]]
        )
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
    if manifest.get("profile") == "glue-slaclip-screen":
        eligible = [
            row for row in results
            if row["candidate_role"] in {
                "tuned_fixed_candidate", "slaclip_target_candidate"
            }
        ]
        fixed_ranked = sorted(
            (row for row in eligible if row["method"] == "baseline"),
            key=lambda row: (-row["selection_score"], row["candidate"]),
        )
        slaclip_ranked = sorted(
            (row for row in eligible if row["method"] == "slaclip"),
            key=lambda row: (-row["selection_score"], row["candidate"]),
        )
        if len(fixed_ranked) != 5 or len(slaclip_ranked) != 5:
            raise CampaignError("focused screen must rank exactly five fixed and five SlaClip candidates")
        selection_payload = {
            "schema_version": 1,
            "inference": "single_seed_150_step_exploratory_screen",
            "selection_metric": "public_holdout_response_only_loss",
            "ranking_rule": "ascending loss, then candidate id",
            "task_test_or_official_glue_evaluation_used": False,
            "best_fixed": fixed_ranked[0],
            "best_slaclip": slaclip_ranked[0],
            "fixed_ranking": fixed_ranked,
            "slaclip_ranking": slaclip_ranked,
            "initial_C_sensitivity_controls": [
                row for row in results
                if row["candidate_role"] == "initial_C_sensitivity_control"
            ],
            "confirmation_requirement": manifest["glue_slaclip_screen"][
                "confirmation_requirement"
            ],
        }
        _with_sha(out / "glue_slaclip_screen_ranking.json", _json_bytes(selection_payload))
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
            "baseline-reproduction-cached", "glue-slaclip-screen",
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
