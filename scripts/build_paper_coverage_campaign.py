#!/usr/bin/env python3
"""Build and summarize PRISM paper-coverage and clipping-regime screens.

These profiles are exploratory screens rather than confirmatory experiments.
``paper-breadth`` preserves the historical 15-arm plan; ``regime-map`` crosses
the paper's available dataset, privacy, rank, and 4B/9B axes; and
``glue-slaclip-screen`` focuses on the first baseline setting whose fixed-C
trajectory exhibited non-saturated clipping and measurable slack.
``glue-high-c-refinement`` then performs a dynamically locked fixed-C boundary
scan and conditional-target refinement.  ``glue-r8-slack-screen`` moves to the
paper's rank-8 GLUE setting, locks two completed source campaigns, and uses a
fresh-seed Stage-2 comparator after a dynamic fixed-C selection.  All arms run inside one immutable
one- or two-lane Slurm allocation, and measured clipping strata are reported
without pretending they are pre-established SlaClip failure thresholds.
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
GLUE_HIGH_C_REFINEMENT_SEED = 44
GLUE_HIGH_C_REFINEMENT_STEPS = 200
GLUE_HIGH_C_FIXED_GRID = (3.0, 5.0, 7.5, 10.0, 15.0)
GLUE_HIGH_C_RHO_QUANTILES = (0.10, 0.25, 0.50, 0.75, 0.90)
GLUE_HIGH_C_RHO_BOUNDS = (0.20, 0.90)
GLUE_HIGH_C_PRIMARY_ETA = 0.02
GLUE_HIGH_C_FAST_ETA = 0.05
GLUE_R8_SLACK_STAGE1_SEED = 45
GLUE_R8_SLACK_STAGE2_SEED = 46
GLUE_R8_SLACK_STEPS = 200
GLUE_R8_SLACK_FIXED_GRID = (0.5, 1.0, 2.0, 5.0, 10.0, 15.0)
GLUE_R8_SLACK_RHO_QUANTILES = (0.10, 0.25, 0.50, 0.75, 0.90)
GLUE_R8_SLACK_RHO_BOUNDS = (0.05, 0.95)
GLUE_R8_SLACK_PRIMARY_ETA = 0.02
GLUE_R8_SLACK_FAST_ETA = 0.05
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

GLUE_HIGH_C_FIXED_CANDIDATES = tuple(
    {
        "id": f"fixed-c{_candidate_id_value(value)}",
        "method": "baseline",
        "initial_c": value,
        "rho": None,
        "eta": None,
        "role": "high_C_fixed_candidate",
        "stage": 1,
        "lane": 0,
    }
    for value in GLUE_HIGH_C_FIXED_GRID
)

GLUE_R8_SLACK_SETTING = next(
    setting for setting in REGIME_SETTINGS
    if setting["id"] == "glue8-4b-eps6-r8"
)
GLUE_R8_SLACK_FIXED_CANDIDATES = tuple(
    {
        "id": f"fixed-c{_candidate_id_value(value)}",
        "method": "baseline",
        "initial_c": value,
        "rho": None,
        "eta": None,
        "role": "slack_fixed_candidate",
        "stage": 1,
        "lane": 0,
    }
    for value in GLUE_R8_SLACK_FIXED_GRID
)

# Job 1413408 is the immutable negative decision source for leaving the
# rank-16/high-C branch.  Hash both the plan/selection chain and the final
# ranking so a later source edit cannot silently change this decision.
GLUE_R8_NEGATIVE_DECISION_SOURCE = {
    "campaign_id": "paper-coverage-74b74a88f044-glue-high-c-refinement-v2",
    "job_id": "1413408",
    "code_sha": "74b74a88f044fcab75124faabd26ee559c94782a",
    "profile": "glue-high-c-refinement",
    "artifact_sha256": {
        "submission_receipt.json": "9d4d7216664ca4d6c3cbeb488f2713b4266c4dfd178e940a6265bbee72f2a568",
        "plans/manifest.json": "10e3ab0455f5dcae4f43e2b16de6f68654c35bb174fa3f0aed8ba9aacd25cc56",
        "selection/high_c_stage1_lock.json": "44bd90947d62e4fb5c54317d7662654e6ecfb50f5d08adb346592d510ca55ac6",
        "plans/stage2-slaclip.tsv": "76f417c8ec967e1cf67624bb725d15eed7cbfd0bf96526a551eab322eaf3b57b",
        "artifacts/glue_high_c_refinement_ranking.json": "4f46b5fd11531705124784ef0ca8850103c0e7e699d140ccea64f690c9a470a4",
        "artifacts/paper_coverage_summary.csv": "bd444a1d370d62ed05cd0be15ecf5aaad0b6082b09474477a484ddd58d9a983b",
        "status/job-1413408.txt": "ea60b83c06f2df8a66024c2c31e30ae343618d1f66edc33571826ed7c9448cc0",
    },
    "stage1_lock_sha256": "44bd90947d62e4fb5c54317d7662654e6ecfb50f5d08adb346592d510ca55ac6",
    "stage2_plan_sha256": "76f417c8ec967e1cf67624bb725d15eed7cbfd0bf96526a551eab322eaf3b57b",
    "expected_best_fixed": "fixed-c15p0",
    "expected_best_fixed_loss": 0.28900413651950657,
    "expected_best_slaclip": "full-sla-q10-eta002",
    "expected_best_slaclip_loss": 0.2973122682981193,
}

# The completed rank-8 arm is usable even though its nine-arm parent campaign
# later timed out: the arm has a completed 500-step status, complete exact
# telemetry, and all pinned official GLUE-validation outputs.  The parent job
# state is deliberately not required to be COMPLETED.
GLUE_R8_BASELINE_SOURCE = {
    "campaign_id": "paper-coverage-8495ac8f0c07-baseline-reproduction-cached-v2",
    "relative_arm_root": "runs/glue8-4b-eps6-r8/fixed-c1-paper-default/seed-42",
    "job_id": "1402286",
    "parent_job_terminal_state": "TIMEOUT",
    "code_sha": "8495ac8f0c07addf910d8f3a2e7eec6a88884a92",
    "run_id": "paper-coverage-glue8-4b-eps6-r8-baseline-seed42_C1_47550ae8cf",
    "config_fingerprint": "47550ae8cf0d88c6cc39f4cfb3fb71771032849140139c2ef11e17378ab8840d",
    "data_sha256": "281bad3305e8be7a7660b64b22704a9462f983b90df5d6f0c1c4abb649d6d091",
    "model_id": MODEL_4B,
    "model_revision": "cc012e0a6d0787b4adcc0fa2c4da74402494554d",
    "raw_records": 500,
    "artifact_sha256": {
        "submission_receipt.json": "3f299a6fb8094e69c53d5a24a435b8d6299e83fc9542732cb5c7df9b33659fdd",
        "plans/manifest.json": "933507b20afbe734f88addee33c710d3bc1834e3816a955e75d59bfc726ef664",
        "adapter/run_status.json": "2034563a132b334fcca293729a9cc42b3949c2cb54b8b9caf5306bb4e5efd99a",
        "adapter/adapter_model.safetensors": "9d68d92b22f029612c9c4855c2cd79b683b723d2a11c6e17e2e0fb4dd6a397b9",
        "results/run_status.json": "17c84326301a01b24f8f77e5a4c2defcfa73aa1a414cebda23cbd22814c839a5",
        "results/research_raw/NON_PRIVATE_train_log.jsonl": "5141126a0cff0e4865b53bf5ce8b27a49dc7810dede68de8f228cd0ddf3fb179",
        "results/research_raw/telemetry_steps.csv": "837911bb231c1ccb3f5c10254611811496873d8b8ac69faa8301c2706029e708",
        "results/research_raw/telemetry_summary.json": "ab48ea38ed3cf6d1e599e07a5f5004787ba62f19341c90141de36fa291a8e864",
        "results/validation/split_manifest.json": "e10042a93c76f0534b9501e62e6ef3c80d93e0a7465efcb0c365febcfa5dc07d",
        "results/evaluation_config.json": "2f6a7aa27586eeb30b2cf79088315c067f254a57c0d5b4c9d7fed4e9b72d9f4c",
        "results/summary.csv": "15e74572136fb963f439c774f4812a013a0710ad53024f5250da5bdff73753b0",
        "results/details.csv": "d495649c4ebe9b11a31998fec0b4c6fe2df276773416404d63c5ae3ba604759b",
        "results/cola.json": "5cc4066bff69ae26ad90a1b0cde4f01ecbdd1ef9f31160d3e2fb75491cf798ed",
        "results/sst2.json": "a1dbf4639fdd953a55afb44d80d8027cd38cd26c0cadd46ecc081288e68565d9",
        "results/mrpc.json": "184f737a8e5617b331408c098d3e793bb654811bdd9bfa734a1e9fea7df32462",
        "results/stsb.json": "8492cb19e29bf55bebd522a8159f3f0655e05a85f9140c86a4e14be63d1b112d",
        "results/qqp.json": "dcec0f103d91afc1a4bc2d9a8ddbc9b74bd0a8854049a6476a6de912349883a1",
        "results/mnli.json": "4cf206bf3b53e0b51a61f8041576ef2a82fab49fd1b64a611b7f6548e9c0b69c",
        "results/qnli.json": "81aa504f336ee0b8ec01139d930cd964e1abd8cdfd233140e0b36d7dda31eb2c",
        "results/rte.json": "c1694ef9744d343bd59d331607e35cd13921fdd846fd14ca18289ee39755b9cb",
        "orchestration-status.txt": "c8428b8682b9e47f382934c0ce716120e632bb3892f8b2f77f41412f4871a5b9",
    },
    "post_burn_in_records": 450,
    "post_burn_in_clip_fraction_mean": 0.48935359997850886,
    "post_burn_in_clip_fraction_median": 0.4909090909090909,
    "post_burn_in_small_gradient_proxy_mean": 0.2137261623205744,
    "post_burn_in_small_gradient_proxy_median": 0.21058819990851185,
    "post_burn_in_conditional_clip_fraction_quantiles": {
        "q10": 0.5221464409097681,
        "q25": 0.5652740332850836,
        "q50": 0.6218267282148673,
        "q75": 0.6790500711141583,
        "q90": 0.731064119780448,
    },
    "official_glue_validation_average": 0.7662401098508378,
    "official_glue_dataset_revision": "bcdcba79d07bc864c1c254ccfcedcce55bcc9a8c",
    "official_glue_content_sha256": "6cedcb0bc71ed180ed50468406f3295a28ace450f0ae8f970abdcc77616af9ef",
}

# Job 1411662's five fixed and five primary Full-SlaClip arms were complete
# before its two C0 controls.  Those ten completed arms are sufficient to
# motivate this boundary refinement and are locked independently of the final
# campaign-level ranking.  The controls are intentionally excluded because
# they cannot affect either primary ranking.
GLUE_HIGH_C_PRECEDING_PRIMARY_SOURCE = {
    "campaign_id": "paper-coverage-9876d652153a-glue-slaclip-screen-v2",
    "job_id": "1411662",
    "code_sha": "9876d652153a31ad35644820a2dcc9b5481620fa",
    "manifest_sha256": "54215469ec16bc369b1075021751417610bba6b7b4c0dc791cd3c3618cc806b5",
    "submission_receipt_sha256": (
        "d399b04fea09f31a9b6aaaa410f1df7df2fa9a7feeaf8187ee6d44e9489b9926"
    ),
    "excluded_roles": ["initial_C_sensitivity_control"],
    "expected_best_fixed": "fixed-c5p0",
    "expected_best_slaclip": "full-sla-c1-rho055",
    "artifact_sha256": {
        "fixed-c0p5": (
            "1d33a2089a766b788c4c662f7f90155d293946c6c5fe5e5dcd4f48d227aa12f3",
            "12d2d90db639110b369899c8ce62ab0f33af3f3471c9820591d5f892a1c6a0a3",
            "2ff7460a0cd90fb0fa3151b56336abde71666f36c85f18c974429b5243eff61f",
            "b9032b1ea329de3296843fbf62625acc32fdbdde002d68393a443c800fac0079",
            "1468dfc24169bedbbdd7ee91e2619803bb334087d69cb8304f621eaf8ea21459",
            "3c69a40c07fe687cc4f19d05316354adb4623e7fea86ba504c7c1945d6b09e61",
        ),
        "fixed-c1p0": (
            "1510f5784dd7bf40205ad47a6308e891985bba737748d275fd44dba6149d372a",
            "f61104d84e6a042eb0208fd5b23f04f2fbbe47972ced6469aba7fe96968908ed",
            "63563ee0e41b0c4620b7f1e5ab24e6bf77a3659b5c466f5c0787a6436c1d5b23",
            "b9032b1ea329de3296843fbf62625acc32fdbdde002d68393a443c800fac0079",
            "f33f20524322d711f90c1e6e98123db2b567a151d55e1562284f91b3b28ccc7c",
            "07fba1ec3f2f9dc1eff597626bb71722360a77370bb34d65a9b3c32ef3ea1ef8",
        ),
        "fixed-c2p0": (
            "795fde48b67972757b64a4e96aa4820821c8f7815a01f1b8b2d4154aaa18fc2c",
            "645c842ff54b2147e1b5cab6fe690d8767035519af277482414512a565a58f60",
            "42628a5ccb7792ecca8232061500d8b47c78ca1df1bbab57759fdbaebfabde30",
            "b9032b1ea329de3296843fbf62625acc32fdbdde002d68393a443c800fac0079",
            "05d03a0db943de60f975924a3587583a5686771ffcd6f50ab82b27d99d5b871c",
            "f8bb6c055e11b4c45472fe39bce5eeb620dd4ecb41de036085752c382a84d17a",
        ),
        "fixed-c3p0": (
            "1d74c049f6c68796c97c42b28b17a98715a1959909ff7b64c578d5c57eba739a",
            "63538e7f89270ad6fb983bb9d354e637847bc066e59a3a5e9dfe3d2c1c9592f3",
            "b04dfbda3b460bd50deff23e3c72f90100a9323dc8fcd1fd5937a50bd84e7749",
            "b9032b1ea329de3296843fbf62625acc32fdbdde002d68393a443c800fac0079",
            "10175f54316570dd5fafd13027bec7ea692fa5440cb8cddcf820068c98cfe85a",
            "6d9f4a9fa05f2f81442a5bbd29be4e8853867dbf04efb9ed83afcfe7c2e3ed68",
        ),
        "fixed-c5p0": (
            "810cb20ee385a7840967178d863ffe27f417cf645c9581f1bcd8f1a3d1e52ce6",
            "6b9078ac92d1442585df275c343c48ca8249172a4037a3767bca546a1a464f1c",
            "e24ef10f53dcc4c24bc256ce237b003e369c8c9310be4e06371b659bad482f3b",
            "b9032b1ea329de3296843fbf62625acc32fdbdde002d68393a443c800fac0079",
            "efdf451bec823e59f8c2d20d0567ba58a953dc77e5fb17c4d4bc01b860b06838",
            "64c0b23f08bddfdbe0097b10fa67592cb61b625b2532d8372e4c4f223355d468",
        ),
        "full-sla-c1-rho055": (
            "89c5bcbaa05d0d382545d03000111637fabd7093ee74aca77efb4a0f4c66d084",
            "9028bd0b92ff9bdf0cfbd78969fc3d63fa272eb33df73f2cd91a6fa010227730",
            "0ac42931dabe3aa0f3921675c5fe3c5460504d4cd6d7246cf6fcc4e0ce5eb022",
            "b9032b1ea329de3296843fbf62625acc32fdbdde002d68393a443c800fac0079",
            "6ff38f608903a757591182dc2ea43d3f486dadccbcc1781717eaf99de0c62bd4",
            "8e7c5b8f928497f311ea9ccf855a1014eda52625e5df36e1d6d031d249a037d3",
        ),
        "full-sla-c1-rho060": (
            "916cc6eacc11a807096a3b903b8a371eda1e8f8c1bc7151b5d2ac024b7a4c2fa",
            "3a0f9b2088698227cc105d1fdfc99c863fed262c4e05a26e3d3bce9574d07f88",
            "b6f25e23bce1586cd5b07f6c361f05175e048628345f611d2031dda35800bf74",
            "b9032b1ea329de3296843fbf62625acc32fdbdde002d68393a443c800fac0079",
            "324261879bcbe6680ffb98ffad39cde1cd742352febb7703c597b0d2e7e5d425",
            "4b56139c33594ae95bb534935d9933b8a6e2364c0b186364bef9c5e4948d08d2",
        ),
        "full-sla-c1-rho065": (
            "6bdb09491ac1c2045043de37ddafce73d4d1f538c0518d76410927f5c48a1215",
            "b16eddefd3bc492a8a70d4353fd8355714d9c467d472e9e09ac28b575d445100",
            "02ab82a19b60730b40f14a629fb578ee1fd593a2b4123d4fb8c6f3a99e6251e5",
            "b9032b1ea329de3296843fbf62625acc32fdbdde002d68393a443c800fac0079",
            "b21c132b546431d68496564fede6f2af203e60f5224b44c13c9861867e6bf963",
            "6957a035a6c2f8ef1a6faf57c01f90f2569c31d92319dffad77fcfec0fc63dcd",
        ),
        "full-sla-c1-rho070": (
            "0c2c8e6d5114310390b66bc4afbbedc9c9d2d71f5174fab16d379edf5ee9534d",
            "d4faee7d283dcec2fcc8dd08e1bb91e2ce548c0b0bed8af7db10694607889b08",
            "c11fddecb608f39c15801441eb0810026e7c511f103162048cd9fadbacf54b0d",
            "b9032b1ea329de3296843fbf62625acc32fdbdde002d68393a443c800fac0079",
            "98a3862e70907838df4b95b62b7b9ec7a33f59d741e06d3533f481fda8b25050",
            "961af5f6df1b9d0ab660f1711a0104b560b11a61f723a05e2dc1e0e36c8ca21b",
        ),
        "full-sla-c1-rho075": (
            "4425746ab734a47d8b81c17a0a4e7aac6c284a9bf7e70de1c85ffef1ef8ac099",
            "b8c38fd64ae6bd10cb1495c219d1b8d63aecf1cd69b33b940e3d2b9ac9741465",
            "c58cc6605d804772cfea4c87f61275678a126721695409122384a475db468029",
            "b9032b1ea329de3296843fbf62625acc32fdbdde002d68393a443c800fac0079",
            "4458d7f2deb8e2bf0bd248e00166a523e48667cf9d7fa6cb0387d8ea80f030e5",
            "130625842e3c8c9adad88af2191dad81662d4a58c1e344ebe497105d3200a075",
        ),
    },
    "artifact_order": (
        "status", "raw_telemetry", "telemetry_summary", "split_manifest",
        "validation_metrics", "validation_curve",
    ),
}


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
    elif profile == "glue-high-c-refinement":
        settings = (BREADTH_SETTINGS[0],)
        candidates = GLUE_HIGH_C_FIXED_CANDIDATES
        screen_steps = GLUE_HIGH_C_REFINEMENT_STEPS
        eval_limit = 0
        screen_seed = GLUE_HIGH_C_REFINEMENT_SEED
    elif profile == "glue-r8-slack-screen":
        settings = (GLUE_R8_SLACK_SETTING,)
        candidates = GLUE_R8_SLACK_FIXED_CANDIDATES
        screen_steps = GLUE_R8_SLACK_STEPS
        eval_limit = 0
        screen_seed = GLUE_R8_SLACK_STAGE1_SEED
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
        "inference_class": (
            "two_seed_exploratory_screen_requires_multi_seed_full_length_confirmation"
            if profile == "glue-r8-slack-screen"
            else "single_seed_exploratory_breadth_screen_requires_fresh_seed_confirmation"
        ),
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
            if profile in {
                "glue-slaclip-screen", "glue-high-c-refinement",
                "glue-r8-slack-screen",
            }
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
            "exploratory": profile in {
                "regime-map", "glue-slaclip-screen", "glue-high-c-refinement",
                "glue-r8-slack-screen",
            },
            "screen_steps": screen_steps,
            "per_task_eval_limit": eval_limit,
            "fixed_C_grid": (
                list(GLUE_SLACLIP_FIXED_GRID)
                if profile == "glue-slaclip-screen"
                else list(GLUE_HIGH_C_FIXED_GRID)
                if profile == "glue-high-c-refinement"
                else list(GLUE_R8_SLACK_FIXED_GRID)
                if profile == "glue-r8-slack-screen"
                else [0.5, 1.0, 2.0, 3.0, 5.0]
                if profile == "regime-map"
                else [1.0]
            ),
            "conditional_rho_grid": (
                list(GLUE_SLACLIP_RHO_GRID)
                if profile == "glue-slaclip-screen"
                else "derived_from_stage1_q10_q25_q50_q75_q90"
                if profile in {"glue-high-c-refinement", "glue-r8-slack-screen"}
                else [0.5, 0.7, 0.8, 0.9, 0.98]
                if profile == "regime-map"
                else [0.9, 0.98]
            ),
            "initial_C_sensitivity_control": (
                {"C_0": [0.5, 2.0], "rho": 0.65, "eta": 0.05}
                if profile == "glue-slaclip-screen"
                else {
                    "C_0": "max(0.1,stage1_best_fixed_C/2)",
                    "rho": "stage1_q50",
                    "eta": GLUE_HIGH_C_PRIMARY_ETA,
                }
                if profile == "glue-high-c-refinement"
                else {
                    "C_0": "max(0.1,stage1_best_fixed_C/2)",
                    "rho": "stage1_q50",
                    "eta": GLUE_R8_SLACK_PRIMARY_ETA,
                }
                if profile == "glue-r8-slack-screen"
                else {"C_0": 2.0, "rho": 0.9, "eta": 0.05}
                if profile == "regime-map"
                else None
            ),
            "interpretation": (
                "descriptive two-seed staged screen; clipping-rate bins are measured "
                "outcomes, not predeclared failure thresholds or confirmatory evidence"
                if profile == "glue-r8-slack-screen"
                else "descriptive one-seed screen; clipping-rate bins are measured "
                "outcomes, not predeclared failure thresholds or confirmatory evidence"
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
        "glue_high_c_refinement": {
            "enabled": profile == "glue-high-c-refinement",
            "stage1": (
                {
                    "seed": GLUE_HIGH_C_REFINEMENT_SEED,
                    "steps": GLUE_HIGH_C_REFINEMENT_STEPS,
                    "fixed_C_grid": list(GLUE_HIGH_C_FIXED_GRID),
                    "burn_in_rule": "exclude steps 1 through 50; summarize steps 51 through 200",
                    "winner_rule": (
                        "ascending step-200 public-holdout response-only per-record loss, "
                        "then ascending C, then candidate id"
                    ),
                }
                if profile == "glue-high-c-refinement" else None
            ),
            "stage2_recipe": (
                {
                    "rho_source": (
                        "winner fixed-C raw_reference_conditional_clip_fraction "
                        "over steps 51 through 200"
                    ),
                    "rho_quantiles": list(GLUE_HIGH_C_RHO_QUANTILES),
                    "rho_bounds": list(GLUE_HIGH_C_RHO_BOUNDS),
                    "rho_transform": "clamp each quantile to [0.20,0.90]",
                    "require_unique_rho_values": True,
                    "primary": {
                        "arms": 5,
                        "initial_C": "stage1_best_fixed_C",
                        "eta": GLUE_HIGH_C_PRIMARY_ETA,
                    },
                    "controls": [
                        {
                            "role": "controller_speed_control",
                            "rho": "q50",
                            "initial_C": "stage1_best_fixed_C",
                            "eta": GLUE_HIGH_C_FAST_ETA,
                        },
                        {
                            "role": "initial_C_sensitivity_control",
                            "rho": "q50",
                            "initial_C": "max(0.1,stage1_best_fixed_C/2)",
                            "eta": GLUE_HIGH_C_PRIMARY_ETA,
                        },
                    ],
                }
                if profile == "glue-high-c-refinement" else None
            ),
            "selection": (
                {
                    "public_holdout_rows": GLUE_SLACLIP_VALIDATION_ROWS,
                    "public_holdout_seed": GLUE_SLACLIP_VALIDATION_SEED,
                    "public_holdout_indices_sha256": (
                        GLUE_SLACLIP_VALIDATION_INDICES_SHA256
                    ),
                    "public_holdout_records_sha256": (
                        GLUE_SLACLIP_VALIDATION_RECORDS_SHA256
                    ),
                    "validation_curve_steps": [0, 50, 100, 150, 200],
                    "primary_metric": "step_200_response_only_mean_per_record_loss",
                    "secondary_metrics": [
                        "full_normalized_validation_loss_auc",
                        "late_window_normalized_validation_loss_auc_steps_100_to_200",
                        "clip_CDF_and_threshold_trajectory",
                        "signal_retention_bias_noise_SNR_MSE",
                        "controller_error_and_C_bound_hits",
                    ],
                    "official_task_evaluation": False,
                }
                if profile == "glue-high-c-refinement" else None
            ),
            "preceding_screen_source": (
                GLUE_HIGH_C_PRECEDING_PRIMARY_SOURCE
                if profile == "glue-high-c-refinement" else None
            ),
            "privacy_scope": (
                {
                    "target_selection": (
                        "NON_PRIVATE data-dependent calibration from exact Stage-1 "
                        "per-record gradient telemetry on the training split"
                    ),
                    "per_run_accounting": (
                        "epsilon=6, delta=1e-5 for each run conditional on its "
                        "already-selected hyperparameters"
                    ),
                    "end_to_end_dp_claim": False,
                    "calibration_and_stage2_training_overlap": True,
                    "publication_warning": (
                        "research_raw telemetry and derived target locks are NON_PRIVATE"
                    ),
                }
                if profile == "glue-high-c-refinement" else None
            ),
            "inference": (
                "single-seed exploratory boundary refinement; no fresh-seed or "
                "official-GLUE claim"
                if profile == "glue-high-c-refinement" else None
            ),
        },
        "glue_r8_slack_screen": {
            "enabled": profile == "glue-r8-slack-screen",
            "motivation_sources": (
                {
                    "rank16_negative_decision": GLUE_R8_NEGATIVE_DECISION_SOURCE,
                    "rank8_complete_baseline": GLUE_R8_BASELINE_SOURCE,
                }
                if profile == "glue-r8-slack-screen" else None
            ),
            "stage1": (
                {
                    "seed": GLUE_R8_SLACK_STAGE1_SEED,
                    "steps": GLUE_R8_SLACK_STEPS,
                    "fixed_C_grid": list(GLUE_R8_SLACK_FIXED_GRID),
                    "burn_in_rule": (
                        "exclude steps 1 through 50; summarize steps 51 through 200"
                    ),
                    "winner_rule": (
                        "ascending step-200 public-holdout response-only per-record "
                        "loss, then ascending C, then candidate id"
                    ),
                    "boundary_rule": (
                        "record a warning and block later confirmation when either "
                        "fixed-C grid boundary wins; "
                        "do not abort this two-stage screen"
                    ),
                }
                if profile == "glue-r8-slack-screen" else None
            ),
            "stage2_recipe": (
                {
                    "seed": GLUE_R8_SLACK_STAGE2_SEED,
                    "fresh_relative_to_stage1": True,
                    "rho_source": (
                        "Stage-1 winner raw_reference_conditional_clip_fraction "
                        "over steps 51 through 200"
                    ),
                    "rho_quantiles": list(GLUE_R8_SLACK_RHO_QUANTILES),
                    "rho_bounds": list(GLUE_R8_SLACK_RHO_BOUNDS),
                    "rho_transform": "clamp each quantile to [0.05,0.95]",
                    "require_unique_rho_values": True,
                    "fresh_fixed_comparator": {
                        "arms": 1,
                        "C": "stage1_best_fixed_C",
                    },
                    "primary": {
                        "arms": 5,
                        "initial_C": "stage1_best_fixed_C",
                        "eta": GLUE_R8_SLACK_PRIMARY_ETA,
                    },
                    "controls": [
                        {
                            "role": "controller_speed_control",
                            "rho": "q50",
                            "initial_C": "stage1_best_fixed_C",
                            "eta": GLUE_R8_SLACK_FAST_ETA,
                        },
                        {
                            "role": "initial_C_sensitivity_control",
                            "rho": "q50",
                            "initial_C": "max(0.1,stage1_best_fixed_C/2)",
                            "eta": GLUE_R8_SLACK_PRIMARY_ETA,
                        },
                    ],
                }
                if profile == "glue-r8-slack-screen" else None
            ),
            "selection": (
                {
                    "public_holdout_rows": GLUE_SLACLIP_VALIDATION_ROWS,
                    "public_holdout_seed": GLUE_SLACLIP_VALIDATION_SEED,
                    "public_holdout_indices_sha256": (
                        GLUE_SLACLIP_VALIDATION_INDICES_SHA256
                    ),
                    "public_holdout_records_sha256": (
                        GLUE_SLACLIP_VALIDATION_RECORDS_SHA256
                    ),
                    "validation_curve_steps": [0, 50, 100, 150, 200],
                    "primary_metric": "step_200_response_only_mean_per_record_loss",
                    "secondary_metrics": [
                        "full_normalized_validation_loss_auc",
                        "late_window_normalized_validation_loss_auc_steps_100_to_200",
                    ],
                    "official_task_evaluation": False,
                }
                if profile == "glue-r8-slack-screen" else None
            ),
            "primary_gate": (
                {
                    "endpoint": "best_slaclip_strictly_lower_than_fresh_fixed",
                    "full_auc": "best_slaclip_not_higher_than_fresh_fixed",
                    "late_auc": "best_slaclip_not_higher_than_fresh_fixed",
                    "boundary_block": "stage1_best_fixed_C_at_grid_boundary",
                }
                if profile == "glue-r8-slack-screen" else None
            ),
            "privacy_scope": (
                {
                    "target_selection": (
                        "NON_PRIVATE data-dependent calibration from exact Stage-1 "
                        "per-record gradient telemetry"
                    ),
                    "per_run_accounting": (
                        "epsilon=6, delta=1e-5 for each run conditional on its "
                        "already-selected hyperparameters"
                    ),
                    "end_to_end_dp_claim": False,
                    "publication_warning": (
                        "research_raw telemetry and derived target locks are NON_PRIVATE"
                    ),
                }
                if profile == "glue-r8-slack-screen" else None
            ),
            "inference": (
                "two-seed exploratory slack screen; no official-GLUE or "
                "multi-seed confirmation claim"
                if profile == "glue-r8-slack-screen" else None
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
    elif profile == "glue-high-c-refinement":
        _verify_high_c_preceding_primary_source(root)
    elif profile == "glue-r8-slack-screen":
        _verify_glue_r8_slack_sources(root)
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
    if profile in {"glue-high-c-refinement", "glue-r8-slack-screen"}:
        _with_sha(
            root / "plans" / "stage1-fixed.tsv",
            _plan_bytes(manifest, 0, include_all=True),
        )
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


def _expected_validation_curve_steps(steps: int) -> set[int]:
    expected = set(range(0, int(steps) + 1, 50))
    expected.add(int(steps))
    return expected


def _validate_focused_curve(
    arm: dict[str, Any],
    status: dict[str, Any],
    curve_path: Path,
) -> dict[int, dict[str, Any]]:
    """Validate and return one public-selection validation trajectory."""

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
        raise CampaignError(
            f"invalid focused-screen validation lock: {arm['arm_id']}"
        )
    curves: dict[int, dict[str, Any]] = {}
    try:
        with curve_path.open(encoding="utf-8") as curve_handle:
            for line_number, curve_line in enumerate(curve_handle, start=1):
                curve = json.loads(curve_line)
                step = int(curve.get("step", -1))
                if step in curves:
                    raise CampaignError(
                        f"duplicate validation curve step for {arm['arm_id']}: {step}"
                    )
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
                _finite(
                    curve.get("loss_mean"),
                    f"{arm['arm_id']}:validation_curve:{step}:loss_mean",
                )
                _finite(
                    curve.get("token_mean_loss"),
                    f"{arm['arm_id']}:validation_curve:{step}:token_mean_loss",
                )
                curves[step] = curve
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise CampaignError(
            f"cannot read focused-screen validation curve: {arm['arm_id']}"
        ) from exc
    expected_steps = _expected_validation_curve_steps(int(arm["steps"]))
    if set(curves) != expected_steps:
        raise CampaignError(
            f"focused-screen validation curve has wrong steps for {arm['arm_id']}: "
            f"{sorted(curves)}"
        )
    endpoint_loss = _finite(
        validation.get("loss_mean"), f"{arm['arm_id']}:validation.loss_mean"
    )
    if not math.isclose(
        _finite(curves[int(arm["steps"])].get("loss_mean"), "curve endpoint"),
        endpoint_loss,
        rel_tol=1e-9,
        abs_tol=1e-8,
    ):
        raise CampaignError(f"endpoint validation and curve disagree: {arm['arm_id']}")
    return curves


def _load_focused_arm(
    root: Path,
    arm: dict[str, Any],
    expected_code_sha: str,
) -> tuple[dict[str, Any], dict[int, dict[str, Any]], dict[int, dict[str, Any]], dict[str, str]]:
    """Strictly bind all selection artifacts for one arm."""

    arm_root = root / arm["relative_root"]
    paths = {
        "status": arm_root / "adapter" / "run_status.json",
        "telemetry_summary": (
            arm_root / "results" / "research_raw" / "telemetry_summary.json"
        ),
        "raw_telemetry": (
            arm_root / "results" / "research_raw" / "NON_PRIVATE_train_log.jsonl"
        ),
        "split_manifest": (
            arm_root / "results" / "validation" / "split_manifest.json"
        ),
        "validation_metrics": (
            arm_root / "results" / "validation" / "validation_metrics.json"
        ),
        "validation_curve": (
            arm_root / "results" / "validation" / "validation_curve.jsonl"
        ),
    }
    try:
        status = json.loads(paths["status"].read_text(encoding="utf-8"))
        telemetry = json.loads(
            paths["telemetry_summary"].read_text(encoding="utf-8")
        )
        split = json.loads(paths["split_manifest"].read_text(encoding="utf-8"))
        validation = json.loads(
            paths["validation_metrics"].read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise CampaignError(f"incomplete focused arm {arm['arm_id']}: {exc}") from exc
    if status.get("data_split") != split or status.get("validation") != validation:
        raise CampaignError(
            f"focused arm status is not bound to split/metrics: {arm['arm_id']}"
        )
    records = _validate_arm_telemetry(
        arm, status, telemetry, paths["raw_telemetry"], expected_code_sha
    )
    curves = _validate_focused_curve(arm, status, paths["validation_curve"])
    hashes = {label: _file_sha256(path) for label, path in paths.items()}
    return status, records, curves, hashes


def _verify_high_c_preceding_primary_source(campaign_root: Path) -> None:
    """Recompute the locked 1411662 primary rankings without its controls."""

    source = GLUE_HIGH_C_PRECEDING_PRIMARY_SOURCE
    source_root = campaign_root.parent / source["campaign_id"]
    source_manifest_path = source_root / "plans" / "manifest.json"
    receipt_path = source_root / "submission_receipt.json"
    if _file_sha256(source_manifest_path) != source["manifest_sha256"]:
        raise CampaignError("preceding GLUE screen manifest hash mismatch")
    if _file_sha256(receipt_path) != source["submission_receipt_sha256"]:
        raise CampaignError("preceding GLUE screen submission receipt hash mismatch")
    try:
        source_manifest = json.loads(
            source_manifest_path.read_text(encoding="utf-8")
        )
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CampaignError("cannot read preceding GLUE screen manifest") from exc
    if (
        receipt.get("campaign_id") != source["campaign_id"]
        or str(receipt.get("current_job_id")) != source["job_id"]
        or receipt.get("code_sha") != source["code_sha"]
        or receipt.get("coverage_profile") != "glue-slaclip-screen"
    ):
        raise CampaignError("preceding GLUE submission identity mismatch")
    if (
        source_manifest.get("profile") != "glue-slaclip-screen"
        or source_manifest.get("code_sha") != source["code_sha"]
        or source_manifest.get("seed") != GLUE_SLACLIP_SCREEN_SEED
    ):
        raise CampaignError("preceding GLUE screen identity mismatch")
    expected_candidates = set(source["artifact_sha256"])
    all_source_arms = source_manifest.get("arms", [])
    primary_arms = [
        arm for arm in all_source_arms
        if arm.get("candidate_id") in expected_candidates
    ]
    excluded_arms = [
        arm for arm in all_source_arms
        if arm.get("candidate_id") not in expected_candidates
    ]
    if (
        len(all_source_arms) != 12
        or len(primary_arms) != 10
        or {arm["candidate_id"] for arm in primary_arms} != expected_candidates
        or any(
            arm.get("role") not in {
                "tuned_fixed_candidate", "slaclip_target_candidate"
            }
            for arm in primary_arms
        )
    ):
        raise CampaignError("preceding GLUE primary-arm set mismatch")
    if (
        len(excluded_arms) != 2
        or {arm.get("role") for arm in excluded_arms}
        != set(source["excluded_roles"])
    ):
        raise CampaignError("preceding GLUE excluded-control set mismatch")
    fixed = []
    adaptive = []
    order = tuple(source["artifact_order"])
    for arm in primary_arms:
        status, _records, _curves, actual_hashes = _load_focused_arm(
            source_root, arm, source["code_sha"]
        )
        expected_hashes = dict(
            zip(order, source["artifact_sha256"][arm["candidate_id"]], strict=True)
        )
        if actual_hashes != expected_hashes:
            raise CampaignError(
                f"preceding primary artifact hash mismatch: {arm['candidate_id']}"
            )
        row = (
            _finite(
                status["validation"].get("loss_mean"),
                f"preceding:{arm['candidate_id']}:endpoint_loss",
            ),
            arm["candidate_id"],
        )
        (fixed if arm["method"] == "baseline" else adaptive).append(row)
    fixed.sort()
    adaptive.sort()
    if (
        len(fixed) != 5
        or len(adaptive) != 5
        or fixed[0][1] != source["expected_best_fixed"]
        or adaptive[0][1] != source["expected_best_slaclip"]
    ):
        raise CampaignError("preceding primary ranking does not match its lock")


def _verify_glue_r8_negative_decision_source(campaign_root: Path) -> None:
    """Fail closed unless job 1413408 still proves the rank-16 gate failed."""

    source = GLUE_R8_NEGATIVE_DECISION_SOURCE
    source_root = campaign_root.parent / source["campaign_id"]
    for relative, expected in source["artifact_sha256"].items():
        actual = _file_sha256(source_root / relative)
        if actual != expected:
            raise CampaignError(
                f"rank-16 negative-decision source hash mismatch: {relative}; "
                f"expected={expected}, actual={actual}"
            )
    try:
        receipt = json.loads(
            (source_root / "submission_receipt.json").read_text(encoding="utf-8")
        )
        manifest = json.loads(
            (source_root / "plans" / "manifest.json").read_text(encoding="utf-8")
        )
        lock = json.loads(
            (source_root / "selection" / "high_c_stage1_lock.json").read_text(
                encoding="utf-8"
            )
        )
        ranking = json.loads(
            (
                source_root
                / "artifacts"
                / "glue_high_c_refinement_ranking.json"
            ).read_text(encoding="utf-8")
        )
        job_status = dict(
            line.split("=", 1)
            for line in (source_root / "status" / "job-1413408.txt")
            .read_text(encoding="utf-8")
            .splitlines()
            if "=" in line
        )
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise CampaignError("cannot read rank-16 negative-decision source") from exc
    if (
        receipt.get("campaign_id") != source["campaign_id"]
        or str(receipt.get("current_job_id")) != source["job_id"]
        or receipt.get("code_sha") != source["code_sha"]
        or receipt.get("coverage_profile") != source["profile"]
        or manifest.get("profile") != source["profile"]
        or manifest.get("code_sha") != source["code_sha"]
        or job_status.get("job_id") != source["job_id"]
        or job_status.get("state") != "completed"
        or job_status.get("phase") != "complete"
        or job_status.get("exit_code") != "0"
        or job_status.get("code_sha") != source["code_sha"]
    ):
        raise CampaignError("rank-16 negative-decision source identity mismatch")
    if (
        lock.get("profile") != source["profile"]
        or lock.get("code_sha") != source["code_sha"]
        or lock.get("manifest_sha256")
        != source["artifact_sha256"]["plans/manifest.json"]
        or lock.get("stage2_plan_sha256") != source["stage2_plan_sha256"]
        or _file_sha256(source_root / "selection" / "high_c_stage1_lock.json")
        != source["stage1_lock_sha256"]
    ):
        raise CampaignError("rank-16 negative-decision selection lock mismatch")
    best_fixed = ranking.get("best_fixed", {})
    best_slaclip = ranking.get("best_slaclip", {})
    if (
        ranking.get("stage1_lock_sha256") != source["stage1_lock_sha256"]
        or ranking.get("stage2_plan_sha256") != source["stage2_plan_sha256"]
        or ranking.get("slaclip_beats_best_fixed_primary") is not False
        or ranking.get("fixed_winner_at_allowed_C_max") is not True
        or ranking.get("task_test_or_official_glue_evaluation_used") is not False
        or best_fixed.get("candidate") != source["expected_best_fixed"]
        or best_slaclip.get("candidate") != source["expected_best_slaclip"]
        or not math.isclose(
            _finite(best_fixed.get("public_validation_loss"), "rank16 fixed loss"),
            source["expected_best_fixed_loss"],
            rel_tol=0.0,
            abs_tol=1e-15,
        )
        or not math.isclose(
            _finite(
                best_slaclip.get("public_validation_loss"), "rank16 SlaClip loss"
            ),
            source["expected_best_slaclip_loss"],
            rel_tol=0.0,
            abs_tol=1e-15,
        )
    ):
        raise CampaignError("rank-16 negative-decision ranking mismatch")
    if not best_slaclip["public_validation_loss"] > best_fixed["public_validation_loss"]:
        raise CampaignError("rank-16 source no longer records a negative primary result")


def _verify_glue_r8_complete_baseline_source(campaign_root: Path) -> None:
    """Validate the completed rank-8 arm independently of its timed-out job."""

    source = GLUE_R8_BASELINE_SOURCE
    campaign = campaign_root.parent / source["campaign_id"]
    arm_root = campaign / source["relative_arm_root"]
    for relative, expected in source["artifact_sha256"].items():
        path = (
            campaign / relative
            if relative in {"submission_receipt.json", "plans/manifest.json"}
            else arm_root / relative
        )
        actual = _file_sha256(path)
        if actual != expected:
            raise CampaignError(
                f"rank-8 baseline source hash mismatch: {relative}; "
                f"expected={expected}, actual={actual}"
            )
    try:
        receipt = json.loads(
            (campaign / "submission_receipt.json").read_text(encoding="utf-8")
        )
        manifest = json.loads(
            (campaign / "plans" / "manifest.json").read_text(encoding="utf-8")
        )
        status = json.loads(
            (arm_root / "adapter" / "run_status.json").read_text(encoding="utf-8")
        )
        result_status = json.loads(
            (arm_root / "results" / "run_status.json").read_text(encoding="utf-8")
        )
        telemetry = json.loads(
            (
                arm_root
                / "results"
                / "research_raw"
                / "telemetry_summary.json"
            ).read_text(encoding="utf-8")
        )
        evaluation = json.loads(
            (arm_root / "results" / "evaluation_config.json").read_text(
                encoding="utf-8"
            )
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise CampaignError("cannot read complete rank-8 baseline source") from exc
    if (
        receipt.get("campaign_id") != source["campaign_id"]
        or str(receipt.get("current_job_id")) != source["job_id"]
        or receipt.get("code_sha") != source["code_sha"]
        or receipt.get("coverage_profile") != "baseline-reproduction-cached"
        or manifest.get("profile") != "baseline-reproduction-cached"
        or manifest.get("code_sha") != source["code_sha"]
    ):
        raise CampaignError("rank-8 baseline campaign identity mismatch")
    matching = [
        arm for arm in manifest.get("arms", [])
        if arm.get("relative_root") == source["relative_arm_root"]
    ]
    if (
        len(matching) != 1
        or matching[0].get("setting_id") != "glue8-4b-eps6-r8"
        or matching[0].get("candidate_id") != "fixed-c1-paper-default"
        or matching[0].get("seed") != 42
        or matching[0].get("steps") != source["raw_records"]
        or matching[0].get("lora_r") != 8
    ):
        raise CampaignError("rank-8 baseline source arm is absent from its manifest")
    expected_status = {
        "state": "completed",
        "update_steps": source["raw_records"],
        "dataset": "glue8",
        "method": "baseline",
        "privacy": "dp",
        "base_model": source["model_id"],
        "model_revision": source["model_revision"],
        "resolved_model_revision": source["model_revision"],
        "data_content_sha256": source["data_sha256"],
        "run_id": source["run_id"],
        "config_fingerprint": source["config_fingerprint"],
        "telemetry_mode": "research_raw",
        "non_private_telemetry": True,
        "training_lora_r": 8,
    }
    for payload_name, payload in (("adapter", status), ("result", result_status)):
        for key, expected in expected_status.items():
            if payload.get(key) != expected:
                raise CampaignError(
                    f"rank-8 {payload_name} status mismatch for {key}: "
                    f"expected={expected!r}, actual={payload.get(key)!r}"
                )
    config = status.get("config")
    if not isinstance(config, dict) or result_status.get("config") != config:
        raise CampaignError("rank-8 status/config artifacts are not identical")
    expected_config = {
        "implementation_git_sha": source["code_sha"],
        "implementation_git_dirty": False,
        "dataset": "glue8",
        "method": "baseline",
        "privacy": "dp",
        "base_model": source["model_id"],
        "model_revision": source["model_revision"],
        "seed": 42,
        "lora_r": 8,
        "total_update_steps": source["raw_records"],
        "batch_size": 64,
        "micro_batch_size": 4,
        "learning_rate": 0.0002,
        "cutoff_len": 384,
        "train_on_inputs": False,
        "dp_epsilon": 6.0,
        "dp_delta": 1e-5,
        "dp_max_grad_norm": 1.0,
        "dp_accountant": "prv",
        "protocol_stage": "final",
        "val_set_size": 0,
        "run_train": True,
        "run_eval": True,
        "telemetry_mode": "research_raw",
        "allow_non_private_telemetry": True,
    }
    for key, expected in expected_config.items():
        if config.get(key) != expected:
            raise CampaignError(
                f"rank-8 baseline config mismatch for {key}: "
                f"expected={expected!r}, actual={config.get(key)!r}"
            )
    raw_sha = source["artifact_sha256"][
        "results/research_raw/NON_PRIVATE_train_log.jsonl"
    ]
    if (
        telemetry.get("summary_schema_version") != 4
        or telemetry.get("NON_PRIVATE_TELEMETRY") is not True
        or telemetry.get("source", {}).get("raw_sha256") != raw_sha
        or telemetry.get("source", {}).get("raw_physical_records")
        != source["raw_records"]
        or telemetry.get("source", {}).get("raw_unique_steps")
        != source["raw_records"]
        or telemetry.get("source", {}).get("raw_duplicate_records") != 0
    ):
        raise CampaignError("rank-8 telemetry summary is not bound to its raw log")
    expected_tasks = ["cola", "sst2", "mrpc", "stsb", "qqp", "mnli", "qnli", "rte"]
    assets = evaluation.get("glue_eval_assets", {})
    if (
        evaluation.get("evaluation_schema_version") != 1
        or evaluation.get("config_fingerprint") != source["config_fingerprint"]
        or evaluation.get("base_model") != source["model_id"]
        or evaluation.get("requested_model_revision") != source["model_revision"]
        or evaluation.get("resolved_model_revision") != source["model_revision"]
        or evaluation.get("tasks") != expected_tasks
        or evaluation.get("fast_dev_run") != 0
        or assets.get("dataset_revision")
        != source["official_glue_dataset_revision"]
        or assets.get("content_sha256") != source["official_glue_content_sha256"]
    ):
        raise CampaignError("rank-8 official GLUE-validation identity mismatch")
    average = _task_average(arm_root / "results" / "summary.csv")
    if not math.isclose(
        average,
        source["official_glue_validation_average"],
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        raise CampaignError("rank-8 official GLUE-validation average mismatch")
    try:
        detail_rows = list(
            csv.DictReader(
                (arm_root / "results" / "details.csv").open(
                    encoding="utf-8", newline=""
                )
            )
        )
    except OSError as exc:
        raise CampaignError("cannot read rank-8 GLUE details") from exc
    if [row.get("task") for row in detail_rows] != expected_tasks:
        raise CampaignError("rank-8 GLUE details task order mismatch")

    raw_path = (
        arm_root / "results" / "research_raw" / "NON_PRIVATE_train_log.jsonl"
    )
    records: dict[int, dict[str, Any]] = {}
    try:
        with raw_path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                record = json.loads(line)
                step = int(record.get("step", -1))
                if step in records:
                    raise CampaignError(f"duplicate rank-8 source step {step}")
                for key, expected in (
                    ("NON_PRIVATE_TELEMETRY", True),
                    ("run_id", source["run_id"]),
                    ("config_fingerprint", source["config_fingerprint"]),
                    ("method", "baseline"),
                    ("privacy", "dp"),
                    ("dataset", "glue8"),
                    ("base_model", source["model_id"]),
                    ("model_revision", source["model_revision"]),
                ):
                    if record.get(key) != expected:
                        raise CampaignError(
                            f"rank-8 raw identity mismatch at line {line_number}: {key}"
                        )
                records[step] = record
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise CampaignError("cannot parse rank-8 baseline telemetry") from exc
    if set(records) != set(range(1, source["raw_records"] + 1)):
        raise CampaignError("rank-8 telemetry does not contain exactly steps 1 through 500")
    post = [records[step] for step in range(51, source["raw_records"] + 1)]
    if len(post) != source["post_burn_in_records"]:
        raise CampaignError("rank-8 burn-in slice has the wrong length")
    try:
        clip = [float(record["raw_clip_fraction"]) for record in post]
        small = [
            float(record["raw_reference_small_gradient_proxy"]) for record in post
        ]
        conditional = [
            float(record["raw_reference_conditional_clip_fraction"])
            for record in post
        ]
    except (KeyError, TypeError, ValueError) as exc:
        raise CampaignError("rank-8 source lacks a target statistic") from exc
    if not all(
        record.get("raw_reference_conditional_clip_fraction_valid") is True
        for record in post
    ) or not all(
        math.isfinite(value) and 0.0 <= value <= 1.0
        for values in (clip, small, conditional)
        for value in values
    ):
        raise CampaignError("rank-8 source target statistics are invalid")
    observed = {
        "post_burn_in_clip_fraction_mean": math.fsum(clip) / len(clip),
        "post_burn_in_clip_fraction_median": _quantile(clip, 0.5),
        "post_burn_in_small_gradient_proxy_mean": math.fsum(small) / len(small),
        "post_burn_in_small_gradient_proxy_median": _quantile(small, 0.5),
    }
    for key, actual in observed.items():
        if not math.isclose(
            actual, float(source[key]), rel_tol=0.0, abs_tol=1e-12
        ):
            raise CampaignError(
                f"rank-8 source statistic mismatch for {key}: "
                f"expected={source[key]}, actual={actual}"
            )
    labels = ("q10", "q25", "q50", "q75", "q90")
    for label, fraction in zip(
        labels, GLUE_R8_SLACK_RHO_QUANTILES, strict=True
    ):
        actual = _quantile(conditional, fraction)
        expected = source["post_burn_in_conditional_clip_fraction_quantiles"][label]
        if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12):
            raise CampaignError(
                f"rank-8 conditional quantile mismatch for {label}: "
                f"expected={expected}, actual={actual}"
            )


def _verify_glue_r8_slack_sources(campaign_root: Path) -> None:
    _verify_glue_r8_negative_decision_source(campaign_root)
    _verify_glue_r8_complete_baseline_source(campaign_root)


def lock_high_c_refinement(root: Path) -> None:
    """Lock the Stage-1 winner and materialize the deterministic Stage-2 plan."""

    manifest_path = root / "plans" / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CampaignError("cannot read high-C refinement manifest") from exc
    if manifest.get("profile") != "glue-high-c-refinement":
        raise CampaignError("high-C lock requires glue-high-c-refinement profile")
    stage1_arms = manifest.get("arms")
    if (
        not isinstance(stage1_arms, list)
        or len(stage1_arms) != 5
        or {arm.get("initial_c") for arm in stage1_arms}
        != set(GLUE_HIGH_C_FIXED_GRID)
        or any(
            arm.get("method") != "baseline"
            or arm.get("role") != "high_C_fixed_candidate"
            or arm.get("seed") != GLUE_HIGH_C_REFINEMENT_SEED
            or arm.get("steps") != GLUE_HIGH_C_REFINEMENT_STEPS
            for arm in stage1_arms
        )
    ):
        raise CampaignError("high-C Stage-1 manifest does not match its recipe")

    ranked = []
    loaded: dict[str, tuple[dict[str, Any], dict[int, dict[str, Any]], dict[int, dict[str, Any]], dict[str, str]]] = {}
    for arm in stage1_arms:
        bound = _load_focused_arm(root, arm, manifest["code_sha"])
        loaded[arm["arm_id"]] = bound
        status, _records, curves, hashes = bound
        endpoint = _finite(
            status["validation"].get("loss_mean"),
            f"{arm['arm_id']}:stage1 endpoint",
        )
        ranked.append(
            {
                "arm_id": arm["arm_id"],
                "candidate_id": arm["candidate_id"],
                "fixed_C": arm["initial_c"],
                "endpoint_loss": endpoint,
                "curve_loss": {
                    str(step): _finite(curve.get("loss_mean"), "stage1 curve")
                    for step, curve in sorted(curves.items())
                },
                "artifact_sha256": hashes,
            }
        )
    ranked.sort(
        key=lambda row: (row["endpoint_loss"], row["fixed_C"], row["candidate_id"])
    )
    best = ranked[0]
    best_arm = next(
        arm for arm in stage1_arms if arm["arm_id"] == best["arm_id"]
    )
    best_records = loaded[best["arm_id"]][1]
    conditional = []
    for step in range(51, GLUE_HIGH_C_REFINEMENT_STEPS + 1):
        record = best_records[step]
        if record.get("raw_reference_conditional_clip_fraction_valid") is not True:
            raise CampaignError(
                f"best fixed arm has invalid conditional clipping proxy at step {step}"
            )
        conditional_value = _finite(
            record.get("raw_reference_conditional_clip_fraction"),
            f"{best['arm_id']}:step{step}:conditional_clip_fraction",
        )
        if not 0.0 <= conditional_value <= 1.0:
            raise CampaignError(
                "best fixed arm has out-of-range conditional clipping proxy "
                f"at step {step}: {conditional_value}"
            )
        conditional.append(conditional_value)
    rho_min, rho_max = GLUE_HIGH_C_RHO_BOUNDS
    labels = ("q10", "q25", "q50", "q75", "q90")
    rho_values = [
        max(rho_min, min(rho_max, _quantile(conditional, fraction)))
        for fraction in GLUE_HIGH_C_RHO_QUANTILES
    ]
    if len({float(value).hex() for value in rho_values}) != 5:
        raise CampaignError(
            "derived high-C rho grid is not unique after clamping to [0.20,0.90]"
        )
    if any(
        not rho_values[index] < rho_values[index + 1]
        for index in range(len(rho_values) - 1)
    ):
        raise CampaignError("derived high-C rho grid is not strictly increasing")

    stage2_arms = []
    for label, rho in zip(labels, rho_values, strict=True):
        candidate_id = f"full-sla-{label}-eta002"
        stage2_arms.append(
            {
                **{
                    key: best_arm[key]
                    for key in (
                        "paper_reference", "dataset", "model_slug", "model_id",
                        "epsilon", "lora_r", "learning_rate", "cutoff_len",
                        "train_on_inputs", "setting_id", "model_revision",
                    )
                },
                "id": candidate_id,
                "candidate_id": candidate_id,
                "arm_id": (
                    f"{best_arm['setting_id']}--{candidate_id}--"
                    f"seed{GLUE_HIGH_C_REFINEMENT_SEED}"
                ),
                "method": "slaclip",
                "initial_c": float(best["fixed_C"]),
                "rho": rho,
                "eta": GLUE_HIGH_C_PRIMARY_ETA,
                "role": "slaclip_target_candidate",
                "stage": 2,
                "rho_source_quantile": label,
                "lane": 0,
                "seed": GLUE_HIGH_C_REFINEMENT_SEED,
                "steps": GLUE_HIGH_C_REFINEMENT_STEPS,
                "eval_limit": 0,
                "relative_root": (
                    f"runs/{best_arm['setting_id']}/{candidate_id}/"
                    f"seed-{GLUE_HIGH_C_REFINEMENT_SEED}"
                ),
            }
        )
    central_rho = rho_values[2]
    controls = (
        (
            "full-sla-q50-eta005-control",
            float(best["fixed_C"]),
            GLUE_HIGH_C_FAST_ETA,
            "controller_speed_control",
        ),
        (
            "full-sla-q50-c0half-eta002-control",
            max(0.1, float(best["fixed_C"]) / 2.0),
            GLUE_HIGH_C_PRIMARY_ETA,
            "initial_C_sensitivity_control",
        ),
    )
    for candidate_id, initial_c, eta, role in controls:
        arm = dict(stage2_arms[2])
        arm.update(
            {
                "id": candidate_id,
                "candidate_id": candidate_id,
                "arm_id": (
                    f"{best_arm['setting_id']}--{candidate_id}--"
                    f"seed{GLUE_HIGH_C_REFINEMENT_SEED}"
                ),
                "initial_c": initial_c,
                "rho": central_rho,
                "eta": eta,
                "role": role,
                "relative_root": (
                    f"runs/{best_arm['setting_id']}/{candidate_id}/"
                    f"seed-{GLUE_HIGH_C_REFINEMENT_SEED}"
                ),
            }
        )
        stage2_arms.append(arm)
    if len(stage2_arms) != 7 or len({arm["arm_id"] for arm in stage2_arms}) != 7:
        raise CampaignError("high-C Stage-2 recipe did not produce seven unique arms")

    stage2_manifest = {**manifest, "arms": stage2_arms}
    stage2_plan_data = _plan_bytes(stage2_manifest, 0, include_all=True)
    stage2_plan_sha256 = hashlib.sha256(stage2_plan_data).hexdigest()
    lock_payload = {
        "schema_version": 1,
        "profile": "glue-high-c-refinement",
        "code_sha": manifest["code_sha"],
        "manifest_sha256": _file_sha256(manifest_path),
        "selection_seed": GLUE_HIGH_C_REFINEMENT_SEED,
        "selection_metric": "step_200_response_only_mean_per_record_loss",
        "ranking_rule": (
            "ascending endpoint loss, then ascending fixed C, then candidate id"
        ),
        "stage1_ranking": ranked,
        "best_fixed": best,
        "burn_in_steps_excluded": [1, 50],
        "conditional_clip_proxy_records": len(conditional),
        "rho_quantiles": {
            label: value for label, value in zip(labels, rho_values, strict=True)
        },
        "rho_bounds": list(GLUE_HIGH_C_RHO_BOUNDS),
        "rho_values_unique_after_clamp": True,
        "stage2_arms": stage2_arms,
        "stage2_plan_sha256": stage2_plan_sha256,
        "primary_endpoint": "step_200_response_only_mean_per_record_loss",
        "secondary_endpoints": {
            "full_normalized_auc": "trapezoid_steps_0_to_200_divided_by_200",
            "late_window_normalized_auc": (
                "trapezoid_steps_100_to_200_divided_by_100"
            ),
        },
        "official_task_evaluation_used": False,
        "NON_PRIVATE_DATA_DEPENDENT_SELECTION": True,
        "privacy_scope": manifest["glue_high_c_refinement"]["privacy_scope"],
        "inference": "single_seed_exploratory_boundary_refinement",
    }
    lock_path = root / "selection" / "high_c_stage1_lock.json"
    _with_sha(lock_path, _json_bytes(lock_payload))
    _with_sha(
        root / "plans" / "stage2-slaclip.tsv",
        stage2_plan_data,
    )
    print(
        f"locked_best_fixed={best['candidate_id']} C={best['fixed_C']} "
        f"stage2_arms={len(stage2_arms)}"
    )


def lock_glue_r8_slack_screen(root: Path) -> None:
    """Lock rank-8 Stage 1 and write the fresh-seed Stage-2 plan."""

    manifest_path = root / "plans" / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CampaignError("cannot read rank-8 slack-screen manifest") from exc
    if manifest.get("profile") != "glue-r8-slack-screen":
        raise CampaignError(
            "rank-8 slack lock requires glue-r8-slack-screen profile"
        )
    stage1_arms = manifest.get("arms")
    if (
        not isinstance(stage1_arms, list)
        or len(stage1_arms) != 6
        or {arm.get("initial_c") for arm in stage1_arms}
        != set(GLUE_R8_SLACK_FIXED_GRID)
        or any(
            arm.get("setting_id") != "glue8-4b-eps6-r8"
            or arm.get("lora_r") != 8
            or arm.get("method") != "baseline"
            or arm.get("role") != "slack_fixed_candidate"
            or arm.get("stage") != 1
            or arm.get("seed") != GLUE_R8_SLACK_STAGE1_SEED
            or arm.get("steps") != GLUE_R8_SLACK_STEPS
            or arm.get("lane") != 0
            for arm in stage1_arms
        )
    ):
        raise CampaignError("rank-8 Stage-1 manifest does not match its recipe")

    ranked = []
    loaded: dict[
        str,
        tuple[
            dict[str, Any], dict[int, dict[str, Any]],
            dict[int, dict[str, Any]], dict[str, str],
        ],
    ] = {}
    for arm in stage1_arms:
        bound = _load_focused_arm(root, arm, manifest["code_sha"])
        loaded[arm["arm_id"]] = bound
        status, _records, curves, hashes = bound
        endpoint = _finite(
            status["validation"].get("loss_mean"),
            f"{arm['arm_id']}:stage1 endpoint",
        )
        ranked.append(
            {
                "arm_id": arm["arm_id"],
                "candidate_id": arm["candidate_id"],
                "fixed_C": arm["initial_c"],
                "endpoint_loss": endpoint,
                "curve_loss": {
                    str(step): _finite(curve.get("loss_mean"), "rank8 stage1 curve")
                    for step, curve in sorted(curves.items())
                },
                "artifact_sha256": hashes,
            }
        )
    ranked.sort(
        key=lambda row: (row["endpoint_loss"], row["fixed_C"], row["candidate_id"])
    )
    best = ranked[0]
    best_arm = next(
        arm for arm in stage1_arms if arm["arm_id"] == best["arm_id"]
    )
    winner_at_grid_min = math.isclose(
        float(best["fixed_C"]), min(GLUE_R8_SLACK_FIXED_GRID),
        rel_tol=0.0, abs_tol=1e-12,
    )
    winner_at_grid_max = math.isclose(
        float(best["fixed_C"]), max(GLUE_R8_SLACK_FIXED_GRID),
        rel_tol=0.0, abs_tol=1e-12,
    )
    winner_at_boundary = winner_at_grid_min or winner_at_grid_max
    best_records = loaded[best["arm_id"]][1]
    conditional = []
    for step in range(51, GLUE_R8_SLACK_STEPS + 1):
        record = best_records[step]
        if record.get("raw_reference_conditional_clip_fraction_valid") is not True:
            raise CampaignError(
                f"rank-8 winner has invalid conditional clipping proxy at step {step}"
            )
        value = _finite(
            record.get("raw_reference_conditional_clip_fraction"),
            f"{best['arm_id']}:step{step}:conditional_clip_fraction",
        )
        if not 0.0 <= value <= 1.0:
            raise CampaignError(
                "rank-8 winner has out-of-range conditional clipping proxy "
                f"at step {step}: {value}"
            )
        conditional.append(value)
    labels = ("q10", "q25", "q50", "q75", "q90")
    rho_min, rho_max = GLUE_R8_SLACK_RHO_BOUNDS
    rho_values = [
        max(rho_min, min(rho_max, _quantile(conditional, fraction)))
        for fraction in GLUE_R8_SLACK_RHO_QUANTILES
    ]
    if len({float(value).hex() for value in rho_values}) != 5:
        raise CampaignError(
            "derived rank-8 rho grid is not unique after clamping to [0.05,0.95]"
        )
    if any(
        not rho_values[index] < rho_values[index + 1]
        for index in range(len(rho_values) - 1)
    ):
        raise CampaignError("derived rank-8 rho grid is not strictly increasing")

    shared = {
        key: best_arm[key]
        for key in (
            "paper_reference", "dataset", "model_slug", "model_id", "epsilon",
            "lora_r", "learning_rate", "cutoff_len", "train_on_inputs",
            "setting_id", "model_revision",
        )
    }
    fixed_id = f"fresh-fixed-c{_candidate_id_value(float(best['fixed_C']))}"
    stage2_arms = [
        {
            **shared,
            "id": fixed_id,
            "candidate_id": fixed_id,
            "arm_id": (
                f"{best_arm['setting_id']}--{fixed_id}--"
                f"seed{GLUE_R8_SLACK_STAGE2_SEED}"
            ),
            "method": "baseline",
            "initial_c": float(best["fixed_C"]),
            "rho": None,
            "eta": None,
            "role": "fresh_fixed_comparator",
            "stage": 2,
            "lane": 0,
            "seed": GLUE_R8_SLACK_STAGE2_SEED,
            "steps": GLUE_R8_SLACK_STEPS,
            "eval_limit": 0,
            "relative_root": (
                f"runs/{best_arm['setting_id']}/{fixed_id}/"
                f"seed-{GLUE_R8_SLACK_STAGE2_SEED}"
            ),
        }
    ]
    for label, rho in zip(labels, rho_values, strict=True):
        candidate_id = f"full-sla-{label}-eta002"
        stage2_arms.append(
            {
                **shared,
                "id": candidate_id,
                "candidate_id": candidate_id,
                "arm_id": (
                    f"{best_arm['setting_id']}--{candidate_id}--"
                    f"seed{GLUE_R8_SLACK_STAGE2_SEED}"
                ),
                "method": "slaclip",
                "initial_c": float(best["fixed_C"]),
                "rho": rho,
                "eta": GLUE_R8_SLACK_PRIMARY_ETA,
                "role": "slaclip_target_candidate",
                "stage": 2,
                "rho_source_quantile": label,
                "lane": 0,
                "seed": GLUE_R8_SLACK_STAGE2_SEED,
                "steps": GLUE_R8_SLACK_STEPS,
                "eval_limit": 0,
                "relative_root": (
                    f"runs/{best_arm['setting_id']}/{candidate_id}/"
                    f"seed-{GLUE_R8_SLACK_STAGE2_SEED}"
                ),
            }
        )
    central_rho = rho_values[2]
    controls = (
        (
            "full-sla-q50-eta005-control",
            float(best["fixed_C"]),
            GLUE_R8_SLACK_FAST_ETA,
            "controller_speed_control",
        ),
        (
            "full-sla-q50-c0half-eta002-control",
            max(0.1, float(best["fixed_C"]) / 2.0),
            GLUE_R8_SLACK_PRIMARY_ETA,
            "initial_C_sensitivity_control",
        ),
    )
    primary_template = stage2_arms[3]
    for candidate_id, initial_c, eta, role in controls:
        arm = dict(primary_template)
        arm.update(
            {
                "id": candidate_id,
                "candidate_id": candidate_id,
                "arm_id": (
                    f"{best_arm['setting_id']}--{candidate_id}--"
                    f"seed{GLUE_R8_SLACK_STAGE2_SEED}"
                ),
                "initial_c": initial_c,
                "rho": central_rho,
                "eta": eta,
                "role": role,
                "relative_root": (
                    f"runs/{best_arm['setting_id']}/{candidate_id}/"
                    f"seed-{GLUE_R8_SLACK_STAGE2_SEED}"
                ),
            }
        )
        stage2_arms.append(arm)
    if len(stage2_arms) != 8 or len({arm["arm_id"] for arm in stage2_arms}) != 8:
        raise CampaignError("rank-8 Stage-2 recipe did not produce eight unique arms")

    stage2_manifest = {**manifest, "arms": stage2_arms}
    stage2_plan_data = _plan_bytes(stage2_manifest, 0, include_all=True)
    stage2_plan_sha256 = hashlib.sha256(stage2_plan_data).hexdigest()
    lock_payload = {
        "schema_version": 1,
        "profile": "glue-r8-slack-screen",
        "code_sha": manifest["code_sha"],
        "manifest_sha256": _file_sha256(manifest_path),
        "stage1_seed": GLUE_R8_SLACK_STAGE1_SEED,
        "stage2_seed": GLUE_R8_SLACK_STAGE2_SEED,
        "stage2_seed_is_fresh": True,
        "selection_metric": "step_200_response_only_mean_per_record_loss",
        "ranking_rule": (
            "ascending endpoint loss, then ascending fixed C, then candidate id"
        ),
        "stage1_ranking": ranked,
        "best_fixed": best,
        "stage1_fixed_winner_at_grid_min": winner_at_grid_min,
        "stage1_fixed_winner_at_grid_max": winner_at_grid_max,
        "stage1_fixed_winner_at_search_boundary": winner_at_boundary,
        "stage1_boundary_warning": (
            "best fixed C is a search-grid boundary; this screen may finish but "
            "must not advance to later confirmation"
            if winner_at_boundary else None
        ),
        "burn_in_steps_excluded": [1, 50],
        "conditional_clip_proxy_records": len(conditional),
        "rho_quantiles": {
            label: value for label, value in zip(labels, rho_values, strict=True)
        },
        "rho_bounds": list(GLUE_R8_SLACK_RHO_BOUNDS),
        "rho_values_unique_after_clamp": True,
        "stage2_arms": stage2_arms,
        "stage2_plan_sha256": stage2_plan_sha256,
        "primary_endpoint": "step_200_response_only_mean_per_record_loss",
        "secondary_endpoints": {
            "full_normalized_auc": "trapezoid_steps_0_to_200_divided_by_200",
            "late_window_normalized_auc": (
                "trapezoid_steps_100_to_200_divided_by_100"
            ),
        },
        "official_task_evaluation_used": False,
        "NON_PRIVATE_DATA_DEPENDENT_SELECTION": True,
        "privacy_scope": manifest["glue_r8_slack_screen"]["privacy_scope"],
        "inference": "two_seed_exploratory_rank8_slack_screen",
    }
    lock_path = root / "selection" / "r8_slack_stage1_lock.json"
    _with_sha(lock_path, _json_bytes(lock_payload))
    _with_sha(root / "plans" / "stage2-slaclip.tsv", stage2_plan_data)
    print(
        f"locked_rank8_best_fixed={best['candidate_id']} C={best['fixed_C']} "
        f"boundary={str(winner_at_boundary).lower()} "
        f"stage2_arms={len(stage2_arms)}"
    )


def _normalized_trapezoid_auc(
    losses_by_step: dict[int, float], start: int, end: int
) -> float:
    points = sorted(step for step in losses_by_step if start <= step <= end)
    if not points or points[0] != start or points[-1] != end or end <= start:
        raise CampaignError(
            f"validation AUC interval is incomplete: start={start}, end={end}, "
            f"points={points}"
        )
    area = 0.0
    for left, right in zip(points, points[1:], strict=False):
        area += (right - left) * (
            losses_by_step[left] + losses_by_step[right]
        ) / 2.0
    return area / float(end - start)


def analyze(root: Path) -> None:
    manifest_path = root / "plans" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    analysis_arms = list(manifest.get("arms", []))
    high_c_lock = None
    r8_slack_lock = None
    if manifest.get("profile") == "glue-high-c-refinement":
        # Recompute the dynamic provenance gate at analysis time.  The
        # immutable writer accepts byte-identical replay but refuses any change
        # to the Stage-1-derived lock, its sidecar, or the Stage-2 plan.  (The
        # preceding-screen source was already verified twice by ``prepare``.)
        lock_high_c_refinement(root)
        lock_path = root / "selection" / "high_c_stage1_lock.json"
        stage2_plan_path = root / "plans" / "stage2-slaclip.tsv"
        try:
            high_c_lock = json.loads(lock_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise CampaignError("high-C refinement lacks its Stage-1 lock") from exc
        if (
            high_c_lock.get("profile") != manifest["profile"]
            or high_c_lock.get("code_sha") != manifest["code_sha"]
            or high_c_lock.get("manifest_sha256") != _file_sha256(manifest_path)
            or not isinstance(high_c_lock.get("stage2_arms"), list)
            or len(high_c_lock["stage2_arms"]) != 7
            or high_c_lock.get("stage2_plan_sha256")
            != _file_sha256(stage2_plan_path)
        ):
            raise CampaignError("high-C refinement selection lock is stale")
        analysis_arms.extend(high_c_lock["stage2_arms"])
        if len(analysis_arms) != 12 or len(
            {arm["arm_id"] for arm in analysis_arms}
        ) != 12:
            raise CampaignError("high-C refinement must analyze 12 unique arms")
    elif manifest.get("profile") == "glue-r8-slack-screen":
        lock_glue_r8_slack_screen(root)
        lock_path = root / "selection" / "r8_slack_stage1_lock.json"
        stage2_plan_path = root / "plans" / "stage2-slaclip.tsv"
        try:
            r8_slack_lock = json.loads(lock_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise CampaignError("rank-8 slack screen lacks its Stage-1 lock") from exc
        if (
            r8_slack_lock.get("profile") != manifest["profile"]
            or r8_slack_lock.get("code_sha") != manifest["code_sha"]
            or r8_slack_lock.get("manifest_sha256") != _file_sha256(manifest_path)
            or r8_slack_lock.get("stage1_seed") != GLUE_R8_SLACK_STAGE1_SEED
            or r8_slack_lock.get("stage2_seed") != GLUE_R8_SLACK_STAGE2_SEED
            or r8_slack_lock.get("stage2_seed_is_fresh") is not True
            or not isinstance(r8_slack_lock.get("stage2_arms"), list)
            or len(r8_slack_lock["stage2_arms"]) != 8
            or r8_slack_lock.get("stage2_plan_sha256")
            != _file_sha256(stage2_plan_path)
        ):
            raise CampaignError("rank-8 slack-screen selection lock is stale")
        analysis_arms.extend(r8_slack_lock["stage2_arms"])
        if len(analysis_arms) != 14 or len(
            {arm["arm_id"] for arm in analysis_arms}
        ) != 14:
            raise CampaignError("rank-8 slack screen must analyze 14 unique arms")
    results = []
    trajectory_rows = []
    validation_curve_rows = []
    focused_artifact_hashes: dict[str, dict[str, str]] = {}
    for arm in analysis_arms:
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
        focused_screen = manifest.get("profile") in {
            "glue-slaclip-screen", "glue-high-c-refinement",
            "glue-r8-slack-screen",
        }
        raw_records = None
        if focused_screen:
            status, raw_records, _validated_curves, artifact_hashes = (
                _load_focused_arm(root, arm, manifest["code_sha"])
            )
            focused_artifact_hashes[arm["arm_id"]] = artifact_hashes
        validation_loss = None
        validation_auc = None
        validation_late_auc = None
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
                            "NON_PRIVATE_SELECTION_METRIC": True,
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
            expected_curve_steps = _expected_validation_curve_steps(int(arm["steps"]))
            if curve_steps != expected_curve_steps:
                raise CampaignError(
                    f"focused-screen validation curve has wrong steps for {arm['arm_id']}: "
                    f"{sorted(curve_steps)}"
                )
            if not math.isclose(
                curve_loss_by_step[int(arm["steps"])], validation_loss,
                rel_tol=1e-9, abs_tol=1e-8
            ):
                raise CampaignError(
                    f"endpoint validation and curve disagree: {arm['arm_id']}"
                )
            validation_auc = _normalized_trapezoid_auc(
                curve_loss_by_step, 0, int(arm["steps"])
            )
            late_start = max(0, int(arm["steps"]) - 100)
            validation_late_auc = _normalized_trapezoid_auc(
                curve_loss_by_step, late_start, int(arm["steps"])
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
            "raw_reference_conditional_clip_fraction_valid",
            "raw_reference_expected_batch_size_normalization",
            "raw_reference_slack_indicator_last",
            "raw_reference_slaclip_num_slots",
            "slack_unclipped_proxy", "slack_clipped_proxy",
            "slack_indicator_noise_std", "slaclip_gamma_t",
            "raw_slack_indicator_noise_residual_l2",
            "raw_slack_indicator_noise_residual_rmse",
            "raw_slack_indicator_noise_residual_first_coordinate",
            "slaclip_eta", "slaclip_num_slots", "slaclip_c_min",
            "slaclip_c_max", "slaclip_beta",
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
            unclipped_signal = raw_record.get("raw_unclipped_signal_norm")
            clipped_signal = raw_record.get("raw_clipped_signal_norm")
            clipping_bias = raw_record.get("raw_clipping_bias_norm")
            current_c = raw_record.get("dp_clip_threshold")
            next_c = raw_record.get("dp_next_clip_threshold")
            quantiles = raw_record.get("raw_global_norm_quantiles")
            if not isinstance(quantiles, dict):
                quantiles = {}
            signal_retention = None
            clipping_bias_ratio = None
            if unclipped_signal is not None:
                denominator = _finite(
                    unclipped_signal,
                    f"{arm['arm_id']}:raw_unclipped_signal_norm",
                )
                if denominator != 0.0:
                    if clipped_signal is not None:
                        signal_retention = _finite(
                            clipped_signal,
                            f"{arm['arm_id']}:raw_clipped_signal_norm",
                        ) / denominator
                    if clipping_bias is not None:
                        clipping_bias_ratio = _finite(
                            clipping_bias,
                            f"{arm['arm_id']}:raw_clipping_bias_norm",
                        ) / denominator
            threshold_delta = None
            threshold_ratio = None
            if current_c is not None and next_c is not None:
                current_c_value = _finite(
                    current_c, f"{arm['arm_id']}:dp_clip_threshold"
                )
                next_c_value = _finite(
                    next_c, f"{arm['arm_id']}:dp_next_clip_threshold"
                )
                threshold_delta = next_c_value - current_c_value
                if current_c_value != 0.0:
                    threshold_ratio = next_c_value / current_c_value
            trajectory_rows.append({
                "NON_PRIVATE_TELEMETRY": True,
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
                "clip_threshold_delta": threshold_delta,
                "clip_threshold_ratio": threshold_ratio,
                "raw_signal_retention_ratio": signal_retention,
                "raw_clipping_bias_ratio": clipping_bias_ratio,
                "raw_global_norm_q10": quantiles.get("0.1"),
                "raw_global_norm_q25": quantiles.get("0.25"),
                "raw_global_norm_q50": quantiles.get("0.5"),
                "raw_global_norm_q75": quantiles.get("0.75"),
                "raw_global_norm_q90": quantiles.get("0.9"),
                "raw_global_norm_q95": quantiles.get("0.95"),
                "raw_global_norm_q99": quantiles.get("0.99"),
            })
        optional_series = lambda name: [
            _finite(record.get(name), f"{arm['arm_id']}:{name}")
            for record in ordered_raw_records
            if record.get(name) is not None
        ]
        conditional_clip_values = optional_series(
            "raw_reference_conditional_clip_fraction"
        )
        controller_error_values = optional_series("slaclip_controller_error")
        observed_cdf_values = optional_series("slaclip_observed_unclipped_proxy")
        target_cdf_values = optional_series("slaclip_target_unclipped_proxy")
        slack_noise_values = optional_series("slack_indicator_noise_std")
        clip_threshold_values = optional_series("dp_clip_threshold")
        results.append(
            {
                "NON_PRIVATE_TELEMETRY": True,
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
                "seed": arm["seed"],
                "selection_metric": selection_metric,
                "selection_value": selection_value,
                "selection_score": selection_score,
                "task_average": task_average,
                "public_validation_loss": validation_loss,
                "full_normalized_validation_loss_auc": validation_auc,
                "late_window_normalized_validation_loss_auc": validation_late_auc,
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
                "conditional_clip_fraction_p10": (
                    _quantile(conditional_clip_values, 0.1)
                    if conditional_clip_values else None
                ),
                "conditional_clip_fraction_median": (
                    _quantile(conditional_clip_values, 0.5)
                    if conditional_clip_values else None
                ),
                "conditional_clip_fraction_p90": (
                    _quantile(conditional_clip_values, 0.9)
                    if conditional_clip_values else None
                ),
                "clip_threshold_mean": metric("dp_clip_threshold"),
                "clip_threshold_last": metric("dp_clip_threshold", "last"),
                "clip_threshold_min": (
                    min(clip_threshold_values) if clip_threshold_values else None
                ),
                "clip_threshold_max": (
                    max(clip_threshold_values) if clip_threshold_values else None
                ),
                "signal_to_noise_mean": metric("raw_signal_to_noise_ratio"),
                "bias_to_noise_mean": metric("raw_clipping_bias_to_noise_ratio"),
                "signal_retention_mean": metric("raw_signal_retention_ratio"),
                "clipping_bias_mean": metric("raw_clipping_bias_norm"),
                "realized_noise_mean": metric("raw_realized_noise_norm"),
                "bias_noise_mse_proxy_mean": metric("raw_bias_noise_squared_error_proxy"),
                "controller_error_mean": (
                    sum(controller_error_values) / len(controller_error_values)
                    if controller_error_values else None
                ),
                "controller_error_abs_mean": (
                    sum(abs(value) for value in controller_error_values)
                    / len(controller_error_values)
                    if controller_error_values else None
                ),
                "controller_error_abs_p90": (
                    _quantile([abs(value) for value in controller_error_values], 0.9)
                    if controller_error_values else None
                ),
                "observed_unclipped_cdf_mean": (
                    sum(observed_cdf_values) / len(observed_cdf_values)
                    if observed_cdf_values else None
                ),
                "target_unclipped_cdf_mean": (
                    sum(target_cdf_values) / len(target_cdf_values)
                    if target_cdf_values else None
                ),
                "slack_indicator_noise_std_mean": (
                    sum(slack_noise_values) / len(slack_noise_values)
                    if slack_noise_values else None
                ),
                "C_hit_min_steps": sum(
                    bool(record.get("slaclip_c_hit_min"))
                    for record in ordered_raw_records
                ),
                "C_hit_max_steps": sum(
                    bool(record.get("slaclip_c_hit_max"))
                    for record in ordered_raw_records
                ),
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
        paired_rank8_fixed = (
            manifest.get("profile") != "glue-r8-slack-screen"
            or row["candidate_role"] == "fresh_fixed_comparator"
        )
        if row["method"] == "baseline" and paired_rank8_fixed:
            best_fixed[row["setting_id"]] = max(
                best_fixed.get(row["setting_id"], float("-inf")), row["selection_score"]
            )
    regime_groups: dict[str, list[dict[str, Any]]] = {}
    for row in results:
        if (
            row["method"] != "slaclip"
            or row["candidate_role"] in {
                "initial_C_sensitivity_control", "controller_speed_control"
            }
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
            "NON_PRIVATE_TELEMETRY": True,
            "clip_regime_bin": label,
            "slaclip_arms": len(deltas),
            "mean_delta_vs_setting_best_fixed": sum(deltas) / len(deltas) if deltas else None,
            "win_rate_vs_setting_best_fixed": sum(value > 0 for value in deltas) / len(deltas) if deltas else None,
            "mean_small_gradient_proxy": (
                sum(row["small_gradient_proxy_mean"] for row in members) / len(members)
                if members else None
            ),
        })
    _with_sha(out / "paper_coverage_summary.json", _json_bytes({
        "schema_version": 2,
        "NON_PRIVATE_TELEMETRY": True,
        "warning": (
            "Contains exact per-record-gradient-derived research statistics; "
            "not a differentially private release artifact."
        ),
        "rows": results,
    }))
    regime_fields = list(regime_rows[0])
    regime_csv = [",".join(regime_fields)]
    for row in regime_rows:
        regime_csv.append(",".join("" if row[key] is None else str(row[key]) for key in regime_fields))
    _with_sha(out / "clipping_regime_summary.csv", ("\n".join(regime_csv) + "\n").encode())
    _with_sha(out / "clipping_regime_summary.json", _json_bytes({
        "schema_version": 1,
        "NON_PRIVATE_TELEMETRY": True,
        "inference": (
            "exploratory_two_seed_staged_descriptive_only"
            if manifest.get("profile") == "glue-r8-slack-screen"
            else "exploratory_one_seed_descriptive_only"
        ),
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
    elif manifest.get("profile") == "glue-high-c-refinement":
        fixed_ranked = sorted(
            (
                row for row in results
                if row["candidate_role"] == "high_C_fixed_candidate"
            ),
            key=lambda row: (
                -row["selection_score"], row["initial_C"], row["candidate"]
            ),
        )
        slaclip_ranked = sorted(
            (
                row for row in results
                if row["candidate_role"] == "slaclip_target_candidate"
            ),
            key=lambda row: (-row["selection_score"], row["candidate"]),
        )
        controls = [
            row for row in results
            if row["candidate_role"] in {
                "controller_speed_control", "initial_C_sensitivity_control"
            }
        ]
        if len(fixed_ranked) != 5 or len(slaclip_ranked) != 5 or len(controls) != 2:
            raise CampaignError(
                "high-C refinement must rank five fixed and five primary "
                "SlaClip candidates plus two controls"
            )
        if high_c_lock is None:
            raise CampaignError("high-C analyzer lost its selection lock")
        locked_best = high_c_lock.get("best_fixed", {})
        if (
            locked_best.get("candidate_id") != fixed_ranked[0]["candidate"]
            or not math.isclose(
                _finite(locked_best.get("endpoint_loss"), "locked best endpoint"),
                fixed_ranked[0]["public_validation_loss"],
                rel_tol=1e-9,
                abs_tol=1e-8,
            )
        ):
            raise CampaignError("final fixed ranking disagrees with Stage-1 lock")
        best_fixed_row = fixed_ranked[0]
        best_slaclip_row = slaclip_ranked[0]
        selection_payload = {
            "schema_version": 1,
            "inference": "single_seed_200_step_exploratory_boundary_refinement",
            "selection_seed": GLUE_HIGH_C_REFINEMENT_SEED,
            "selection_metric": "step_200_response_only_mean_per_record_loss",
            "ranking_rule": "ascending endpoint loss, then candidate id",
            "primary_endpoint": "step_200_response_only_mean_per_record_loss",
            "secondary_endpoints": {
                "full_normalized_validation_loss_auc": (
                    "trapezoid steps 0 through 200 divided by 200"
                ),
                "late_window_normalized_validation_loss_auc": (
                    "trapezoid steps 100 through 200 divided by 100"
                ),
                "selection_role": "diagnostic_only_not_used_to_choose_winner",
            },
            "task_test_or_official_glue_evaluation_used": False,
            "privacy_scope": manifest["glue_high_c_refinement"]["privacy_scope"],
            "stage1_lock_sha256": _file_sha256(
                root / "selection" / "high_c_stage1_lock.json"
            ),
            "stage2_plan_sha256": _file_sha256(
                root / "plans" / "stage2-slaclip.tsv"
            ),
            "arm_artifact_sha256": focused_artifact_hashes,
            "best_fixed": best_fixed_row,
            "best_slaclip": best_slaclip_row,
            "best_slaclip_loss_delta_vs_best_fixed": (
                best_slaclip_row["public_validation_loss"]
                - best_fixed_row["public_validation_loss"]
            ),
            "slaclip_beats_best_fixed_primary": (
                best_slaclip_row["public_validation_loss"]
                < best_fixed_row["public_validation_loss"]
            ),
            "fixed_winner_at_allowed_C_max": math.isclose(
                float(best_fixed_row["initial_C"]), 15.0,
                rel_tol=0.0, abs_tol=1e-12,
            ),
            "fixed_ranking": fixed_ranked,
            "slaclip_ranking": slaclip_ranked,
            "controls": controls,
            "fresh_seed_confirmation_gate": (
                "advance only if best Full SlaClip has lower primary endpoint "
                "loss than best fixed; secondary endpoints remain diagnostic"
            ),
        }
        _with_sha(
            out / "glue_high_c_refinement_ranking.json",
            _json_bytes(selection_payload),
        )
    elif manifest.get("profile") == "glue-r8-slack-screen":
        stage1_ranked = sorted(
            (
                row for row in results
                if row["candidate_role"] == "slack_fixed_candidate"
            ),
            key=lambda row: (
                -row["selection_score"], row["initial_C"], row["candidate"]
            ),
        )
        fresh_fixed_rows = [
            row for row in results
            if row["candidate_role"] == "fresh_fixed_comparator"
        ]
        slaclip_ranked = sorted(
            (
                row for row in results
                if row["candidate_role"] == "slaclip_target_candidate"
            ),
            key=lambda row: (-row["selection_score"], row["candidate"]),
        )
        controls = [
            row for row in results
            if row["candidate_role"] in {
                "controller_speed_control", "initial_C_sensitivity_control"
            }
        ]
        if (
            len(stage1_ranked) != 6
            or len(fresh_fixed_rows) != 1
            or len(slaclip_ranked) != 5
            or len(controls) != 2
        ):
            raise CampaignError(
                "rank-8 slack screen must rank six Stage-1 fixed arms, one "
                "fresh fixed comparator, five primary SlaClip arms, and two controls"
            )
        if r8_slack_lock is None:
            raise CampaignError("rank-8 analyzer lost its selection lock")
        locked_best = r8_slack_lock.get("best_fixed", {})
        if (
            locked_best.get("candidate_id") != stage1_ranked[0]["candidate"]
            or not math.isclose(
                _finite(
                    locked_best.get("endpoint_loss"),
                    "locked rank-8 Stage-1 endpoint",
                ),
                stage1_ranked[0]["public_validation_loss"],
                rel_tol=1e-9,
                abs_tol=1e-8,
            )
        ):
            raise CampaignError(
                "rank-8 final Stage-1 ranking disagrees with its lock"
            )
        fresh_fixed = fresh_fixed_rows[0]
        best_slaclip = slaclip_ranked[0]
        if (
            fresh_fixed["seed"] != GLUE_R8_SLACK_STAGE2_SEED
            or best_slaclip["seed"] != GLUE_R8_SLACK_STAGE2_SEED
            or fresh_fixed["initial_C"] != float(locked_best["fixed_C"])
        ):
            raise CampaignError("rank-8 Stage-2 fresh comparator identity mismatch")
        endpoint_pass = (
            best_slaclip["public_validation_loss"]
            < fresh_fixed["public_validation_loss"]
        )
        full_auc_pass = (
            best_slaclip["full_normalized_validation_loss_auc"]
            <= fresh_fixed["full_normalized_validation_loss_auc"]
        )
        late_auc_pass = (
            best_slaclip["late_window_normalized_validation_loss_auc"]
            <= fresh_fixed["late_window_normalized_validation_loss_auc"]
        )
        performance_gate = endpoint_pass and full_auc_pass and late_auc_pass
        boundary_block = (
            r8_slack_lock.get("stage1_fixed_winner_at_search_boundary") is True
        )
        block_reasons = []
        if not endpoint_pass:
            block_reasons.append("best_slaclip_endpoint_not_strictly_lower")
        if not full_auc_pass:
            block_reasons.append("best_slaclip_full_auc_higher")
        if not late_auc_pass:
            block_reasons.append("best_slaclip_late_auc_higher")
        if boundary_block:
            block_reasons.append("stage1_fixed_winner_at_search_boundary")
        selection_payload = {
            "schema_version": 1,
            "inference": "two_seed_200_step_exploratory_rank8_slack_screen",
            "stage1_seed": GLUE_R8_SLACK_STAGE1_SEED,
            "stage2_seed": GLUE_R8_SLACK_STAGE2_SEED,
            "stage2_seed_is_fresh": True,
            "selection_metric": "step_200_response_only_mean_per_record_loss",
            "ranking_rule": "ascending endpoint loss, then candidate id",
            "primary_endpoint": "step_200_response_only_mean_per_record_loss",
            "secondary_endpoints": {
                "full_normalized_validation_loss_auc": (
                    "trapezoid steps 0 through 200 divided by 200"
                ),
                "late_window_normalized_validation_loss_auc": (
                    "trapezoid steps 100 through 200 divided by 100"
                ),
                "gate_role": "both must be no worse than fresh fixed",
            },
            "task_test_or_official_glue_evaluation_used": False,
            "privacy_scope": manifest["glue_r8_slack_screen"]["privacy_scope"],
            "stage1_lock_sha256": _file_sha256(
                root / "selection" / "r8_slack_stage1_lock.json"
            ),
            "stage2_plan_sha256": _file_sha256(
                root / "plans" / "stage2-slaclip.tsv"
            ),
            "arm_artifact_sha256": focused_artifact_hashes,
            "stage1_best_fixed": stage1_ranked[0],
            "stage1_fixed_ranking": stage1_ranked,
            "stage1_fixed_winner_at_grid_min": r8_slack_lock.get(
                "stage1_fixed_winner_at_grid_min"
            ),
            "stage1_fixed_winner_at_grid_max": r8_slack_lock.get(
                "stage1_fixed_winner_at_grid_max"
            ),
            "stage1_fixed_winner_at_search_boundary": boundary_block,
            "stage1_boundary_warning": r8_slack_lock.get(
                "stage1_boundary_warning"
            ),
            "fresh_fixed": fresh_fixed,
            "best_slaclip": best_slaclip,
            "slaclip_ranking": slaclip_ranked,
            "controls": controls,
            "best_slaclip_loss_delta_vs_fresh_fixed": (
                best_slaclip["public_validation_loss"]
                - fresh_fixed["public_validation_loss"]
            ),
            "primary_gate": {
                "endpoint_strictly_lower": endpoint_pass,
                "full_auc_not_worse": full_auc_pass,
                "late_auc_not_worse": late_auc_pass,
                "performance_gate_passed": performance_gate,
                "boundary_blocked": boundary_block,
                "confirmation_allowed": performance_gate and not boundary_block,
                "block_reasons": block_reasons,
            },
            "fresh_seed_confirmation_gate": (
                "advance only when the endpoint is strictly lower and both full "
                "and late AUC are no worse than the fresh fixed comparator, and "
                "the Stage-1 fixed winner is not a search-grid boundary"
            ),
        }
        _with_sha(
            out / "glue_r8_slack_screen_ranking.json",
            _json_bytes(selection_payload),
        )
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
            "glue-high-c-refinement", "glue-r8-slack-screen",
        ),
        default="paper-breadth",
    )
    prep.add_argument("--model-12b-revision")
    lock = sub.add_parser("lock-high-c-refinement")
    lock.add_argument("--campaign-root", required=True, type=Path)
    lock_r8 = sub.add_parser("lock-r8-slack-screen")
    lock_r8.add_argument("--campaign-root", required=True, type=Path)
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
    elif args.command == "lock-high-c-refinement":
        lock_high_c_refinement(args.campaign_root)
    elif args.command == "lock-r8-slack-screen":
        lock_glue_r8_slack_screen(args.campaign_root)
    else:
        analyze(args.campaign_root)


if __name__ == "__main__":
    main()
