from __future__ import annotations

import csv
import json
import math
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "summarize_telemetry.py"


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")


def test_summarizer_flattens_metrics_merges_safe_log_and_handles_missing_fields(tmp_path: Path) -> None:
    raw_path = tmp_path / "NON_PRIVATE_train_log.jsonl"
    safe_path = tmp_path / "train_log.jsonl"
    csv_path = tmp_path / "steps.csv"
    json_path = tmp_path / "summary.json"
    common = {
        "NON_PRIVATE_TELEMETRY": True,
        "run_id": "run-a",
        "config_fingerprint": "abc123",
        "method": "slaclip",
        "privacy": "dp",
        "dataset": "math10k",
    }
    _write_jsonl(
        raw_path,
        [
            {
                **common,
                "step": 1,
                "loss_mean": 2.0,
                "dp_clip_threshold": 1.0,
                "dp_next_clip_threshold": 1.1,
                "slaclip_gamma_t": 0.6,
                "slaclip_c_hit_min": False,
                "slaclip_c_hit_max": False,
                "raw_global_norm_quantiles": {"0.1": 0.2, "0.5": 0.8, "0.99": 3.0},
                "raw_clip_fraction": 0.4,
                "raw_clip_coefficient_mean": 0.8,
                "raw_clip_coefficient_min": 0.2,
                "raw_unclipped_signal_norm": 4.0,
                "raw_clipped_signal_norm": 3.0,
                "raw_clipping_bias_norm": 1.0,
                "raw_realized_noise_norm": 2.0,
                "raw_signal_to_noise_ratio": 1.5,
                "raw_slack_indicator": [0.2, 0.4],
                "raw_slack_indicator_noise_residual": [0.1, -0.1],
                "raw_slack_indicator_noise_residual_l2": 0.141421356,
                "raw_slack_indicator_noise_residual_rmse": 0.1,
                "raw_slack_indicator_noise_residual_first_coordinate": 0.1,
                "raw_unclipped_clipped_cosine": 0.95,
                "raw_clipped_noisy_cosine": 0.25,
                "raw_clipping_bias_to_noise_ratio": 0.5,
                "raw_bias_noise_squared_error_proxy": 5.0,
                "dp_noisy_tangent_gradient_norm": 3.5,
                "dp_factor_product_update_norm": 0.25,
                "eps_spent": 1.0,
            },
            {
                **common,
                "step": 2,
                "loss_mean": 1.8,
                "dp_clip_threshold": 1.1,
                "dp_next_clip_threshold": 1.2,
                "raw_clip_fraction": 0.3,
                "slaclip_c_hit_min": True,
                "slaclip_c_hit_max": False,
            },
            # A resumed legacy log may contain a duplicate; default policy keeps the last.
            {
                **common,
                "step": 2,
                "loss_mean": 1.7,
                "dp_clip_threshold": 1.1,
                "dp_next_clip_threshold": 1.25,
                "raw_clip_fraction": 0.25,
                "slaclip_c_hit_min": True,
                "slaclip_c_hit_max": False,
                "raw_global_norm_quantiles": {"0.5": 0.7},
                "eps_spent": 1.6,
            },
            {
                **common,
                "step": 4,
                "loss_mean": 1.5,
                "dp_clip_threshold": 1.25,
                "dp_next_clip_threshold": 1.3,
            },
        ],
    )
    _write_jsonl(
        safe_path,
        [
            {
                "run_id": "run-a",
                "config_fingerprint": "abc123",
                "method": "slaclip",
                "privacy": "dp",
                "dataset": "math10k",
                "step": 1,
                "dp_noise_multiplier": 0.9,
                "slaclip_controller": "slaclip",
                "slaclip_beta": 0.75,
                "slaclip_small_gradient_proxy_noisy": 0.6,
                "slaclip_remaining_mass_proxy_noisy": 0.4,
                "slaclip_target_unclipped_proxy_preprojection": 0.7,
                "slaclip_target_unclipped_proxy": 0.7,
                "slaclip_target_clipped_proxy": 0.3,
                "slaclip_observed_unclipped_proxy": 0.4,
            },
            {
                "run_id": "run-a",
                "config_fingerprint": "abc123",
                "method": "slaclip",
                "privacy": "dp",
                "dataset": "math10k",
                "step": 2,
                "dp_noise_multiplier": 0.9,
                "dp_update_clip_coef_min": 0.95,
            },
            {
                "run_id": "run-a",
                "config_fingerprint": "abc123",
                "method": "slaclip",
                "privacy": "dp",
                "dataset": "math10k",
                "step": 3,
                "dp_noise_multiplier": 0.9,
            },
        ],
    )

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            str(raw_path),
            "--safe-log",
            str(safe_path),
            "--csv-out",
            str(csv_path),
            "--json-out",
            str(json_path),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "NON-PRIVATE" in result.stdout

    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["step"] for row in rows] == ["1", "2", "4"]
    assert rows[1]["loss_mean"] == "1.7"
    assert rows[0]["raw_global_norm_q10"] == "0.2"
    assert rows[1]["raw_global_norm_q10"] == ""
    assert rows[1]["raw_global_norm_q50"] == "0.7"
    assert rows[1]["dp_noise_multiplier"] == "0.9"
    assert rows[1]["dp_update_clip_coef_min"] == "0.95"
    assert float(rows[0]["raw_signal_retention_ratio"]) == 0.75
    assert float(rows[0]["raw_clipping_bias_ratio"]) == 0.25
    assert json.loads(rows[0]["raw_slack_indicator_json"]) == [0.2, 0.4]
    assert json.loads(rows[0]["raw_slack_indicator_noise_residual_json"]) == [
        0.1,
        -0.1,
    ]
    assert float(rows[0]["raw_unclipped_clipped_cosine"]) == 0.95
    assert float(rows[0]["raw_clipped_noisy_cosine"]) == 0.25
    assert float(rows[0]["raw_bias_noise_squared_error_proxy"]) == 5.0
    assert math.isclose(float(rows[0]["clip_threshold_delta"]), 0.1)
    assert math.isclose(float(rows[1]["epsilon_increment"]), 0.6)
    assert float(rows[0]["slaclip_target_non_small_clip_fraction"]) == 0.75
    assert float(rows[0]["raw_clip_fraction_reference_target"]) == 0.3
    assert rows[0]["raw_clip_fraction_error_target_kind"] == (
        "full_dynamic_clipped_proxy"
    )
    assert math.isclose(float(rows[0]["raw_clip_fraction_error"]), 0.1)
    assert math.isclose(float(rows[0]["slaclip_controller_error"]), 0.3)

    summary = json.loads(json_path.read_text(encoding="utf-8"))
    assert summary["NON_PRIVATE_TELEMETRY"] is True
    assert summary["run_identity"]["run_id"] == "run-a"
    assert summary["source"]["raw_duplicate_records"] == 1
    assert summary["source"]["safe_steps_without_raw_record"] == 1
    assert summary["steps"] == {
        "count": 3,
        "first": 1,
        "last": 4,
        "missing_count": 1,
        "missing": [3],
    }
    assert summary["metrics"]["loss_mean"]["last"] == 1.5
    assert summary["metrics"]["raw_slack_indicator_noise_residual_rmse"][
        "mean"
    ] == 0.1
    assert summary["metrics"]["raw_clip_fraction_error"]["mean"] == pytest.approx(
        0.1
    )
    assert summary["field_coverage"]["slaclip_gamma_t"] == {"present": 1, "missing": 2}
    assert summary["boolean_metrics"]["slaclip_c_hit_min"] == {
        "count": 2,
        "missing": 1,
        "true_count": 1,
        "false_count": 1,
        "true_rate": 0.5,
    }
    assert summary["boolean_metrics"]["slaclip_c_hit_max"]["true_rate"] == 0.0


def test_summarizer_rejects_unmarked_raw_record(tmp_path: Path) -> None:
    raw_path = tmp_path / "raw.jsonl"
    _write_jsonl(raw_path, [{"step": 1, "loss_mean": 1.0}])
    result = subprocess.run(
        [sys.executable, str(SCRIPT), str(raw_path), "--format", "json"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "NON_PRIVATE_TELEMETRY=true" in result.stderr
    assert not (tmp_path / "telemetry_summary.json").exists()


def test_summarizer_duplicate_error_policy_is_strict(tmp_path: Path) -> None:
    raw_path = tmp_path / "raw.jsonl"
    _write_jsonl(
        raw_path,
        [
            {"NON_PRIVATE_TELEMETRY": True, "step": 1},
            {"NON_PRIVATE_TELEMETRY": True, "step": 1},
        ],
    )
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            str(raw_path),
            "--duplicate-policy",
            "error",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "duplicate step 1" in result.stderr
