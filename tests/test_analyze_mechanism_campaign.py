from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "analyze_mechanism_campaign.py"
TASKS = ("gsm8k", "AQuA", "mawps", "SVAMP")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _safe_record(*, step: int, fingerprint: str, threshold: float) -> dict:
    return {
        "telemetry_schema_version": 3,
        "run_id": f"run-{fingerprint}",
        "config_fingerprint": fingerprint,
        "method": "slaclip",
        "privacy": "dp",
        "telemetry_mode": "research_raw",
        "step": step,
        "dp_clip_threshold": threshold,
        "dp_next_clip_threshold": threshold + 0.1,
        "eps_spent": float(step),
    }


def _telemetry_row(
    *,
    step: int,
    fingerprint: str,
    method: str,
    threshold: float,
    loss: float,
    exact_oracle: bool = False,
) -> dict:
    row = {
        "NON_PRIVATE_TELEMETRY": "True",
        "telemetry_schema_version": 3,
        "run_id": f"run-{fingerprint}",
        "config_fingerprint": fingerprint,
        "method": method,
        "privacy": "dp",
        "step": step,
        "loss_mean": loss,
        "dp_clip_threshold": threshold,
        "dp_next_clip_threshold": threshold + (0.1 if method == "slaclip" else 0.0),
        "dp_expected_batch_size": 2.0,
        "dp_std_per_factor": 0.1 * threshold,
        "eps_spent": float(step),
        "raw_realized_batch_size": 2,
        "raw_clip_fraction": 0.5 if method == "slaclip" else 1.0,
        "raw_clip_coefficient_mean": 0.7,
        "raw_global_norm_q50": 1.0,
        "raw_global_norm_q95": 1.8,
        "raw_global_norm_q99": 1.9,
        "raw_unclipped_signal_norm": 2.0,
        "raw_clipped_signal_norm": 1.0,
        "raw_clipping_bias_norm": 1.1,
        "raw_realized_noise_norm": 3.0 * threshold,
        "raw_signal_to_noise_ratio": 1.0 / (3.0 * threshold),
        "dp_noisy_tangent_gradient_norm": (1.0 + (3.0 * threshold) ** 2) ** 0.5,
        "dp_factor_product_update_norm": 0.2,
        "raw_global_norm_hist_counts_json": json.dumps([1, 1]),
        "raw_global_norm_hist_edges_json": json.dumps([0.0, 1.0, 2.0]),
        "raw_global_norm_hist_overflow": 0,
        "slack_indicator_json": "",
        "slack_indicator_noise_std": "",
        "slaclip_num_slots": "",
        "slack_unclipped_proxy": "",
        "slaclip_target_unclipped_proxy": "",
        "slaclip_controller_error": "",
    }
    if method == "slaclip":
        # With K=2 and C around one, each histogram bin gives a valid interval
        # for these noised values.  Exact raw slots are intentionally absent so
        # the legacy histogram-bounds fallback is exercised.
        row.update(
            {
                "slack_indicator_json": json.dumps([0.4, 0.1]),
                "slack_indicator_noise_std": 0.2,
                "slaclip_num_slots": 2,
                "slack_unclipped_proxy": 0.4,
                "slaclip_target_unclipped_proxy": 0.3,
                "slaclip_controller_error": -0.1,
            }
        )
        if exact_oracle:
            row["raw_slack_indicator_json"] = json.dumps([0.3, 0.1])
    return row


def _prediction_rows(seed: int, method: str, task: str) -> list[dict]:
    # The identities are identical across a pair while flags produce both
    # directions of a paired flip.
    if method == "baseline":
        flags = [True, False, False]
    else:
        flags = [False, True, False] if seed == 1 else [True, True, False]
    rows = []
    for index, flag in enumerate(flags):
        if task.casefold() == "aqua":
            prediction = "A" if flag else "B"
        elif index == 2:
            # This is the repository evaluator's real "no numeric answer"
            # representation; json.dump emits Infinity and json.load restores
            # it as a non-finite float.
            prediction = float("inf")
        else:
            prediction = float(index if flag else index + 10)
        rows.append(
            {
                "instruction": f"question-{index}",
                "input": "",
                "answer": str(index),
                "pred": prediction,
                "flag": flag,
            }
        )
    return rows


def _make_campaign(
    tmp_path: Path,
    *,
    unsafe_safe_log: bool = False,
    exact_oracle: bool = False,
) -> Path:
    campaign = tmp_path / "campaign"
    _write_json(
        campaign / "final" / "manifest.json",
        {"expected_update_steps": 3},
    )
    _write_json(
        campaign / "selection" / "selection.json",
        {
            "selected_slaclip": {"candidate_id": "sla-selected", "params": {}},
            "selection_protocol": {
                "common_config": {"base_model": "example/model-4b"}
            },
        },
    )
    for seed in (1, 2):
        for method in ("baseline", "slaclip"):
            run_root = campaign / "final" / "model-4b" / f"seed-{seed}" / method
            fingerprint = f"fp-{seed}-{method}"
            config = {
                "method": method,
                "seed": seed,
                "base_model": "example/model-4b",
                "model_revision": "revision-a",
                "batch_size": 2,
                "dp_max_grad_norm": 1.0,
                "dp_epsilon": 6.0,
                "dp_delta": 1e-5,
                "output_dir": str(run_root / "adapter"),
                "result_dir": str(run_root / "results"),
                "run_id": f"run-{fingerprint}",
                "run_name": f"name-{fingerprint}",
                "config_fingerprint": fingerprint,
            }
            if method == "slaclip":
                config.update(
                    {
                        # A controller's initial C is intentionally allowed to
                        # differ from the paired fixed-C reference.
                        "dp_max_grad_norm": 1.5,
                        "slaclip_beta": 0.99,
                        "slaclip_eta": 0.15,
                        "slaclip_num_slots": 2,
                    }
                )
            _write_json(
                run_root / "adapter" / "run_status.json",
                {
                    "state": "completed",
                    "update_steps": 3,
                    "run_id": f"run-{fingerprint}",
                    "config_fingerprint": fingerprint,
                    "method": method,
                    "base_model": "example/model-4b",
                    "model_revision": "revision-a",
                    "config": config,
                },
            )
            (run_root / "orchestration-status.txt").write_text(
                "\n".join(
                    [
                        "state=completed",
                        f"candidate_id={'sla-selected' if method == 'slaclip' else 'fixed-c1'}",
                        f"seed={seed}",
                        "",
                    ]
                ),
                encoding="utf-8",
            )
            telemetry = [
                _telemetry_row(
                    step=step,
                    fingerprint=fingerprint,
                    method=method,
                    threshold=1.0 + (0.1 * step if method == "slaclip" else 0.0),
                    loss=1.5 - 0.1 * step - (0.02 if method == "slaclip" else 0.0),
                    exact_oracle=exact_oracle,
                )
                for step in (1, 2, 3)
            ]
            _write_csv(
                run_root / "results" / "research_raw" / "telemetry_steps.csv",
                telemetry,
            )
            task_values = {
                task: (2.0 / 3.0 if method == "slaclip" else 1.0 / 3.0)
                for task in TASKS
            }
            _write_csv(
                run_root / "results" / "summary.csv",
                [{**task_values, "Average": sum(task_values.values()) / len(task_values)}],
            )
            for task in TASKS:
                _write_json(
                    run_root / "results" / f"{task}.json",
                    _prediction_rows(seed, method, task),
                )
            if method == "slaclip":
                safe_records = [
                    _safe_record(
                        step=step,
                        fingerprint=fingerprint,
                        threshold=1.0 + 0.1 * step,
                    )
                    for step in (1, 2, 3)
                ]
                if unsafe_safe_log and seed == 1:
                    safe_records[0]["loss_mean"] = 1.0
                safe_path = run_root / "adapter" / "train_log.jsonl"
                safe_path.write_text(
                    "".join(json.dumps(record) + "\n" for record in safe_records),
                    encoding="utf-8",
                )
    return campaign


def test_mechanism_analyzer_writes_hashed_outputs_and_histogram_cdf_bounds(
    tmp_path: Path,
) -> None:
    campaign = _make_campaign(tmp_path)
    output = campaign / "deep-analysis-v1"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--campaign-root",
            str(campaign),
            "--output-dir",
            str(output),
            "--window",
            "2",
            "--smooth-window",
            "2",
            "--strict",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "NON_PRIVATE" in result.stdout

    required = {
        "step_metrics.csv",
        "run_summaries.csv",
        "paired_mechanism.csv",
        "cdf_diagnostics.csv",
        "loss_accuracy_association.csv",
        "prediction_transitions.csv",
        "replay_schedule.json",
        "manifest.json",
    }
    assert required == {path.name for path in output.iterdir()}

    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["analysis_status"] == "COMPLETE"
    assert manifest["NON_PRIVATE_TELEMETRY"] is True
    assert manifest["run_count"] == 4
    assert manifest["row_counts"]["step_metrics.csv"] == 12
    assert manifest["row_counts"]["cdf_diagnostics.csv"] == 12
    for name, expected_hash in manifest["outputs"].items():
        assert _sha256(output / name) == expected_hash

    with (output / "cdf_diagnostics.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        cdf_rows = list(csv.DictReader(handle))
    assert len(cdf_rows) == 12
    assert {row["oracle_source"] for row in cdf_rows} == {"fixed_histogram_bounds"}
    assert all(float(row["oracle_lower"]) <= float(row["oracle_upper"]) for row in cdf_rows)
    assert all(row["exact_noise_residual"] == "" for row in cdf_rows)
    assert all(row["exact_standardized_noise_residual"] == "" for row in cdf_rows)
    assert all(row["exact_within_95pct"] == "" for row in cdf_rows)
    assert all(
        row["noise_95_band_intersects_oracle"]
        == row["legacy_histogram_interval_intersects_noised_95pct_band"]
        for row in cdf_rows
    )

    schedule = json.loads((output / "replay_schedule.json").read_text(encoding="utf-8"))
    assert schedule["num_steps"] == 3
    assert schedule["source_seed_count"] == 2
    assert schedule["schema_version"] == 2
    assert schedule["schedule_privacy_class"] == "DP_DERIVED_FROM_2_SLACLIP_RUNS"
    assert schedule["control_interpretation"] == "DATA_DEPENDENT_MECHANISTIC_CONTROL"
    assert "source_privacy_class" not in schedule
    accounting = schedule["privacy_accounting"]
    assert accounting["conditional_replay_given_fixed_schedule"] == {
        "epsilon": 6.0,
        "delta": 1e-5,
        "interpretation": (
            "Only the replay gradient mechanism, conditional on treating "
            "the already-derived schedule as fixed."
        ),
    }
    assert accounting["schedule_source_basic_composition"][
        "epsilon_upper_bound"
    ] == 12.0
    assert accounting["schedule_source_basic_composition"][
        "delta_upper_bound"
    ] == 2e-5
    assert accounting["schedule_plus_one_replay_basic_composition"][
        "epsilon_upper_bound"
    ] == 18.0
    assert math.isclose(
        accounting["schedule_plus_one_replay_basic_composition"][
            "delta_upper_bound"
        ],
        3e-5,
        rel_tol=0.0,
        abs_tol=1e-20,
    )
    assert accounting["research_raw_bundle_privacy_class"] == "NON_PRIVATE"
    assert schedule["clip_thresholds"] == [1.1, 1.2, 1.3]
    assert all(value > 0.0 for value in schedule["clip_thresholds"])
    digest_payload = dict(schedule)
    expected_digest = digest_payload.pop("schedule_content_sha256")
    canonical = json.dumps(
        digest_payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    assert hashlib.sha256(canonical).hexdigest() == expected_digest
    assert len(schedule["source_logs"]) == 2
    assert all(len(item["sha256"]) == 64 for item in schedule["source_logs"])

    with (output / "prediction_transitions.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        transitions = list(csv.DictReader(handle))
    assert len(transitions) == 8
    assert all(row["NON_PRIVATE_TELEMETRY"] == "True" for row in transitions)
    for row in transitions:
        if row["task"] == "AQuA":
            assert row["baseline_parse_failures"] == "0"
            assert row["slaclip_parse_failures"] == "0"
        else:
            assert row["baseline_parse_failures"] == "1"
            assert row["slaclip_parse_failures"] == "1"


def test_parse_failure_handles_real_numeric_evaluator_sentinels_and_aqua() -> None:
    spec = importlib.util.spec_from_file_location(
        "test_analyze_mechanism_campaign_module",
        SCRIPT,
    )
    assert spec is not None and spec.loader is not None
    analyzer = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = analyzer
    spec.loader.exec_module(analyzer)

    for prediction in (
        None,
        "",
        "   ",
        "not-a-number",
        float("inf"),
        float("-inf"),
        float("nan"),
        "Infinity",
        "NaN",
    ):
        assert analyzer._parse_failed({"pred": prediction}, task="gsm8k")
    for prediction in (0, -1.25, "3.5"):
        assert not analyzer._parse_failed({"pred": prediction}, task="SVAMP")
    assert not analyzer._parse_failed({"pred": "C"}, task="AQuA")
    assert analyzer._parse_failed({"pred": "Z"}, task="AQuA")
    assert analyzer._parse_failed({"pred": float("inf")}, task="AQuA")


def test_exact_cdf_diagnostics_export_exact_gaussian_coverage(
    tmp_path: Path,
) -> None:
    campaign = _make_campaign(tmp_path, exact_oracle=True)
    output = campaign / "deep-analysis-exact"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--campaign-root",
            str(campaign),
            "--output-dir",
            str(output),
            "--window",
            "2",
            "--smooth-window",
            "2",
            "--strict",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    with (output / "cdf_diagnostics.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert rows
    assert {row["oracle_source"] for row in rows} == {
        "exact_raw_slack_indicator"
    }
    assert all(row["exact_noise_residual"] != "" for row in rows)
    assert all(row["exact_standardized_noise_residual"] != "" for row in rows)
    assert all(row["exact_within_95pct"] == "True" for row in rows)
    for row in rows:
        residual = float(row["exact_noise_residual"])
        standardized = float(row["exact_standardized_noise_residual"])
        assert math.isclose(
            standardized,
            residual / float(row["noise_std"]),
            rel_tol=0.0,
            abs_tol=1e-12,
        )


def test_strict_replay_source_rejects_non_private_field(tmp_path: Path) -> None:
    campaign = _make_campaign(tmp_path, unsafe_safe_log=True)
    output = campaign / "deep-analysis-v1"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--campaign-root",
            str(campaign),
            "--output-dir",
            str(output),
            "--strict",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "non-private fields" in result.stderr
    assert not (output / "manifest.json").exists()
