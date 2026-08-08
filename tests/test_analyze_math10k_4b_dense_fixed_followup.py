from __future__ import annotations

import importlib.util
import csv
import json
import math
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "analyze_math10k_4b_dense_fixed_followup.py"
SPEC = importlib.util.spec_from_file_location("dense_fixed_analysis", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_plan_parser_preserves_locked_identity(tmp_path: Path) -> None:
    arm = tmp_path / "arm"
    plan = tmp_path / "plan.tsv"
    plan.write_text(
        f"train|dense-selection|fixed-c1p5|dense-fixed-selection|42|baseline|1.5|NA|NA|NA|{arm}\n"
        f"train|dense-final|fixed-c1p5|dense-best-fixed|191|baseline|1.5|NA|NA|NA|{arm}-final\n",
        encoding="utf-8",
    )

    records = MODULE._read_plan(plan)

    assert [(record.phase, record.seed, record.clip) for record in records] == [
        ("dense-selection", 42, 1.5),
        ("dense-final", 191, 1.5),
    ]
    assert all(record.method == "baseline" for record in records)


def test_plan_parser_rejects_unfingerprinted_shape(tmp_path: Path) -> None:
    plan = tmp_path / "bad.tsv"
    plan.write_text("train|too|few|fields\n", encoding="utf-8")
    with pytest.raises(MODULE.AnalysisError, match="invalid plan record"):
        MODULE._read_plan(plan)


def test_five_seed_paired_interval_uses_seed_as_unit() -> None:
    differences = [0.02, 0.01, -0.01, 0.03, 0.0]
    result = MODULE._paired_summary(differences)

    assert result["n_independent_seeds"] == 5
    assert result["wins"] == 3
    assert result["ties"] == 1
    assert result["losses"] == 1
    assert math.isclose(result["mean_difference"], 0.01)
    assert result["ci95_low"] < result["mean_difference"] < result["ci95_high"]


def test_paired_interval_refuses_step_level_pseudoreplication() -> None:
    with pytest.raises(MODULE.AnalysisError, match="exactly five independent seeds"):
        MODULE._paired_summary([0.01] * 300)


def test_strict_telemetry_rejects_stale_fingerprint(tmp_path: Path) -> None:
    arm = tmp_path / "arm"
    raw = arm / "results" / "research_raw"
    raw.mkdir(parents=True)
    fingerprint = "f" * 64
    rows = []
    for step in range(1, 301):
        rows.append(
            {
                "NON_PRIVATE_TELEMETRY": "True",
                "config_fingerprint": fingerprint,
                "method": "baseline",
                "dataset": "math10k",
                "privacy": "dp",
                "step": str(step),
                **{name: "1.0" for name in MODULE.MECHANISM_METRICS},
            }
        )
    with (raw / "telemetry_steps.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "NON_PRIVATE_TELEMETRY": True,
        "steps": {"count": 300, "first": 1, "last": 300, "missing_count": 0},
        "source": {
            "raw_physical_records": 300,
            "raw_unique_steps": 300,
            "safe_physical_records": 300,
            "safe_unique_steps": 300,
            "raw_duplicate_records": 0,
            "safe_duplicate_records": 0,
        },
        "run_identity": {
            "method": "baseline",
            "config_fingerprint": fingerprint,
            "dataset": "math10k",
            "privacy": "dp",
        },
        "metrics": {
            name: {"count": 300, "missing": 0, "mean": 1.0}
            for name in MODULE.MECHANISM_METRICS
        },
    }
    summary_path = raw / "telemetry_summary.json"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    spec = MODULE.Spec("dense-selection", "fixed-c1", "dense", 42, "baseline", 1.0, None, None, arm)
    status = {"config_fingerprint": fingerprint}

    means, observed = MODULE._telemetry(spec, status, strict=True)
    assert len(means) == len(MODULE.MECHANISM_METRICS)
    assert len(observed) == 300

    summary["run_identity"]["config_fingerprint"] = "0" * 64
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    with pytest.raises(MODULE.AnalysisError, match="run identity mismatch"):
        MODULE._telemetry(spec, status, strict=True)


def test_final_metrics_are_bound_to_selection_and_recomputed(tmp_path: Path) -> None:
    arm = tmp_path / "arm"
    result = arm / "results"
    result.mkdir(parents=True)
    raw = {"gsm8k": 0.6, "AQuA": 0.4, "mawps": 0.5, "SVAMP": 0.8}
    clean = {**raw, "mawps": 0.55}
    payload = {
        "primary_metric": "clean_three_task_macro_accuracy",
        "selection_sha256": "a" * 64,
        "raw_task_accuracy": raw,
        "decontaminated_task_accuracy": clean,
        "clean_three_task_macro_accuracy": (0.6 + 0.4 + 0.8) / 3,
        "decontaminated_four_task_macro_accuracy": (0.6 + 0.4 + 0.55 + 0.8) / 4,
        "paper_raw_four_task_macro_accuracy": sum(raw.values()) / 4,
        "clean_mawps_records": 185,
    }
    path = result / "decontaminated_metrics.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    spec = MODULE.Spec("final", "candidate", "role", 191, "baseline", 1.5, None, None, arm)

    metrics = MODULE._final_metrics(spec, "a" * 64)
    assert metrics["paper_raw_four_task_macro_accuracy"] == pytest.approx(0.575)

    with pytest.raises(MODULE.AnalysisError, match="locked source selection"):
        MODULE._final_metrics(spec, "b" * 64)
