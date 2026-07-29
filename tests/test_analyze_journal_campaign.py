from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "analyze_journal_campaign.py"
SPEC = importlib.util.spec_from_file_location("analyze_journal_campaign", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
analyzer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analyzer)


FRESH_SEEDS = [17, 29, 47, 71, 101]
BEST_SEEDS = [17, 29, 47]
MODELS = [
    {
        "slug": "gemma-3-4b-pt",
        "base_model": "google/gemma-3-4b-pt",
        "model_revision": "4" * 40,
    },
    {
        "slug": "gemma-2-9b",
        "base_model": "google/gemma-2-9b",
        "model_revision": "9" * 40,
    },
]


def _canonical_sha(payload) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _summary_values(model_index: int, seed_index: int, arm: str) -> dict[str, float]:
    base = 0.45 + 0.1 * model_index + 0.002 * seed_index
    delta = {"baseline": 0.0, "slaclip": 0.01, "best-fixed": 0.004}[arm]
    values = {
        "gsm8k": base + delta,
        "AQuA": base + 0.02 + delta,
        "mawps": base + 0.04 + delta,
        "SVAMP": base + 0.06 + delta,
    }
    values["Average"] = sum(values.values()) / 4.0
    return values


def _write_run(
    campaign: Path,
    *,
    model: dict,
    model_index: int,
    seed: int,
    seed_index: int,
    artifact_arm: str,
    logical_arm: str,
    selected_params: dict,
    best_fixed_params: dict,
) -> None:
    root = campaign / "final" / model["slug"] / f"seed-{seed}" / artifact_arm
    method = "slaclip" if logical_arm == "slaclip" else "baseline"
    config = {
        "dataset": "math10k",
        "privacy": "dp",
        "method": method,
        "seed": seed,
        "base_model": model["base_model"],
        "model_revision": model["model_revision"],
        "resolved_model_revision": model["model_revision"],
        "protocol_stage": "final",
        "validation_data_is_public": False,
        "val_set_size": 0,
        "validation_seed": 1729,
        "validation_batch_size": 8,
        "total_update_steps": 300,
        "batch_size": 64,
        "micro_batch_size": 4,
        "learning_rate": 0.0003,
        "cutoff_len": 256,
        "train_on_inputs": True,
        "dp_epsilon": 6.0,
        "dp_delta": 1e-5,
        "dp_accountant": "prv",
        "dp_grad_sample_mode": "functorch",
        "dp_secure_mode": False,
        "lora_r": 16,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "target_modules": ["q_proj", "k_proj", "v_proj", "up_proj", "down_proj"],
        "telemetry_mode": "research_raw",
        "implementation_git_sha": "a" * 40,
        "data_content_sha256": "b" * 64,
        "dp_max_grad_norm": 1.0,
        "slaclip_beta": 0.5,
        "slaclip_eta": 0.5,
        "slaclip_num_slots": 15,
        "slaclip_c_min": 0.1,
        "slaclip_c_max": 15.0,
        "output_dir": str(root / "adapter"),
        "result_dir": str(root / "results"),
        "run_id": f"{model['slug']}-{seed}-{logical_arm}",
        "config_fingerprint": f"fp-{model['slug']}-{seed}-{logical_arm}",
    }
    if logical_arm == "slaclip":
        config.update(selected_params)
    elif logical_arm == "best-fixed":
        config.update(best_fixed_params)
    fingerprint = config["config_fingerprint"]
    status = {
        "state": "completed",
        "update_steps": 300,
        "method": method,
        "privacy": "dp",
        "dataset": "math10k",
        "base_model": model["base_model"],
        "model_revision": model["model_revision"],
        "config_fingerprint": fingerprint,
        "config": config,
        "privacy_accounting": {
            "target_epsilon": 6.0,
            "target_delta": 1e-5,
            "epsilon_spent": 5.9988,
            "completed_update_steps": 300,
        },
    }
    _write_json(root / "adapter" / "run_status.json", status)
    values = _summary_values(model_index, seed_index, logical_arm)
    summary_path = root / "results" / "summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[*analyzer.TASKS, "Average"], lineterminator="\n")
        writer.writeheader()
        writer.writerow(values)
    telemetry = {
        "summary_schema_version": 2,
        "NON_PRIVATE_TELEMETRY": True,
        "warning": "Contains exact statistics derived from training examples; NON-PRIVATE and not a DP release.",
        "steps": {"count": 300, "first": 1, "last": 300, "missing_count": 0, "missing": []},
        "run_identity": {
            "config_fingerprint": fingerprint,
            "method": method,
            "base_model": model["base_model"],
            "model_revision": model["model_revision"],
        },
        "source": {"raw_sha256": hashlib.sha256(fingerprint.encode()).hexdigest()},
        "metrics": {
            "raw_clip_fraction": {
                "count": 300,
                "first": 1.0,
                "last": 0.98 if logical_arm == "slaclip" else 1.0,
                "mean": 0.99 if logical_arm == "slaclip" else 1.0,
                "missing": 0,
            },
            "dp_clip_threshold": {
                "count": 300,
                "first": config["dp_max_grad_norm"],
                "last": 2.0 if logical_arm == "slaclip" else config["dp_max_grad_norm"],
                "mean": 1.5 if logical_arm == "slaclip" else config["dp_max_grad_norm"],
                "missing": 0,
            },
        },
        "boolean_metrics": {
            "slaclip_c_hit_max": {
                "count": 300 if logical_arm == "slaclip" else 0,
                "true_count": 0,
                "false_count": 300 if logical_arm == "slaclip" else 0,
            }
        },
    }
    _write_json(root / "results" / "research_raw" / "telemetry_summary.json", telemetry)


def _build_campaign(tmp_path: Path, *, best_fixed_c: float = 1.0) -> Path:
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    selected_params = {
        "dp_max_grad_norm": 1.0,
        "slaclip_beta": 0.975,
        "slaclip_eta": 0.15,
        "slaclip_num_slots": 15,
        "slaclip_c_min": 0.1,
        "slaclip_c_max": 15.0,
    }
    best_id = "fixed-c1" if best_fixed_c == 1.0 else "fixed-c15"
    selection = {
        "schema_version": 1,
        "stage": "stage2",
        "required_seeds": [42, 43, 44],
        "canonical_fixed_candidate_id": "fixed-c1",
        "selection_protocol": {
            "common_config": {
                "dataset": "math10k",
                "privacy": "dp",
                "base_model": MODELS[0]["base_model"],
                "model_revision": MODELS[0]["model_revision"],
                "batch_size": 64,
                "micro_batch_size": 4,
                "learning_rate": 0.0003,
                "lora_r": 16,
                "lora_alpha": 16,
                "lora_dropout": 0.05,
                "cutoff_len": 256,
                "train_on_inputs": True,
                "total_update_steps": 300,
                "dp_epsilon": 6.0,
                "dp_delta": 1e-5,
                "val_set_size": 500,
                "validation_seed": 1729,
                "validation_batch_size": 8,
                "protocol_stage": "selection",
                "validation_data_is_public": True,
                "telemetry_mode": "research_raw",
            }
        },
        "selected_slaclip": {
            "candidate_id": "sla-selected",
            "family": "slaclip",
            "method": "slaclip",
            "params": selected_params,
        },
        "best_fixed": {
            "candidate_id": best_id,
            "family": "fixed",
            "method": "baseline",
            "params": {"dp_max_grad_norm": best_fixed_c},
        },
    }
    selection_sha = _canonical_sha(selection)
    _write_json(campaign / "selection" / "selection.json", selection)
    (campaign / "selection" / "selected.env").write_text(
        f"PRISM_SELECTION_SHA256={selection_sha}\n"
        "PRISM_SELECTED_CANDIDATE_ID=sla-selected\n"
        f"PRISM_SELECTED_BEST_FIXED_CANDIDATE_ID={best_id}\n",
        encoding="utf-8",
    )
    final_manifest = {
        "schema_version": 1,
        "selection_sha256": selection_sha,
        "expected_update_steps": 300,
        "target_epsilon": 6.0,
        "target_delta": 1e-5,
        "fresh_seeds": FRESH_SEEDS,
        "best_fixed_seeds": BEST_SEEDS,
        "models": MODELS,
    }
    _write_json(campaign / "final" / "manifest.json", final_manifest)
    for model_index, model in enumerate(MODELS):
        for seed_index, seed in enumerate(FRESH_SEEDS):
            _write_run(
                campaign,
                model=model,
                model_index=model_index,
                seed=seed,
                seed_index=seed_index,
                artifact_arm="baseline",
                logical_arm="baseline",
                selected_params=selected_params,
                best_fixed_params={"dp_max_grad_norm": best_fixed_c},
            )
            _write_run(
                campaign,
                model=model,
                model_index=model_index,
                seed=seed,
                seed_index=seed_index,
                artifact_arm="slaclip",
                logical_arm="slaclip",
                selected_params=selected_params,
                best_fixed_params={"dp_max_grad_norm": best_fixed_c},
            )
        if best_fixed_c != 1.0:
            for seed_index, seed in enumerate(BEST_SEEDS):
                _write_run(
                    campaign,
                    model=model,
                    model_index=model_index,
                    seed=seed,
                    seed_index=seed_index,
                    artifact_arm="best-fixed",
                    logical_arm="best-fixed",
                    selected_params=selected_params,
                    best_fixed_params={"dp_max_grad_norm": best_fixed_c},
                )
    return campaign


def _analyze(campaign: Path, *, allow_incomplete: bool = False):
    return analyzer.analyze_campaign(
        campaign_root=campaign,
        selection_path=Path("selection/selection.json"),
        final_root=Path("final"),
        output_dir=Path("journal"),
        allow_incomplete=allow_incomplete,
    )


def test_complete_alias_campaign_writes_formal_tables(tmp_path: Path) -> None:
    campaign = _build_campaign(tmp_path)
    manifest = _analyze(campaign)
    assert manifest["analysis_status"] == "FORMAL_COMPLETE"
    assert manifest["complete"] is True
    assert manifest["expected_runs"] == manifest["observed_runs"] == 26
    assert manifest["best_fixed_uses_baseline_alias"] is True

    with (campaign / "journal" / "accuracy_by_run.csv").open(newline="") as handle:
        accuracy_rows = list(csv.DictReader(handle))
    assert len(accuracy_rows) == 26 * 5
    assert all(row["raw_exact_telemetry_status"].startswith("NON_PRIVATE") for row in accuracy_rows)

    with (campaign / "journal" / "paired_accuracy_summary.csv").open(newline="") as handle:
        paired = list(csv.DictReader(handle))
    assert len(paired) == 2 * 3 * 5
    target = next(
        row
        for row in paired
        if row["model_slug"] == "gemma-3-4b-pt"
        and row["comparison"] == "slaclip_vs_baseline"
        and row["metric"] == "macro"
    )
    assert target["n"] == "5"
    assert float(target["paired_mean_delta"]) == pytest.approx(0.01)
    assert (target["wins"], target["ties"], target["losses"]) == ("5", "0", "0")

    with (campaign / "journal" / "telemetry_by_run.csv").open(newline="") as handle:
        telemetry_rows = list(csv.DictReader(handle))
    assert len(telemetry_rows) == 26
    assert all(row["NON_PRIVATE_TELEMETRY"] == "True" for row in telemetry_rows)
    assert "metrics__raw_clip_fraction__mean" in telemetry_rows[0]


def test_noncanonical_best_fixed_requires_and_aggregates_its_own_runs(tmp_path: Path) -> None:
    campaign = _build_campaign(tmp_path, best_fixed_c=1.5)
    manifest = _analyze(campaign)
    assert manifest["complete"] is True
    assert manifest["best_fixed_uses_baseline_alias"] is False
    with (campaign / "journal" / "accuracy_by_run.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    fixed = [row for row in rows if row["arm"] == "best-fixed" and row["metric"] == "macro"]
    assert len(fixed) == 6
    assert all(row["artifact_alias"] == "False" for row in fixed)


def test_default_analysis_fails_closed_on_missing_result(tmp_path: Path) -> None:
    campaign = _build_campaign(tmp_path)
    missing = campaign / "final" / MODELS[0]["slug"] / "seed-101" / "slaclip" / "results" / "summary.csv"
    missing.unlink()
    with pytest.raises(analyzer.IncompleteRun, match="missing task accuracy summary"):
        _analyze(campaign)


def test_allow_incomplete_writes_a_nonreportable_snapshot(tmp_path: Path) -> None:
    campaign = _build_campaign(tmp_path)
    missing = campaign / "final" / MODELS[0]["slug"] / "seed-101" / "slaclip" / "results" / "summary.csv"
    missing.unlink()
    manifest = _analyze(campaign, allow_incomplete=True)
    assert manifest["complete"] is False
    assert manifest["analysis_status"] == "INCOMPLETE_SNAPSHOT_DO_NOT_REPORT"
    assert manifest["observed_runs"] == 25
    assert len(manifest["missing_or_incomplete_runs"]) == 1
    text = (campaign / "journal" / "accuracy_by_run.csv").read_text()
    assert "INCOMPLETE_SNAPSHOT_DO_NOT_REPORT" in text


def test_selection_parameter_mismatch_is_never_ignored(tmp_path: Path) -> None:
    campaign = _build_campaign(tmp_path)
    status_path = (
        campaign
        / "final"
        / MODELS[0]["slug"]
        / "seed-17"
        / "slaclip"
        / "adapter"
        / "run_status.json"
    )
    status = json.loads(status_path.read_text())
    status["config"]["slaclip_beta"] = 0.95
    _write_json(status_path, status)
    with pytest.raises(analyzer.AnalysisError, match="does not match selection"):
        _analyze(campaign, allow_incomplete=True)


def test_selection_sha_mismatch_fails_before_reading_runs(tmp_path: Path) -> None:
    campaign = _build_campaign(tmp_path)
    manifest_path = campaign / "final" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["selection_sha256"] = "0" * 64
    _write_json(manifest_path, manifest)
    with pytest.raises(analyzer.AnalysisError, match="selection SHA"):
        _analyze(campaign)


def test_formal_settings_must_be_constant_across_seeds(tmp_path: Path) -> None:
    campaign = _build_campaign(tmp_path)
    for arm in ("baseline", "slaclip"):
        status_path = (
            campaign
            / "final"
            / MODELS[0]["slug"]
            / "seed-29"
            / arm
            / "adapter"
            / "run_status.json"
        )
        status = json.loads(status_path.read_text())
        # The two paired arms still agree with one another, but this undeclared
        # per-seed drift must not enter a formal multi-seed estimate.
        status["config"]["prism_floor_factor"] = 0.6
        _write_json(status_path, status)
    with pytest.raises(analyzer.AnalysisError, match="vary across seeds"):
        _analyze(campaign)


def test_nonfinite_telemetry_is_never_written_to_journal_tables(tmp_path: Path) -> None:
    campaign = _build_campaign(tmp_path)
    telemetry_path = (
        campaign
        / "final"
        / MODELS[0]["slug"]
        / "seed-17"
        / "slaclip"
        / "results"
        / "research_raw"
        / "telemetry_summary.json"
    )
    telemetry = json.loads(telemetry_path.read_text())
    telemetry["metrics"]["raw_clip_fraction"]["mean"] = float("nan")
    telemetry_path.write_text(json.dumps(telemetry, allow_nan=True), encoding="utf-8")
    with pytest.raises(analyzer.AnalysisError, match="must be finite"):
        _analyze(campaign)
