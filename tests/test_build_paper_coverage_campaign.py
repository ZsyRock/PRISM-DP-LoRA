from __future__ import annotations

import importlib.util
import hashlib
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "build_paper_coverage_campaign",
    ROOT / "scripts" / "build_paper_coverage_campaign.py",
)
assert SPEC and SPEC.loader
campaign = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(campaign)

CODE_SHA = "1" * 40
REV_4B = "2" * 40
REV_9B = "3" * 40


def test_manifest_covers_paper_settings_and_full_slaclip() -> None:
    manifest = campaign.build_manifest(CODE_SHA, REV_4B, REV_9B, profile="paper-breadth")
    arms = manifest["arms"]
    assert len(arms) == 15
    assert {arm["setting_id"] for arm in arms} == {
        "glue8-4b-eps6-r16",
        "glue8-4b-eps3-r16",
        "math10k-9b-eps6-r16",
        "math10k-4b-eps6-r8",
        "math10k-4b-eps6-r32",
    }
    assert {arm["candidate_id"] for arm in arms} == {
        "fixed-c1",
        "full-sla-rho090",
        "full-sla-rho098",
    }
    assert sum(arm["lane"] == 0 for arm in arms) == 6
    assert sum(arm["lane"] == 1 for arm in arms) == 9
    for arm in arms:
        assert arm["seed"] == 42
        assert arm["initial_c"] == 1.0
        assert arm["model_revision"] == (REV_9B if arm["model_id"] == campaign.MODEL_9B else REV_4B)
        if arm["method"] == "slaclip":
            assert arm["rho"] in {0.9, 0.98}
            assert arm["eta"] == 0.05
        else:
            assert arm["rho"] is None
            assert arm["eta"] is None


def test_prepare_is_immutable_and_writes_exact_lane_counts(tmp_path: Path) -> None:
    campaign.prepare(tmp_path, CODE_SHA, REV_4B, REV_9B, "paper-breadth")
    campaign.prepare(tmp_path, CODE_SHA, REV_4B, REV_9B, "paper-breadth")
    assert len((tmp_path / "plans" / "lane-0.tsv").read_text().splitlines()) == 6
    assert len((tmp_path / "plans" / "lane-1.tsv").read_text().splitlines()) == 9
    assert len((tmp_path / "plans" / "sequential.tsv").read_text().splitlines()) == 15
    assert all(
        line.startswith("0|")
        for line in (tmp_path / "plans" / "sequential.tsv").read_text().splitlines()
    )
    assert (tmp_path / "plans" / "manifest.json.sha256").is_file()
    with pytest.raises(campaign.CampaignError, match="refusing to overwrite"):
        campaign.prepare(tmp_path, "4" * 40, REV_4B, REV_9B, "paper-breadth")


@pytest.mark.parametrize("bad", ["main", "A" * 40, "0" * 39, "g" * 40])
def test_manifest_rejects_unpinned_revisions(bad: str) -> None:
    with pytest.raises(campaign.CampaignError, match="full lowercase commit SHA"):
        campaign.build_manifest(CODE_SHA, REV_4B, bad)


def test_regime_map_crosses_paper_axes_and_clipping_grids() -> None:
    manifest = campaign.build_manifest(CODE_SHA, REV_4B, REV_9B, profile="regime-map")
    arms = manifest["arms"]
    assert manifest["schema_version"] == 2
    assert manifest["profile"] == "regime-map"
    assert len(arms) == 99
    assert len({arm["setting_id"] for arm in arms}) == 9
    assert {arm["epsilon"] for arm in arms} == {3.0, 6.0}
    assert {arm["lora_r"] for arm in arms} == {8, 16, 32}
    assert {arm["dataset"] for arm in arms} == {"glue8", "math10k"}
    assert {arm["model_id"] for arm in arms} == {campaign.MODEL_4B, campaign.MODEL_9B}
    assert {arm["initial_c"] for arm in arms if arm["method"] == "baseline"} == {
        0.5, 1.0, 2.0, 3.0, 5.0,
    }
    assert {arm["rho"] for arm in arms if arm["method"] == "slaclip"} == {
        0.5, 0.7, 0.8, 0.9, 0.98,
    }
    assert all(arm["steps"] == 150 and arm["eval_limit"] == 512 for arm in arms)
    assert any(
        arm["candidate_id"] == "full-sla-c2-rho090"
        and arm["initial_c"] == 2.0
        and arm["rho"] == 0.9
        for arm in arms
    )
    assert sum(arm["lane"] == 0 for arm in arms) == 44
    assert sum(arm["lane"] == 1 for arm in arms) == 55


def test_baseline_reproduction_covers_all_paper_dataset_model_settings() -> None:
    manifest = campaign.build_manifest(
        CODE_SHA,
        REV_4B,
        REV_9B,
        profile="baseline-reproduction",
        model_12b_revision="4" * 40,
    )
    arms = manifest["arms"]
    assert len(arms) == 10
    assert manifest["baseline_reproduction"]["covered_settings"] == 10
    assert manifest["baseline_reproduction"]["excluded_setting"] is None
    assert {arm["model_id"] for arm in arms} == {
        campaign.MODEL_4B,
        campaign.MODEL_9B,
        campaign.MODEL_12B,
    }
    assert sum(arm["dataset"] == "glue8" for arm in arms) == 4
    assert sum(arm["dataset"] == "math10k" for arm in arms) == 6
    assert {arm["epsilon"] for arm in arms} == {3.0, 6.0}
    assert {arm["lora_r"] for arm in arms} == {8, 16, 32}
    for arm in arms:
        assert arm["method"] == "baseline"
        assert arm["initial_c"] == 1.0
        assert arm["rho"] is None and arm["eta"] is None
        assert arm["eval_limit"] == 0
        if arm["dataset"] == "glue8":
            assert arm["steps"] == 500
            assert arm["learning_rate"] == 0.0002
            assert arm["cutoff_len"] == 384
            assert arm["train_on_inputs"] is False
        else:
            assert arm["steps"] == 300
            assert arm["learning_rate"] == 0.0003
            assert arm["cutoff_len"] == 256
            assert arm["train_on_inputs"] is True


def test_cached_baseline_profile_is_explicitly_incomplete() -> None:
    manifest = campaign.build_manifest(
        CODE_SHA, REV_4B, REV_9B, profile="baseline-reproduction-cached"
    )
    assert len(manifest["arms"]) == 9
    assert manifest["baseline_reproduction"]["covered_settings"] == 9
    assert "12B" in manifest["baseline_reproduction"]["excluded_setting"]
    assert campaign.MODEL_12B not in {arm["model_id"] for arm in manifest["arms"]}
    sequential = campaign._plan_bytes(manifest, 0, include_all=True).decode().splitlines()
    settings = [line.split("|")[2] for line in sequential]
    assert settings[:3] == [
        "glue8-4b-eps6-r16",
        "math10k-4b-eps6-r16",
        "math10k-9b-eps6-r16",
    ]
    assert all(line.startswith("0|") for line in sequential)


def test_glue_slaclip_screen_is_predeclared_and_balanced() -> None:
    paper_config = json.loads((ROOT / "configs" / "glue8_paper.json").read_text())
    assert paper_config["slaclip_target_non_small_clip_fraction"] == 0.5
    manifest = campaign.build_manifest(
        CODE_SHA, REV_4B, REV_9B, profile="glue-slaclip-screen"
    )
    arms = manifest["arms"]
    assert len(arms) == 12
    assert manifest["seed"] == 43
    assert {arm["setting_id"] for arm in arms} == {"glue8-4b-eps6-r16"}
    assert {arm["model_id"] for arm in arms} == {campaign.MODEL_4B}
    assert all(
        arm["steps"] == 150 and arm["eval_limit"] == 0 and arm["seed"] == 43
        for arm in arms
    )
    fixed = [arm for arm in arms if arm["role"] == "tuned_fixed_candidate"]
    adaptive = [arm for arm in arms if arm["role"] == "slaclip_target_candidate"]
    controls = [
        arm for arm in arms if arm["role"] == "initial_C_sensitivity_control"
    ]
    assert {arm["initial_c"] for arm in fixed} == {0.5, 1.0, 2.0, 3.0, 5.0}
    assert all(arm["rho"] is None and arm["eta"] is None for arm in fixed)
    assert {arm["rho"] for arm in adaptive} == {0.55, 0.60, 0.65, 0.70, 0.75}
    assert all(arm["initial_c"] == 1.0 and arm["eta"] == 0.05 for arm in adaptive)
    assert {(arm["initial_c"], arm["rho"]) for arm in controls} == {
        (0.5, 0.65), (2.0, 0.65),
    }
    assert len(fixed) == 5 and len(adaptive) == 5 and len(controls) == 2
    assert sum(arm["lane"] == 0 for arm in arms) == 6
    assert sum(arm["lane"] == 1 for arm in arms) == 6
    source = manifest["glue_slaclip_screen"]["baseline_source"]
    assert source["job_id"] == "1402286"
    assert source["raw_telemetry_sha256"] == (
        "ebe0085350f0d3dc4b7cbe90cbc18dd3a9179056cc9e6f899fe99785260312e8"
    )
    assert source["post_burn_in_conditional_clip_fraction_q10"] == pytest.approx(
        0.5524977719630925
    )
    assert source["post_burn_in_conditional_clip_fraction_median"] == pytest.approx(
        0.6401230104328237
    )
    assert source["post_burn_in_conditional_clip_fraction_q90"] == pytest.approx(
        0.7290751684816961
    )
    assert manifest["glue_slaclip_screen"]["calibration_selection_overlap"] is True
    assert "conditional-clipping q10-q90" in manifest["glue_slaclip_screen"][
        "target_derivation"
    ]
    assert manifest["glue_slaclip_screen"]["selection"] == {
        "seed": 43,
        "public_holdout_rows": 800,
        "public_holdout_stratification": "100 rows per GLUE8 task",
        "public_holdout_seed": 1729,
        "public_holdout_indices_sha256": (
            "34a59e5cf4d98300f3d484d9c82b37172b850939bc19e002c22428be22a12805"
        ),
        "public_holdout_records_sha256": (
            "43f7a3d0db422b2331a59d3611e777faf99d0bf434a405ef98a8ea7b1c582fad"
        ),
        "metric": "response_only_mean_per_record_causal_lm_loss",
        "official_task_evaluation": False,
    }
    sequential = campaign._plan_bytes(manifest, 0, include_all=True).decode().splitlines()
    assert len(sequential) == 12
    assert all(line.startswith("0|") for line in sequential)
    assert all("full-sla-c0p5-rho065" not in line for line in sequential[:10])
    assert all("full-sla-c2p0-rho065" not in line for line in sequential[:10])
    assert "full-sla-c0p5-rho065" in sequential[-2]
    assert "full-sla-c2p0-rho065" in sequential[-1]


def _write_focused_campaign(root_path: Path) -> dict:
    manifest = campaign.build_manifest(
        CODE_SHA, REV_4B, REV_9B, profile="glue-slaclip-screen"
    )
    (root_path / "plans").mkdir()
    (root_path / "plans" / "manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    for index, arm in enumerate(manifest["arms"]):
        root = root_path / arm["relative_root"]
        (root / "adapter").mkdir(parents=True)
        (root / "results" / "research_raw").mkdir(parents=True)
        (root / "results" / "validation").mkdir(parents=True)
        validation_loss = 1.0 + index / 100.0
        run_id = f"focused-arm-{index}"
        fingerprint = f"{index + 1:064x}"
        manifest_sha = f"{index + 101:064x}"
        split = {
            "protocol_stage": "selection",
            "validation_data_is_public": True,
            "validation_rows": 800,
            "seed": 1729,
            "validation_indices_sha256": (
                campaign.GLUE_SLACLIP_VALIDATION_INDICES_SHA256
            ),
            "validation_record_hashes_sha256": (
                campaign.GLUE_SLACLIP_VALIDATION_RECORDS_SHA256
            ),
            "manifest_sha256": manifest_sha,
        }
        validation = {
            **split,
            "PUBLIC_VALIDATION_DATA": True,
            "NON_PRIVATE_SELECTION_METRIC": True,
            "records": 800,
            "selection_metric": "response_only_mean_per_record_causal_lm_loss",
            "loss_definition": (
                "response_only_per_record_mean_of_nonignored_next_token_losses"
            ),
            "loss_mean": validation_loss,
        }
        config = {
            "implementation_git_sha": CODE_SHA,
            "implementation_git_dirty": False,
            "dataset": arm["dataset"],
            "method": arm["method"],
            "privacy": "dp",
            "base_model": arm["model_id"],
            "model_revision": arm["model_revision"],
            "seed": arm["seed"],
            "lora_r": arm["lora_r"],
            "total_update_steps": 150,
            "batch_size": 64,
            "micro_batch_size": 4,
            "learning_rate": arm["learning_rate"],
            "cutoff_len": arm["cutoff_len"],
            "train_on_inputs": arm["train_on_inputs"],
            "val_set_size": 800,
            "validation_seed": 1729,
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
            "slaclip_target_non_small_clip_fraction": (
                arm["rho"] if arm["method"] == "slaclip" else 0.5
            ),
            "run_train": True,
            "run_eval": False,
            "resume": True,
            "checkpoint_every": 25,
        }
        if arm["method"] == "slaclip":
            config.update({
                "slaclip_eta": arm["eta"],
                "slaclip_c_min": 0.1,
                "slaclip_c_max": 15.0,
            })
        status = {
            "state": "completed",
            "update_steps": 150,
            "dataset": arm["dataset"],
            "method": arm["method"],
            "privacy": "dp",
            "base_model": arm["model_id"],
            "model_revision": arm["model_revision"],
            "resolved_model_revision": arm["model_revision"],
            "data_content_sha256": campaign.GLUE_SLACLIP_SOURCE["data_sha256"],
            "telemetry_mode": "research_raw",
            "non_private_telemetry": True,
            "run_id": run_id,
            "config_fingerprint": fingerprint,
            "config": config,
            "data_split": split,
            "validation": validation,
        }
        (root / "adapter" / "run_status.json").write_text(
            json.dumps(status), encoding="utf-8"
        )
        raw_records = [
            {
                "NON_PRIVATE_TELEMETRY": True,
                "run_id": run_id,
                "config_fingerprint": fingerprint,
                "method": arm["method"],
                "privacy": "dp",
                "dataset": arm["dataset"],
                "base_model": arm["model_id"],
                "model_revision": arm["model_revision"],
                "step": step,
                "raw_clip_fraction": 0.4 + step / 1000.0,
                "raw_reference_small_gradient_proxy": 0.2,
            }
            for step in range(1, 151)
        ]
        raw_text = "".join(json.dumps(row) + "\n" for row in raw_records)
        raw_path = root / "results" / "research_raw" / "NON_PRIVATE_train_log.jsonl"
        raw_path.write_text(raw_text, encoding="utf-8")
        raw_sha = hashlib.sha256(raw_text.encode()).hexdigest()
        metrics = {
            name: {
                "count": 150,
                "missing": 0,
                "mean": 0.5 + index / 100.0,
                "last": 0.6 + index / 100.0,
            }
            for name in campaign.ANALYSIS_REQUIRED_METRICS
        }
        telemetry = {
            "summary_schema_version": 4,
            "NON_PRIVATE_TELEMETRY": True,
            "source": {
                "raw_sha256": raw_sha,
                "raw_physical_records": 150,
                "raw_unique_steps": 150,
                "raw_duplicate_records": 0,
            },
            "steps": {
                "count": 150,
                "first": 1,
                "last": 150,
                "missing_count": 0,
                "missing": [],
            },
            "run_identity": {
                "run_id": run_id,
                "config_fingerprint": fingerprint,
                "method": arm["method"],
                "privacy": "dp",
                "dataset": arm["dataset"],
                "base_model": arm["model_id"],
                "model_revision": arm["model_revision"],
                "resolved_model_revision": arm["model_revision"],
            },
            "metrics": metrics,
        }
        (root / "results" / "research_raw" / "telemetry_summary.json").write_text(
            json.dumps(telemetry), encoding="utf-8"
        )
        curve_records = [
            {
                "PUBLIC_VALIDATION_DATA": True,
                "NON_PRIVATE_SELECTION_METRIC": True,
                "records": 800,
                "run_id": run_id,
                "config_fingerprint": fingerprint,
                "planned_update_steps": 150,
                "manifest_sha256": manifest_sha,
                "selection_metric": "response_only_mean_per_record_causal_lm_loss",
                "loss_definition": (
                    "response_only_per_record_mean_of_nonignored_next_token_losses"
                ),
                "validation_indices_sha256": (
                    campaign.GLUE_SLACLIP_VALIDATION_INDICES_SHA256
                ),
                "validation_record_hashes_sha256": (
                    campaign.GLUE_SLACLIP_VALIDATION_RECORDS_SHA256
                ),
                "step": step,
                "loss_mean": validation_loss + (150 - step) / 1000.0,
                "token_mean_loss": validation_loss,
                "supervised_tokens": 3200,
            }
            for step in (0, 50, 100, 150)
        ]
        (root / "results" / "validation" / "validation_curve.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in curve_records), encoding="utf-8"
        )
    return manifest


def test_glue_slaclip_analyzer_uses_locked_public_holdout(tmp_path: Path) -> None:
    _write_focused_campaign(tmp_path)

    campaign.analyze(tmp_path)
    ranking = json.loads(
        (tmp_path / "artifacts" / "glue_slaclip_screen_ranking.json").read_text()
    )
    assert ranking["best_fixed"]["candidate"] == "fixed-c0p5"
    assert ranking["best_slaclip"]["candidate"] == "full-sla-c1-rho055"
    assert len(ranking["fixed_ranking"]) == 5
    assert len(ranking["slaclip_ranking"]) == 5
    assert len(ranking["initial_C_sensitivity_controls"]) == 2
    regimes = json.loads(
        (tmp_path / "artifacts" / "clipping_regime_summary.json").read_text()
    )
    assert sum(row["slaclip_arms"] for row in regimes["rows"]) == 5
    validation_curve = (
        tmp_path / "artifacts" / "public_validation_curve.csv"
    ).read_text().splitlines()
    assert len(validation_curve) == 1 + 12 * 4


def test_glue_slaclip_analyzer_rejects_candidate_identity_mismatch(tmp_path: Path) -> None:
    manifest = _write_focused_campaign(tmp_path)
    first = manifest["arms"][0]
    status_path = tmp_path / first["relative_root"] / "adapter" / "run_status.json"
    status = json.loads(status_path.read_text())
    status["config"]["dp_max_grad_norm"] = 99.0
    status_path.write_text(json.dumps(status), encoding="utf-8")
    with pytest.raises(campaign.CampaignError, match="arm config mismatch"):
        campaign.analyze(tmp_path)


def test_glue_slaclip_analyzer_rejects_stale_summary(tmp_path: Path) -> None:
    manifest = _write_focused_campaign(tmp_path)
    first = manifest["arms"][0]
    summary_path = (
        tmp_path / first["relative_root"]
        / "results" / "research_raw" / "telemetry_summary.json"
    )
    summary = json.loads(summary_path.read_text())
    summary["source"]["raw_sha256"] = "0" * 64
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    with pytest.raises(campaign.CampaignError, match="stale or incomplete"):
        campaign.analyze(tmp_path)


def test_full_baseline_profile_requires_pinned_12b_revision() -> None:
    with pytest.raises(campaign.CampaignError, match="requires a pinned 12B"):
        campaign.build_manifest(
            CODE_SHA, REV_4B, REV_9B, profile="baseline-reproduction"
        )


def test_task_average_supports_glue_and_math_headers(tmp_path: Path) -> None:
    glue = tmp_path / "glue.csv"
    glue.write_text("method,GLUE8_Avg\nslaclip,0.75\n", encoding="utf-8")
    math = tmp_path / "math.csv"
    math.write_text("gsm8k,Average\n0.6,0.55\n", encoding="utf-8")
    assert campaign._task_average(glue) == pytest.approx(0.75)
    assert campaign._task_average(math) == pytest.approx(0.55)


@pytest.mark.parametrize(
    ("value", "expected"),
    [(0.69, "lt_70pct"), (0.7, "70_to_lt_90pct"), (0.9, "90_to_lt_98pct"), (0.98, "ge_98pct")],
)
def test_clip_regime_bins_are_explicit(value: float, expected: str) -> None:
    assert campaign._clip_bin(value) == expected


def test_paper_coverage_timeout_signal_reaches_job_steps() -> None:
    wrapper = (ROOT / "scripts" / "submit_paper_coverage_campaign.sh").read_text()
    lane = (ROOT / "slurm" / "paper_coverage_lane.sh").read_text()
    assert "--signal=TERM@180" in wrapper
    assert "--signal=B:TERM@180" not in wrapper
    assert 'state=interrupted' in lane
    assert 'write_arm_status "${state}" "${code}"' in lane
    assert 'write_arm_status completed 0\n  CURRENT_ARM_STATUS=""' in lane
