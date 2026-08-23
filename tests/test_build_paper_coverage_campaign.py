from __future__ import annotations

import importlib.util
import hashlib
import json
from pathlib import Path
from typing import Callable

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


def test_cached_baseline_gap_profile_runs_only_two_missing_4b_settings() -> None:
    manifest = campaign.build_manifest(
        CODE_SHA, REV_4B, REV_9B, profile="baseline-gap-fill-cached"
    )
    arms = manifest["arms"]
    assert [arm["setting_id"] for arm in arms] == [
        "glue8-4b-eps6-r32",
        "math10k-4b-eps6-r32",
    ]
    assert [arm["steps"] for arm in arms] == [500, 300]
    assert all(
        arm["model_id"] == campaign.MODEL_4B
        and arm["model_revision"] == REV_4B
        and arm["method"] == "baseline"
        and arm["initial_c"] == 1.0
        and arm["seed"] == 42
        and arm["eval_limit"] == 0
        for arm in arms
    )
    metadata = manifest["baseline_reproduction"]
    assert metadata["full_length"] is True
    assert metadata["covered_settings"] == 2
    assert metadata["gap_fill"] == {
        "settings": ["glue8-4b-eps6-r32", "math10k-4b-eps6-r32"],
        "completed_external_settings": 7,
        "expected_cached_coverage_after_merge": 9,
        "merge_requires_hash_validated_external_evidence": True,
    }
    assert "12B" in metadata["excluded_setting"]
    assert len(
        campaign._plan_bytes(manifest, 0, include_all=True).decode().splitlines()
    ) == 2


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


def test_glue_high_c_refinement_predeclares_dynamic_two_stage_recipe() -> None:
    manifest = campaign.build_manifest(
        CODE_SHA, REV_4B, REV_9B, profile="glue-high-c-refinement"
    )
    arms = manifest["arms"]
    assert len(arms) == 5
    assert manifest["seed"] == 44
    assert {arm["initial_c"] for arm in arms} == {3.0, 5.0, 7.5, 10.0, 15.0}
    assert all(
        arm["method"] == "baseline"
        and arm["role"] == "high_C_fixed_candidate"
        and arm["stage"] == 1
        and arm["seed"] == 44
        and arm["steps"] == 200
        and arm["lane"] == 0
        for arm in arms
    )
    refinement = manifest["glue_high_c_refinement"]
    assert refinement["stage2_recipe"]["rho_quantiles"] == [
        0.10, 0.25, 0.50, 0.75, 0.90
    ]
    assert refinement["stage2_recipe"]["rho_bounds"] == [0.20, 0.90]
    assert refinement["stage2_recipe"]["require_unique_rho_values"] is True
    assert refinement["stage2_recipe"]["primary"] == {
        "arms": 5,
        "initial_C": "stage1_best_fixed_C",
        "eta": 0.02,
    }
    assert refinement["selection"]["validation_curve_steps"] == [
        0, 50, 100, 150, 200
    ]
    assert refinement["selection"]["primary_metric"].startswith("step_200")
    source = refinement["preceding_screen_source"]
    assert source["job_id"] == "1411662"
    assert source["manifest_sha256"] == (
        "54215469ec16bc369b1075021751417610bba6b7b4c0dc791cd3c3618cc806b5"
    )
    assert source["submission_receipt_sha256"] == (
        "d399b04fea09f31a9b6aaaa410f1df7df2fa9a7feeaf8187ee6d44e9489b9926"
    )
    assert len(source["artifact_sha256"]) == 10
    assert source["expected_best_fixed"] == "fixed-c5p0"
    assert source["expected_best_slaclip"] == "full-sla-c1-rho055"
    assert all(len(value) == 6 for value in source["artifact_sha256"].values())
    assert refinement["privacy_scope"]["end_to_end_dp_claim"] is False
    assert "NON_PRIVATE" in refinement["privacy_scope"]["target_selection"]
    plan = campaign._plan_bytes(manifest, 0, include_all=True).decode().splitlines()
    assert len(plan) == 5 and all(line.startswith("0|") for line in plan)


def test_glue_r8_slack_screen_predeclares_fresh_seed_two_stage_recipe() -> None:
    manifest = campaign.build_manifest(
        CODE_SHA, REV_4B, REV_9B, profile="glue-r8-slack-screen"
    )
    arms = manifest["arms"]
    assert len(arms) == 6
    assert manifest["seed"] == 45
    assert manifest["inference_class"].startswith("two_seed_exploratory")
    assert {arm["setting_id"] for arm in arms} == {"glue8-4b-eps6-r8"}
    assert {arm["lora_r"] for arm in arms} == {8}
    assert {arm["initial_c"] for arm in arms} == {
        0.5, 1.0, 2.0, 5.0, 10.0, 15.0,
    }
    assert all(
        arm["method"] == "baseline"
        and arm["role"] == "slack_fixed_candidate"
        and arm["stage"] == 1
        and arm["seed"] == 45
        and arm["steps"] == 200
        and arm["lane"] == 0
        for arm in arms
    )
    screen = manifest["glue_r8_slack_screen"]
    assert screen["stage2_recipe"]["seed"] == 46
    assert screen["stage2_recipe"]["fresh_relative_to_stage1"] is True
    assert screen["stage2_recipe"]["rho_quantiles"] == [
        0.10, 0.25, 0.50, 0.75, 0.90,
    ]
    assert screen["stage2_recipe"]["rho_bounds"] == [0.05, 0.95]
    assert "not assumed bounded" in screen["stage2_recipe"][
        "rho_source_value_domain"
    ]
    assert screen["stage2_recipe"]["rho_transform"] == (
        "compute five raw quantiles, then project each to [0.05,0.95]"
    )
    assert screen["stage2_recipe"]["primary"] == {
        "arms": 5,
        "initial_C": "stage1_best_fixed_C",
        "eta": 0.02,
    }
    assert screen["primary_gate"] == {
        "endpoint": "best_slaclip_strictly_lower_than_fresh_fixed",
        "full_auc": "best_slaclip_not_higher_than_fresh_fixed",
        "late_auc": "best_slaclip_not_higher_than_fresh_fixed",
        "boundary_block": "stage1_best_fixed_C_at_grid_boundary",
    }
    assert screen["motivation_sources"]["rank16_negative_decision"]["job_id"] == (
        "1413408"
    )
    baseline = screen["motivation_sources"]["rank8_complete_baseline"]
    assert baseline["job_id"] == "1402286"
    assert baseline["parent_job_terminal_state"] == "TIMEOUT"
    assert baseline["raw_records"] == 500
    assert baseline["official_glue_validation_average"] == pytest.approx(
        0.7662401098508378
    )
    assert len(campaign._plan_bytes(manifest, 0, include_all=True).splitlines()) == 6


def _write_focused_campaign(
    root_path: Path,
    *,
    profile: str = "glue-slaclip-screen",
    validation_losses: dict[str, float] | None = None,
    arms_override: list[dict] | None = None,
    index_offset: int = 0,
) -> dict:
    if arms_override is None:
        manifest = campaign.build_manifest(
            CODE_SHA, REV_4B, REV_9B, profile=profile
        )
        (root_path / "plans").mkdir()
        (root_path / "plans" / "manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )
        arms = manifest["arms"]
    else:
        manifest = json.loads(
            (root_path / "plans" / "manifest.json").read_text(encoding="utf-8")
        )
        arms = arms_override
    for local_index, arm in enumerate(arms):
        index = index_offset + local_index
        steps = int(arm["steps"])
        root = root_path / arm["relative_root"]
        (root / "adapter").mkdir(parents=True)
        (root / "results" / "research_raw").mkdir(parents=True)
        (root / "results" / "validation").mkdir(parents=True)
        validation_loss = (
            validation_losses[arm["candidate_id"]]
            if validation_losses is not None
            else 1.0 + index / 100.0
        )
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
            "total_update_steps": steps,
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
            "update_steps": steps,
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
                "raw_reference_conditional_clip_fraction": (
                    0.30 + index / 100.0 + step / 2000.0
                ),
                "raw_reference_conditional_clip_fraction_valid": True,
            }
            for step in range(1, steps + 1)
        ]
        raw_text = "".join(json.dumps(row) + "\n" for row in raw_records)
        raw_path = root / "results" / "research_raw" / "NON_PRIVATE_train_log.jsonl"
        raw_path.write_text(raw_text, encoding="utf-8")
        raw_sha = hashlib.sha256(raw_text.encode()).hexdigest()
        metrics = {
            name: {
                "count": steps,
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
                "raw_physical_records": steps,
                "raw_unique_steps": steps,
                "raw_duplicate_records": 0,
            },
            "steps": {
                "count": steps,
                "first": 1,
                "last": steps,
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
                "planned_update_steps": steps,
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
                "loss_mean": validation_loss + (steps - step) / 1000.0,
                "token_mean_loss": validation_loss,
                "supervised_tokens": 3200,
            }
            for step in sorted(campaign._expected_validation_curve_steps(steps))
        ]
        (root / "results" / "validation" / "split_manifest.json").write_text(
            json.dumps(split), encoding="utf-8"
        )
        (root / "results" / "validation" / "validation_metrics.json").write_text(
            json.dumps(validation), encoding="utf-8"
        )
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


def _write_high_c_stage1_and_lock(tmp_path: Path) -> dict:
    losses = {
        "fixed-c3p0": 0.50,
        "fixed-c5p0": 0.45,
        "fixed-c7p5": 0.40,
        "fixed-c10p0": 0.35,
        "fixed-c15p0": 0.38,
    }
    _write_focused_campaign(
        tmp_path,
        profile="glue-high-c-refinement",
        validation_losses=losses,
    )
    campaign.lock_high_c_refinement(tmp_path)
    campaign.lock_high_c_refinement(tmp_path)
    return json.loads(
        (tmp_path / "selection" / "high_c_stage1_lock.json").read_text()
    )


def _write_r8_stage1(
    tmp_path: Path, *, boundary_winner: bool = False
) -> dict:
    losses = {
        "fixed-c0p5": 0.60,
        "fixed-c1p0": 0.55,
        "fixed-c2p0": 0.50,
        "fixed-c5p0": 0.40,
        "fixed-c10p0": 0.45,
        "fixed-c15p0": 0.35 if boundary_winner else 0.46,
    }
    return _write_focused_campaign(
        tmp_path,
        profile="glue-r8-slack-screen",
        validation_losses=losses,
    )


def _write_r8_stage1_and_lock(
    tmp_path: Path, *, boundary_winner: bool = False
) -> dict:
    _write_r8_stage1(tmp_path, boundary_winner=boundary_winner)
    campaign.lock_glue_r8_slack_screen(tmp_path)
    campaign.lock_glue_r8_slack_screen(tmp_path)
    return json.loads(
        (tmp_path / "selection" / "r8_slack_stage1_lock.json").read_text()
    )


def _replace_r8_winner_conditional_proxy(
    tmp_path: Path,
    manifest: dict,
    value_for_post_burn_in_index: Callable[[int], float],
) -> None:
    winner = next(
        arm for arm in manifest["arms"] if arm["candidate_id"] == "fixed-c5p0"
    )
    raw_path = (
        tmp_path / winner["relative_root"]
        / "results" / "research_raw" / "NON_PRIVATE_train_log.jsonl"
    )
    records = [json.loads(line) for line in raw_path.read_text().splitlines()]
    for record in records:
        step = int(record["step"])
        if step >= 51:
            record["raw_reference_conditional_clip_fraction"] = (
                value_for_post_burn_in_index(step - 51)
            )
    raw_text = "".join(json.dumps(record) + "\n" for record in records)
    raw_path.write_text(raw_text, encoding="utf-8")
    summary_path = raw_path.with_name("telemetry_summary.json")
    summary = json.loads(summary_path.read_text())
    summary["source"]["raw_sha256"] = hashlib.sha256(raw_text.encode()).hexdigest()
    summary_path.write_text(json.dumps(summary), encoding="utf-8")


def test_r8_slack_lock_builds_fresh_seed_comparator_and_dynamic_plan(
    tmp_path: Path,
) -> None:
    lock = _write_r8_stage1_and_lock(tmp_path)
    assert lock["best_fixed"]["candidate_id"] == "fixed-c5p0"
    assert lock["best_fixed"]["fixed_C"] == 5.0
    assert lock["stage1_seed"] == 45
    assert lock["stage2_seed"] == 46
    assert lock["stage2_seed_is_fresh"] is True
    assert lock["conditional_clip_proxy_records"] == 150
    assert lock["stage1_fixed_winner_at_search_boundary"] is False
    assert lock["stage1_fixed_winner_at_grid_min"] is False
    assert lock["stage1_fixed_winner_at_grid_max"] is False
    assert lock["stage1_boundary_warning"] is None
    rhos = list(lock["rho_quantiles"].values())
    assert rhos == sorted(rhos) and len(set(rhos)) == 5
    assert all(0.05 <= rho <= 0.95 for rho in rhos)
    arms = lock["stage2_arms"]
    fresh = [arm for arm in arms if arm["role"] == "fresh_fixed_comparator"]
    primary = [arm for arm in arms if arm["role"] == "slaclip_target_candidate"]
    controls = [
        arm for arm in arms
        if arm["role"] in {
            "controller_speed_control", "initial_C_sensitivity_control",
        }
    ]
    assert len(fresh) == 1 and len(primary) == 5 and len(controls) == 2
    assert fresh[0]["method"] == "baseline"
    assert fresh[0]["initial_c"] == 5.0
    assert fresh[0]["seed"] == 46
    assert all(
        arm["method"] == "slaclip"
        and arm["initial_c"] == 5.0
        and arm["eta"] == 0.02
        and arm["seed"] == 46
        for arm in primary
    )
    assert len((tmp_path / "plans" / "stage2-slaclip.tsv").read_text().splitlines()) == 8
    assert lock["stage2_plan_sha256"] == hashlib.sha256(
        (tmp_path / "plans" / "stage2-slaclip.tsv").read_bytes()
    ).hexdigest()


def test_r8_slack_lock_projects_finite_proxy_above_one_after_quantiles(
    tmp_path: Path,
) -> None:
    manifest = _write_r8_stage1(tmp_path)
    _replace_r8_winner_conditional_proxy(
        tmp_path,
        manifest,
        lambda index: 0.3 + 0.8 * index / 149.0,
    )

    campaign.lock_glue_r8_slack_screen(tmp_path)
    lock = json.loads(
        (tmp_path / "selection" / "r8_slack_stage1_lock.json").read_text()
    )
    raw = lock["raw_conditional_clip_proxy_quantiles"]
    projected = lock["rho_quantiles"]
    assert raw["q90"] > 1.0
    assert projected["q90"] == 0.95
    assert list(projected.values()) == sorted(projected.values())
    assert len(set(projected.values())) == 5


def test_r8_slack_lock_rejects_duplicate_rhos_after_projection(
    tmp_path: Path,
) -> None:
    manifest = _write_r8_stage1(tmp_path)
    _replace_r8_winner_conditional_proxy(
        tmp_path,
        manifest,
        lambda index: 0.8 + 0.4 * index / 149.0,
    )

    with pytest.raises(
        campaign.CampaignError,
        match="rho grid is not unique after clamping",
    ):
        campaign.lock_glue_r8_slack_screen(tmp_path)


def test_r8_slack_analyzer_uses_fresh_comparator_and_strict_gate(
    tmp_path: Path,
) -> None:
    lock = _write_r8_stage1_and_lock(tmp_path)
    stage2_losses = {
        arm["candidate_id"]: 0.42 + index / 100.0
        for index, arm in enumerate(lock["stage2_arms"])
    }
    fresh = next(
        arm for arm in lock["stage2_arms"]
        if arm["role"] == "fresh_fixed_comparator"
    )
    best = next(
        arm for arm in lock["stage2_arms"]
        if arm["candidate_id"] == "full-sla-q10-eta002"
    )
    stage2_losses[fresh["candidate_id"]] = 0.40
    stage2_losses[best["candidate_id"]] = 0.35
    _write_focused_campaign(
        tmp_path,
        profile="glue-r8-slack-screen",
        validation_losses=stage2_losses,
        arms_override=lock["stage2_arms"],
        index_offset=100,
    )
    campaign.analyze(tmp_path)
    ranking = json.loads(
        (tmp_path / "artifacts" / "glue_r8_slack_screen_ranking.json").read_text()
    )
    assert ranking["stage1_best_fixed"]["candidate"] == "fixed-c5p0"
    assert ranking["fresh_fixed"]["candidate"] == fresh["candidate_id"]
    assert ranking["best_slaclip"]["candidate"] == best["candidate_id"]
    assert ranking["best_slaclip"]["delta_vs_setting_best_fixed"] == pytest.approx(
        0.05
    )
    assert ranking["primary_gate"] == {
        "endpoint_strictly_lower": True,
        "full_auc_not_worse": True,
        "late_auc_not_worse": True,
        "performance_gate_passed": True,
        "boundary_blocked": False,
        "confirmation_allowed": True,
        "block_reasons": [],
    }
    assert len(ranking["arm_artifact_sha256"]) == 14
    assert len(
        (tmp_path / "artifacts" / "baseline_telemetry_steps.csv")
        .read_text().splitlines()
    ) == 1 + 14 * 200


def test_r8_slack_boundary_winner_blocks_later_confirmation(
    tmp_path: Path,
) -> None:
    lock = _write_r8_stage1_and_lock(tmp_path, boundary_winner=True)
    assert lock["best_fixed"]["fixed_C"] == 15.0
    assert lock["stage1_fixed_winner_at_search_boundary"] is True
    assert lock["stage1_fixed_winner_at_grid_min"] is False
    assert lock["stage1_fixed_winner_at_grid_max"] is True
    stage2_losses = {
        arm["candidate_id"]: 0.42 + index / 100.0
        for index, arm in enumerate(lock["stage2_arms"])
    }
    fresh = next(
        arm for arm in lock["stage2_arms"]
        if arm["role"] == "fresh_fixed_comparator"
    )
    best = next(
        arm for arm in lock["stage2_arms"]
        if arm["candidate_id"] == "full-sla-q10-eta002"
    )
    stage2_losses[fresh["candidate_id"]] = 0.40
    stage2_losses[best["candidate_id"]] = 0.35
    _write_focused_campaign(
        tmp_path,
        profile="glue-r8-slack-screen",
        validation_losses=stage2_losses,
        arms_override=lock["stage2_arms"],
        index_offset=100,
    )
    campaign.analyze(tmp_path)
    gate = json.loads(
        (tmp_path / "artifacts" / "glue_r8_slack_screen_ranking.json").read_text()
    )["primary_gate"]
    assert gate["performance_gate_passed"] is True
    assert gate["boundary_blocked"] is True
    assert gate["confirmation_allowed"] is False
    assert gate["block_reasons"] == ["stage1_fixed_winner_at_search_boundary"]


def test_high_c_stage1_lock_derives_unique_stage2_plan(tmp_path: Path) -> None:
    lock = _write_high_c_stage1_and_lock(tmp_path)
    assert lock["best_fixed"]["candidate_id"] == "fixed-c10p0"
    assert lock["best_fixed"]["fixed_C"] == 10.0
    assert lock["conditional_clip_proxy_records"] == 150
    assert list(lock["rho_quantiles"]) == ["q10", "q25", "q50", "q75", "q90"]
    rhos = list(lock["rho_quantiles"].values())
    assert rhos == sorted(rhos) and len(set(rhos)) == 5
    assert all(0.20 <= rho <= 0.90 for rho in rhos)
    arms = lock["stage2_arms"]
    primary = [arm for arm in arms if arm["role"] == "slaclip_target_candidate"]
    assert len(primary) == 5
    assert all(
        arm["initial_c"] == 10.0
        and arm["eta"] == 0.02
        and arm["steps"] == 200
        and arm["seed"] == 44
        and arm["lane"] == 0
        for arm in primary
    )
    speed = next(arm for arm in arms if arm["role"] == "controller_speed_control")
    sensitivity = next(
        arm for arm in arms if arm["role"] == "initial_C_sensitivity_control"
    )
    assert speed["initial_c"] == 10.0 and speed["eta"] == 0.05
    assert sensitivity["initial_c"] == 5.0 and sensitivity["eta"] == 0.02
    assert speed["rho"] == sensitivity["rho"] == lock["rho_quantiles"]["q50"]
    stage2_plan = (tmp_path / "plans" / "stage2-slaclip.tsv").read_text().splitlines()
    assert len(stage2_plan) == 7
    assert all(line.startswith("0|") for line in stage2_plan)
    assert lock["stage2_plan_sha256"] == hashlib.sha256(
        (tmp_path / "plans" / "stage2-slaclip.tsv").read_bytes()
    ).hexdigest()
    assert (tmp_path / "selection" / "high_c_stage1_lock.json.sha256").is_file()
    assert (tmp_path / "plans" / "stage2-slaclip.tsv.sha256").is_file()


def test_high_c_analyzer_uses_endpoint_primary_and_reports_auc(tmp_path: Path) -> None:
    lock = _write_high_c_stage1_and_lock(tmp_path)
    stage2_losses = {
        arm["candidate_id"]: 0.34 + index / 100.0
        for index, arm in enumerate(lock["stage2_arms"])
    }
    _write_focused_campaign(
        tmp_path,
        profile="glue-high-c-refinement",
        validation_losses=stage2_losses,
        arms_override=lock["stage2_arms"],
        index_offset=100,
    )
    campaign.analyze(tmp_path)
    ranking = json.loads(
        (tmp_path / "artifacts" / "glue_high_c_refinement_ranking.json").read_text()
    )
    assert ranking["best_fixed"]["candidate"] == "fixed-c10p0"
    assert ranking["best_slaclip"]["candidate"] == "full-sla-q10-eta002"
    assert ranking["slaclip_beats_best_fixed_primary"] is True
    assert len(ranking["fixed_ranking"]) == 5
    assert len(ranking["slaclip_ranking"]) == 5
    assert len(ranking["controls"]) == 2
    assert ranking["stage2_plan_sha256"] == lock["stage2_plan_sha256"]
    assert ranking["privacy_scope"]["end_to_end_dp_claim"] is False
    assert len(ranking["arm_artifact_sha256"]) == 12
    assert all(
        row["full_normalized_validation_loss_auc"] is not None
        and row["late_window_normalized_validation_loss_auc"] is not None
        for row in ranking["fixed_ranking"] + ranking["slaclip_ranking"]
    )
    curve_lines = (
        tmp_path / "artifacts" / "public_validation_curve.csv"
    ).read_text().splitlines()
    summary_header = (
        tmp_path / "artifacts" / "paper_coverage_summary.csv"
    ).read_text().splitlines()[0]
    regime_header = (
        tmp_path / "artifacts" / "clipping_regime_summary.csv"
    ).read_text().splitlines()[0]
    telemetry_lines = (
        tmp_path / "artifacts" / "baseline_telemetry_steps.csv"
    ).read_text().splitlines()
    assert len(curve_lines) == 1 + 12 * 5
    assert len(telemetry_lines) == 1 + 12 * 200
    assert "NON_PRIVATE_TELEMETRY" in summary_header
    assert "NON_PRIVATE_TELEMETRY" in regime_header
    assert "NON_PRIVATE_SELECTION_METRIC" in curve_lines[0]
    assert "NON_PRIVATE_TELEMETRY" in telemetry_lines[0]
    assert "raw_signal_retention_ratio" in telemetry_lines[0]
    assert "raw_reference_conditional_clip_fraction_valid" in telemetry_lines[0]
    assert "raw_slack_indicator_noise_residual_rmse" in telemetry_lines[0]


def test_high_c_analyzer_rejects_stage2_plan_tampering(tmp_path: Path) -> None:
    lock = _write_high_c_stage1_and_lock(tmp_path)
    stage2_losses = {
        arm["candidate_id"]: 0.34 + index / 100.0
        for index, arm in enumerate(lock["stage2_arms"])
    }
    _write_focused_campaign(
        tmp_path,
        profile="glue-high-c-refinement",
        validation_losses=stage2_losses,
        arms_override=lock["stage2_arms"],
        index_offset=100,
    )
    plan_path = tmp_path / "plans" / "stage2-slaclip.tsv"
    plan_path.write_bytes(plan_path.read_bytes() + b"\n")
    with pytest.raises(campaign.CampaignError, match="immutable artifact"):
        campaign.analyze(tmp_path)


def test_high_c_analyzer_binds_stage2_metrics_to_status(tmp_path: Path) -> None:
    lock = _write_high_c_stage1_and_lock(tmp_path)
    stage2_losses = {
        arm["candidate_id"]: 0.34 + index / 100.0
        for index, arm in enumerate(lock["stage2_arms"])
    }
    _write_focused_campaign(
        tmp_path,
        profile="glue-high-c-refinement",
        validation_losses=stage2_losses,
        arms_override=lock["stage2_arms"],
        index_offset=100,
    )
    arm = lock["stage2_arms"][0]
    metrics_path = (
        tmp_path / arm["relative_root"]
        / "results" / "validation" / "validation_metrics.json"
    )
    metrics = json.loads(metrics_path.read_text())
    metrics["loss_mean"] += 1.0
    metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
    with pytest.raises(campaign.CampaignError, match="not bound to split/metrics"):
        campaign.analyze(tmp_path)


def test_high_c_lock_rejects_nonunique_clamped_rho_grid(tmp_path: Path) -> None:
    losses = {
        "fixed-c3p0": 0.50,
        "fixed-c5p0": 0.45,
        "fixed-c7p5": 0.40,
        "fixed-c10p0": 0.35,
        "fixed-c15p0": 0.38,
    }
    manifest = _write_focused_campaign(
        tmp_path,
        profile="glue-high-c-refinement",
        validation_losses=losses,
    )
    winner = next(arm for arm in manifest["arms"] if arm["candidate_id"] == "fixed-c10p0")
    raw_path = (
        tmp_path / winner["relative_root"]
        / "results" / "research_raw" / "NON_PRIVATE_train_log.jsonl"
    )
    records = [json.loads(line) for line in raw_path.read_text().splitlines()]
    for record in records:
        record["raw_reference_conditional_clip_fraction"] = 0.1
    raw_text = "".join(json.dumps(record) + "\n" for record in records)
    raw_path.write_text(raw_text, encoding="utf-8")
    summary_path = raw_path.with_name("telemetry_summary.json")
    summary = json.loads(summary_path.read_text())
    summary["source"]["raw_sha256"] = hashlib.sha256(raw_text.encode()).hexdigest()
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    with pytest.raises(campaign.CampaignError, match="rho grid is not unique"):
        campaign.lock_high_c_refinement(tmp_path)


@pytest.mark.parametrize(
    "nonfinite_proxy", [float("nan"), float("inf"), float("-inf")]
)
def test_high_c_lock_rejects_nonfinite_conditional_proxy(
    tmp_path: Path, nonfinite_proxy: float
) -> None:
    losses = {
        "fixed-c3p0": 0.50,
        "fixed-c5p0": 0.45,
        "fixed-c7p5": 0.40,
        "fixed-c10p0": 0.35,
        "fixed-c15p0": 0.38,
    }
    manifest = _write_focused_campaign(
        tmp_path,
        profile="glue-high-c-refinement",
        validation_losses=losses,
    )
    winner = next(
        arm for arm in manifest["arms"] if arm["candidate_id"] == "fixed-c10p0"
    )
    raw_path = (
        tmp_path / winner["relative_root"]
        / "results" / "research_raw" / "NON_PRIVATE_train_log.jsonl"
    )
    records = [json.loads(line) for line in raw_path.read_text().splitlines()]
    records[50]["raw_reference_conditional_clip_fraction"] = nonfinite_proxy
    raw_text = "".join(json.dumps(record) + "\n" for record in records)
    raw_path.write_text(raw_text, encoding="utf-8")
    summary_path = raw_path.with_name("telemetry_summary.json")
    summary = json.loads(summary_path.read_text())
    summary["source"]["raw_sha256"] = hashlib.sha256(raw_text.encode()).hexdigest()
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    with pytest.raises(campaign.CampaignError, match="not finite"):
        campaign.lock_high_c_refinement(tmp_path)


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


def test_paper_coverage_shell_contract_uses_queue_friendly_a100_defaults() -> None:
    wrapper = (ROOT / "scripts" / "submit_paper_coverage_campaign.sh").read_text()
    worker = (ROOT / "slurm" / "paper_coverage_campaign.sbatch").read_text()
    lane = (ROOT / "slurm" / "paper_coverage_lane.sh").read_text()
    assert "glue-high-c-refinement" in wrapper
    assert "glue-r8-slack-screen" in wrapper
    assert "baseline-gap-fill-cached" in wrapper
    assert "DEFAULT_PARTITION=a100" in wrapper
    assert "DEFAULT_WALLTIME=1-00:00:00" in wrapper
    assert "DEFAULT_GPU_TYPE=a100" in wrapper
    assert "DEFAULT_GPU_LANES=1" in wrapper
    assert "DEFAULT_CPUS_PER_TASK=8" in wrapper
    assert "DEFAULT_HOST_MEMORY=80G" in wrapper
    assert "DEFAULT_STEP_MEMORY=76G" in wrapper
    assert "DEFAULT_HOST_MEMORY=48G" not in wrapper
    assert "DEFAULT_GPU_TYPE=h200" not in wrapper
    assert 'COVERAGE_PROFILE="${PRISM_COVERAGE_PROFILE:-}"' in wrapper
    assert "PRISM_COVERAGE_PROFILE must be set explicitly" in wrapper
    assert '"step_gres": step_gres' in wrapper
    assert '"step_memory": step_memory' in wrapper
    assert "requires PRISM_GPU_LANES=1" in wrapper
    assert "stage1-high-c-fixed-screen" in worker
    assert "lock-high-c-refinement" in worker
    assert "lock-r8-slack-screen" in worker
    assert "stage2-derived-full-slaclip-screen" in worker
    assert "stage1-fixed.tsv" in worker and "stage2-slaclip.tsv" in worker
    assert worker.count("verify_plan_sidecar") >= 3
    assert "sbatch" not in worker
    assert 'RUN_REAL_SMOKE="${12:-true}"' in lane
    assert "expected_curve_steps" in lane
