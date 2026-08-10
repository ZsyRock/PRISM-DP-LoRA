from __future__ import annotations

import csv
import importlib.util
import json
import stat
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "build_math10k_4b_low_target_campaign.py"
SPEC = importlib.util.spec_from_file_location("low_target_campaign", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
low = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(low)


CODE_SHA = "a" * 40
MODEL_REVISION = "b" * 40
MODEL_ID = "google/gemma-3-4b-pt"
MANIFEST_SHA = "c" * 64


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _prepare(tmp_path: Path) -> tuple[Path, dict]:
    campaign = tmp_path / "campaign"
    low.prepare(
        campaign_root=campaign,
        code_sha=CODE_SHA,
        model_id=MODEL_ID,
        model_revision=MODEL_REVISION,
    )
    registry = json.loads(
        (campaign / "screen" / "candidate_registry.json").read_text(
            encoding="utf-8"
        )
    )
    return campaign, registry


def _split(*, marker: str = "shared") -> dict:
    return {
        "schema_version": 2,
        "manifest_sha256": MANIFEST_SHA,
        "protocol_stage": "selection",
        "validation_data_is_public": True,
        "seed": 1729,
        "validation_rows": 500,
        "validation_indices_sha256": marker,
    }


def _raw_rows(
    candidate: dict,
    *,
    steps: int,
    clip_fraction: float,
    target_override_step: int | None = None,
) -> list[dict]:
    params = candidate["params"]
    rows = []
    for step in range(1, steps + 1):
        threshold = float(params["dp_max_grad_norm"])
        row = {
            "NON_PRIVATE_TELEMETRY": True,
            "telemetry_mode": "research_raw",
            "method": candidate["method"],
            "privacy": "dp",
            "dataset": "math10k",
            "base_model": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "step": step,
            "raw_clip_fraction": clip_fraction,
            "dp_clip_threshold": threshold,
            "dp_next_clip_threshold": threshold,
            "raw_clipping_bias_norm": 2.0,
            "raw_realized_noise_norm": 4.0,
            "raw_signal_to_noise_ratio": 0.25,
            "raw_bias_noise_squared_error_proxy": 20.0,
        }
        if candidate["family"] == "slaclip":
            rho = float(params["slaclip_target_non_small_clip_fraction"])
            noisy_z = 0.0
            dynamic_target = rho * (1.0 - noisy_z)
            if step == target_override_step:
                dynamic_target -= 0.1
            row.update(
                {
                    "slaclip_small_gradient_proxy_noisy": noisy_z,
                    "slaclip_target_clipped_proxy": dynamic_target,
                    "slaclip_c_hit_min": False,
                    "slaclip_c_hit_max": False,
                    "raw_reference_small_gradient_proxy": 0.0,
                    "slack_indicator_noise_std": 0.03,
                }
            )
        rows.append(row)
    return rows


def _screen_accuracy(candidate: dict) -> float:
    c0 = float(candidate["params"]["dp_max_grad_norm"])
    rho = float(candidate["params"]["slaclip_target_non_small_clip_fraction"])
    return {
        (2.0, 0.8): 0.56,
        (3.0, 0.8): 0.54,
        (2.0, 0.9): 0.53,
        (3.0, 0.9): 0.57,
    }[(c0, rho)]


def _write_screen_run(
    campaign: Path,
    registry: dict,
    candidate: dict,
    *,
    split_marker: str = "shared",
    clip_fraction: float | None = None,
    target_override_step: int | None = None,
) -> None:
    paths = candidate["stage1_run"]
    common = registry["selection"]["common_config"]
    config = {
        **common,
        **candidate["params"],
        "method": candidate["method"],
        "seed": low.STAGE1_SEED,
        "resolved_model_revision": MODEL_REVISION,
        "implementation_git_dirty": False,
        "data_content_sha256": low.MATH10K_DATA_SHA256,
        "run_train": True,
        "dp_grad_sample_mode": "functorch",
        "lora_r": 16,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "target_modules": ["q_proj", "k_proj", "v_proj", "up_proj", "down_proj"],
        "prism_floor_mode": "scalar",
        "prism_floor_factor": 0.5,
        "prism_cond_max": 10000.0,
        "prism_lift_fix": "both",
        "prism_debias_second_moment": False,
        "config_fingerprint": f"screen-fingerprint-{candidate['id']}",
    }
    _write_json(
        campaign / paths["run_status"],
        {
            "state": "completed",
            "update_steps": low.STAGE1_STEPS,
            "method": candidate["method"],
            "privacy": "dp",
            "dataset": "math10k",
            "base_model": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "resolved_model_revision": MODEL_REVISION,
            "data_content_sha256": low.MATH10K_DATA_SHA256,
            "config_fingerprint": config["config_fingerprint"],
            "non_private_telemetry": True,
            "privacy_accounting": {
                "accountant": "prv",
                "grad_sample_mode": "functorch",
                "secure_mode": False,
                "scope": "single_training_run",
                "completed_update_steps": low.STAGE1_STEPS,
                "target_epsilon": 6.0,
                "target_delta": 1e-5,
                "epsilon_spent": 5.99,
                "noise_multiplier": 0.5,
                "sample_rate": 0.0068,
                "expected_batch_size": 64.0,
            },
            "config": config,
        },
    )
    _write_json(campaign / paths["split_manifest"], _split(marker=split_marker))
    _write_json(
        campaign / paths["validation_metrics"],
        {
            "numeric_exact_accuracy": _screen_accuracy(candidate),
            "loss_mean": 0.3,
            "records": 500,
            "seed": 1729,
            "validation_data_is_public": True,
            "PUBLIC_VALIDATION_DATA": True,
            "protocol_stage": "selection",
            "selection_metric": "public_math10k_numeric_exact_match_accuracy",
            "loss_definition": (
                "response_only_per_record_mean_of_nonignored_next_token_losses"
            ),
            "manifest_sha256": MANIFEST_SHA,
        },
    )
    rho = float(candidate["params"]["slaclip_target_non_small_clip_fraction"])
    raw_rows = _raw_rows(
        candidate,
        steps=low.STAGE1_STEPS,
        clip_fraction=rho if clip_fraction is None else clip_fraction,
        target_override_step=target_override_step,
    )
    for row in raw_rows:
        row["resolved_model_revision"] = MODEL_REVISION
        row["config_fingerprint"] = config["config_fingerprint"]
    _write_jsonl(campaign / paths["raw_log"], raw_rows)


def _write_all_screen_runs(campaign: Path, registry: dict) -> None:
    for candidate in registry["candidates"]:
        _write_screen_run(campaign, registry, candidate)


def test_prepare_locks_four_arm_150_step_screen_and_fixed_c2_control(
    tmp_path: Path,
) -> None:
    campaign, registry = _prepare(tmp_path)
    candidates = registry["candidates"]
    assert len(candidates) == 4
    assert {item["family"] for item in candidates} == {"slaclip"}
    assert {
        (
            item["params"]["dp_max_grad_norm"],
            item["params"]["slaclip_target_non_small_clip_fraction"],
            item["params"]["slaclip_eta"],
        )
        for item in candidates
    } == {
        (2.0, 0.8, 0.1),
        (3.0, 0.8, 0.1),
        (2.0, 0.9, 0.1),
        (3.0, 0.9, 0.1),
    }
    assert registry["selection"]["common_config"]["total_update_steps"] == 150
    assert registry["selection"]["tracking_audit"]["burn_in_steps_excluded"] == 50
    assert registry["confirmation_seeds"] == [401, 433, 467, 503, 547]
    fixed = registry["predeclared_strongest_fixed_c2"]
    assert fixed["id"] == "fixed-c2"
    assert fixed["params"]["dp_max_grad_norm"] == 2.0
    assert registry["historical_test_results_available_before_protocol"] is True
    assert registry["inference_class"] == (
        "locked_internal_paired_confirmation_not_untouched_external_replication"
    )

    rows = (campaign / "plans" / "stage1.tsv").read_text(
        encoding="utf-8"
    ).splitlines()
    assert len(rows) == 4
    assert all(len(row.split("|")) == 11 for row in rows)
    assert all(row.split("|")[1] == "selection" for row in rows)
    assert all(row.split("|")[4] == "42" for row in rows)
    assert stat.S_IMODE(
        (campaign / "screen" / "candidate_registry.json").stat().st_mode
    ) == 0o600

    low.prepare(
        campaign_root=campaign,
        code_sha=CODE_SHA,
        model_id=MODEL_ID,
        model_revision=MODEL_REVISION,
    )
    with pytest.raises(low.CampaignError, match="overwrite immutable"):
        low.prepare(
            campaign_root=campaign,
            code_sha="d" * 40,
            model_id=MODEL_ID,
            model_revision=MODEL_REVISION,
        )


def test_lock_selects_per_rho_and_writes_exact_15_arm_same_a100_plan(
    tmp_path: Path,
) -> None:
    campaign, registry = _prepare(tmp_path)
    _write_all_screen_runs(campaign, registry)
    result = low.lock(campaign_root=campaign)
    selection = json.loads(
        (campaign / "selection" / "selection.json").read_text(encoding="utf-8")
    )
    assert result["selected_rho_0p8"] == "sla-c2-r0p8-e0p1"
    assert result["selected_rho_0p9"] == "sla-c3-r0p9-e0p1"
    assert selection["predeclared_strongest_fixed_c2"]["candidate_id"] == "fixed-c2"
    assert selection["test_assets_accessed"] is False
    assert selection["historical_test_results_available_before_protocol"] is True

    rows = (campaign / "plans" / "final.tsv").read_text(
        encoding="utf-8"
    ).splitlines()
    assert len(rows) == 15
    fields = [row.split("|") for row in rows]
    assert {int(row[4]) for row in fields} == set(low.CONFIRMATION_SEEDS)
    assert {row[3] for row in fields} == {
        "slaclip-rho-0p8",
        "slaclip-rho-0p9",
        "predeclared-strongest-fixed-c2",
    }
    fixed_rows = [row for row in fields if row[3] == "predeclared-strongest-fixed-c2"]
    assert len(fixed_rows) == 5
    assert all(row[2] == "fixed-c2" and row[5] == "baseline" for row in fixed_rows)
    assert all(float(row[6]) == 2.0 for row in fixed_rows)


def test_lock_rejects_wrong_dynamic_target_formula(tmp_path: Path) -> None:
    campaign, registry = _prepare(tmp_path)
    _write_all_screen_runs(campaign, registry)
    candidate = next(
        item for item in registry["candidates"] if item["id"] == "sla-c2-r0p8-e0p1"
    )
    _write_screen_run(
        campaign,
        registry,
        candidate,
        target_override_step=low.STAGE1_BURN_IN_STEP + 1,
    )
    with pytest.raises(low.CampaignError, match=r"rho\*\(1-z_t\)"):
        low.lock(campaign_root=campaign)


def test_lock_rejects_different_public_split(tmp_path: Path) -> None:
    campaign, registry = _prepare(tmp_path)
    _write_all_screen_runs(campaign, registry)
    _write_screen_run(
        campaign,
        registry,
        registry["candidates"][-1],
        split_marker="different",
    )
    with pytest.raises(low.CampaignError, match="one identical public split"):
        low.lock(campaign_root=campaign)


def _final_config(record: dict, candidate: dict, registry: dict) -> dict:
    return {
        **registry["selection"]["common_config"],
        **candidate["params"],
        "method": candidate["method"],
        "seed": int(record["seed"]),
        "protocol_stage": "final",
        "total_update_steps": low.FINAL_STEPS,
        "val_set_size": 0,
        "run_train": True,
        "run_eval": True,
        "resolved_model_revision": MODEL_REVISION,
        "implementation_git_dirty": False,
        "data_content_sha256": low.MATH10K_DATA_SHA256,
        "dp_grad_sample_mode": "functorch",
        "raw_hist_bins": 128,
        "raw_hist_max": 30.0,
        "lora_r": 16,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "target_modules": ["q_proj", "k_proj", "v_proj", "up_proj", "down_proj"],
        "prism_floor_mode": "scalar",
        "prism_floor_factor": 0.5,
        "prism_cond_max": 10000.0,
        "prism_lift_fix": "both",
        "prism_debias_second_moment": False,
        "config_fingerprint": f"fingerprint-{candidate['id']}-{record['seed']}",
    }


def _write_final_run(
    record: dict,
    candidate: dict,
    registry: dict,
    *,
    task_accuracy: dict[str, float],
) -> None:
    arm = Path(record["arm_root"])
    config = _final_config(record, candidate, registry)
    split_manifest = {
        "schema_version": 2,
        "protocol_stage": "final",
        "validation_data_is_public": False,
        "seed": 1729,
        "requested_validation_rows": 0,
        "validation_rows": 0,
        "source_rows": 9919,
        "train_rows": 9919,
        "source_content_sha256": low.MATH10K_DATA_SHA256,
    }
    _write_json(
        arm / "adapter" / "run_status.json",
        {
            "state": "completed",
            "update_steps": low.FINAL_STEPS,
            "method": candidate["method"],
            "dataset": "math10k",
            "privacy": "dp",
            "base_model": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "resolved_model_revision": MODEL_REVISION,
            "data_content_sha256": low.MATH10K_DATA_SHA256,
            "data_split": split_manifest,
            "non_private_telemetry": True,
            "privacy_accounting": {
                "accountant": "prv",
                "grad_sample_mode": "functorch",
                "secure_mode": False,
                "scope": "single_training_run",
                "completed_update_steps": low.FINAL_STEPS,
                "target_epsilon": 6.0,
                "target_delta": 1e-5,
                "epsilon_spent": 5.99,
                "noise_multiplier": 0.5,
                "sample_rate": 0.0065,
                "expected_batch_size": 64.0,
            },
            "config": config,
        },
    )
    results = arm / "results"
    _write_json(results / "validation" / "split_manifest.json", split_manifest)
    _write_json(
        results / "evaluation_config.json",
        {
            "evaluation_schema_version": 1,
            "dataset": "math10k",
            "base_model": MODEL_ID,
            "requested_model_revision": MODEL_REVISION,
            "resolved_model_revision": MODEL_REVISION,
            "config_fingerprint": config["config_fingerprint"],
            "tasks": ["gsm8k", "AQuA", "mawps", "SVAMP"],
            "batch_size": 8,
            "num_beams": 4,
            "max_new_tokens": 256,
            "max_input_length": 1024,
            "test_assets": {
                task: {"rows": rows, "sha256": sha256}
                for task, (rows, sha256) in low.EXPECTED_TEST_ASSETS.items()
            },
        },
    )
    four_task = sum(task_accuracy.values()) / 4
    results.mkdir(parents=True, exist_ok=True)
    with (results / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["gsm8k", "AQuA", "mawps", "SVAMP", "Average"])
        writer.writerow(
            [
                task_accuracy["gsm8k"],
                task_accuracy["AQuA"],
                task_accuracy["mawps"],
                task_accuracy["SVAMP"],
                four_task,
            ]
        )
    clip = (
        float(candidate["params"]["slaclip_target_non_small_clip_fraction"])
        if candidate["family"] == "slaclip"
        else 0.9
    )
    raw_rows = _raw_rows(
        candidate,
        steps=low.FINAL_STEPS,
        clip_fraction=clip,
    )
    for row in raw_rows:
        row["resolved_model_revision"] = MODEL_REVISION
        row["config_fingerprint"] = config["config_fingerprint"]
    _write_jsonl(
        results / "research_raw" / "NON_PRIVATE_train_log.jsonl",
        raw_rows,
    )


def _write_all_final_runs(campaign: Path, registry: dict) -> None:
    candidates = low._candidate_map(registry)
    plan = low._parse_plan(campaign / "plans" / "final.tsv", campaign.resolve())
    base_by_seed = {
        401: 0.50,
        433: 0.51,
        467: 0.49,
        503: 0.52,
        547: 0.48,
    }
    for record in plan:
        candidate = candidates[record["candidate_id"]]
        clean = base_by_seed[record["seed"]]
        if record["role"] == "slaclip-rho-0p8":
            clean += 0.02
        elif record["role"] == "slaclip-rho-0p9":
            clean += 0.01
        tasks = {
            "gsm8k": clean,
            "AQuA": clean,
            "mawps": 0.9,
            "SVAMP": clean,
        }
        _write_final_run(record, candidate, registry, task_accuracy=tasks)


def _completed_campaign(tmp_path: Path) -> tuple[Path, dict]:
    campaign, registry = _prepare(tmp_path)
    _write_all_screen_runs(campaign, registry)
    low.lock(campaign_root=campaign)
    _write_all_final_runs(campaign, registry)
    return campaign, registry


def test_analyze_emits_primary_ci_and_exploratory_guard(tmp_path: Path) -> None:
    campaign, _ = _completed_campaign(tmp_path)
    result = low.analyze(campaign_root=campaign, output_dir=Path("artifacts"))
    payload = json.loads(Path(result["analysis"]).read_text(encoding="utf-8"))

    rho8 = payload["comparisons"]["0.8"]
    rho9 = payload["comparisons"]["0.9"]
    assert rho8["analysis_role"] == "secondary_exploratory"
    assert rho8["accuracy_superiority_supported"] is None
    assert rho9["analysis_role"] == "primary_locked_internal"
    assert rho9["accuracy_superiority_supported"] is True
    assert rho9["mean_delta"] == pytest.approx(0.01)
    assert rho9["strict_slaclip_wins"] == 5
    assert rho9["paired_t_95_ci"] == pytest.approx([0.01, 0.01])
    assert rho9["fixed_candidate_id"] == "fixed-c2"
    assert rho9["fixed_c2_mean_accuracy"] == pytest.approx(0.50)
    assert "tracking_mae" in rho9["slaclip_trajectory_stability"]
    assert rho9["mechanism_attribution"]["cdf_signal_informative"] is False
    assert payload["historical_test_results_available_before_protocol"] is True
    assert payload["inference_class"] == (
        "locked_internal_paired_confirmation_not_untouched_external_replication"
    )
    assert (campaign / "artifacts" / "final_paired_results.csv").is_file()
    assert (campaign / "artifacts" / "final_paired_results.json.sha256").is_file()


def test_analyze_rejects_test_asset_identity_tamper(tmp_path: Path) -> None:
    campaign, _ = _completed_campaign(tmp_path)
    plan = low._parse_plan(campaign / "plans" / "final.tsv", campaign.resolve())
    first = plan[0]
    config_path = Path(first["arm_root"]) / "results" / "evaluation_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["test_assets"]["gsm8k"]["sha256"] = "0" * 64
    _write_json(config_path, config)
    with pytest.raises(low.CampaignError, match="test asset mismatch"):
        low.analyze(campaign_root=campaign)


def test_analyze_rejects_privacy_accounting_tamper(tmp_path: Path) -> None:
    campaign, _ = _completed_campaign(tmp_path)
    plan = low._parse_plan(campaign / "plans" / "final.tsv", campaign.resolve())
    first = plan[0]
    status_path = Path(first["arm_root"]) / "adapter" / "run_status.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    status["privacy_accounting"]["epsilon_spent"] = 6.1
    _write_json(status_path, status)
    with pytest.raises(low.CampaignError, match="privacy accounting mismatch"):
        low.analyze(campaign_root=campaign)


def test_analyze_rejects_wrong_final_decoding_protocol(tmp_path: Path) -> None:
    campaign, _ = _completed_campaign(tmp_path)
    plan = low._parse_plan(campaign / "plans" / "final.tsv", campaign.resolve())
    first = plan[0]
    config_path = Path(first["arm_root"]) / "results" / "evaluation_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["num_beams"] = 1
    _write_json(config_path, config)
    with pytest.raises(low.CampaignError, match="evaluation config mismatch"):
        low.analyze(campaign_root=campaign)


def test_worker_runs_both_phases_sequentially_in_one_allocation() -> None:
    worker = (ROOT / "slurm" / "math10k_4b_low_target_campaign.sbatch").read_text(
        encoding="utf-8"
    )
    assert "selection) steps=150" in worker
    assert "final) steps=300" in worker
    assert '--steps "${steps}"' in worker
    assert 'run_plan "${STAGE1_PLAN}" 4 4' in worker
    assert 'run_plan "${FINAL_PLAN}" 15 15' in worker
    assert "tracking-control" not in worker
    assert not any(line.startswith("#SBATCH") for line in worker.splitlines())
    assert "sbatch " not in worker
    assert "CUDA_VISIBLE_DEVICES" not in worker


def test_wrapper_requests_exactly_one_a100_and_no_array() -> None:
    wrapper = (
        ROOT / "scripts" / "submit_math10k_4b_low_target_campaign.sh"
    ).read_text(encoding="utf-8")
    assert 'GPU_GRES="gpu:a100:1"' in wrapper
    assert 'CPUS_PER_TASK="8"' in wrapper
    assert 'HOST_MEMORY="128G"' in wrapper
    assert 'WALLTIME="2-12:00:00"' in wrapper
    assert "--nodes=1" in wrapper
    assert "--ntasks=1" in wrapper
    assert "--array" not in wrapper
    assert "--export=NONE" in wrapper
    assert 'state="${state%% *}"' in wrapper
