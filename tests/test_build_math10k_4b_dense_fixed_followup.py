from __future__ import annotations

import importlib.util
import json
import stat
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "build_math10k_4b_dense_fixed_followup.py"
SPEC = importlib.util.spec_from_file_location("dense_fixed_followup", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
dense = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(dense)


SOURCE_SHA = "feba5968285dc1651bf7726327932b3f625ace22"


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _candidate(identifier: str, family: str, clip: float, *, rho=None, eta=None):
    params = {"dp_max_grad_norm": clip}
    method = "baseline"
    if family == "slaclip":
        method = "slaclip"
        params.update(
            {
                "slaclip_target_non_small_clip_fraction": rho,
                "slaclip_eta": eta,
                "slaclip_num_slots": 15,
                "slaclip_c_min": 0.1,
                "slaclip_c_max": 15.0,
            }
        )
    return {
        "id": identifier,
        "family": family,
        "method": method,
        "params": params,
        "runs": {
            str(seed): {
                "run_status": f"screen/runs/{identifier}/seed-{seed}/adapter/run_status.json",
                "validation_metrics": (
                    f"screen/runs/{identifier}/seed-{seed}/results/validation/validation_metrics.json"
                ),
                "split_manifest": (
                    f"screen/runs/{identifier}/seed-{seed}/results/validation/split_manifest.json"
                ),
            }
            for seed in dense.SELECTION_SEEDS
        },
    }


def _plan_line(root: Path, phase: str, candidate: dict, role: str, seed: int) -> str:
    params = candidate["params"]
    rho = params.get("slaclip_target_non_small_clip_fraction", "NA")
    eta = params.get("slaclip_eta", "NA")
    arm = (
        root / "screen" / "runs" / candidate["id"] / f"seed-{seed}"
        if phase == "stage2"
        else root / "final" / "gemma-3-4b-pt" / f"seed-{seed}" / role
    )
    return "|".join(
        str(value)
        for value in (
            "train",
            phase,
            candidate["id"],
            role,
            seed,
            candidate["method"],
            params["dp_max_grad_norm"],
            rho,
            eta,
            "NA",
            arm,
        )
    )


def _make_training_run(
    *,
    root: Path,
    candidate: dict,
    seed: int,
    accuracy: float,
    loss: float,
    common: dict,
    split: dict,
) -> None:
    run = candidate["runs"][str(seed)]
    status_path = root / run["run_status"]
    metrics_path = root / run["validation_metrics"]
    split_path = root / run["split_manifest"]
    split_sha = dense._payload_sha256(split)
    metrics = {
        "selection_metric": dense.EXPECTED_METRIC,
        "loss_definition": dense.EXPECTED_LOSS,
        "numeric_exact_accuracy": accuracy,
        "numeric_exact_correct": round(500 * accuracy),
        "numeric_parse_failures": 0,
        "loss_mean": loss,
        "validation_data_is_public": True,
        "PUBLIC_VALIDATION_DATA": True,
        "protocol_stage": "selection",
        "records": 500,
        "manifest_sha256": split_sha,
    }
    fingerprint = f"fingerprint-{candidate['id']}-{seed}"
    config = {
        **common,
        **candidate["params"],
        "method": candidate["method"],
        "seed": seed,
    }
    status = {
        "state": "completed",
        "update_steps": 300,
        "method": candidate["method"],
        "run_id": f"{candidate['id']}-seed-{seed}",
        "config_fingerprint": fingerprint,
        "config": config,
        "privacy_accounting": {
            "completed_update_steps": 300,
            "epsilon_spent": 5.997,
        },
        "validation": metrics,
    }
    _write_json(status_path, status)
    _write_json(metrics_path, metrics)
    _write_json(split_path, split)
    log_path = status_path.parent / "train_log.jsonl"
    rows = []
    for step in range(1, 301):
        if candidate["method"] == "baseline":
            current = next_clip = candidate["params"]["dp_max_grad_norm"]
            row = {
                "step": step,
                "method": "baseline",
                "config_fingerprint": fingerprint,
                "dp_clip_threshold": current,
                "dp_next_clip_threshold": next_clip,
            }
        else:
            current = candidate["params"]["dp_max_grad_norm"] + (step - 1) * 0.001
            next_clip = candidate["params"]["dp_max_grad_norm"] + step * 0.001
            row = {
                "step": step,
                "method": "slaclip",
                "config_fingerprint": fingerprint,
                "dp_clip_threshold": current,
                "dp_next_clip_threshold": next_clip,
                "slaclip_c_min": 0.1,
                "slaclip_c_max": 15.0,
                "slaclip_eta": candidate["params"]["slaclip_eta"],
                "slaclip_target_non_small_clip_fraction": candidate["params"][
                    "slaclip_target_non_small_clip_fraction"
                ],
                "slaclip_num_slots": 15,
                "slaclip_controller_error": 0.01,
                "slack_indicator": [0.0] * 15,
            }
        rows.append(json.dumps(row, sort_keys=True, allow_nan=False))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def _fixture(tmp_path: Path, *, sla_accuracy=None, fixed_c2_accuracy=None):
    source = tmp_path / "source"
    campaign = tmp_path / "followup"
    source.mkdir()
    common = {
        "dataset": "math10k",
        "privacy": "dp",
        "base_model": "google/gemma-3-4b-pt",
        "model_revision": "c" * 40,
        "implementation_git_sha": SOURCE_SHA,
        "total_update_steps": 300,
        "batch_size": 64,
        "micro_batch_size": 4,
        "learning_rate": 0.0003,
        "val_set_size": 500,
        "validation_seed": 1729,
        "validation_data_is_public": True,
        "protocol_stage": "selection",
        "dp_epsilon": 6.0,
        "dp_delta": 1e-5,
        "telemetry_mode": "research_raw",
    }
    protocol = {
        "name": "source-refinement",
        "protocol_stage": "selection",
        "validation_data_is_public": True,
        "selection_metric": dense.EXPECTED_METRIC,
        "loss_definition": dense.EXPECTED_LOSS,
        "required_update_steps": 300,
        "target_epsilon": 6.0,
        "epsilon_tolerance": 0.02,
        "stage1_seed": 42,
        "stage2_seeds": list(dense.SELECTION_SEEDS),
        "common_config": common,
    }
    candidates = [
        _candidate("fixed-c1", "fixed", 1.0),
        _candidate("fixed-c2", "fixed", 2.0),
        _candidate("sla-a", "slaclip", 1.5, rho=0.985, eta=0.05),
        _candidate("sla-b", "slaclip", 1.75, rho=0.99, eta=0.05),
        _candidate("sla-c", "slaclip", 1.25, rho=0.995, eta=0.1),
    ]
    mapped = {candidate["id"]: candidate for candidate in candidates}
    registry = {
        "schema_version": 1,
        "selection_protocol": protocol,
        "candidates": candidates,
    }
    ranking_ids = ["sla-a", "sla-b", "sla-c", "fixed-c1", "fixed-c2"]
    selection = {
        "schema_version": 1,
        "stage": "stage2",
        "required_seeds": list(dense.SELECTION_SEEDS),
        "registry_sha256": dense._payload_sha256(registry),
        "selection_protocol_sha256": dense._payload_sha256(protocol),
        "selection_protocol": protocol,
        "split_manifest_sha256": "placeholder",
        "ranking": [
            {
                "candidate_id": identifier,
                "method": mapped[identifier]["method"],
                "params": mapped[identifier]["params"],
            }
            for identifier in ranking_ids
        ],
        "selected_slaclip": {
            "candidate_id": "sla-a",
            "method": "slaclip",
            "params": mapped["sla-a"]["params"],
        },
        "best_fixed": {
            "candidate_id": "fixed-c1",
            "method": "baseline",
            "params": mapped["fixed-c1"]["params"],
        },
    }
    split = {
        "schema_version": 2,
        "validation_data_is_public": True,
        "seed": 1729,
        "validation_rows": 500,
        "validation_indices_sha256": "d" * 64,
    }
    selection["split_manifest_sha256"] = dense._payload_sha256(split)
    _write_json(source / "screen" / "candidate_registry.json", registry)
    _write_json(source / "selection" / "selection.json", selection)
    stage2_lines = [
        _plan_line(source, "stage2", mapped[identifier], "five-seed-confirmation", seed)
        for seed in dense.SELECTION_SEEDS[1:]
        for identifier in ranking_ids
    ]
    (source / "plans").mkdir(parents=True)
    (source / "plans" / "stage2.tsv").write_text(
        "\n".join(stage2_lines) + "\n", encoding="utf-8"
    )
    final_lines = [
        _plan_line(source, "final", mapped[identifier], role, seed)
        for seed in dense.FRESH_SEEDS
        for identifier, role in (("sla-a", "slaclip"), ("fixed-c2", "baseline"))
    ]
    (source / "plans" / "final.tsv").write_text(
        "\n".join(final_lines) + "\n", encoding="utf-8"
    )

    accuracies = {
        1.0: [0.50] * 5,
        2.0: fixed_c2_accuracy if fixed_c2_accuracy is not None else [0.51] * 5,
    }
    selected_accuracies = sla_accuracy or [0.55, 0.55, 0.55, 0.53, 0.56]
    for candidate, values, loss in (
        (mapped["fixed-c1"], accuracies[1.0], 0.34),
        (mapped["fixed-c2"], accuracies[2.0], 0.33),
        (mapped["sla-a"], selected_accuracies, 0.29),
    ):
        for seed, accuracy in zip(dense.SELECTION_SEEDS, values, strict=True):
            _make_training_run(
                root=source,
                candidate=candidate,
                seed=seed,
                accuracy=accuracy,
                loss=loss,
                common=common,
                split=split,
            )
    return source, campaign, common, split


def _write_dense_results(campaign: Path, common: dict, split: dict) -> None:
    accuracies = {
        1.5: [0.54] * 5,
        1.75: [0.52] * 5,
        1.25: [0.53] * 5,
    }
    losses = {1.5: 0.30, 1.75: 0.31, 1.25: 0.32}
    for clip in dense.DENSE_C_VALUES:
        candidate = _candidate(dense._candidate_id(clip), "fixed", clip)
        # New runs live below follow-up selection/, unlike source screen/.
        for seed in dense.SELECTION_SEEDS:
            candidate["runs"][str(seed)] = {
                "run_status": (
                    f"selection/runs/{candidate['id']}/seed-{seed}/adapter/run_status.json"
                ),
                "validation_metrics": (
                    f"selection/runs/{candidate['id']}/seed-{seed}/results/validation/validation_metrics.json"
                ),
                "split_manifest": (
                    f"selection/runs/{candidate['id']}/seed-{seed}/results/validation/split_manifest.json"
                ),
            }
        for seed, accuracy in zip(dense.SELECTION_SEEDS, accuracies[clip], strict=True):
            _make_training_run(
                root=campaign,
                candidate=candidate,
                seed=seed,
                accuracy=accuracy,
                loss=losses[clip],
                common=common,
                split=split,
            )


def _prepare(source: Path, campaign: Path):
    return dense.prepare(
        source_root=source,
        campaign_root=campaign,
        experiment_code_sha=SOURCE_SHA,
    )


def _lock(source: Path, campaign: Path):
    return dense.lock(
        source_root=source,
        campaign_root=campaign,
        experiment_code_sha=SOURCE_SHA,
    )


def test_prepare_writes_exact_ordered_15_arm_immutable_plan(tmp_path: Path) -> None:
    source, campaign, _common, _split = _fixture(tmp_path)
    payload = _prepare(source, campaign)
    plan = campaign / "plans" / "dense-selection.tsv"
    lines = plan.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 15
    observed = [(line.split("|")[2], int(line.split("|")[4])) for line in lines]
    assert observed == [
        (dense._candidate_id(clip), seed)
        for clip in (1.5, 1.75, 1.25)
        for seed in (42, 43, 44, 45, 46)
    ]
    assert all(line.split("|")[0:2] == ["train", "dense-selection"] for line in lines)
    assert all(line.split("|")[3] == "dense-fixed-selection" for line in lines)
    assert all(line.split("|")[5] == "baseline" for line in lines)
    assert payload["selection_design"]["complete_fixed_clip_grid"] == [1.0, 1.25, 1.5, 1.75, 2.0]
    assert payload["selection_design"]["fresh_seeds"] == [191, 223, 257, 293, 331]
    for artifact in (plan, campaign / "selection" / "dense-fixed-protocol.json"):
        assert stat.S_IMODE(artifact.stat().st_mode) == 0o600
        assert artifact.with_name(artifact.name + ".sha256").exists()
    before = plan.read_bytes()
    assert _prepare(source, campaign) == payload
    assert plan.read_bytes() == before


def test_lock_ranks_five_points_and_enables_exact_five_fresh_arms(tmp_path: Path) -> None:
    source, campaign, common, split = _fixture(tmp_path)
    _prepare(source, campaign)
    _write_dense_results(campaign, common, split)
    payload = _lock(source, campaign)
    assert [row["clip_threshold"] for row in payload["fixed_grid_ranking"]] == [
        1.5,
        1.25,
        1.75,
        2.0,
        1.0,
    ]
    assert payload["dense_best_fixed"]["clip_threshold"] == 1.5
    assert payload["gate"]["passed"] is True
    assert payload["gate"]["observed_mean_accuracy_delta"] == pytest.approx(0.008)
    assert payload["gate"]["observed_strict_paired_wins"] == 4
    assert len(payload["gate"]["per_seed_pairing"]) == 5
    assert payload["trajectory_log_compliance"]["compliant"] is True
    assert len(payload["trajectory_log_compliance"]["seed_audits"]) == 5
    final = campaign / "plans" / "dense-final.tsv"
    lines = final.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 5
    for line, seed in zip(lines, dense.FRESH_SEEDS, strict=True):
        fields = line.split("|")
        assert fields[0:6] == [
            "train",
            "dense-final",
            "fixed-c1p5",
            "dense-best-fixed",
            str(seed),
            "baseline",
        ]
        assert fields[6:10] == ["1.5", "NA", "NA", "NA"]
    locked = campaign / "selection" / "dense-best-fixed.json"
    assert json.loads(locked.read_text(encoding="utf-8")) == payload
    assert locked.with_name(locked.name + ".sha256").exists()
    assert final.with_name(final.name + ".sha256").exists()
    assert _lock(source, campaign) == payload


def test_failed_performance_gate_records_reasons_and_empty_final_plan(tmp_path: Path) -> None:
    source, campaign, common, split = _fixture(
        tmp_path, sla_accuracy=[0.541, 0.541, 0.539, 0.539, 0.54]
    )
    _prepare(source, campaign)
    _write_dense_results(campaign, common, split)
    payload = _lock(source, campaign)
    assert payload["gate"]["passed"] is False
    assert payload["gate"]["observed_strict_paired_wins"] == 2
    assert any("mean accuracy delta" in reason for reason in payload["gate"]["reasons"])
    assert any("strict paired wins" in reason for reason in payload["gate"]["reasons"])
    assert payload["fresh_final_plan"]["record_count"] == 0
    assert (campaign / "plans" / "dense-final.tsv").read_bytes() == b""


def test_gate_pass_with_dense_best_c2_reuses_source_fresh_arms(tmp_path: Path) -> None:
    source, campaign, common, split = _fixture(
        tmp_path,
        sla_accuracy=[0.61] * 5,
        fixed_c2_accuracy=[0.60] * 5,
    )
    _prepare(source, campaign)
    _write_dense_results(campaign, common, split)
    payload = _lock(source, campaign)
    assert payload["dense_best_fixed"]["clip_threshold"] == 2.0
    assert payload["gate"]["passed"] is True
    assert payload["fresh_final_plan"]["status"] == "empty-reuse-source-fixed-c2"
    assert payload["fresh_final_plan"]["record_count"] == 0
    reuse = payload["fresh_final_plan"]["source_fixed_c2_reuse"]
    assert reuse["enabled"] is True
    assert reuse["candidate_id"] == "fixed-c2"
    assert [arm["seed"] for arm in reuse["registered_arms"]] == [191, 223, 257, 293, 331]
    assert all(arm["arm_root"].endswith("/baseline") for arm in reuse["registered_arms"])
    assert (campaign / "plans" / "dense-final.tsv").read_bytes() == b""


def test_slaclip_clip_bound_violation_fails_gate_with_audit_reason(tmp_path: Path) -> None:
    source, campaign, common, split = _fixture(tmp_path)
    _prepare(source, campaign)
    _write_dense_results(campaign, common, split)
    path = source / "screen" / "runs" / "sla-a" / "seed-42" / "adapter" / "train_log.jsonl"
    rows = path.read_text(encoding="utf-8").splitlines()
    first = json.loads(rows[0])
    first["dp_next_clip_threshold"] = 15.5
    rows[0] = json.dumps(first, sort_keys=True)
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    payload = _lock(source, campaign)
    assert payload["gate"]["passed"] is False
    assert payload["trajectory_log_compliance"]["compliant"] is False
    assert any("outside [C_min,C_max]" in reason for reason in payload["trajectory_log_compliance"]["reasons"])
    assert any("trajectory/log" in reason for reason in payload["gate"]["reasons"])
    assert (campaign / "plans" / "dense-final.tsv").read_bytes() == b""


def test_rejects_wrong_frozen_sha_and_immutable_plan_tampering(tmp_path: Path) -> None:
    source, campaign, _common, _split = _fixture(tmp_path)
    with pytest.raises(dense.FollowupError, match="frozen source mechanism commit"):
        dense.prepare(
            source_root=source,
            campaign_root=campaign,
            experiment_code_sha="a" * 40,
        )
    _prepare(source, campaign)
    plan = campaign / "plans" / "dense-selection.tsv"
    plan.write_text("tampered\n", encoding="utf-8")
    with pytest.raises(dense.FollowupError, match="immutable artifact"):
        _prepare(source, campaign)
