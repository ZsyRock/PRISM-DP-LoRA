from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "select_validation_candidates.py"
SPEC = importlib.util.spec_from_file_location("select_validation_candidates", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
selector = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(selector)


def _canonical_sha(payload) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_json(path: Path, payload, *, allow_nan: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=allow_nan) + "\n",
        encoding="utf-8",
    )


def _split_manifest(*, discriminator: str = "shared") -> dict:
    payload = {
        "schema_version": 1,
        "algorithm": "sha256_ranked_stratified_prompt_group_v1",
        "seed": 1729,
        "source_rows": 9919,
        "train_rows": 8919,
        "validation_rows": 1000,
        "requested_validation_rows": 1000,
        "stratum_targets": {"math10k": 1000},
        "validation_indices": list(range(1000)),
        "validation_indices_sha256": "1" * 64,
        "train_record_hashes_sha256": "2" * 64,
        "validation_record_hashes_sha256": hashlib.sha256(discriminator.encode()).hexdigest(),
        "source_content_sha256": "3" * 64,
        "protocol_stage": "selection",
        "validation_data_is_public": True,
    }
    payload["manifest_sha256"] = _canonical_sha(payload)
    return payload


def _make_run(
    campaign: Path,
    *,
    candidate_id: str,
    family: str,
    params: dict,
    seed: int,
    loss: float,
    split: dict,
    epsilon: float = 5.9988,
    state: str = "completed",
    common_config: dict | None = None,
) -> dict[str, str]:
    root = campaign / "screen" / "runs" / candidate_id / f"seed-{seed}"
    status_path = root / "adapter" / "run_status.json"
    validation_dir = root / "results" / "validation"
    metrics_path = validation_dir / "validation_metrics.json"
    split_path = validation_dir / "split_manifest.json"
    method = "baseline" if family == "fixed" else "slaclip"
    config = {
        "protocol_stage": "selection",
        "validation_data_is_public": True,
        "seed": seed,
        "method": method,
        "total_update_steps": 300,
        **(common_config or {}),
        **params,
    }
    metrics = {
        **split,
        "metric_schema_version": 1,
        "selection_metric": selector.EXPECTED_METRIC,
        "loss_definition": selector.EXPECTED_LOSS_DEFINITION,
        "loss_mean": loss,
        "token_mean_loss": loss + 0.1,
        "token_perplexity": 3.0,
        "records": split["validation_rows"],
        "supervised_tokens": 4000,
        "NON_PRIVATE_SELECTION_METRIC": True,
    }
    status = {
        "state": state,
        "privacy": "dp",
        "method": method,
        "update_steps": 300,
        "config": config,
        "config_fingerprint": f"fingerprint-{candidate_id}-{seed}",
        "data_split": split,
        "validation": metrics,
        "privacy_accounting": {
            "target_epsilon": 6.0,
            "epsilon_spent": epsilon,
            "completed_update_steps": 300,
        },
    }
    _write_json(status_path, status, allow_nan=True)
    _write_json(metrics_path, metrics, allow_nan=True)
    _write_json(split_path, split)
    return {
        "run_status": str(status_path.relative_to(campaign)),
        "validation_metrics": str(metrics_path.relative_to(campaign)),
        "split_manifest": str(split_path.relative_to(campaign)),
    }


def _build_campaign(tmp_path: Path) -> tuple[Path, Path]:
    campaign = tmp_path / "campaign"
    (campaign / "screen").mkdir(parents=True)
    shared_split = _split_manifest()
    common_config = {
        "dataset": "math10k",
        "privacy": "dp",
        "base_model": "google/gemma-3-4b-pt",
        "model_revision": "4" * 40,
        "batch_size": 64,
        "micro_batch_size": 4,
        "total_update_steps": 300,
        "dp_epsilon": 6.0,
        "dp_delta": 1e-5,
        "val_set_size": 1000,
        "protocol_stage": "selection",
        "validation_data_is_public": True,
    }
    definitions = [
        ("fixed-c1", "fixed", {"dp_max_grad_norm": 1.0}, {42: 2.0, 43: 2.0, 44: 2.0}),
        ("fixed-c15", "fixed", {"dp_max_grad_norm": 1.5}, {42: 1.7, 43: 2.1, 44: 2.5}),
        (
            "sla-a",
            "slaclip",
            {
                "dp_max_grad_norm": 1.0,
                "slaclip_beta": 0.99,
                "slaclip_eta": 0.15,
                "slaclip_num_slots": 15,
                "slaclip_c_min": 0.1,
                "slaclip_c_max": 15.0,
            },
            {42: 1.5, 43: 2.0, 44: 2.0},
        ),
        (
            "sla-b",
            "slaclip",
            {
                "dp_max_grad_norm": 1.0,
                "slaclip_beta": 0.975,
                "slaclip_eta": 0.15,
                "slaclip_num_slots": 15,
                "slaclip_c_min": 0.1,
                "slaclip_c_max": 15.0,
            },
            {42: 1.6, 43: 1.5, 44: 2.4},
        ),
        (
            "sla-c",
            "slaclip",
            {
                "dp_max_grad_norm": 1.5,
                "slaclip_beta": 0.99,
                "slaclip_eta": 0.15,
                "slaclip_num_slots": 15,
                "slaclip_c_min": 0.1,
                "slaclip_c_max": 15.0,
            },
            {42: 1.9},
        ),
    ]
    candidates = []
    for candidate_id, family, params, losses in definitions:
        runs = {
            str(seed): _make_run(
                campaign,
                candidate_id=candidate_id,
                family=family,
                params=params,
                seed=seed,
                loss=loss,
                split=shared_split,
                common_config=common_config,
            )
            for seed, loss in losses.items()
        }
        candidates.append(
            {
                "id": candidate_id,
                "family": family,
                "method": "baseline" if family == "fixed" else "slaclip",
                "params": params,
                "runs": runs,
            }
        )
    registry = {
        "schema_version": 1,
        "selection_protocol": {
            "name": "prism_slaclip_k15_public_validation_v1",
            "protocol_stage": "selection",
            "validation_data_is_public": True,
            "selection_metric": selector.EXPECTED_METRIC,
            "loss_definition": selector.EXPECTED_LOSS_DEFINITION,
            "required_update_steps": 300,
            "target_epsilon": 6.0,
            "epsilon_tolerance": 0.02,
            "stage1_seed": 42,
            "stage2_seeds": [42, 43, 44],
            "common_config": common_config,
        },
        "candidates": candidates,
    }
    registry_path = campaign / "screen" / "candidate_registry.json"
    _write_json(registry_path, registry)
    return campaign, registry_path


def _select(campaign: Path, registry: Path, stage: str, name: str, *, stage1=None):
    output = campaign / "selection" / name
    return selector.select_candidates(
        campaign_root=campaign,
        registry_path=registry,
        stage=stage,
        output_path=output,
        top_slaclip=2,
        stage1_selection_path=stage1,
    )


def test_stage1_and_stage2_are_deterministic_and_ignore_unselected_runs(tmp_path: Path) -> None:
    campaign, registry = _build_campaign(tmp_path)
    stage1_path = campaign / "selection" / "stage1-selection.json"
    stage1 = _select(campaign, registry, "stage1", stage1_path.name)
    assert stage1["best_fixed"]["candidate_id"] == "fixed-c15"
    assert [item["candidate_id"] for item in stage1["top_slaclip"]] == ["sla-a", "sla-b"]

    # sla-c deliberately has no stage-2 seeds; stage2 must only inspect the
    # immutable top-two set selected in stage1.
    stage2_path = campaign / "selection" / "selection.json"
    stage2 = _select(campaign, registry, "stage2", stage2_path.name, stage1=stage1_path)
    assert stage2["required_seeds"] == [42, 43, 44]
    # Stage1 favored C=1.5, but the same three-seed evidence correctly locks
    # the canonical C=1 control as the stronger fixed threshold.
    assert stage2["stage1_best_fixed_candidate_id"] == "fixed-c15"
    assert stage2["canonical_fixed_candidate_id"] == "fixed-c1"
    assert [item["candidate_id"] for item in stage2["fixed_ranking"]] == ["fixed-c1", "fixed-c15"]
    assert stage2["best_fixed"]["candidate_id"] == "fixed-c1"
    # Both arithmetic means are exactly 11/6; candidate id is the declared tie-break.
    assert stage2["selected_slaclip"]["candidate_id"] == "sla-a"
    selected_env = (campaign / "selection" / "selected.env").read_text()
    assert "PRISM_SELECTED_CANDIDATE_ID=sla-a\n" in selected_env
    assert "PRISM_SELECTED_BEST_FIXED_CANDIDATE_ID=fixed-c1\n" in selected_env
    assert "PRISM_SELECTED_BEST_FIXED_DP_MAX_GRAD_NORM=1.0\n" in selected_env
    assert stage2_path.with_suffix(".csv").exists()
    csv_text = stage2_path.with_suffix(".csv").read_text()
    assert "ranking_group,group_rank" in csv_text
    assert "fixed-c1,fixed,baseline" in csv_text

    # Byte-identical reruns are accepted and leave the locked selection intact.
    before = stage2_path.read_bytes()
    rerun = _select(campaign, registry, "stage2", stage2_path.name, stage1=stage1_path)
    assert rerun == stage2
    assert stage2_path.read_bytes() == before


def test_numeric_exact_accuracy_is_primary_and_loss_breaks_ties(tmp_path: Path) -> None:
    campaign, registry_path = _build_campaign(tmp_path)
    registry = json.loads(registry_path.read_text())
    registry["selection_protocol"]["selection_metric"] = selector.NUMERIC_EXACT_METRIC
    accuracy_by_candidate = {
        "fixed-c1": 0.55,
        "fixed-c15": 0.50,
        "sla-a": 0.80,
        "sla-b": 0.85,
        "sla-c": 0.80,
    }
    for candidate in registry["candidates"]:
        accuracy = accuracy_by_candidate[candidate["id"]]
        for run in candidate["runs"].values():
            metrics_path = campaign / run["validation_metrics"]
            status_path = campaign / run["run_status"]
            metrics = json.loads(metrics_path.read_text())
            status = json.loads(status_path.read_text())
            metrics.update(
                {
                    "selection_metric": selector.NUMERIC_EXACT_METRIC,
                    "numeric_exact_accuracy": accuracy,
                    "numeric_exact_correct": int(accuracy * metrics["records"]),
                    "numeric_parse_failures": 0,
                }
            )
            status["validation"] = metrics
            _write_json(metrics_path, metrics)
            _write_json(status_path, status)
    _write_json(registry_path, registry)

    stage1_path = campaign / "selection" / "numeric-stage1-selection.json"
    stage1 = _select(campaign, registry_path, "stage1", stage1_path.name)
    assert stage1["ranking_rule"].startswith(
        "descending_validation_numeric_exact_accuracy"
    )
    assert stage1["best_fixed"]["candidate_id"] == "fixed-c1"
    assert [item["candidate_id"] for item in stage1["top_slaclip"]] == [
        "sla-b",
        "sla-a",
    ]

    stage2 = _select(
        campaign,
        registry_path,
        "stage2",
        "numeric-selection.json",
        stage1=stage1_path,
    )
    assert stage2["selected_slaclip"]["candidate_id"] == "sla-b"
    assert stage2["selected_slaclip"]["mean_validation_accuracy"] == 0.85
    assert "mean_validation_accuracy" in (
        campaign / "selection" / "numeric-selection.csv"
    ).read_text()


def test_refuses_to_overwrite_a_different_selection(tmp_path: Path) -> None:
    campaign, registry = _build_campaign(tmp_path)
    stage1_path = campaign / "selection" / "stage1-selection.json"
    _select(campaign, registry, "stage1", stage1_path.name)
    stage1_path.write_text('{"tampered":true}\n', encoding="utf-8")
    with pytest.raises(selector.SelectionError, match="refusing to overwrite inconsistent"):
        _select(campaign, registry, "stage1", stage1_path.name)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("incomplete", "not completed"),
        ("nan_loss", "finite"),
        ("epsilon", "outside"),
        ("not_public", "public selection protocol"),
    ],
)
def test_rejects_invalid_screen_evidence(tmp_path: Path, mutation: str, message: str) -> None:
    campaign, registry_path = _build_campaign(tmp_path)
    registry = json.loads(registry_path.read_text())
    run = registry["candidates"][0]["runs"]["42"]
    status_path = campaign / run["run_status"]
    metrics_path = campaign / run["validation_metrics"]
    status = json.loads(status_path.read_text())
    metrics = json.loads(metrics_path.read_text())
    if mutation == "incomplete":
        status["state"] = "running"
    elif mutation == "nan_loss":
        metrics["loss_mean"] = float("nan")
        status["validation"]["loss_mean"] = float("nan")
    elif mutation == "epsilon":
        status["privacy_accounting"]["epsilon_spent"] = 5.5
    elif mutation == "not_public":
        status["config"]["validation_data_is_public"] = False
    _write_json(status_path, status, allow_nan=True)
    _write_json(metrics_path, metrics, allow_nan=True)
    with pytest.raises(selector.SelectionError, match=message):
        _select(campaign, registry_path, "stage1", "stage1-selection.json")


def test_rejects_mixed_split_manifests(tmp_path: Path) -> None:
    campaign, registry_path = _build_campaign(tmp_path)
    registry = json.loads(registry_path.read_text())
    candidate = registry["candidates"][1]
    run = candidate["runs"]["42"]
    different = _split_manifest(discriminator="different")
    status_path = campaign / run["run_status"]
    metrics_path = campaign / run["validation_metrics"]
    split_path = campaign / run["split_manifest"]
    status = json.loads(status_path.read_text())
    metrics = json.loads(metrics_path.read_text())
    status["data_split"] = different
    status["validation"].update(different)
    metrics.update(different)
    _write_json(status_path, status)
    _write_json(metrics_path, metrics)
    _write_json(split_path, different)
    with pytest.raises(selector.SelectionError, match="share one split manifest"):
        _select(campaign, registry_path, "stage1", "stage1-selection.json")


def test_rejects_any_registered_input_outside_screen(tmp_path: Path) -> None:
    campaign, registry_path = _build_campaign(tmp_path)
    registry = json.loads(registry_path.read_text())
    candidate = registry["candidates"][0]
    run = candidate["runs"]["42"]
    source = campaign / run["validation_metrics"]
    forbidden = campaign / "evaluation" / "validation_metrics.json"
    forbidden.parent.mkdir(parents=True)
    forbidden.write_bytes(source.read_bytes())
    run["validation_metrics"] = str(forbidden.relative_to(campaign))
    _write_json(registry_path, registry)
    with pytest.raises(selector.SelectionError, match="must remain inside"):
        _select(campaign, registry_path, "stage1", "stage1-selection.json")


def test_allows_portable_campaign_below_tests_named_ancestor(tmp_path: Path) -> None:
    campaign, registry_path = _build_campaign(tmp_path / "tests" / "portable-account")
    selection = _select(
        campaign,
        registry_path,
        "stage1",
        "stage1-selection.json",
    )
    assert selection["best_fixed"]["candidate_id"] == "fixed-c15"


def test_stage2_requires_all_fixed_control_seeds(tmp_path: Path) -> None:
    campaign, registry_path = _build_campaign(tmp_path)
    registry = json.loads(registry_path.read_text())
    fixed_c1 = next(item for item in registry["candidates"] if item["id"] == "fixed-c1")
    del fixed_c1["runs"]["44"]
    _write_json(registry_path, registry)
    stage1_path = campaign / "selection" / "stage1-selection.json"
    _select(campaign, registry_path, "stage1", stage1_path.name)
    with pytest.raises(selector.SelectionError, match="no registered run for seed 44"):
        _select(campaign, registry_path, "stage2", "selection.json", stage1=stage1_path)


def test_rejects_nontransferable_identity_in_candidate_params(tmp_path: Path) -> None:
    campaign, registry_path = _build_campaign(tmp_path)
    registry = json.loads(registry_path.read_text())
    registry["candidates"][0]["params"]["base_model"] = "google/gemma-3-4b-pt"
    _write_json(registry_path, registry)
    with pytest.raises(selector.SelectionError, match="exactly transferable fields"):
        _select(campaign, registry_path, "stage1", "stage1-selection.json")


def test_common_config_is_checked_against_each_screen_run(tmp_path: Path) -> None:
    campaign, registry_path = _build_campaign(tmp_path)
    registry = json.loads(registry_path.read_text())
    run = registry["candidates"][0]["runs"]["42"]
    status_path = campaign / run["run_status"]
    status = json.loads(status_path.read_text())
    status["config"]["batch_size"] = 32
    _write_json(status_path, status)
    with pytest.raises(selector.SelectionError, match="common config batch_size"):
        _select(campaign, registry_path, "stage1", "stage1-selection.json")
