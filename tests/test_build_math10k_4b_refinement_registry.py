from __future__ import annotations

import importlib.util
import json
import stat
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "build_math10k_4b_refinement_registry.py"
SPEC = importlib.util.spec_from_file_location(
    "build_math10k_4b_refinement_registry", SCRIPT
)
assert SPEC is not None and SPEC.loader is not None
registry_builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(registry_builder)

SELECTOR_SCRIPT = ROOT / "scripts" / "select_validation_candidates.py"
SELECTOR_SPEC = importlib.util.spec_from_file_location(
    "selector_for_refinement_registry_test", SELECTOR_SCRIPT
)
assert SELECTOR_SPEC is not None and SELECTOR_SPEC.loader is not None
selector = importlib.util.module_from_spec(SELECTOR_SPEC)
SELECTOR_SPEC.loader.exec_module(selector)


MODEL_ID = "google/gemma-3-4b-pt"
MODEL_REVISION = "a" * 40
CODE_SHA = "b" * 40


def _build(tmp_path: Path):
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    output = Path("screen/candidate_registry.json")
    payload = registry_builder.build_registry(
        campaign_root=campaign,
        output=output,
        model_id=MODEL_ID,
        model_revision=MODEL_REVISION,
        code_sha=CODE_SHA,
    )
    return campaign, campaign / output, payload


def test_builds_exact_refinement_grid_and_selector_run_schema(tmp_path: Path) -> None:
    campaign, output, payload = _build(tmp_path)
    assert json.loads(output.read_text(encoding="utf-8")) == payload
    assert stat.S_IMODE(output.stat().st_mode) == 0o600
    assert set(payload) == {"schema_version", "selection_protocol", "candidates"}
    assert payload["schema_version"] == selector.REGISTRY_SCHEMA_VERSION

    protocol = payload["selection_protocol"]
    assert set(protocol) == {
        "name",
        "protocol_stage",
        "validation_data_is_public",
        "selection_metric",
        "loss_definition",
        "required_update_steps",
        "target_epsilon",
        "epsilon_tolerance",
        "stage1_seed",
        "stage2_seeds",
        "common_config",
    }
    assert protocol["selection_metric"] == selector.NUMERIC_EXACT_METRIC
    assert protocol["loss_definition"] == selector.EXPECTED_LOSS_DEFINITION
    assert protocol["required_update_steps"] == 300
    assert protocol["target_epsilon"] == 6.0
    assert protocol["epsilon_tolerance"] == 0.02
    assert protocol["stage1_seed"] == 42
    assert protocol["stage2_seeds"] == [42, 43, 44, 45, 46]
    assert selector._validate_protocol(payload) == protocol

    common = protocol["common_config"]
    assert common["dataset"] == "math10k"
    assert common["base_model"] == MODEL_ID
    assert common["model_revision"] == MODEL_REVISION
    assert common["implementation_git_sha"] == CODE_SHA
    assert common["total_update_steps"] == 300
    assert common["val_set_size"] == 500
    assert common["dp_epsilon"] == 6.0
    assert common["dp_delta"] == 1e-5
    assert common["batch_size"] == 64
    assert common["micro_batch_size"] == 4
    assert common["learning_rate"] == 0.0003
    assert common["telemetry_mode"] == "research_raw"
    assert common["allow_non_private_telemetry"] is True
    forbidden_common = (
        selector.FIXED_PARAM_KEYS
        | (selector.SLACLIP_PARAM_KEYS - {"dp_max_grad_norm"})
        | {"method", "seed"}
    )
    assert not forbidden_common.intersection(common)

    candidates = payload["candidates"]
    assert len(candidates) == 27
    assert len({candidate["id"] for candidate in candidates}) == 27
    mapped = selector._candidate_map(payload)
    assert set(mapped) == {candidate["id"] for candidate in candidates}
    fixed = [candidate for candidate in candidates if candidate["family"] == "fixed"]
    slaclip = [
        candidate for candidate in candidates if candidate["family"] == "slaclip"
    ]
    assert [(item["id"], item["params"]) for item in fixed] == [
        ("fixed-c1", {"dp_max_grad_norm": 1.0}),
        ("fixed-c2", {"dp_max_grad_norm": 2.0}),
    ]
    assert len(slaclip) == 25

    main_grid = {
        (
            candidate["params"]["dp_max_grad_norm"],
            candidate["params"]["slaclip_target_non_small_clip_fraction"],
            candidate["params"]["slaclip_eta"],
        )
        for candidate in slaclip
        if candidate["id"] != "sla-c2-r0p97-e0p15"
    }
    assert main_grid == {
        (c0, rho, eta)
        for c0 in (1.25, 1.5, 1.75, 2.0)
        for rho in (0.985, 0.99, 0.995)
        for eta in (0.05, 0.1)
    }
    anchor = mapped["sla-c2-r0p97-e0p15"]
    assert anchor["params"] == {
        "dp_max_grad_norm": 2.0,
        "slaclip_target_non_small_clip_fraction": 0.97,
        "slaclip_eta": 0.15,
        "slaclip_num_slots": 15,
        "slaclip_c_min": 0.1,
        "slaclip_c_max": 15.0,
    }
    for candidate in candidates:
        if candidate["family"] == "slaclip":
            assert candidate["params"]["slaclip_num_slots"] == 15
            assert candidate["params"]["slaclip_c_min"] == 0.1
            assert candidate["params"]["slaclip_c_max"] == 15.0
        assert set(candidate["runs"]) == {"42", "43", "44", "45", "46"}
        for seed, run in candidate["runs"].items():
            prefix = f"screen/runs/{candidate['id']}/seed-{seed}"
            assert run == {
                "run_status": f"{prefix}/adapter/run_status.json",
                "validation_metrics": (
                    f"{prefix}/results/validation/validation_metrics.json"
                ),
                "split_manifest": f"{prefix}/results/validation/split_manifest.json",
            }
            assert str(campaign) not in "".join(run.values())


def test_byte_identical_rerun_preserves_locked_file(tmp_path: Path) -> None:
    campaign, output, first = _build(tmp_path)
    before = output.read_bytes()
    inode = output.stat().st_ino
    second = registry_builder.build_registry(
        campaign_root=campaign,
        output=output,
        model_id=MODEL_ID,
        model_revision=MODEL_REVISION,
        code_sha=CODE_SHA,
    )
    assert second == first
    assert output.read_bytes() == before
    assert output.stat().st_ino == inode


def test_refuses_to_replace_inconsistent_registry(tmp_path: Path) -> None:
    campaign, output, _ = _build(tmp_path)
    output.write_text('{"tampered":true}\n', encoding="utf-8")
    with pytest.raises(registry_builder.RegistryError, match="refusing to overwrite"):
        registry_builder.build_registry(
            campaign_root=campaign,
            output=output,
            model_id=MODEL_ID,
            model_revision=MODEL_REVISION,
            code_sha=CODE_SHA,
        )
    assert output.read_text(encoding="utf-8") == '{"tampered":true}\n'


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("model_id", "", "model_id"),
        ("model_revision", "main", "model_revision"),
        ("code_sha", "abc123", "code_sha"),
    ],
)
def test_rejects_unlocked_inputs(
    tmp_path: Path, field: str, value: str, message: str
) -> None:
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    kwargs = {
        "campaign_root": campaign,
        "output": Path("screen/candidate_registry.json"),
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "code_sha": CODE_SHA,
    }
    kwargs[field] = value
    with pytest.raises(registry_builder.RegistryError, match=message):
        registry_builder.build_registry(**kwargs)


@pytest.mark.parametrize(
    "output",
    [
        Path("candidate_registry.json"),
        Path("selection/candidate_registry.json"),
        Path("screen/evaluation/candidate_registry.json"),
        Path("screen/not-the-registry.json"),
    ],
)
def test_rejects_selector_incompatible_output_paths(
    tmp_path: Path, output: Path
) -> None:
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    with pytest.raises(registry_builder.RegistryError):
        registry_builder.build_registry(
            campaign_root=campaign,
            output=output,
            model_id=MODEL_ID,
            model_revision=MODEL_REVISION,
            code_sha=CODE_SHA,
        )
