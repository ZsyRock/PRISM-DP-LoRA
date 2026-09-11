from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


weighted = load_module("weighted_target_screen")
campaign = load_module("build_paper_coverage_campaign")


def records(early=1.0, late=0.7, steps=200):
    # Deliberately unrelated small-gradient values guard against the old inverse map.
    return {
        step: {"raw_clip_fraction": early if step <= 25 else late,
               "raw_reference_small_gradient_proxy": 0.4,
               "raw_reference_conditional_clip_fraction": 0.91}
        for step in range(1, steps + 1)
    }


def test_direct_weighted_target_is_079_without_inverse_small_mass_adjustment():
    target = weighted.derive_target(records())
    assert target["rho"] == pytest.approx(0.79)
    assert target["inverse_small_mass_adjustment"] is False
    assert target["early_steps"] == [1, 25]
    assert target["late_steps"] == [176, 200]
    # Full controller's effective global target at z=.4 is .474, not .79.
    assert target["rho"] * (1.0 - 0.4) == pytest.approx(0.474)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -0.1, 1.1])
def test_reject_invalid_actual_clipping(value):
    data = records()
    data[1]["raw_clip_fraction"] = value
    with pytest.raises(ValueError, match="finite"):
        weighted.derive_target(data)


def test_reject_nonconsecutive_or_too_short_trajectory():
    data = records()
    del data[10]
    with pytest.raises(ValueError, match="consecutive"):
        weighted.derive_target(data)
    with pytest.raises(ValueError, match="50"):
        weighted.derive_target(records(steps=49))


@pytest.mark.parametrize("value", [0.0, 1.0])
def test_endpoint_target_is_explicitly_flagged_without_silent_replacement(value):
    result = weighted.derive_target(records(value, value))
    assert result["rho"] == value
    assert result["degenerate_target"] is True


def make_manifest():
    return weighted.build_manifest(campaign, "1" * 40, weighted.MODEL_REVISION, "2" * 40)


def test_manifest_has_six_fixed_arms_and_single_a100_plan():
    manifest = make_manifest()
    assert manifest["profile"] == weighted.PROFILE
    assert len(manifest["arms"]) == 6
    assert {(a["setting_id"], a["initial_c"]) for a in manifest["arms"]} == {
        ("glue8-4b-eps6-r8", 15), ("glue8-4b-eps6-r8", 40), ("glue8-4b-eps6-r8", 80),
        ("glue8-4b-eps6-r16", 15), ("glue8-4b-eps6-r16", 30), ("glue8-4b-eps6-r16", 60),
    }
    assert all(a["steps"] == 200 and a["seed"] == 47 and a["lane"] == 0 for a in manifest["arms"])
    assert manifest["full_slaclip"]["C_max"] == 100
    recipe = manifest["weighted_target_screen"]
    assert recipe["total_planned_arms"] == 18
    assert recipe["resources"] == {"gpus": 1, "gpu_type": "a100", "memory": "80G", "time": "24:00:00", "lanes": 1}
    assert recipe["journal_confirmation"] is False
    assert recipe["inverse_small_mass_adjustment"] is False


def test_source_hash_tamper_is_rejected(tmp_path):
    path = tmp_path / "source.json"
    path.write_text('{"clipping": 0.8}')
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    weighted._check_hash(campaign, path, digest)
    path.write_text('{"clipping": 0.9}')
    with pytest.raises(campaign.CampaignError, match="hash mismatch"):
        weighted._check_hash(campaign, path, digest)


def test_lock_creates_fresh_matched_stage2_and_refuses_changed_results(tmp_path, monkeypatch):
    manifest = make_manifest()
    campaign._with_sha(tmp_path / "plans/manifest.json", campaign._json_bytes(manifest))
    source_data = {"default": {}, "tuned": {}}
    for sid in weighted.SETTING_ORDER:
        source_data["default"][sid] = {"target": weighted.derive_target(records(steps=500)),
                                        "provenance": {"source": "default", "sha256": "3" * 64}}
        if sid in weighted.PRIOR_WINNERS:
            setting = next(s for s in campaign.REGIME_SETTINGS if s["id"] == sid)
            arm = weighted._arm({**setting, "model_revision": weighted.MODEL_REVISION}, "prior",
                                weighted.PRIOR_WINNERS[sid], 47, 200, "weighted_stage1_fixed")
            source_data["tuned"][sid] = {
                "arm": arm, "target": weighted.derive_target(records(0.8, 0.2)),
                "provenance": {"source": "prior", "sha256": "4" * 64}, "endpoint_loss": 0.25,
            }
    calls = []
    def verify(builder, root):
        calls.append(root)
        return source_data
    monkeypatch.setattr(weighted, "verify_sources", verify)
    losses = {15.0: 0.4, 30.0: 0.25, 40.0: 0.2, 60.0: 0.3, 80.0: 0.1}
    def load(root, arm, sha):
        return ({"validation": {"loss_mean": losses[arm["initial_c"]]}},
                records(0.8, 0.2), {}, {"raw_telemetry": "5" * 64})
    builder = SimpleNamespace(**vars(campaign))
    builder._load_focused_arm = load
    result = weighted.lock(builder, tmp_path)
    assert len(calls) == 1
    assert len(result["stage2_arms"]) == 12
    assert sum(a["method"] == "baseline" for a in result["stage2_arms"]) == 4
    assert all(a["steps"] == 150 and a["seed"] == 48 and a["lane"] == 0 for a in result["stage2_arms"])
    assert all(a["c_max"] == 100 for a in result["stage2_arms"])
    setting = next(s for s in result["settings"] if s["setting_id"].endswith("r8"))
    assert setting["selected_C"] == 80
    assert setting["boundary_winner"] is True
    assert setting["journal_confirmation"] is False
    for sid in weighted.SETTING_ORDER:
        arms = [a for a in result["stage2_arms"] if a["setting_id"] == sid]
        assert len({a["initial_c"] for a in arms}) == 1
        assert {a["role"] for a in arms} == {"fresh_fixed_comparator", "weighted_default_target", "weighted_tuned_target"}
    plan = tmp_path / "plans/stage2-weighted.tsv"
    assert len(plan.read_text().splitlines()) == 12
    assert hashlib.sha256(plan.read_bytes()).hexdigest() == result["stage2_plan_sha256"]
    assert weighted.lock(builder, tmp_path) == result
    losses[80.0] = 0.9
    with pytest.raises(campaign.CampaignError, match="refusing to overwrite"):
        weighted.lock(builder, tmp_path)


def test_lock_rejects_mutated_stage1_recipe(tmp_path, monkeypatch):
    manifest = make_manifest()
    manifest["arms"][0]["initial_c"] = 14.0
    campaign._with_sha(tmp_path / "plans/manifest.json", campaign._json_bytes(manifest))
    monkeypatch.setattr(weighted, "verify_sources", lambda *args: {"default": {}, "tuned": {}})
    with pytest.raises(campaign.CampaignError, match="recipe changed"):
        weighted.lock(campaign, tmp_path)
