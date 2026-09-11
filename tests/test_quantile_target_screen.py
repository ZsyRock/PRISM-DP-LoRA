from __future__ import annotations

from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("quantile_target_screen", ROOT / "scripts/quantile_target_screen.py")
assert SPEC and SPEC.loader
quantile = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(quantile)
CAMPAIGN_SPEC = importlib.util.spec_from_file_location("quantile_test_campaign", ROOT / "scripts/build_paper_coverage_campaign.py")
assert CAMPAIGN_SPEC and CAMPAIGN_SPEC.loader
campaign = importlib.util.module_from_spec(CAMPAIGN_SPEC)
CAMPAIGN_SPEC.loader.exec_module(campaign)


def records(values=None):
    values = [i / 99 for i in range(100)] if values is None else values
    return {step: {"step": step, "raw_clip_fraction": value,
                   "raw_reference_small_gradient_proxy": 0.8,
                   "raw_reference_conditional_clip_fraction": 0.99}
            for step, value in enumerate(values, start=1)}


def test_linear_empirical_percentiles_exact_on_fractional_trajectory():
    result = quantile.derive_targets(records())
    assert [entry["rho"] for entry in result["targets"]] == pytest.approx([0.25, 0.5, 0.75])
    assert result["source_steps"] == 100
    assert result["source_step_range"] == [1, 100]
    assert result["realized_clip_min"] == 0.0
    assert result["realized_clip_max"] == 1.0
    assert result["realized_clip_mean"] == pytest.approx(0.5)
    assert result["method"] == "linear_empirical_quantile_all_steps"
    assert result["flags"] == []


def test_empirical_distribution_not_min_max_interval_interpolation():
    result = quantile.derive_targets(records([0.1] * 40 + [0.4] * 40 + [0.9] * 20))
    assert [entry["rho"] for entry in result["targets"]] == [0.1, 0.4, 0.4]
    assert result["duplicate_targets"] is True
    assert result["targets"][2]["duplicate_of_quantile"] == 0.5
    assert result["unique_targets"][1]["quantiles"] == [0.5, 0.75]


def test_uses_all_steps_including_initial_high_clipping():
    result = quantile.derive_targets(records([1.0] * 25 + [0.2] * 75))
    assert [entry["rho"] for entry in result["targets"]] == pytest.approx([0.2, 0.2, 0.4])
    assert result["source_window"] == "all_steps_no_burn_in"


def test_never_inverts_small_gradient_mass_or_uses_conditional_telemetry():
    data = records()
    before = deepcopy(data)
    result = quantile.derive_targets(data)
    assert data == before
    assert result["targets"][1]["rho"] == pytest.approx(0.5)
    assert result["targets"][1]["rho"] * (1 - 0.8) == pytest.approx(0.1)
    assert result["inverse_small_mass_adjustment"] is False


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"), -0.001, 1.001, None, "bad"])
def test_rejects_invalid_clipping_observations(value):
    data = records()
    data[20]["raw_clip_fraction"] = value
    with pytest.raises(ValueError, match="finite"):
        quantile.derive_targets(data)


def test_rejects_missing_clipping_field():
    data = records()
    del data[20]["raw_clip_fraction"]
    with pytest.raises(ValueError, match="every step"):
        quantile.derive_targets(data)


@pytest.mark.parametrize("edit", ["gap", "shifted", "string_key", "bool_key", "short"])
def test_requires_at_least_50_unique_consecutive_integer_steps(edit):
    data = records()
    if edit == "gap":
        data.pop(50)
    elif edit == "shifted":
        data[101] = data.pop(1)
    elif edit == "string_key":
        data["50"] = data.pop(50)
    elif edit == "bool_key":
        data[True] = data.pop(1)
    else:
        data = records([0.5] * 49)
    with pytest.raises(ValueError, match="50 consecutive"):
        quantile.derive_targets(data)


def test_embedded_duplicate_step_cannot_hide_behind_unique_map_keys():
    data = records()
    data[50]["step"] = 49
    with pytest.raises(ValueError, match="unique trajectory key"):
        quantile.derive_targets(data)


@pytest.mark.parametrize("value,flag", [(0.0, "zero_percentile_target"), (1.0, "saturated_percentile_at_one")])
def test_boundary_and_duplicate_targets_are_flagged_not_adjusted(value, flag):
    result = quantile.derive_targets(records([value] * 50))
    assert [entry["rho"] for entry in result["targets"]] == [value] * 3
    assert len(result["unique_targets"]) == 1
    assert result["unique_targets"][0]["quantiles"] == [0.25, 0.5, 0.75]
    assert result["unique_targets"][0]["degenerate_target"] is True
    assert flag in result["flags"]
    assert "constant_clipping_trajectory" in result["flags"]
    assert "duplicate_percentile_targets_deduplicated" in result["flags"]


def test_saturated_percentile_flag_even_when_not_entire_source_saturated():
    result = quantile.derive_targets(records([0.2] * 20 + [1.0] * 80))
    assert result["all_steps_clipped"] is False
    assert result["saturated_source"] is False
    assert all(entry["saturated_target"] for entry in result["targets"])
    assert "saturated_percentile_at_one" in result["flags"]


@pytest.mark.parametrize("probabilities", [(), (-0.1,), (1.1,), (float("nan"),), (0.5, 0.5), (None,)])
def test_invalid_quantile_requests_rejected(probabilities):
    with pytest.raises(ValueError, match="quantile"):
        quantile.derive_targets(records(), probabilities)


def test_custom_endpoint_quantiles_are_exact():
    result = quantile.derive_targets(records(), (0.0, 1.0))
    assert [entry["rho"] for entry in result["targets"]] == [0.0, 1.0]


def sources():
    return {
        f"setting-{rank}": {
            "records": records(),
            "provenance": {"telemetry_sha256": str(rank) * 64, "source_kind": "default"},
            "initial_c": 1.0,
        }
        for rank in (1, 2, 3, 4)
    }


def test_manifest_is_deterministic_self_verifying_and_detached_from_inputs():
    data = sources()
    before = deepcopy(data)
    result = quantile.build_target_manifest(data)
    assert data == before
    assert len(result["settings"]) == 4
    quantile.verify_target_manifest(result)
    assert result == quantile.build_target_manifest(dict(reversed(list(data.items()))))
    data["setting-1"]["provenance"]["telemetry_sha256"] = "changed"
    data["setting-1"]["records"][1]["raw_clip_fraction"] = 1.0
    quantile.verify_target_manifest(result)
    assert result["settings"][0]["source_provenance"]["telemetry_sha256"] == "1" * 64
    assert result["NON_PRIVATE_CALIBRATION"] is True
    assert result["end_to_end_dp_claim"] is False


@pytest.mark.parametrize("field", ["rho", "source", "metadata", "digest"])
def test_manifest_tampering_is_rejected(field):
    result = quantile.build_target_manifest(sources())
    if field == "rho":
        result["settings"][0]["target_summary"]["targets"][0]["rho"] = 0.99
    elif field == "source":
        result["settings"][0]["source_provenance"]["telemetry_sha256"] = "changed"
    elif field == "metadata":
        result["settings"][0]["source_metadata"]["initial_c"] = 2.0
    else:
        result["content_sha256"] = "bad"
    with pytest.raises(ValueError, match="hash mismatch"):
        quantile.verify_target_manifest(result)


@pytest.mark.parametrize("data", [{}, {"": {}}, {"setting": {"records": records()}}, {"setting": {"provenance": {"sha": "x"}}}])
def test_manifest_requires_nonempty_sources_and_provenance(data):
    with pytest.raises(ValueError):
        quantile.build_target_manifest(data)


def test_campaign_recipe_pins_default_targets_and_matched_fixed_controls():
    weighted = campaign._weighted_target_module()
    result = quantile.build_manifest(campaign, "1" * 40, weighted.MODEL_REVISION, "2" * 40)
    assert result["profile"] == quantile.PROFILE
    assert len(result["arms"]) == 20
    assert result["weighted_target_screen"] == {"enabled": False}
    assert result["arms"][0]["setting_id"] == "glue8-4b-eps6-r32"
    for sid in quantile.SETTING_ORDER:
        arms = [arm for arm in result["arms"] if arm["setting_id"] == sid]
        assert len(arms) == 5
        assert [arm["role"] for arm in arms] == ["quantile_default_fixed", "fresh_fixed_comparator"] + ["quantile_default_target"] * 3
        assert arms[0]["initial_c"] == 1.0
        assert {arm["initial_c"] for arm in arms[1:]} == {quantile.SELECTED_FIXED_C[sid]}
        assert [arm["rho"] for arm in arms[2:]] == list(quantile.PINNED_DEFAULT_TARGETS[sid])
        assert [arm["quantile"] for arm in arms[2:]] == [0.25, 0.5, 0.75]
    assert all(arm["seed"] == 49 and arm["steps"] == 150 and arm["lane"] == 0 and arm["c_max"] == 100 for arm in result["arms"])
    recipe = result["quantile_target_screen"]
    assert recipe["target_source"] == "paper_default_C1"
    assert recipe["source_steps"] == 500
    assert recipe["resources"] == {"gpus": 1, "gpu_type": "a100", "memory": "80G", "time": "24:00:00", "lanes": 1}
    assert recipe["deferred_math"]["requested_percentiles"] == [1.0, 1.0, 1.0]
    assert recipe["journal_confirmation"] is False
    assert recipe["official_task_evaluation"] is False


def setup_source_verification(tmp_path, monkeypatch):
    """Use real selection checks with synthetic but cryptographically bound files."""
    weighted = campaign._weighted_target_module()
    builder = SimpleNamespace(**vars(campaign))
    builder._weighted_target_module = lambda: weighted
    previous_root = tmp_path / quantile.PREVIOUS_CAMPAIGN
    previous_manifest = weighted.build_manifest(builder, quantile.PREVIOUS_CODE, weighted.MODEL_REVISION, "0" * 40)
    source_data = {"default": {}, "tuned": {}}
    trajectory = records([i / 499 for i in range(500)])
    rho = tuple(entry["rho"] for entry in quantile.derive_targets(trajectory)["targets"])
    monkeypatch.setattr(quantile, "PINNED_DEFAULT_TARGETS", {sid: rho for sid in quantile.SETTING_ORDER})
    for sid in quantile.SETTING_ORDER:
        source_data["default"][sid] = {
            "records": trajectory, "provenance": {"sha": "d" * 64, "setting": sid},
            "target": weighted.derive_target(trajectory),
        }
        if sid in weighted.PRIOR_WINNERS:
            source_data["tuned"][sid] = {
                "arm": {"initial_c": quantile.SELECTED_FIXED_C[sid]},
                "records": trajectory, "provenance": {"sha": "t" * 64, "setting": sid},
                "target": weighted.derive_target(trajectory),
            }
    monkeypatch.setattr(weighted, "verify_sources", lambda *args: source_data)
    artifacts = {}
    rankings = {}
    focused = {}
    for arm in previous_manifest["arms"]:
        loss = 0.25 if arm["initial_c"] == quantile.SELECTED_FIXED_C[arm["setting_id"]] else 0.5
        hashes = {"status": hashlib.sha256(arm["arm_id"].encode()).hexdigest()}
        artifacts[arm["arm_id"]] = hashes
        focused[arm["arm_id"]] = ({"validation": {"loss_mean": loss}}, trajectory, {}, hashes)
        rankings.setdefault(arm["setting_id"], []).append({"C": arm["initial_c"], "validation_loss": loss, "arm_id": arm["arm_id"]})
    selections = []
    for sid in quantile.SETTING_ORDER:
        if sid in rankings:
            ranking = sorted(rankings[sid], key=lambda item: (item["validation_loss"], item["C"], item["arm_id"]))
            winner = next(arm for arm in previous_manifest["arms"] if arm["arm_id"] == ranking[0]["arm_id"])
            provenance = {"campaign_id": quantile.PREVIOUS_CAMPAIGN, "relative_root": winner["relative_root"],
                          "code_sha": quantile.PREVIOUS_CODE, "hashes": artifacts[winner["arm_id"]]}
        else:
            ranking = []
            provenance = source_data["tuned"][sid]["provenance"]
        selections.append({
            "setting_id": sid, "selected_C": quantile.SELECTED_FIXED_C[sid],
            "default_source": source_data["default"][sid]["provenance"],
            "default_target": source_data["default"][sid]["target"],
            "tuned_source": provenance, "tuned_target": weighted.derive_target(trajectory),
            "fixed_ranking": ranking, "boundary_winner": False,
        })
    manifest_bytes = campaign._json_bytes(previous_manifest)
    manifest_hash = hashlib.sha256(manifest_bytes).hexdigest()
    lock = {"code_sha": quantile.PREVIOUS_CODE, "manifest_sha256": manifest_hash,
            "settings": selections, "stage1_artifacts": artifacts}
    lock_bytes = campaign._json_bytes(lock)
    monkeypatch.setattr(quantile, "PREVIOUS_MANIFEST_SHA256", manifest_hash)
    monkeypatch.setattr(quantile, "PREVIOUS_LOCK_SHA256", hashlib.sha256(lock_bytes).hexdigest())
    campaign._with_sha(previous_root / "plans/manifest.json", manifest_bytes)
    campaign._with_sha(previous_root / "selection/weighted-targets.lock.json", lock_bytes)
    builder._load_focused_arm = lambda root, arm, sha: focused[arm["arm_id"]]
    return builder, tmp_path / "new-quantile-campaign", source_data, focused, previous_root


def test_source_verifier_returns_recomputable_json_without_raw_records(tmp_path, monkeypatch):
    builder, root, _, _, _ = setup_source_verification(tmp_path, monkeypatch)
    verified = quantile.verify_sources(builder, root)
    quantile.verify_target_manifest(verified["targets"])
    serialized = json.dumps(verified, allow_nan=False)
    assert "raw_clip_fraction\"" not in serialized
    assert len(verified["targets"]["settings"]) == 4
    assert len(verified["previous_selection"]["stage1_artifacts"]) == 6


@pytest.mark.parametrize("tamper", ["lock", "manifest", "selected_winner", "artifact", "default_rho", "default_provenance", "prior_fixed"])
def test_source_verifier_rejects_tampered_selection_or_targets(tmp_path, monkeypatch, tamper):
    builder, root, sources_data, focused, previous_root = setup_source_verification(tmp_path, monkeypatch)
    if tamper in ("lock", "manifest"):
        path = previous_root / ("selection/weighted-targets.lock.json" if tamper == "lock" else "plans/manifest.json")
        path.write_text(path.read_text() + " ")
    elif tamper == "selected_winner":
        arm_id = next(key for key in focused if "r8--weighted-fixed-c40" in key)
        focused[arm_id][0]["validation"]["loss_mean"] = 1.0
    elif tamper == "artifact":
        focused[next(iter(focused))][3]["status"] = "changed"
    elif tamper == "default_rho":
        sources_data["default"]["glue8-4b-eps6-r8"]["records"] = records([0.9] * 500)
    elif tamper == "default_provenance":
        sources_data["default"]["glue8-4b-eps6-r8"]["provenance"]["sha"] = "changed"
    else:
        sources_data["tuned"]["glue8-4b-eps6-r32"]["arm"]["initial_c"] = 14.0
    with pytest.raises(campaign.CampaignError):
        quantile.verify_sources(builder, root)
