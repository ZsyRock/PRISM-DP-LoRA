from __future__ import annotations

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


interval = load("range_target_test_module", "scripts/range_target_screen.py")
campaign = load("range_campaign_test_module", "scripts/build_paper_coverage_campaign.py")


def records(values=None):
    values = [i / 99 for i in range(100)] if values is None else values
    return {step: {"step": step, "raw_clip_fraction": value,
                   "raw_reference_small_gradient_proxy": 0.8,
                   "raw_reference_conditional_clip_fraction": 0.99}
            for step, value in enumerate(values, start=1)}


def test_skewed_clipping_trajectory_uses_range_positions_not_empirical_quantiles():
    data = records([0.1] * 40 + [0.4] * 40 + [0.9] * 20)
    result = interval.derive_targets(data)
    assert [entry["rho"] for entry in result["targets"]] == pytest.approx([0.3, 0.5, 0.7])
    empirical = interval._EMPIRICAL.derive_targets(data)
    assert [entry["rho"] for entry in empirical["targets"]] == [0.1, 0.4, 0.4]
    assert result["empirical_clipping_percentiles"] is False
    assert result["gradient_norm_quantiles"] is False
    assert result["method"] == interval.RANGE_METHOD
    assert result["flags"] == []


def test_uses_initial_maximum_and_rare_minimum_without_burnin():
    result = interval.derive_targets(records([1.0] + [0.5] * 98 + [0.2]))
    assert [entry["rho"] for entry in result["targets"]] == pytest.approx([0.4, 0.6, 0.8])
    assert result["source_step_range"] == [1, 100]
    assert result["source_window"] == "all_steps_no_burn_in"


def test_never_inverts_small_gradient_mass_or_mutates_input():
    data = records()
    before = deepcopy(data)
    result = interval.derive_targets(data)
    assert data == before
    assert result["targets"][1]["rho"] == 0.5
    assert result["inverse_small_mass_adjustment"] is False
    assert result["controller_global_target"] == "p_star_t=rho_q*(1-z_t)"


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -0.001, 1.001, None, "bad"])
def test_invalid_observations_rejected(value):
    data = records()
    data[20]["raw_clip_fraction"] = value
    with pytest.raises(ValueError, match="finite"):
        interval.derive_targets(data)


@pytest.mark.parametrize("edit", ["gap", "shifted", "string_key", "bool_key", "short", "embedded"])
def test_invalid_step_sequences_rejected(edit):
    data = records()
    if edit == "gap":
        data.pop(50)
    elif edit == "shifted":
        data[101] = data.pop(1)
    elif edit == "string_key":
        data["50"] = data.pop(50)
    elif edit == "bool_key":
        data[True] = data.pop(1)
    elif edit == "embedded":
        data[50]["step"] = 49
    else:
        data = records([0.5] * 49)
    with pytest.raises(ValueError):
        interval.derive_targets(data)


@pytest.mark.parametrize("value", [0.0, 0.4, 1.0])
def test_constant_ranges_are_deduplicated_not_silently_changed(value):
    result = interval.derive_targets(records([value] * 50))
    assert [entry["rho"] for entry in result["targets"]] == [value] * 3
    assert len(result["unique_targets"]) == 1
    assert result["unique_targets"][0]["range_positions"] == [0.25, 0.5, 0.75]
    assert result["targets"][2]["duplicate_of_range_position"] == 0.25
    assert "constant_clipping_trajectory" in result["flags"]
    assert "duplicate_range_targets_deduplicated" in result["flags"]


@pytest.mark.parametrize("positions", [(), (-0.1,), (1.1,), (float("nan"),), (0.5, 0.5), (None,)])
def test_invalid_range_positions_rejected(positions):
    with pytest.raises(ValueError, match="positions"):
        interval.derive_targets(records(), positions)


def test_custom_endpoints_are_exact():
    result = interval.derive_targets(records(), (0.0, 1.0))
    assert [entry["rho"] for entry in result["targets"]] == [0.0, 1.0]


def sources():
    return {sid: {"records": records([lower] + [upper] * 499),
                  "provenance": {"sha256": str(index) * 64},
                  "source_initial_c": 1.0, "source_seed": 42,
                  "source_kind": "paper_default_C1", "screen_initial_c": 15.0}
            for index, (sid, (lower, upper)) in enumerate(interval.PINNED_DEFAULT_RANGES.items(), start=1)}


def test_sealed_manifest_deterministic_detached_and_nonprivate():
    data = sources()
    before = deepcopy(data)
    result = interval.build_target_manifest(data)
    assert data == before
    assert result == interval.build_target_manifest(dict(reversed(list(data.items()))))
    interval.verify_target_manifest(result)
    assert result["NON_PRIVATE_CALIBRATION"] is True
    assert result["end_to_end_dp_claim"] is False
    data[next(iter(data))]["provenance"]["sha256"] = "changed"
    interval.verify_target_manifest(result)


@pytest.mark.parametrize("field", ["rho", "source", "metadata", "digest"])
def test_sealed_manifest_detects_tampering(field):
    result = interval.build_target_manifest(sources())
    if field == "rho":
        result["settings"][0]["target_summary"]["targets"][0]["rho"] = 0.99
    elif field == "source":
        result["settings"][0]["source_provenance"]["sha256"] = "changed"
    elif field == "metadata":
        result["settings"][0]["source_metadata"]["source_initial_c"] = 2.0
    else:
        result["content_sha256"] = "bad"
    with pytest.raises(ValueError, match="hash mismatch"):
        interval.verify_target_manifest(result)


@pytest.mark.parametrize("data", [{}, {"": {}}, {"setting": {"records": records()}}, {"setting": {"provenance": {"sha": "x"}}}])
def test_missing_provenance_or_sources_rejected(data):
    with pytest.raises(ValueError):
        interval.build_target_manifest(data)


def test_manifest_20_matched_seed50_arms_uses_default_c0_and_three_range_positions():
    weighted = campaign._weighted_target_module()
    result = interval.build_manifest(campaign, "1" * 40, weighted.MODEL_REVISION, "2" * 40)
    assert result["profile"] == interval.PROFILE
    assert result["weighted_target_screen"] == {"enabled": False}
    assert result["quantile_target_screen"] == {"enabled": False}
    assert len(result["arms"]) == 20
    for sid in interval.SETTING_ORDER:
        arms = [arm for arm in result["arms"] if arm["setting_id"] == sid]
        assert len(arms) == 5
        assert [arm["role"] for arm in arms] == ["range_default_fixed", "fresh_fixed_comparator"] + ["range_default_target"] * 3
        assert arms[0]["initial_c"] == 1.0
        assert arms[1]["initial_c"] == interval.SELECTED_FIXED_C[sid]
        assert {arm["initial_c"] for arm in arms[2:]} == {1.0}
        assert [arm["range_position"] for arm in arms[2:]] == [0.25, 0.5, 0.75]
        assert [arm["rho"] for arm in arms[2:]] == list(interval.PINNED_DEFAULT_TARGETS[sid])
        assert all(arm["eta"] == 0.05 for arm in arms[2:])
        assert all("quantile" not in arm for arm in arms)
    assert all(arm["seed"] == 50 and arm["steps"] == 150 and arm["lane"] == 0 and arm["c_max"] == 100 for arm in result["arms"])
    recipe = result["range_target_screen"]
    assert recipe["C0"] == 1.0
    assert recipe["source_steps"] == 500
    assert recipe["K"] == 15
    assert recipe["resources"] == {"gpus": 1, "gpu_type": "a100", "memory": "80G", "time": "24:00:00", "lanes": 1}
    assert recipe["deferred_math"]["constant_one_configurations"] == 5
    assert len(set(recipe["deferred_math"]["nonconstant_range_targets"])) == 3
    assert recipe["official_task_evaluation"] is False
    assert recipe["journal_confirmation"] is False


def setup_verified_summary():
    empirical = interval._EMPIRICAL
    source_manifest = empirical.build_target_manifest(sources())
    audit = {"targets": source_manifest, "previous_selection": {"verified": True}}
    helper = SimpleNamespace(verify_sources=lambda *args: audit,
                             verify_target_manifest=empirical.verify_target_manifest)
    builder = SimpleNamespace(_quantile_target_module=lambda: helper,
                              CampaignError=campaign.CampaignError)
    return builder, audit


def test_source_verifier_converts_sealed_verified_minmax_and_retains_provenance(tmp_path):
    builder, previous = setup_verified_summary()
    before = deepcopy(previous)
    result = interval.verify_sources(builder, tmp_path)
    interval.verify_target_manifest(result["targets"])
    assert previous == before
    assert result["previous_selection"] == {"verified": True}
    assert result["profile"] == interval.PROFILE
    assert "raw_clip_fraction\"" not in json.dumps(result, allow_nan=False)
    for entry in result["targets"]["settings"]:
        sid = entry["setting_id"]
        assert entry["source_metadata"]["screen_initial_c"] == 1.0
        assert [item["rho"] for item in entry["target_summary"]["targets"]] == list(interval.PINNED_DEFAULT_TARGETS[sid])


def test_source_verifier_rejects_broken_old_digest(tmp_path):
    builder, previous = setup_verified_summary()
    previous["targets"]["settings"][0]["target_summary"]["realized_clip_min"] = 0.2
    with pytest.raises(ValueError, match="hash mismatch"):
        interval.verify_sources(builder, tmp_path)


@pytest.mark.parametrize("tamper", ["min", "steps", "seed", "initial_c", "settings"])
def test_source_verifier_rejects_resealed_but_unexpected_baselines(tmp_path, tamper):
    builder, previous = setup_verified_summary()
    entry = previous["targets"]["settings"][0]
    if tamper == "min":
        entry["target_summary"]["realized_clip_min"] = 0.2
    elif tamper == "steps":
        entry["target_summary"]["source_steps"] = 499
    elif tamper == "seed":
        entry["source_metadata"]["source_seed"] = 99
    elif tamper == "initial_c":
        entry["source_metadata"]["source_initial_c"] = 2.0
    else:
        previous["targets"]["settings"].pop()
    import hashlib
    content = {key: value for key, value in previous["targets"].items() if key != "content_sha256"}
    previous["targets"]["content_sha256"] = hashlib.sha256(interval._EMPIRICAL._canonical_bytes(content)).hexdigest()
    with pytest.raises(campaign.CampaignError):
        interval.verify_sources(builder, tmp_path)
