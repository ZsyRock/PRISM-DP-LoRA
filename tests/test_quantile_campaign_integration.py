"""Run the quantile screen through the real campaign artifact validators.

Only historical scratch-source verification is mocked in the synthetic tests;
manifest planning, status, telemetry, validation, and matched comparisons use
the production implementation. Historical-source integrity has separate tests.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "quantile_campaign_fixtures",
    ROOT / "tests" / "test_build_paper_coverage_campaign.py",
)
assert SPEC and SPEC.loader
fixtures = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(fixtures)
campaign = fixtures.campaign

EXPECTED = {
    "glue8-4b-eps6-r8": (
        40.0, (0.4444444444444444, 0.5, 0.5488402678144428),
    ),
    "glue8-4b-eps6-r16": (
        30.0, (0.4633017163504969, 0.5096189419163892, 0.5689655172413793),
    ),
    "glue8-4b-eps6-r32": (
        15.0, (0.4909090909090909, 0.5410800385728062, 0.6068788171006108),
    ),
    "glue8-4b-eps3-r16": (
        30.0, (0.4741902834008097, 0.5250069463739928, 0.5797101449275363),
    ),
}


def _make_manifest():
    quantile = campaign._quantile_target_module()
    weighted = campaign._weighted_target_module()
    manifest = campaign.build_manifest(
        fixtures.CODE_SHA, weighted.MODEL_REVISION, fixtures.REV_9B,
        profile=quantile.PROFILE,
    )
    return quantile, manifest


def _write_quantile_campaign(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    quantile, manifest = _make_manifest()
    monkeypatch.setattr(campaign, "_quantile_target_module", lambda: quantile)
    monkeypatch.setattr(
        quantile, "verify_sources",
        lambda *args: {"synthetic_source": True, "settings": []},
    )
    campaign._with_sha(tmp_path / "plans/manifest.json", campaign._json_bytes(manifest))
    # All adaptive arms beat the weak default C=1, but only q25 beats the
    # selected tuned fixed. Comparing only to C=1 would falsely report 12 wins.
    losses = {}
    for arm in manifest["arms"]:
        if arm["method"] == "baseline":
            loss = 0.8 if arm["initial_c"] == 1.0 else 0.4
        else:
            loss = {"q25": 0.35, "q50": 0.4, "q75": 0.45}[
                arm["candidate_id"].rsplit("-", 1)[-1]
            ]
        losses[arm["candidate_id"]] = loss
    fixtures._write_focused_campaign(
        tmp_path, arms_override=manifest["arms"], validation_losses=losses,
    )
    for arm in manifest["arms"]:
        path = tmp_path / arm["relative_root"] / "adapter/run_status.json"
        status = json.loads(path.read_text())
        status["config"].update(raw_hist_bins=512, raw_hist_max=200.0)
        if arm["method"] == "slaclip":
            status["config"]["slaclip_c_max"] = arm["c_max"]
        path.write_text(json.dumps(status))
    return manifest


def test_quantile_manifest_matches_requested_empirical_targets_and_budget():
    _, manifest = _make_manifest()
    arms = manifest["arms"]
    assert manifest["profile"] == "glue-quantile-target-screen"
    assert len(arms) == len({arm["arm_id"] for arm in arms}) == 20
    assert arms[0]["setting_id"] == "glue8-4b-eps6-r32"
    assert {arm["setting_id"] for arm in arms} == set(EXPECTED)
    assert all(arm["seed"] == 49 and arm["steps"] == 150 for arm in arms)
    assert all(arm["lane"] == 0 and arm["c_max"] == 100 for arm in arms)
    assert sum(arm["steps"] for arm in arms) == 3000
    assert manifest["full_slaclip"]["C_max"] == 100.0
    assert manifest["full_slaclip"]["K"] == 15
    assert "rho*(1-z_t)" in manifest["full_slaclip"]["target_semantics"]
    for sid, (selected_c, expected_rhos) in EXPECTED.items():
        matching = [arm for arm in arms if arm["setting_id"] == sid]
        fixed = [arm for arm in matching if arm["method"] == "baseline"]
        adaptive = sorted(
            (arm for arm in matching if arm["method"] == "slaclip"),
            key=lambda arm: arm["rho"],
        )
        assert len(fixed) == 2 and len(adaptive) == 3
        assert {arm["initial_c"] for arm in fixed} == {1.0, selected_c}
        assert [arm["rho"] for arm in adaptive] == pytest.approx(expected_rhos)
        assert all(arm["initial_c"] == selected_c for arm in adaptive)
        assert all(arm["eta"] == 0.05 for arm in adaptive)
        assert all(arm["role"] == "quantile_default_target" for arm in adaptive)
        assert [arm["quantile"] for arm in adaptive] == [0.25, 0.5, 0.75]
        assert {arm["candidate_id"] for arm in adaptive} == {
            "quantile-default-q25", "quantile-default-q50", "quantile-default-q75",
        }
    plan = campaign._plan_bytes(manifest, 0, include_all=True).decode().splitlines()
    assert len(plan) == 20
    assert [line.split("|")[2] for line in plan[:5]] == ["glue8-4b-eps6-r32"] * 5
    assert campaign._plan_bytes(manifest, 1).strip() == b""


def test_quantile_analysis_keeps_tuned_and_default_comparisons_distinct(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    _write_quantile_campaign(tmp_path, monkeypatch)
    campaign.analyze(tmp_path)
    report = json.loads((tmp_path / "artifacts/quantile_target_comparison.json").read_text())
    assert report["accuracy_evaluated"] is False
    assert report["journal_confirmation_ready"] is False
    rows = report["comparisons"]
    assert len(rows) == 12
    assert sum(row["endpoint_win_vs_tuned"] for row in rows) == 4
    assert sum(row["endpoint_win_vs_default"] for row in rows) == 12
    for row in rows:
        assert row["seed"] == 49 and row["steps"] == 150
        assert row["tuned_fixed_validation_loss"] == pytest.approx(0.4)
        assert row["default_fixed_validation_loss"] == pytest.approx(0.8)
        q = row["candidate"].rsplit("-", 1)[-1]
        expected_loss = {"q25": 0.35, "q50": 0.4, "q75": 0.45}[q]
        assert row["slaclip_validation_loss"] == pytest.approx(expected_loss)
        assert row["loss_improvement_vs_tuned"] == pytest.approx(0.4 - expected_loss)
        assert row["loss_improvement_vs_default"] == pytest.approx(0.8 - expected_loss)
        assert row["full_auc_not_worse_vs_tuned"] is (q != "q75")
        assert row["late_auc_not_worse_vs_tuned"] is (q != "q75")
        assert row["C0_and_tuned_fixed_C"] == EXPECTED[row["setting_id"]][0]
    artifacts = tmp_path / "artifacts"
    assert len((artifacts / "paper_coverage_summary.csv").read_text().splitlines()) == 21
    assert len((artifacts / "baseline_telemetry_steps.csv").read_text().splitlines()) == 3001
    assert len((artifacts / "public_validation_curve.csv").read_text().splitlines()) == 81


@pytest.mark.parametrize("field,value", [
    ("slaclip_c_max", 15.0),
    ("seed", 48),
    ("total_update_steps", 200),
    ("slaclip_target_non_small_clip_fraction", 0.79),
])
def test_quantile_analysis_rejects_unmatched_or_stale_arm_configuration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str, value: float,
):
    manifest = _write_quantile_campaign(tmp_path, monkeypatch)
    adaptive = next(arm for arm in manifest["arms"] if arm["method"] == "slaclip")
    path = tmp_path / adaptive["relative_root"] / "adapter/run_status.json"
    status = json.loads(path.read_text())
    status["config"][field] = value
    path.write_text(json.dumps(status))
    with pytest.raises(campaign.CampaignError, match=field):
        campaign.analyze(tmp_path)
