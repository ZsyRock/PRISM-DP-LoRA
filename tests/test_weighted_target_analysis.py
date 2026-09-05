"""Exercise weighted-screen postprocessing with the real artifact validators."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "focused_campaign_fixtures",
    ROOT / "tests" / "test_build_paper_coverage_campaign.py",
)
assert SPEC and SPEC.loader
fixtures = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(fixtures)
campaign = fixtures.campaign


def _write_weighted_arms(root: Path, arms: list[dict], losses: dict[str, float],
                         *, index_offset: int = 0) -> None:
    fixtures._write_focused_campaign(
        root, arms_override=arms, validation_losses=losses,
        index_offset=index_offset,
    )
    # Extend the shared historical fixture's configuration to the new profile.
    # Leave all status/raw/summary/curve identity checks on the real code path.
    for arm in arms:
        path = root / arm["relative_root"] / "adapter/run_status.json"
        status = json.loads(path.read_text())
        status["config"].update(raw_hist_bins=512, raw_hist_max=200.0)
        if arm["method"] == "slaclip":
            status["config"]["slaclip_c_max"] = arm["c_max"]
        path.write_text(json.dumps(status))


def test_weighted_analyzer_validates_all_arms_and_compares_only_fresh_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    weighted = campaign._weighted_target_module()
    monkeypatch.setattr(campaign, "_weighted_target_module", lambda: weighted)
    manifest = campaign.build_manifest(
        fixtures.CODE_SHA, weighted.MODEL_REVISION, fixtures.REV_9B,
        profile=weighted.PROFILE,
    )
    campaign._with_sha(tmp_path / "plans/manifest.json", campaign._json_bytes(manifest))
    sources = {"default": {}, "tuned": {}}
    for sid in weighted.SETTING_ORDER:
        sources["default"][sid] = {
            "target": {"rho": 0.6}, "provenance": {"synthetic_source": "default"},
        }
        if sid in weighted.PRIOR_WINNERS:
            setting = next(s for s in campaign.REGIME_SETTINGS if s["id"] == sid)
            arm = weighted._arm(
                {**setting, "model_revision": weighted.MODEL_REVISION},
                "synthetic-prior", weighted.PRIOR_WINNERS[sid], 47, 200,
                "weighted_stage1_fixed",
            )
            sources["tuned"][sid] = {
                "arm": arm, "target": {"rho": 0.2},
                "provenance": {"synthetic_source": "tuned"}, "endpoint_loss": 0.1,
            }
    monkeypatch.setattr(weighted, "verify_sources", lambda *args: sources)
    # Stage1 has a much lower endpoint than every Stage2 arm. An accidental
    # comparison against the historical 200-step baseline would reverse wins.
    stage1_losses = {
        arm["candidate_id"]: (0.05 if arm["initial_c"] in {30, 80} else 0.1)
        for arm in manifest["arms"]
    }
    _write_weighted_arms(tmp_path, manifest["arms"], stage1_losses)
    lock = weighted.lock(campaign, tmp_path)
    assert len(lock["stage2_arms"]) == 12
    assert any(arm["initial_c"] == 80 for arm in lock["stage2_arms"])
    stage2_losses = {
        "weighted-fresh-fixed": 0.40,
        "weighted-full-default-rho": 0.35,
        "weighted-full-tuned-rho": 0.45,
    }
    _write_weighted_arms(tmp_path, lock["stage2_arms"], stage2_losses, index_offset=100)
    campaign.analyze(tmp_path)
    report = json.loads((tmp_path / "artifacts/weighted_target_comparison.json").read_text())
    assert len(report["comparisons"]) == 8
    assert report["journal_confirmation_ready"] is False
    assert report["accuracy_evaluated"] is False
    for row in report["comparisons"]:
        assert row["seed"] == 48 and row["steps"] == 150
        assert row["fixed_validation_loss"] == pytest.approx(0.40)
        expected = 0.05 if row["candidate"] == "weighted-full-default-rho" else -0.05
        assert row["loss_improvement"] == pytest.approx(expected)
        assert row["endpoint_win"] is (expected > 0)
    assert len((tmp_path / "artifacts/paper_coverage_summary.csv").read_text().splitlines()) == 19
    assert len((tmp_path / "artifacts/baseline_telemetry_steps.csv").read_text().splitlines()) == 3001
    # Exercise the real C_max identity validator, not a mocked arm loader.
    adaptive = next(arm for arm in lock["stage2_arms"] if arm["method"] == "slaclip")
    path = tmp_path / adaptive["relative_root"] / "adapter/run_status.json"
    status = json.loads(path.read_text())
    status["config"]["slaclip_c_max"] = 15.0
    path.write_text(json.dumps(status))
    with pytest.raises(campaign.CampaignError, match="slaclip_c_max"):
        campaign.analyze(tmp_path)
