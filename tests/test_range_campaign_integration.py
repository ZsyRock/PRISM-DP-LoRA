"""Exercise interval-target planning, validation selection and fail-closed analysis."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "range_campaign_fixtures", ROOT / "tests/test_build_paper_coverage_campaign.py",
)
assert SPEC and SPEC.loader
fixtures = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(fixtures)
campaign = fixtures.campaign


def _write_campaign(tmp_path, monkeypatch):
    profile = campaign._range_target_module()
    manifest = campaign.build_manifest(
        fixtures.CODE_SHA, campaign._weighted_target_module().MODEL_REVISION,
        fixtures.REV_9B, profile=profile.PROFILE,
    )
    monkeypatch.setattr(campaign, "_range_target_module", lambda: profile)
    monkeypatch.setattr(profile, "verify_sources", lambda *a: {"synthetic_source": True})
    campaign._with_sha(tmp_path / "plans/manifest.json", campaign._json_bytes(manifest))
    losses = {}
    for arm in manifest["arms"]:
        if arm["method"] == "baseline":
            loss = 0.8 if arm["initial_c"] == 1 else 0.4
        else:
            loss = {0.25: 0.45, 0.5: 0.35, 0.75: 0.9}[arm["range_position"]]
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


def test_interval_profile_preserves_order_and_one_lane():
    manifest = campaign.build_manifest(
        fixtures.CODE_SHA, campaign._weighted_target_module().MODEL_REVISION,
        fixtures.REV_9B, profile="glue-range-target-screen",
    )
    rows = campaign._plan_bytes(manifest, 0, include_all=True).decode().splitlines()
    assert len(rows) == 20
    assert [r.split("|")[2] for r in rows[:5]] == ["glue8-4b-eps6-r32"] * 5
    assert campaign._plan_bytes(manifest, 1).strip() == b""
    assert {a["initial_c"] for a in manifest["arms"] if a["method"] == "slaclip"} == {1.0}
    assert all(a["seed"] == 50 and a["steps"] == 150 for a in manifest["arms"])


def test_interval_analysis_selects_best_validation_candidate_and_keeps_losers(tmp_path, monkeypatch):
    _write_campaign(tmp_path, monkeypatch)
    campaign.analyze(tmp_path)
    report = json.loads((tmp_path / "artifacts/range_target_comparison.json").read_text())
    lock = json.loads((tmp_path / "selection/best-range-targets.lock.json").read_text())
    assert report["accuracy_evaluated"] is False
    assert report["journal_confirmation_ready"] is False
    assert report["test_metrics_used_for_selection"] is False
    assert report["target_formula"] == "rho_q=r_min+q*(r_max-r_min)"
    assert len(report["comparisons"]) == 12
    assert sum(r["endpoint_win_vs_default"] for r in report["comparisons"]) == 8
    assert sum(r["endpoint_win_vs_tuned"] for r in report["comparisons"]) == 4
    assert report["selected_default_wins"] == report["selected_tuned_wins"] == 4
    assert len(lock["settings"]) == 4
    assert len(lock["arm_artifact_sha256"]) == 20
    for selected in lock["settings"]:
        assert selected["selected_arm"]["range_position"] == 0.5
        assert selected["selected_arm"]["initial_c"] == 1.0
        assert selected["selected_validation_loss"] == pytest.approx(0.35)
        assert [r["range_position"] for r in selected["ranking"]] == [0.5, 0.25, 0.75]
    artifacts = tmp_path / "artifacts"
    assert len((artifacts / "baseline_telemetry_steps.csv").read_text().splitlines()) == 3001
    assert len((artifacts / "public_validation_curve.csv").read_text().splitlines()) == 81
    # Identical replay is safe and does not silently overwrite a selection.
    campaign.analyze(tmp_path)


@pytest.mark.parametrize("field,value", [
    ("seed", 49), ("total_update_steps", 200),
    ("slaclip_c_max", 15.0), ("slaclip_target_non_small_clip_fraction", 0.79),
    ("dp_max_grad_norm", 15.0),
])
def test_interval_analysis_rejects_mismatching_run_config(tmp_path, monkeypatch, field, value):
    manifest = _write_campaign(tmp_path, monkeypatch)
    arm = next(a for a in manifest["arms"] if a["method"] == "slaclip")
    path = tmp_path / arm["relative_root"] / "adapter/run_status.json"
    status = json.loads(path.read_text())
    status["config"][field] = value
    path.write_text(json.dumps(status))
    with pytest.raises(campaign.CampaignError, match=field):
        campaign.analyze(tmp_path)


def test_interval_analysis_rejects_changed_plan(tmp_path, monkeypatch):
    manifest = _write_campaign(tmp_path, monkeypatch)
    manifest["arms"][2]["rho"] = 0.79
    path = tmp_path / "plans/manifest.json"
    path.write_text(json.dumps(manifest))
    with pytest.raises(campaign.CampaignError, match="range plan"):
        campaign.analyze(tmp_path)


def test_interval_selection_ties_break_by_rho_not_report_order(tmp_path, monkeypatch):
    manifest = _write_campaign(tmp_path, monkeypatch)
    campaign.analyze(tmp_path)
    with (tmp_path / "artifacts/paper_coverage_summary.csv").open() as source:
        import csv
        rows = list(csv.DictReader(source))
    for row in rows:
        row["initial_C"] = float(row["initial_C"])
        row["rho"] = float(row["rho"]) if row["rho"] else None
        for key in ("public_validation_loss", "full_normalized_validation_loss_auc",
                    "late_window_normalized_validation_loss_auc"):
            row[key] = 0.4
    _, lock = campaign._build_range_screen_reports(manifest, rows[::-1], {}, "a" * 64, "b" * 64)
    for selected in lock["settings"]:
        assert selected["selected_arm"]["range_position"] == 0.25
        assert selected["selected_beats_default"] is False
        assert selected["selected_beats_tuned"] is False


def test_interval_profile_accepted_by_cli():
    args = campaign.parser().parse_args([
        "prepare", "--campaign-root", "/unused", "--code-sha", fixtures.CODE_SHA,
        "--model-4b-revision", fixtures.REV_4B, "--model-9b-revision", fixtures.REV_9B,
        "--profile", "glue-range-target-screen",
    ])
    assert args.profile == "glue-range-target-screen"
