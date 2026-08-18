from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "build_paper_coverage_campaign",
    ROOT / "scripts" / "build_paper_coverage_campaign.py",
)
assert SPEC and SPEC.loader
campaign = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(campaign)

CODE_SHA = "1" * 40
REV_4B = "2" * 40
REV_9B = "3" * 40


def test_manifest_covers_paper_settings_and_full_slaclip() -> None:
    manifest = campaign.build_manifest(CODE_SHA, REV_4B, REV_9B)
    arms = manifest["arms"]
    assert len(arms) == 15
    assert {arm["setting_id"] for arm in arms} == {
        "glue8-4b-eps6-r16",
        "glue8-4b-eps3-r16",
        "math10k-9b-eps6-r16",
        "math10k-4b-eps6-r8",
        "math10k-4b-eps6-r32",
    }
    assert {arm["candidate_id"] for arm in arms} == {
        "fixed-c1",
        "full-sla-rho090",
        "full-sla-rho098",
    }
    assert sum(arm["lane"] == 0 for arm in arms) == 6
    assert sum(arm["lane"] == 1 for arm in arms) == 9
    for arm in arms:
        assert arm["seed"] == 42
        assert arm["initial_c"] == 1.0
        assert arm["model_revision"] == (REV_9B if arm["model_id"] == campaign.MODEL_9B else REV_4B)
        if arm["method"] == "slaclip":
            assert arm["rho"] in {0.9, 0.98}
            assert arm["eta"] == 0.05
        else:
            assert arm["rho"] is None
            assert arm["eta"] is None


def test_prepare_is_immutable_and_writes_exact_lane_counts(tmp_path: Path) -> None:
    campaign.prepare(tmp_path, CODE_SHA, REV_4B, REV_9B)
    campaign.prepare(tmp_path, CODE_SHA, REV_4B, REV_9B)
    assert len((tmp_path / "plans" / "lane-0.tsv").read_text().splitlines()) == 6
    assert len((tmp_path / "plans" / "lane-1.tsv").read_text().splitlines()) == 9
    assert (tmp_path / "plans" / "manifest.json.sha256").is_file()
    with pytest.raises(campaign.CampaignError, match="refusing to overwrite"):
        campaign.prepare(tmp_path, "4" * 40, REV_4B, REV_9B)


@pytest.mark.parametrize("bad", ["main", "A" * 40, "0" * 39, "g" * 40])
def test_manifest_rejects_unpinned_revisions(bad: str) -> None:
    with pytest.raises(campaign.CampaignError, match="full lowercase commit SHA"):
        campaign.build_manifest(CODE_SHA, REV_4B, bad)
