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
    manifest = campaign.build_manifest(CODE_SHA, REV_4B, REV_9B, profile="paper-breadth")
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
    campaign.prepare(tmp_path, CODE_SHA, REV_4B, REV_9B, "paper-breadth")
    campaign.prepare(tmp_path, CODE_SHA, REV_4B, REV_9B, "paper-breadth")
    assert len((tmp_path / "plans" / "lane-0.tsv").read_text().splitlines()) == 6
    assert len((tmp_path / "plans" / "lane-1.tsv").read_text().splitlines()) == 9
    assert len((tmp_path / "plans" / "sequential.tsv").read_text().splitlines()) == 15
    assert all(
        line.startswith("0|")
        for line in (tmp_path / "plans" / "sequential.tsv").read_text().splitlines()
    )
    assert (tmp_path / "plans" / "manifest.json.sha256").is_file()
    with pytest.raises(campaign.CampaignError, match="refusing to overwrite"):
        campaign.prepare(tmp_path, "4" * 40, REV_4B, REV_9B, "paper-breadth")


@pytest.mark.parametrize("bad", ["main", "A" * 40, "0" * 39, "g" * 40])
def test_manifest_rejects_unpinned_revisions(bad: str) -> None:
    with pytest.raises(campaign.CampaignError, match="full lowercase commit SHA"):
        campaign.build_manifest(CODE_SHA, REV_4B, bad)


def test_regime_map_crosses_paper_axes_and_clipping_grids() -> None:
    manifest = campaign.build_manifest(CODE_SHA, REV_4B, REV_9B, profile="regime-map")
    arms = manifest["arms"]
    assert manifest["schema_version"] == 2
    assert manifest["profile"] == "regime-map"
    assert len(arms) == 99
    assert len({arm["setting_id"] for arm in arms}) == 9
    assert {arm["epsilon"] for arm in arms} == {3.0, 6.0}
    assert {arm["lora_r"] for arm in arms} == {8, 16, 32}
    assert {arm["dataset"] for arm in arms} == {"glue8", "math10k"}
    assert {arm["model_id"] for arm in arms} == {campaign.MODEL_4B, campaign.MODEL_9B}
    assert {arm["initial_c"] for arm in arms if arm["method"] == "baseline"} == {
        0.5, 1.0, 2.0, 3.0, 5.0,
    }
    assert {arm["rho"] for arm in arms if arm["method"] == "slaclip"} == {
        0.5, 0.7, 0.8, 0.9, 0.98,
    }
    assert all(arm["steps"] == 150 and arm["eval_limit"] == 512 for arm in arms)
    assert any(
        arm["candidate_id"] == "full-sla-c2-rho090"
        and arm["initial_c"] == 2.0
        and arm["rho"] == 0.9
        for arm in arms
    )
    assert sum(arm["lane"] == 0 for arm in arms) == 44
    assert sum(arm["lane"] == 1 for arm in arms) == 55


def test_baseline_reproduction_covers_all_paper_dataset_model_settings() -> None:
    manifest = campaign.build_manifest(
        CODE_SHA,
        REV_4B,
        REV_9B,
        profile="baseline-reproduction",
        model_12b_revision="4" * 40,
    )
    arms = manifest["arms"]
    assert len(arms) == 10
    assert manifest["baseline_reproduction"]["covered_settings"] == 10
    assert manifest["baseline_reproduction"]["excluded_setting"] is None
    assert {arm["model_id"] for arm in arms} == {
        campaign.MODEL_4B,
        campaign.MODEL_9B,
        campaign.MODEL_12B,
    }
    assert sum(arm["dataset"] == "glue8" for arm in arms) == 4
    assert sum(arm["dataset"] == "math10k" for arm in arms) == 6
    assert {arm["epsilon"] for arm in arms} == {3.0, 6.0}
    assert {arm["lora_r"] for arm in arms} == {8, 16, 32}
    for arm in arms:
        assert arm["method"] == "baseline"
        assert arm["initial_c"] == 1.0
        assert arm["rho"] is None and arm["eta"] is None
        assert arm["eval_limit"] == 0
        if arm["dataset"] == "glue8":
            assert arm["steps"] == 500
            assert arm["learning_rate"] == 0.0002
            assert arm["cutoff_len"] == 384
            assert arm["train_on_inputs"] is False
        else:
            assert arm["steps"] == 300
            assert arm["learning_rate"] == 0.0003
            assert arm["cutoff_len"] == 256
            assert arm["train_on_inputs"] is True


def test_cached_baseline_profile_is_explicitly_incomplete() -> None:
    manifest = campaign.build_manifest(
        CODE_SHA, REV_4B, REV_9B, profile="baseline-reproduction-cached"
    )
    assert len(manifest["arms"]) == 9
    assert manifest["baseline_reproduction"]["covered_settings"] == 9
    assert "12B" in manifest["baseline_reproduction"]["excluded_setting"]
    assert campaign.MODEL_12B not in {arm["model_id"] for arm in manifest["arms"]}
    sequential = campaign._plan_bytes(manifest, 0, include_all=True).decode().splitlines()
    settings = [line.split("|")[2] for line in sequential]
    assert settings[:3] == [
        "glue8-4b-eps6-r16",
        "math10k-4b-eps6-r16",
        "math10k-9b-eps6-r16",
    ]
    assert all(line.startswith("0|") for line in sequential)


def test_full_baseline_profile_requires_pinned_12b_revision() -> None:
    with pytest.raises(campaign.CampaignError, match="requires a pinned 12B"):
        campaign.build_manifest(
            CODE_SHA, REV_4B, REV_9B, profile="baseline-reproduction"
        )


def test_task_average_supports_glue_and_math_headers(tmp_path: Path) -> None:
    glue = tmp_path / "glue.csv"
    glue.write_text("method,GLUE8_Avg\nslaclip,0.75\n", encoding="utf-8")
    math = tmp_path / "math.csv"
    math.write_text("gsm8k,Average\n0.6,0.55\n", encoding="utf-8")
    assert campaign._task_average(glue) == pytest.approx(0.75)
    assert campaign._task_average(math) == pytest.approx(0.55)


@pytest.mark.parametrize(
    ("value", "expected"),
    [(0.69, "lt_70pct"), (0.7, "70_to_lt_90pct"), (0.9, "90_to_lt_98pct"), (0.98, "ge_98pct")],
)
def test_clip_regime_bins_are_explicit(value: float, expected: str) -> None:
    assert campaign._clip_bin(value) == expected
