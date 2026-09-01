from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "analyze_baseline_landscape",
    ROOT / "scripts" / "analyze_baseline_landscape.py",
)
assert SPEC and SPEC.loader
landscape = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(landscape)

CODE_SHA = "1" * 40
MODEL_REVISION = "2" * 40
FINGERPRINT = "3" * 64


def _json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _arm(arm_id: str, *, relative_root: str | None = None, steps: int = 20) -> dict[str, Any]:
    return {
        "arm_id": arm_id,
        "setting_id": "glue8-4b-eps6-r16",
        "relative_root": relative_root or f"runs/{arm_id}",
        "method": "baseline",
        "dataset": "glue8",
        "model_id": "example/model-4b",
        "model_revision": MODEL_REVISION,
        "epsilon": 6.0,
        "lora_r": 16,
        "initial_c": 1.0,
        "seed": 42,
        "steps": steps,
    }


def _manifest(campaign: Path, arms: list[dict[str, Any]], profile: str = "test-screen") -> None:
    _json(
        campaign / "plans" / "manifest.json",
        {
            "schema_version": 1,
            "profile": profile,
            "code_sha": CODE_SHA,
            "arms": arms,
        },
    )


def _write_completed(
    campaign: Path,
    arm: dict[str, Any],
    *,
    clips: list[float] | None = None,
    conditionals: list[float] | None = None,
    raw_run_id: str | None = None,
    omit_step: int | None = None,
    official_utility: float | None = None,
    realized_batch_size: int | None = None,
) -> None:
    steps = arm["steps"]
    run_id = f"run-{arm['arm_id']}"
    config = {
        "config_fingerprint": FINGERPRINT,
        "run_id": run_id,
        "method": "baseline",
        "dataset": arm["dataset"],
        "base_model": arm["model_id"],
        "resolved_model_revision": arm["model_revision"],
        "dp_epsilon": arm["epsilon"],
        "lora_r": arm["lora_r"],
        "dp_max_grad_norm": arm["initial_c"],
        "seed": arm["seed"],
        "total_update_steps": steps,
        "implementation_git_sha": CODE_SHA,
    }
    status = {
        "state": "completed",
        "run_id": run_id,
        "config_fingerprint": FINGERPRINT,
        "config": config,
        "method": "baseline",
        "privacy": "dp",
        "dataset": arm["dataset"],
        "base_model": arm["model_id"],
        "model_revision": arm["model_revision"],
        "resolved_model_revision": arm["model_revision"],
        "non_private_telemetry": True,
        "training_lora_r": arm["lora_r"],
        "update_steps": steps,
        "data_content_sha256": "4" * 64,
    }
    arm_root = campaign / arm["relative_root"]
    _json(arm_root / "adapter" / "run_status.json", status)
    _json(arm_root / "results" / "run_status.json", status)
    if clips is None:
        clips = [0.2] * (steps // 2) + [0.8] * (steps - steps // 2)
    if conditionals is None:
        conditionals = [0.10 + 0.04 * index for index in range(steps)]
    assert len(clips) == steps and len(conditionals) == steps
    records = []
    for index in range(steps):
        step = index + 1
        if step == omit_step:
            continue
        record = {
                "NON_PRIVATE_TELEMETRY": True,
                "step": step,
                "run_id": raw_run_id or run_id,
                "config_fingerprint": FINGERPRINT,
                "method": "baseline",
                "privacy": "dp",
                "dataset": arm["dataset"],
                "base_model": arm["model_id"],
                "model_revision": arm["model_revision"],
                "resolved_model_revision": arm["model_revision"],
                "dp_clip_threshold": arm["initial_c"],
                "dp_next_clip_threshold": arm["initial_c"],
                "raw_clip_fraction": clips[index],
                "raw_reference_small_gradient_proxy": 0.20,
                "raw_reference_remaining_mass_proxy": 0.80,
                "raw_reference_conditional_clip_fraction": conditionals[index],
                "raw_reference_conditional_clip_fraction_valid": True,
                "dp_noise_multiplier": 0.10,
                "dp_expected_batch_size": 100.0,
                "raw_reference_expected_batch_size_normalization": 100.0,
                "raw_reference_slaclip_num_slots": 4,
            }
        if realized_batch_size is not None:
            expected_clip_mass = clips[index] * realized_batch_size / 100.0
            record.update({
                "telemetry_schema_version": 7,
                "raw_realized_batch_size": realized_batch_size,
                "raw_reference_conditional_normalization": "expected_batch_size",
                "raw_reference_realized_to_expected_batch_ratio": (
                    realized_batch_size / 100.0
                ),
                "raw_reference_expected_normalized_clip_mass": expected_clip_mass,
            })
        records.append(record)
    raw = arm_root / "results" / "research_raw" / "NON_PRIVATE_train_log.jsonl"
    raw.parent.mkdir(parents=True, exist_ok=True)
    raw.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    raw_sha = hashlib.sha256(raw.read_bytes()).hexdigest()
    _json(
        raw.parent / "telemetry_summary.json",
        {
            "NON_PRIVATE_TELEMETRY": True,
            "run_identity": {
                "run_id": run_id,
                "config_fingerprint": FINGERPRINT,
                "method": "baseline",
                "privacy": "dp",
                "dataset": arm["dataset"],
                "base_model": arm["model_id"],
                "resolved_model_revision": arm["model_revision"],
            },
            "source": {
                "raw_sha256": raw_sha,
                "raw_physical_records": len(records),
                "raw_unique_steps": len(records),
                "raw_duplicate_records": 0,
            },
            "steps": {
                "count": len(records),
                "first": 1,
                "last": records[-1]["step"],
                "missing": [] if omit_step is None else [omit_step],
                "missing_count": 0 if omit_step is None else 1,
            },
        },
    )
    if official_utility is not None:
        summary = arm_root / "results" / "summary.csv"
        with summary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["GLUE8_Avg"])
            writer.writeheader()
            writer.writerow({"GLUE8_Avg": official_utility})


def _write_running(campaign: Path, arm: dict[str, Any]) -> None:
    _json(
        campaign / arm["relative_root"] / "adapter" / "run_status.json",
        {"state": "running"},
    )


def _main(campaigns: list[Path], output: Path, *extra: str) -> int:
    args: list[str] = []
    for campaign in campaigns:
        args.extend(("--campaign-root", str(campaign)))
    args.extend(("--output-dir", str(output), "--minimum-burn-in-steps", "0"))
    args.extend(extra)
    return landscape.main(args)


def test_strict_landscape_statistics_and_manifest_only_discovery(tmp_path: Path) -> None:
    campaign = tmp_path / "campaign"
    arm = _arm("eligible")
    _manifest(campaign, [arm])
    _write_completed(campaign, arm, official_utility=0.77)
    # A convincing-looking but unregistered directory must never be scanned.
    rogue = _arm("rogue")
    _write_completed(campaign, rogue, official_utility=0.99)

    output = tmp_path / "out"
    assert _main([campaign], output) == 0
    report = json.loads((output / "baseline_landscape.json").read_text())
    assert report["counts"] == {
        "registered_baseline_arms": 1,
        "included_completed_fixed_arms": 1,
        "recommended_settings": 1,
        "incomplete_arms": 0,
        "invalid_arms": 0,
    }
    row = report["landscape"][0]
    assert row["arm_id"] == "eligible"
    assert row["burn_in_steps"] == 2
    assert row["analysis_steps"] == 18
    assert row["official_utility_metric"] == "GLUE8_Avg"
    assert row["official_utility"] == 0.77
    assert row["summary_sha256"]
    assert row["clip_median"] < 0.90
    assert row["clip_IQR"] >= 0.05
    assert row["small_proxy_to_noise_ratio"] == 100.0
    assert row["projected_rho_unique_count"] == 5
    assert row["screen_eligible"] is True
    assert row["conditional_proxy_valid_fraction"] == 1.0
    assert "NON_PRIVATE calibration" in report["warning"]
    assert (output / "baseline_landscape.csv").is_file()
    assert (output / "baseline_coverage.csv").is_file()
    assert (output / "recommended_settings.csv").is_file()


def test_conditional_proxy_above_one_is_preserved_then_projected(tmp_path: Path) -> None:
    campaign = tmp_path / "campaign"
    arm = _arm("conditional-above-one")
    _manifest(campaign, [arm])
    conditionals = [0.80 + 0.04 * index for index in range(arm["steps"])]
    _write_completed(campaign, arm, conditionals=conditionals)

    output = tmp_path / "out"
    assert _main([campaign], output) == 0
    row = json.loads((output / "baseline_landscape.json").read_text())[
        "landscape"
    ][0]
    assert row["conditional_proxy_raw_q90"] > 1.0
    assert row["projected_rho_q90"] == 0.95
    assert row["projected_rho_unique_count"] < 5
    assert row["eligible_five_unique_projected_rhos"] is False
    assert row["screen_eligible"] is False
    assert "five_unique_projected_rhos_unavailable" in row["eligibility_reasons"]
    recommended = list(csv.DictReader((output / "recommended_settings.csv").open()))
    assert recommended == []


def test_landscape_recomputes_conditional_rho_in_expected_batch_coordinates(
    tmp_path: Path,
) -> None:
    campaign = tmp_path / "campaign"
    arm = _arm("expected-batch-normalization")
    _manifest(campaign, [arm])
    clips = [0.20 + 0.02 * index for index in range(arm["steps"])]
    conditionals = [clip * 50.0 / 100.0 / 0.80 for clip in clips]
    _write_completed(
        campaign,
        arm,
        clips=clips,
        conditionals=conditionals,
        realized_batch_size=50,
    )

    output = tmp_path / "out"
    assert _main([campaign], output) == 0
    row = json.loads((output / "baseline_landscape.json").read_text())[
        "landscape"
    ][0]
    assert row["conditional_proxy_normalization"] == (
        "recomputed_expected_batch_size"
    )
    post = conditionals[row["burn_in_steps"]:]
    assert row["conditional_proxy_raw_median"] == pytest.approx(
        landscape._quantile(post, 0.50)
    )


def test_incomplete_is_reported_and_allow_incomplete_changes_exit_only(
    tmp_path: Path,
) -> None:
    campaign = tmp_path / "campaign"
    complete = _arm("complete")
    running = _arm("running")
    _manifest(campaign, [complete, running])
    _write_completed(campaign, complete)
    _write_running(campaign, running)

    output = tmp_path / "out"
    assert _main([campaign], output) == 2
    report = json.loads((output / "baseline_landscape.json").read_text())
    assert report["counts"]["incomplete_arms"] == 1
    assert len(report["landscape"]) == 1
    rejected = next(row for row in report["coverage"] if row["arm_id"] == "running")
    assert rejected["classification"] == "incomplete"
    assert rejected["included_in_landscape"] is False
    assert "not 'completed'" in rejected["reason"]

    assert _main([campaign], output, "--allow-incomplete") == 0
    report_again = json.loads((output / "baseline_landscape.json").read_text())
    assert report_again == report


def test_completed_identity_mismatch_is_invalid_even_when_incomplete_allowed(
    tmp_path: Path,
) -> None:
    campaign = tmp_path / "campaign"
    arm = _arm("wrong-identity")
    _manifest(campaign, [arm])
    _write_completed(campaign, arm, raw_run_id="another-run")

    output = tmp_path / "out"
    assert _main([campaign], output, "--allow-incomplete") == 2
    report = json.loads((output / "baseline_landscape.json").read_text())
    assert report["counts"]["invalid_arms"] == 1
    assert report["landscape"] == []
    assert report["coverage"][0]["classification"] == "invalid"
    assert "identity mismatch for raw step 1.run_id" in report["coverage"][0]["reason"]


def test_completed_raw_steps_must_be_exactly_one_through_planned_steps(
    tmp_path: Path,
) -> None:
    campaign = tmp_path / "campaign"
    arm = _arm("missing-step")
    _manifest(campaign, [arm])
    _write_completed(campaign, arm, omit_step=7)

    output = tmp_path / "out"
    assert _main([campaign], output, "--allow-incomplete") == 2
    report = json.loads((output / "baseline_landscape.json").read_text())
    assert report["counts"]["invalid_arms"] == 1
    assert report["landscape"] == []
    assert "exactly 1..planned_steps" in report["coverage"][0]["reason"]


def test_baseline_reproduction_requires_one_row_official_summary(tmp_path: Path) -> None:
    campaign = tmp_path / "campaign"
    arm = _arm("paper-baseline")
    _manifest(campaign, [arm], profile="baseline-reproduction-cached")
    _write_completed(campaign, arm)

    output = tmp_path / "out"
    assert _main([campaign], output, "--allow-incomplete") == 2
    report = json.loads((output / "baseline_landscape.json").read_text())
    assert report["counts"]["invalid_arms"] == 1
    assert "missing official utility summary" in report["coverage"][0]["reason"]


def test_gap_recovery_profiles_require_one_row_official_summary(
    tmp_path: Path,
) -> None:
    profiles = (
        "baseline-gap-fill-all-cached",
        "baseline-gap-fill-math-only-cached",
    )
    for profile in profiles:
        campaign = tmp_path / profile
        arm = _arm(f"{profile}-paper-baseline")
        _manifest(campaign, [arm], profile=profile)
        _write_completed(campaign, arm)

        output = tmp_path / f"out-{profile}"
        assert _main([campaign], output, "--allow-incomplete") == 2
        report = json.loads((output / "baseline_landscape.json").read_text())
        assert report["counts"]["invalid_arms"] == 1
        assert "missing official utility summary" in report["coverage"][0]["reason"]
