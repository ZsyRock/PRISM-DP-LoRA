from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _script(name: str):
    spec = importlib.util.spec_from_file_location(
        f"snr_regression_{name}", ROOT / "scripts" / f"{name}.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _records(clip_threshold: float) -> list[dict]:
    # Keep the last indicator's signal/noise fixed while changing C. Both z
    # and its noise scale by 1/(C+1e-6), so their SNR must remain exactly four.
    z = 0.08 / (clip_threshold + 1e-6)
    remaining = 1.0 - z
    return [
        {
            "step": step,
            "telemetry_schema_version": 7,
            "dp_clip_threshold": clip_threshold,
            "raw_clip_fraction": (0.15 + 0.60 * (step - 1) / 199) * remaining,
            "raw_reference_small_gradient_proxy": z,
            "raw_reference_remaining_mass_proxy": remaining,
            "raw_reference_conditional_clip_fraction": 0.15 + 0.60 * (step - 1) / 199,
            "raw_reference_conditional_clip_fraction_valid": True,
            "raw_reference_conditional_normalization": "expected_batch_size",
            "raw_reference_expected_normalized_clip_mass":
                (0.15 + 0.60 * (step - 1) / 199) * remaining,
            "raw_reference_expected_batch_size_normalization": 100.0,
            "raw_realized_batch_size": 100,
            "dp_expected_batch_size": 100.0,
            "dp_noise_multiplier": 0.5,
            "raw_reference_slaclip_num_slots": 16,
        }
        for step in range(1, 201)
    ]


@pytest.mark.parametrize("clip_threshold", [0.1, 1.0, 15.0])
def test_landscape_snr_is_invariant_to_small_proxy_coordinate_scaling(
    clip_threshold: float,
) -> None:
    landscape = _script("analyze_baseline_landscape")
    row = landscape._analyze_records(
        _records(clip_threshold),
        {"C": clip_threshold, "steps": 200},
        burn_in_fraction=0.0,
        minimum_burn_in_steps=0,
        rho_min=0.05,
        rho_max=0.95,
        clip_median_max=0.9,
        clip_iqr_min=0.05,
        clip_half_delta_min=0.05,
        small_proxy_noise_ratio_min=2.0,
        conditional_valid_fraction_min=0.99,
    )
    assert row["cdf_slack_noise_std_estimate_median"] == pytest.approx(0.02)
    assert row["small_gradient_proxy_noise_std_estimate_median"] == pytest.approx(
        0.02 / (clip_threshold + 1e-6)
    )
    assert row["small_proxy_to_noise_ratio"] == pytest.approx(4.0)
    assert row["small_proxy_noise_ratio_normalization"] == "z_over_sigma_z"
    assert row["eligible_small_proxy_noise_ratio"] is True


def test_high_c_selection_does_not_false_block_on_mixed_noise_units() -> None:
    campaign = _script("build_paper_coverage_campaign")
    results = []
    trajectories = []
    for setting_id in campaign.GLUE_TARGET_BASELINE_SETTING_IDS:
        for c_value in campaign.GLUE_TARGET_BASELINE_FIXED_GRID:
            candidate = f"fixed-c{c_value}"
            results.append({
                "setting_id": setting_id,
                "candidate": candidate,
                "method": "baseline",
                "candidate_role": "target_calibration_fixed_candidate",
                "initial_C": c_value,
                "selection_score": 1.0 if c_value == 15.0 else 0.0,
            })
            if c_value == 15.0:
                trajectories.extend(
                    {**row, "setting_id": setting_id, "candidate_id": candidate}
                    for row in _records(c_value)
                )
    selection = campaign._build_glue_target_baseline_selection(
        {
            "profile": "glue-target-baseline-screen",
            "glue_target_baseline_screen": {"privacy_scope": {}},
        },
        results,
        trajectories,
        {},
    )
    for setting in selection["settings"]:
        assert setting["small_gradient_proxy_median_z"] < 2.0 * 0.02
        assert setting["cdf_slack_noise_std_estimate_median"] == pytest.approx(0.02)
        assert setting["small_gradient_proxy_noise_std_estimate_median"] == pytest.approx(
            0.02 / (15.0 + 1e-6)
        )
        assert setting["small_proxy_to_noise_ratio"] == pytest.approx(4.0)
        assert setting["target_grid_identifiability_gates"][
            "small_proxy_to_noise_ratio_at_least_2"
        ] is True
        assert setting["target_grid_identifiable"] is True
        assert setting["adaptive_exploratory_screen_allowed"] is True
