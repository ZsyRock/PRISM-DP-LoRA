from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest
import torch

from prism_cli.optim.prism import PRISM


def _run(mode: str, slots: int = 4):
    p_a = torch.nn.Parameter(torch.eye(2, 3))
    p_b = torch.nn.Parameter(torch.eye(4, 2))
    optimizer = PRISM(
        [p_a, p_b], lr=1e-3, use_adaptive=False,
        clipping_method="baseline", telemetry_mode=mode,
        slaclip_num_slots=slots,
    )
    optimizer.dp_begin(max_grad_norm=1.0, expected_batch_size=8, noise_multiplier=0.8)
    # Orthogonal factor chart: this coordinate is a unit tangent direction.
    # One sample is just below C so the first bin has measurable smoothing bias.
    p_a.grad_sample = torch.zeros(4, 2, 3)
    p_b.grad_sample = torch.zeros(4, 4, 2)
    p_b.grad_sample[:, 2, 0] = torch.tensor([0.0, 0.25, 0.95, 1.5])
    optimizer.dp_accumulate()
    torch.manual_seed(5081)
    optimizer.dp_finalize(noise_multiplier=0.8)
    return optimizer, p_a.detach().clone(), p_b.detach().clone(), torch.random.get_rng_state()


def test_reference_cdf_uses_expected_batch_and_records_bin_bias() -> None:
    optimizer, *_ = _run("research_raw")
    raw = optimizer.last_raw_log
    indicator = raw["raw_reference_slack_indicator"]
    assert len(indicator) == 4
    assert all(a >= b for a, b in zip(indicator, indicator[1:]))
    assert raw["raw_reference_slack_indicator_first"] == indicator[0]
    assert raw["raw_reference_slack_indicator_last"] == pytest.approx(indicator[-1])
    # Normalization is expected B=8 even though the actual batch contains four.
    assert raw["raw_reference_expected_normalized_unclipped_mass"] == pytest.approx(3 / 8)
    assert indicator[0] == pytest.approx((1 + 1 + 0.20) / 8, abs=1e-6)
    assert raw["raw_reference_unclipped_cdf_bias"] == pytest.approx(-0.10, abs=1e-6)
    assert raw["raw_reference_slack_indicator_noise_std"] == pytest.approx(0.2)
    assert raw["raw_reference_small_gradient_proxy_noise_std"] == pytest.approx(
        0.2 / (1.0 + 1e-6)
    )
    assert not any(key.startswith("raw_reference_") for key in optimizer.last_log)


def test_enabling_reference_cdf_changes_neither_model_noise_nor_rng() -> None:
    safe, safe_a, safe_b, safe_rng = _run("dp_safe")
    raw, raw_a, raw_b, raw_rng = _run("research_raw")
    no_reference, no_ref_a, no_ref_b, no_ref_rng = _run("research_raw", slots=0)
    for a, b, rng in ((raw_a, raw_b, raw_rng), (no_ref_a, no_ref_b, no_ref_rng)):
        assert torch.equal(a, safe_a)
        assert torch.equal(b, safe_b)
        assert torch.equal(rng, safe_rng)
    # The declared observation mode is metadata, not a mechanism output.
    mechanism_logs = [
        {key: value for key, value in opt.last_log.items() if key != "telemetry_mode"}
        for opt in (safe, raw, no_reference)
    ]
    assert mechanism_logs[0] == mechanism_logs[1] == mechanism_logs[2]
    assert safe.last_raw_log == {}
    assert "raw_reference_slack_indicator" not in no_reference.last_raw_log
    assert raw.last_raw_log["raw_realized_noise_norm"] == no_reference.last_raw_log[
        "raw_realized_noise_norm"
    ]


def test_summarizer_preserves_reference_vector_and_optional_old_logs() -> None:
    path = Path(__file__).resolve().parents[1] / "scripts" / "summarize_telemetry.py"
    spec = importlib.util.spec_from_file_location("reference_cdf_summarizer", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    optimizer, *_ = _run("research_raw")
    flattened = module.flatten_record(optimizer.last_raw_log)
    assert json.loads(flattened["raw_reference_slack_indicator_json"]) == optimizer.last_raw_log[
        "raw_reference_slack_indicator"
    ]
    assert flattened["raw_reference_unclipped_cdf_bias"] == pytest.approx(-0.10, abs=1e-6)
    assert "raw_reference_slack_indicator_json" in module.PREFERRED_COLUMNS
    assert "raw_reference_slack_indicator_json" not in module.flatten_record({"step": 1})
