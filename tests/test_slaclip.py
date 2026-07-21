from __future__ import annotations

import math

import pytest
import torch

from prism_cli.optim.prism import PRISM, _factorized_delta_fro_norm_sq, _tangent_fro_norm_sq
from prism_cli.slaclip import automatic_num_slots, build_slack_vectors, update_slaclip_threshold


def test_extended_query_respects_original_clip_norm() -> None:
    clip = 2.0
    norms = torch.linspace(0.0, 4.0, 1001)
    slack, _ = build_slack_vectors(norms, clip, num_slots=13)
    clipped_norm_sq = torch.minimum(norms, torch.tensor(clip)).square()
    extended_norm_sq = clipped_norm_sq + slack.square().sum(dim=1)
    assert torch.all(extended_norm_sq <= clip**2 + 1e-5)


def test_tangent_norm_is_gauge_invariant() -> None:
    torch.manual_seed(5)
    a = torch.randn(5, 2)
    b = torch.randn(4, 2)
    da = torch.randn(5, 2)
    db = torch.randn(4, 2)
    gauge = torch.tensor([[1.7, 0.2], [-0.3, 0.9]])
    gauge_inv_t = torch.linalg.inv(gauge).T

    original = _tangent_fro_norm_sq(da, db, a, b)
    transformed = _tangent_fro_norm_sq(
        da @ gauge,
        db @ gauge_inv_t,
        a @ gauge,
        b @ gauge_inv_t,
    )
    assert transformed == pytest.approx(float(original), rel=1e-5, abs=1e-5)


def test_automatic_num_slots_uses_paper_bound() -> None:
    k = automatic_num_slots(expected_batch_size=64, noise_multiplier=1.0)
    expected = math.floor((64 / (2 * 2.5758293035489004)) ** (2 / 3))
    assert k == expected


def test_full_slaclip_controller_uses_near_zero_coordinate() -> None:
    indicator = torch.tensor([0.25, 0.50])
    c_next, gamma_t = update_slaclip_threshold(
        1.0,
        indicator,
        eta=0.5,
        beta=0.5,
        c_min=0.1,
        c_max=10.0,
    )
    assert gamma_t == pytest.approx(0.75, abs=1e-6)
    assert c_next == pytest.approx(math.exp(0.5 * (0.75 - 0.25)), rel=1e-6)


def _make_optimizer(
    telemetry_mode: str = 'dp_safe',
    *,
    num_slots: int = 3,
    clipping_method: str = 'baseline',
) -> tuple[PRISM, torch.nn.Parameter, torch.nn.Parameter]:
    torch.manual_seed(7)
    # Parameter order and shapes match PEFT lora_A [r,in], lora_B [out,r].
    p_a = torch.nn.Parameter(torch.randn(2, 3))
    p_b = torch.nn.Parameter(torch.randn(4, 2))
    opt = PRISM(
        [p_a, p_b],
        lr=1e-3,
        clipping_method=clipping_method,
        slaclip_num_slots=num_slots,
        telemetry_mode=telemetry_mode,
        use_adaptive=False,
    )
    return opt, p_a, p_b


def _set_grad_samples(p_a: torch.nn.Parameter, p_b: torch.nn.Parameter, g_a: torch.Tensor, g_b: torch.Tensor) -> None:
    p_a.grad_sample = g_a.clone()
    p_b.grad_sample = g_b.clone()


def test_tangent_accumulation_is_microbatch_invariant() -> None:
    torch.manual_seed(11)
    grad_a = torch.randn(6, 2, 3)
    grad_b = torch.randn(6, 4, 2)

    full, p_a_full, p_b_full = _make_optimizer()
    full.dp_begin(max_grad_norm=1.3, expected_batch_size=6, noise_multiplier=1.0)
    _set_grad_samples(p_a_full, p_b_full, grad_a, grad_b)
    assert full.dp_accumulate() == 6
    full_state = [
        (st['prism'].dp_accum_A.clone(), st['prism'].dp_accum_B.clone())
        for st in full.state.values()
    ]

    split, p_a_split, p_b_split = _make_optimizer()
    split.dp_begin(max_grad_norm=1.3, expected_batch_size=6, noise_multiplier=1.0)
    for start, end in ((0, 2), (2, 5), (5, 6)):
        _set_grad_samples(p_a_split, p_b_split, grad_a[start:end], grad_b[start:end])
        assert split.dp_accumulate() == end - start
    split_state = [
        (st['prism'].dp_accum_A.clone(), st['prism'].dp_accum_B.clone())
        for st in split.state.values()
    ]

    assert len(full_state) == len(split_state) == 1
    assert torch.allclose(full_state[0][0], split_state[0][0], atol=1e-5, rtol=1e-5)
    assert torch.allclose(full_state[0][1], split_state[0][1], atol=1e-5, rtol=1e-5)


def test_dp_safe_mode_does_not_emit_exact_gradient_statistics() -> None:
    torch.manual_seed(13)
    opt, p_a, p_b = _make_optimizer('dp_safe')
    opt.dp_begin(max_grad_norm=1.0, expected_batch_size=4, noise_multiplier=0.8)
    _set_grad_samples(p_a, p_b, torch.randn(4, 2, 3), torch.randn(4, 4, 2))
    opt.dp_accumulate()
    opt.dp_finalize(noise_multiplier=0.8)

    forbidden = {
        'raw_clip_fraction',
        'raw_clipped_signal_norm',
        'raw_realized_noise_norm',
        'dp_noise_norm',
        'dp_signal_norm',
        'dp_clip_frac',
        'dp_realized_batch_size',
    }
    assert forbidden.isdisjoint(opt.last_log)
    assert opt.last_raw_log == {}
    assert 'slack_indicator' not in opt.last_log
    assert 'slaclip_num_slots' not in opt.last_log
    assert 'dp_noisy_tangent_gradient_norm' in opt.last_log


def test_slaclip_emits_noisy_indicator_and_updates_threshold() -> None:
    torch.manual_seed(14)
    opt, p_a, p_b = _make_optimizer('dp_safe', clipping_method='slaclip')
    opt.dp_begin(max_grad_norm=1.0, expected_batch_size=4, noise_multiplier=0.8)
    _set_grad_samples(p_a, p_b, torch.randn(4, 2, 3), torch.randn(4, 4, 2))
    opt.dp_accumulate()
    opt.dp_finalize(noise_multiplier=0.8)

    assert 'slack_indicator' in opt.last_log
    assert opt.last_log['slaclip_num_slots'] == 3
    assert 'slaclip_gamma_t' in opt.last_log
    assert opt.last_log['dp_next_clip_threshold'] == pytest.approx(opt.current_clip)


def test_research_raw_mode_is_explicitly_marked() -> None:
    torch.manual_seed(17)
    opt, p_a, p_b = _make_optimizer('research_raw')
    opt.dp_begin(max_grad_norm=1.0, expected_batch_size=4, noise_multiplier=0.8)
    _set_grad_samples(p_a, p_b, torch.randn(4, 2, 3), torch.randn(4, 4, 2))
    opt.dp_accumulate()
    opt.dp_finalize(noise_multiplier=0.8)

    assert opt.last_raw_log['NON_PRIVATE_TELEMETRY'] is True
    assert 'raw_clip_fraction' in opt.last_raw_log
    assert 'raw_global_norm_hist_counts' in opt.last_raw_log
    assert 'raw_realized_noise_norm' in opt.last_raw_log
    assert 'raw_unclipped_signal_norm' in opt.last_raw_log
    assert opt.last_raw_log['raw_realized_batch_size'] == 4


def test_optimizer_checkpoint_restores_slaclip_runtime() -> None:
    opt, _, _ = _make_optimizer()
    opt.current_clip = 0.75
    opt.slaclip_num_slots = 9
    opt.raw_hist_max = 6.0
    saved = opt.state_dict()

    restored, _, _ = _make_optimizer(num_slots=9)
    restored.load_state_dict(saved)
    assert restored.current_clip == pytest.approx(0.75)
    assert restored.slaclip_num_slots == 9
    assert restored.raw_hist_max == pytest.approx(6.0)


def test_optimizer_checkpoint_rejects_different_slaclip_slots() -> None:
    source, _, _ = _make_optimizer(num_slots=9)
    saved = source.state_dict()
    target, _, _ = _make_optimizer(num_slots=3)
    with pytest.raises(ValueError, match='slaclip_num_slots'):
        target.load_state_dict(saved)


def test_factorized_delta_norm_matches_dense_matrix() -> None:
    torch.manual_seed(101)
    a_old = torch.randn(7, 2)
    b_old = torch.randn(5, 2)
    a_new = torch.randn(7, 2)
    b_new = torch.randn(5, 2)
    expected = torch.linalg.matrix_norm(a_new @ b_new.T - a_old @ b_old.T).square()
    actual = _factorized_delta_fro_norm_sq(a_new, b_new, a_old, b_old)
    assert actual == pytest.approx(float(expected), rel=1e-5, abs=1e-5)


def test_empty_poisson_batch_still_applies_gaussian_mechanism() -> None:
    torch.manual_seed(103)
    opt, p_a, p_b = _make_optimizer('dp_safe')
    before_a = p_a.detach().clone()
    before_b = p_b.detach().clone()
    opt.dp_begin(max_grad_norm=1.0, expected_batch_size=4, noise_multiplier=0.8)
    assert opt.dp_finalize(noise_multiplier=0.8) == 0
    assert not (torch.equal(before_a, p_a) and torch.equal(before_b, p_b))
    assert opt.last_raw_log == {}
    assert 'slack_indicator' not in opt.last_log


def test_empty_poisson_batch_still_updates_slaclip_controller() -> None:
    torch.manual_seed(107)
    opt, _, _ = _make_optimizer('dp_safe', clipping_method='slaclip')
    opt.dp_begin(max_grad_norm=1.0, expected_batch_size=4, noise_multiplier=0.8)
    assert opt.dp_finalize(noise_multiplier=0.8) == 0
    assert 'slack_indicator' in opt.last_log
    assert 'slaclip_gamma_t' in opt.last_log
    assert opt.last_log['dp_next_clip_threshold'] == pytest.approx(opt.current_clip)
