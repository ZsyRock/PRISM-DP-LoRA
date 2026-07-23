from __future__ import annotations

import math

import pytest
import torch

from prism_cli.optim.prism import (
    PRISM,
    _dp_isometric_chart_lift,
    _factorized_delta_fro_norm_sq,
    _factorized_isotropic_tangent_noise,
    _full_column_rank_qr,
    _right_solve_gram,
    _right_solve_transpose,
    _tangent_fro_norm_sq,
)
from prism_cli.slaclip import (
    automatic_num_slots,
    build_slack_vectors,
    slaclip_q_threshold_update,
    slack_indicator_noise_std,
    update_slaclip_threshold,
)


def test_extended_query_respects_original_clip_norm() -> None:
    clip = 2.0
    norms = torch.linspace(0.0, 4.0, 1001)
    slack, _ = build_slack_vectors(norms, clip, num_slots=13)
    clipped_norm_sq = torch.minimum(norms, torch.tensor(clip)).square()
    extended_norm_sq = clipped_norm_sq + slack.square().sum(dim=1)
    assert torch.all(extended_norm_sq <= clip**2 + 1e-5)


def test_actual_prism_tangent_plus_slack_respects_joint_bound() -> None:
    torch.manual_seed(3)
    clip = 1.7
    a = torch.randn(7, 3)
    b = torch.randn(5, 3)
    da = torch.randn(11, 7, 3)
    db = torch.randn(11, 5, 3)
    tangent_norms = torch.sqrt(torch.clamp(_tangent_fro_norm_sq(da, db, a, b), min=0.0))
    slack, _ = build_slack_vectors(tangent_norms, clip, num_slots=8)
    joint_sq = torch.minimum(tangent_norms, torch.tensor(clip)).square() + slack.square().sum(dim=1)
    assert torch.all(joint_sq <= clip**2 + 1e-5)


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


def test_qr_sampler_is_exactly_isotropic_in_intrinsic_tangent_coordinates() -> None:
    torch.manual_seed(37)
    # Deliberately non-orthonormal, moderately ill-conditioned factors.  The
    # QR sampler must cancel their coordinate scaling without eps damping.
    a = torch.randn(7, 3) @ torch.diag(torch.tensor([3.0, 0.3, 0.03]))
    b = torch.randn(5, 3) @ torch.tensor(
        [[2.0, 0.4, 0.0], [0.0, 0.5, -0.1], [0.0, 0.0, 0.04]]
    )
    qa, ra, _ = _full_column_rank_qr(
        a,
        gram_rcond=1e-8,
        factor_name='test A',
    )
    qb, rb, _ = _full_column_rank_qr(
        b,
        gram_rcond=1e-8,
        factor_name='test B',
    )
    u = torch.randn_like(a)
    v = torch.randn_like(b)
    noise_a, noise_b = _factorized_isotropic_tangent_noise(u, v, qa, ra, rb)

    intrinsic = noise_a @ b.T + a @ noise_b.T
    u_perp = u - qa @ (qa.T @ u)
    expected = u_perp @ qb.T + qa @ v.T
    assert torch.allclose(intrinsic, expected, atol=2e-5, rtol=2e-5)

    # The two summands are orthogonal isometric Gaussian coordinate blocks,
    # so this identity is stronger than merely checking the expected energy.
    expected_norm_sq = u_perp.square().sum() + v.square().sum()
    actual_norm_sq = _tangent_fro_norm_sq(noise_a, noise_b, a, b)
    assert actual_norm_sq == pytest.approx(
        float(expected_norm_sq),
        rel=3e-5,
        abs=3e-5,
    )


@pytest.mark.parametrize('batched', [False, True])
def test_right_solve_gram_matches_exact_linear_solve(batched: bool) -> None:
    torch.manual_seed(39)
    _, r = torch.linalg.qr(torch.randn(6, 3), mode='reduced')
    x = torch.randn(4, 5, 3) if batched else torch.randn(5, 3)
    actual = _right_solve_gram(x, r)
    gram = r.T @ r
    expected = torch.linalg.solve(gram, x.transpose(-2, -1)).transpose(-2, -1)
    assert torch.allclose(actual, expected, atol=2e-5, rtol=2e-5)


def test_dp_chart_and_paper_half_lift_have_same_intrinsic_tangent() -> None:
    torch.manual_seed(43)
    a = torch.randn(7, 3)
    b = torch.randn(5, 3)
    qa, ra, _ = _full_column_rank_qr(a, gram_rcond=1e-8, factor_name='test A')
    qb, rb, _ = _full_column_rank_qr(b, gram_rcond=1e-8, factor_name='test B')
    intrinsic_grad = torch.randn(4, 7, 5)
    g_a = intrinsic_grad @ b
    g_b = intrinsic_grad.transpose(-2, -1) @ a

    chart_a, chart_b = _dp_isometric_chart_lift(g_a, g_b, qa, ra, rb)
    y_a = _right_solve_gram(g_a, rb)
    y_b = _right_solve_gram(g_b, ra)
    half_a = y_a - 0.5 * (qa @ (qa.T @ y_a))
    half_b = y_b - 0.5 * (qb @ (qb.T @ y_b))

    chart_intrinsic = chart_a @ b.T + a @ chart_b.transpose(-2, -1)
    half_intrinsic = half_a @ b.T + a @ half_b.transpose(-2, -1)
    assert torch.allclose(chart_intrinsic, half_intrinsic, atol=3e-5, rtol=3e-5)
    assert torch.allclose(
        qa.T @ chart_a,
        torch.zeros_like(qa.T @ chart_a),
        atol=2e-5,
        rtol=0.0,
    )
    assert torch.allclose(
        _tangent_fro_norm_sq(chart_a, chart_b, a, b),
        _tangent_fro_norm_sq(half_a, half_b, a, b),
        atol=5e-4,
        rtol=5e-5,
    )


def test_query_and_noise_share_one_reconstructible_dp_chart() -> None:
    torch.manual_seed(47)
    clip = 1.3
    a = torch.randn(7, 3)
    b = torch.randn(5, 3)
    qa, ra, _ = _full_column_rank_qr(a, gram_rcond=1e-8, factor_name='test A')
    qb, rb, _ = _full_column_rank_qr(b, gram_rcond=1e-8, factor_name='test B')
    query_a, query_b = _dp_isometric_chart_lift(
        torch.randn(7, 3),
        torch.randn(5, 3),
        qa,
        ra,
        rb,
    )
    query_norm = torch.sqrt(
        torch.clamp(_tangent_fro_norm_sq(query_a, query_b, a, b), min=0.0)
    )
    coef = min(1.0, clip / float(query_norm))
    clipped_a = query_a * coef
    clipped_b = query_b * coef
    noise_a, noise_b = _factorized_isotropic_tangent_noise(
        torch.randn_like(a),
        torch.randn_like(b),
        qa,
        ra,
        rb,
    )
    released_a = clipped_a + noise_a
    released_b = clipped_b + noise_b
    released_intrinsic = released_a @ b.T + a @ released_b.T

    # The factor pair consumed by the adaptive optimizer is a deterministic,
    # unique chart lift of the intrinsic Gaussian release.
    reconstructed_a = _right_solve_transpose(released_intrinsic @ qb, rb)
    reconstructed_a = reconstructed_a - qa @ (qa.T @ reconstructed_a)
    reconstructed_b = _right_solve_transpose(released_intrinsic.T @ qa, ra)
    assert torch.allclose(reconstructed_a, released_a, atol=3e-5, rtol=3e-5)
    assert torch.allclose(reconstructed_b, released_b, atol=3e-5, rtol=3e-5)

    slack, _ = build_slack_vectors(query_norm.reshape(1), clip, num_slots=8)
    clipped_norm_sq = _tangent_fro_norm_sq(clipped_a, clipped_b, a, b)
    assert clipped_norm_sq + slack.square().sum() <= clip**2 + 1e-5


def test_dp_begin_rejects_rank_deficient_factor_before_release() -> None:
    opt, _, p_b = _make_optimizer()
    with torch.no_grad():
        p_b[:, 1].copy_(p_b[:, 0])
    with pytest.raises(RuntimeError, match='full-column-rank'):
        opt.dp_begin(
            max_grad_norm=1.0,
            expected_batch_size=4,
            noise_multiplier=0.8,
        )


def test_exact_dp_tangent_noise_is_independent_of_optimizer_eps() -> None:
    first, first_a, first_b = _make_optimizer(eps=1e-8)
    second, second_a, second_b = _make_optimizer(eps=1.0)
    for opt in (first, second):
        opt.dp_begin(
            max_grad_norm=1.0,
            expected_batch_size=4,
            noise_multiplier=0.8,
        )
    torch.manual_seed(41)
    first.dp_finalize(noise_multiplier=0.8)
    torch.manual_seed(41)
    second.dp_finalize(noise_multiplier=0.8)
    assert torch.equal(first_a, second_a)
    assert torch.equal(first_b, second_b)


def test_automatic_num_slots_uses_paper_bound() -> None:
    k = automatic_num_slots(expected_batch_size=64, noise_multiplier=1.0)
    expected = math.floor((64 / (2 * 2.5758293035489004)) ** (2 / 3))
    assert k == expected


def test_math10k_actual_privacy_parameters_select_eight_slots() -> None:
    assert automatic_num_slots(expected_batch_size=63.9935, noise_multiplier=0.516357) == 8
    assert slack_indicator_noise_std(0.516357, 8, 63.9935) == pytest.approx(0.022822, rel=1e-4)


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


@pytest.mark.parametrize(
    ('unclipped_proxy', 'direction'),
    [(0.0, 1), (0.01, 0), (0.20, -1)],
)
def test_slaclip_q_99_has_correct_direction_and_fixed_point(
    unclipped_proxy: float,
    direction: int,
) -> None:
    update = slaclip_q_threshold_update(
        1.0,
        torch.tensor([unclipped_proxy, 0.0]),
        eta=0.5,
        target_clip_fraction=0.99,
        c_min=0.1,
        c_max=15.0,
    )
    assert update.target_unclipped_proxy == pytest.approx(0.01)
    if direction > 0:
        assert update.next_clip > 1.0
    elif direction < 0:
        assert update.next_clip < 1.0
    else:
        assert update.next_clip == pytest.approx(1.0)


def test_slaclip_q_99_reports_both_configured_bounds() -> None:
    upper = slaclip_q_threshold_update(
        14.0,
        torch.tensor([-100.0]),
        eta=10.0,
        target_clip_fraction=0.99,
        c_min=0.1,
        c_max=15.0,
    )
    lower = slaclip_q_threshold_update(
        0.2,
        torch.tensor([100.0]),
        eta=10.0,
        target_clip_fraction=0.99,
        c_min=0.1,
        c_max=15.0,
    )
    assert upper.next_clip == 15.0 and upper.hit_upper_bound and not upper.hit_lower_bound
    assert lower.next_clip == 0.1 and lower.hit_lower_bound and not lower.hit_upper_bound


def _make_optimizer(
    telemetry_mode: str = 'dp_safe',
    *,
    eps: float = 1e-8,
    num_slots: int = 3,
    clipping_method: str = 'baseline',
    target_clip_fraction: float = 0.99,
    c_min: float = 0.1,
    c_max: float = 50.0,
) -> tuple[PRISM, torch.nn.Parameter, torch.nn.Parameter]:
    torch.manual_seed(7)
    # Parameter order and shapes match PEFT lora_A [r,in], lora_B [out,r].
    p_a = torch.nn.Parameter(torch.randn(2, 3))
    p_b = torch.nn.Parameter(torch.randn(4, 2))
    opt = PRISM(
        [p_a, p_b],
        lr=1e-3,
        eps=eps,
        clipping_method=clipping_method,
        slaclip_num_slots=num_slots,
        slaclip_target_clip_fraction=target_clip_fraction,
        slaclip_c_min=c_min,
        slaclip_c_max=c_max,
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
    assert opt.last_log['dp_tangent_noise_sampler'] == 'qr_exact_full_rank'
    assert opt.last_log['dp_tangent_query_chart'] == 'qr_asymmetric_isometric'
    assert opt.last_log['dp_factor_relative_gram_eigenvalue_min'] > opt.last_log['dp_factor_rank_rcond']


def test_baseline_update_ignores_all_slaclip_only_parameters() -> None:
    grad_a = torch.randn(4, 2, 3, generator=torch.Generator().manual_seed(131))
    grad_b = torch.randn(4, 4, 2, generator=torch.Generator().manual_seed(137))
    first, first_a, first_b = _make_optimizer(
        'dp_safe',
        clipping_method='baseline',
        num_slots=1,
        target_clip_fraction=0.99,
        c_min=0.1,
        c_max=15.0,
    )
    second, second_a, second_b = _make_optimizer(
        'dp_safe',
        clipping_method='baseline',
        num_slots=99,
        target_clip_fraction=0.1,
        c_min=0.01,
        c_max=100.0,
    )
    for opt, p_a, p_b in ((first, first_a, first_b), (second, second_a, second_b)):
        opt.dp_begin(max_grad_norm=1.0, expected_batch_size=4, noise_multiplier=0.8)
        _set_grad_samples(p_a, p_b, grad_a, grad_b)
        opt.dp_accumulate()
        torch.manual_seed(139)
        opt.dp_finalize(noise_multiplier=0.8)

    assert torch.equal(first_a, second_a)
    assert torch.equal(first_b, second_b)
    assert first.current_clip == second.current_clip == 1.0
    assert 'slack_indicator' not in first.last_log
    assert 'slack_indicator' not in second.last_log


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


def test_slaclip_q_99_optimizer_records_proxy_target_noise_and_bounds() -> None:
    torch.manual_seed(15)
    opt, p_a, p_b = _make_optimizer(
        'dp_safe',
        clipping_method='slaclip_q',
        target_clip_fraction=0.99,
        c_min=0.1,
        c_max=15.0,
    )
    opt.dp_begin(max_grad_norm=1.0, expected_batch_size=4, noise_multiplier=0.8)
    _set_grad_samples(p_a, p_b, torch.randn(4, 2, 3), torch.randn(4, 4, 2))
    opt.dp_accumulate()
    opt.dp_finalize(noise_multiplier=0.8)

    assert opt.last_log['slaclip_controller'] == 'slaclip_q'
    assert opt.last_log['slaclip_target_clip_fraction'] == pytest.approx(0.99)
    assert opt.last_log['slaclip_target_unclipped_proxy'] == pytest.approx(0.01)
    assert opt.last_log['slaclip_c_min'] == pytest.approx(0.1)
    assert opt.last_log['slaclip_c_max'] == pytest.approx(15.0)
    assert opt.last_log['slack_indicator_noise_std'] == pytest.approx(0.8 * math.sqrt(3) / 4)
    assert 0.1 <= opt.current_clip <= 15.0


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


def test_optimizer_checkpoint_rejects_different_slaclip_q_target() -> None:
    source, _, _ = _make_optimizer(
        clipping_method='slaclip_q',
        target_clip_fraction=0.99,
        c_max=15.0,
    )
    saved = source.state_dict()
    target, _, _ = _make_optimizer(
        clipping_method='slaclip_q',
        target_clip_fraction=0.98,
        c_max=15.0,
    )
    with pytest.raises(ValueError, match='slaclip_target_clip_fraction'):
        target.load_state_dict(saved)


def test_automatic_slots_resolve_once_and_restore_from_checkpoint() -> None:
    source, _, _ = _make_optimizer(
        clipping_method='slaclip_q',
        num_slots=0,
        c_max=15.0,
    )
    source.dp_begin(
        max_grad_norm=1.0,
        expected_batch_size=63.9935,
        noise_multiplier=0.516357,
    )
    source.dp_finalize(noise_multiplier=0.516357)
    assert source.slaclip_num_slots == 8
    saved = source.state_dict()

    restored, _, _ = _make_optimizer(
        clipping_method='slaclip_q',
        num_slots=0,
        c_max=15.0,
    )
    restored.load_state_dict(saved)
    assert restored.slaclip_num_slots == 8


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
