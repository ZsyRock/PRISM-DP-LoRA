from __future__ import annotations
import math
import types
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple
import torch
from torch import Tensor

from ..slaclip import (
    automatic_num_slots,
    build_slack_vectors,
    full_slaclip_threshold_update,
    resolve_full_slaclip_target,
    slaclip_q_threshold_update,
    slack_indicator_noise_std,
)

class _LoraShapeHelper:

    @staticmethod
    def move_lora_dim_to_last(x: Tensor, lora_dim: int) -> Tuple[Tensor, Tuple[int, ...]]:
        x2 = torch.moveaxis(x, lora_dim, -1)
        shape_after_move = tuple(x2.shape)
        x2 = x2.reshape(-1, x2.shape[-1])
        return (x2, shape_after_move)

    @staticmethod
    def restore_param_shape(x2d: Tensor, ref: Tensor, lora_dim: int) -> Tensor:
        _, shape_after_move = _LoraShapeHelper.move_lora_dim_to_last(ref, lora_dim)
        out = x2d.reshape(shape_after_move)
        out = torch.moveaxis(out, -1, lora_dim)
        return out

    @staticmethod
    def get_grad_tensor(p: torch.nn.Parameter) -> Tensor:
        if p.grad is None:
            return torch.zeros_like(p.data)
        return p.grad.data

    @staticmethod
    def _as_grad_sample_tensor(gs: object) -> Optional[Tensor]:
        if gs is None:
            return None
        if isinstance(gs, Tensor):
            return gs
        if isinstance(gs, (list, tuple)):
            parts = [g for g in gs if isinstance(g, Tensor)]
            if not parts:
                return None
            return torch.cat(parts, dim=0)
        return None

    @staticmethod
    def get_grad_sample_tensor(p: torch.nn.Parameter) -> Optional[Tensor]:
        return _LoraShapeHelper._as_grad_sample_tensor(getattr(p, 'grad_sample', None))

    @staticmethod
    def clear_grad_sample(p: torch.nn.Parameter) -> None:
        if hasattr(p, 'grad_sample'):
            setattr(p, 'grad_sample', None)

    @staticmethod
    def move_lora_dim_to_last_grad_sample(gs_raw: object, lora_dim_in_param: int) -> Tensor:
        gs = _LoraShapeHelper._as_grad_sample_tensor(gs_raw)
        if gs is None:
            raise ValueError('grad_sample is None or unsupported type')
        if gs.ndim < 2:
            raise ValueError(f'grad_sample must have >=2 dims, got {gs.ndim}')
        if lora_dim_in_param >= 0:
            lora_dim_in_gs = lora_dim_in_param + 1
        else:
            lora_dim_in_gs = lora_dim_in_param
        gs2 = torch.moveaxis(gs, lora_dim_in_gs, -1)
        bs = gs2.shape[0]
        r = gs2.shape[-1]
        return gs2.reshape(bs, -1, r)

def _sym(M: Tensor) -> Tensor:
    return 0.5 * (M + M.T)

def _psd_eigh(M: Tensor) -> Tuple[Tensor, Tensor]:
    Mf = _sym(M.float())
    try:
        w, v = torch.linalg.eigh(Mf)
    except Exception:
        n = Mf.shape[0]
        diag_mean = float(Mf.diagonal().abs().mean().item()) if n > 0 else 0.0
        jitter = 1e-06 * (diag_mean + 1.0)
        J = torch.eye(n, device=Mf.device, dtype=Mf.dtype) * jitter
        try:
            w, v = torch.linalg.eigh(Mf + J)
        except Exception:
            w_cpu, v_cpu = torch.linalg.eigh((Mf + J).double().cpu())
            w = w_cpu.to(device=Mf.device, dtype=Mf.dtype)
            v = v_cpu.to(device=Mf.device, dtype=Mf.dtype)
    w = torch.clamp(w, min=0.0)
    return (w.to(M.dtype), v.to(M.dtype))

def _psd_pinv(M: Tensor, rcond: float=1e-06) -> Tensor:
    w, v = _psd_eigh(M)
    max_w = torch.max(w)
    if float(max_w) <= 0.0:
        return torch.zeros_like(M)
    cutoff = float(rcond) * float(max_w)
    inv = torch.zeros_like(w)
    mask = w > cutoff
    inv[mask] = 1.0 / w[mask]
    out = v * inv.unsqueeze(0) @ v.T
    return _sym(out)

def _full_column_rank_qr(
    X: Tensor,
    *,
    gram_rcond: float,
    factor_name: str,
) -> Tuple[Tensor, Tensor, Dict[str, float]]:
    """Return a thin QR factorization after a fail-closed rank check.

    PRISM's rank-``r`` tangent mechanism assumes that both LoRA factors have
    full column rank.  Silently damping or truncating a singular Gram matrix
    changes the covariance of the Gaussian mechanism while leaving the
    accountant unchanged.  We instead check the same *relative Gram
    eigenvalue* criterion historically controlled by ``rcond`` and refuse to
    make a release outside the paper's smooth rank-``r`` manifold.
    """

    if X.ndim != 2:
        raise RuntimeError(f'{factor_name} must be a matrix, got {X.ndim}D')
    rows, rank = X.shape
    if rows < rank:
        raise RuntimeError(
            f'{factor_name} cannot have full column rank: shape={tuple(X.shape)}'
        )
    Xf = X.float()
    if not bool(torch.isfinite(Xf).all().item()):
        raise RuntimeError(f'{factor_name} contains non-finite values')
    Q, R = torch.linalg.qr(Xf, mode='reduced')
    # R has the same singular values as X, while its r-by-r SVD is far
    # cheaper than a second decomposition of a tall LoRA factor each step.
    try:
        singular_values = torch.linalg.svdvals(R)
    except Exception:
        singular_values = torch.linalg.svdvals(R.double().cpu()).to(
            device=Xf.device,
            dtype=Xf.dtype,
        )
    s_max = float(torch.max(singular_values).item()) if rank else 0.0
    s_min = float(torch.min(singular_values).item()) if rank else 0.0
    relative_gram_eigenvalue = 0.0 if s_max <= 0.0 else (s_min / s_max) ** 2
    if s_max <= 0.0 or relative_gram_eigenvalue <= float(gram_rcond):
        raise RuntimeError(
            'PRISM DP tangent release requires numerically full-column-rank '
            f'LoRA factors; {factor_name} has shape={tuple(X.shape)}, '
            f's_min={s_min:.6g}, s_max={s_max:.6g}, '
            f'(s_min/s_max)^2={relative_gram_eigenvalue:.6g} <= '
            f'rcond={float(gram_rcond):.6g}. The rank-r manifold and its '
            'isotropic tangent Gaussian are undefined at a rank-deficient '
            'factor, so the release is aborted before privacy accounting.'
        )
    return (
        Q,
        R,
        {
            's_min': s_min,
            's_max': s_max,
            'relative_gram_eigenvalue': relative_gram_eigenvalue,
        },
    )


def _gram_inverse_from_qr(R: Tensor) -> Tensor:
    """Compute ``(X.T @ X)^-1`` from ``X = Q @ R`` without a Gram EVD."""

    if R.ndim != 2 or R.shape[0] != R.shape[1]:
        raise ValueError(f'R must be square, got shape={tuple(R.shape)}')
    eye = torch.eye(R.shape[0], device=R.device, dtype=R.dtype)
    R_inv = torch.linalg.solve_triangular(R, eye, upper=True)
    gram_inv = _sym(R_inv @ R_inv.T)
    if not bool(torch.isfinite(gram_inv).all().item()):
        raise RuntimeError(
            'PRISM full-rank Gram inverse is non-finite; aborting before '
            'per-record gradient accumulation and privacy accounting'
        )
    return gram_inv


def _right_solve_transpose(X: Tensor, R: Tensor) -> Tensor:
    """Return ``X @ R^-T`` using a triangular solve."""

    Xt = X.transpose(-2, -1)
    return torch.linalg.solve_triangular(R, Xt, upper=True).transpose(-2, -1)


def _right_solve_gram(X: Tensor, R: Tensor) -> Tensor:
    """Return ``X @ (R.T @ R)^-1`` for a 2D or batched 3D ``X``.

    Keeping both solves triangular avoids explicitly multiplying a possibly
    ill-conditioned Gram inverse into the private per-record query.
    """

    if X.ndim not in {2, 3}:
        raise ValueError(f'X must be 2D or 3D, got {X.ndim}D')
    Xt = X.transpose(-2, -1)
    # first = X @ R^-1, obtained from R.T @ first.T = X.T
    first = torch.linalg.solve_triangular(
        R.T,
        Xt,
        upper=False,
    ).transpose(-2, -1)
    # result = first @ R^-T, obtained from R @ result.T = first.T
    return _right_solve_transpose(first, R)


def _dp_isometric_chart_lift(
    gA: Tensor,
    gB: Tensor,
    QA: Tensor,
    RA: Tensor,
    RB: Tensor,
) -> Tuple[Tensor, Tensor]:
    """Lift a factor-gradient query into the same chart as PRISM noise.

    The paper's intrinsic tangent projection has many equivalent factor
    lifts.  For a rigorous factor-wise implementation of the Gaussian release
    and its adaptive post-processing, the private query and Eq. (19) noise must
    inhabit the same linear chart.  We use the unique representative satisfying
    ``QA.T @ dA = 0``:

    ``dA = (I-QA QA.T) gA (B.T B)^-1`` and
    ``dB = gB (A.T A)^-1``.

    Its induced matrix is the same intrinsic projected tangent as the
    symmetric half lift when the factor gradients obey the LoRA chain rule.
    """

    YA = _right_solve_gram(gA, RB)
    dA = YA - _apply_projector_Q(QA, YA)
    dB = _right_solve_gram(gB, RA)
    return dA, dB


def _factorized_isotropic_tangent_noise(
    U: Tensor,
    V: Tensor,
    QA: Tensor,
    RA: Tensor,
    RB: Tensor,
) -> Tuple[Tensor, Tensor]:
    """Sample-factor map whose induced tangent noise is exactly isotropic.

    With ``A = QA @ RA`` and ``B = QB @ RB``, these lifts induce

    ``(I-QA QA.T) U QB.T + QA V.T``.

    Thus i.i.d. standard-normal ``U,V`` yield the orthogonal projection of a
    dense standard Gaussian onto the rank-``r`` tangent space.  In particular,
    no ``eps`` damping is permitted here: it would reduce covariance in some
    intrinsic directions below the scale assumed by the Gaussian accountant.
    """

    U_perp = U - _apply_projector_Q(QA, U)
    noise_A = _right_solve_transpose(U_perp, RB)
    noise_B = _right_solve_transpose(V, RA)
    return noise_A, noise_B

def _psd_invsqrt_clamped(M: Tensor, eps: float, floor: float, cond_max: Optional[float]=None, cond_strategy: str='raise_small') -> Tuple[Tensor, Dict[str, float]]:
    w, v = _psd_eigh(M)
    w = w + float(eps)
    floor_t = torch.tensor(float(floor), device=w.device, dtype=w.dtype)
    w2 = torch.maximum(w, floor_t)
    if cond_max is not None and cond_max > 1.0:
        strategy = str(cond_strategy).lower()
        if strategy in {'raise_small', 'floor', 'safe'}:
            max_w = torch.max(w2)
            cond_floor = max_w / float(cond_max)
            w2 = torch.maximum(w2, cond_floor)
        elif strategy in {'cap_large', 'legacy', 'old'}:
            cap = floor_t * float(cond_max)
            w2 = torch.minimum(w2, cap)
        else:
            raise ValueError(f'Unknown cond_strategy={cond_strategy!r}')
    invsqrt = 1.0 / torch.sqrt(w2)
    out = v * invsqrt.unsqueeze(0) @ v.T
    stats = {'eig_min': float(torch.min(w).item()), 'eig_max': float(torch.max(w).item()), 'eig_min_clamped': float(torch.min(w2).item()), 'eig_max_clamped': float(torch.max(w2).item())}
    return (_sym(out), stats)

def _apply_projector(A: Tensor, M_pinv: Tensor, X: Tensor) -> Tensor:
    if X.ndim == 2:
        return A @ (M_pinv @ (A.T @ X))
    if X.ndim == 3:
        AtX = torch.matmul(A.T, X)
        tmp = torch.matmul(M_pinv, AtX)
        return torch.matmul(A, tmp)
    raise ValueError(f'X must be 2D or 3D, got {X.ndim}D')

def _apply_projector_Q(Q: Tensor, X: Tensor) -> Tensor:
    if X.ndim == 2:
        return Q @ (Q.T @ X)
    if X.ndim == 3:
        QtX = torch.matmul(Q.T, X)
        return torch.matmul(Q, QtX)
    raise ValueError(f'X must be 2D or 3D, got {X.ndim}D')

def _procrustes_align(A_new: Tensor, B_new: Tensor, A_ref: Tensor, B_ref: Tensor) -> Tuple[Tensor, Tensor]:
    C = A_new.float().T @ A_ref.float() + B_new.float().T @ B_ref.float()
    U, _, Vh = torch.linalg.svd(C, full_matrices=False)
    Q = (U @ Vh).to(A_new.dtype)
    return (A_new @ Q, B_new @ Q)

def _retract_rank_r(A: Tensor, B: Tensor, dA: Tensor, dB: Tensor, eta: float, r: int, align_to: Optional[Tuple[Tensor, Tensor]]=None) -> Tuple[Tensor, Tensor]:
    if r <= 0:
        raise ValueError('rank r must be positive')
    A0 = A.float()
    B0 = B.float()
    dA0 = dA.float()
    dB0 = dB.float()
    U = torch.cat([A0, dA0], dim=1)
    V = torch.cat([B0, dB0], dim=1)
    Q_u, R_u = torch.linalg.qr(U, mode='reduced')
    Q_v, R_v = torch.linalg.qr(V, mode='reduced')
    I = torch.eye(r, device=A.device, dtype=torch.float32)
    K = torch.zeros((2 * r, 2 * r), device=A.device, dtype=torch.float32)
    K[:r, :r] = I
    K[:r, r:] = float(eta) * I
    K[r:, :r] = float(eta) * I
    core = R_u @ K @ R_v.T
    Uc, S, Vhc = torch.linalg.svd(core, full_matrices=False)
    rr = min(r, S.numel())
    Uc_r = Uc[:, :rr]
    Vc_r = Vhc.T[:, :rr]
    S_r = S[:rr]
    S_sqrt = torch.diag(torch.sqrt(torch.clamp(S_r, min=0.0)))
    A_new = Q_u @ Uc_r @ S_sqrt
    B_new = Q_v @ Vc_r @ S_sqrt
    if rr < r:
        A_new = torch.cat([A_new, torch.zeros((A_new.shape[0], r - rr), device=A.device, dtype=A_new.dtype)], dim=1)
        B_new = torch.cat([B_new, torch.zeros((B_new.shape[0], r - rr), device=B.device, dtype=B_new.dtype)], dim=1)
    A_new = A_new.to(dtype=A.dtype)
    B_new = B_new.to(dtype=B.dtype)
    if align_to is not None:
        A_ref, B_ref = align_to
        A_new, B_new = _procrustes_align(A_new, B_new, A_ref, B_ref)
    return (A_new, B_new)

def _tangent_fro_norm_sq(dA: Tensor, dB: Tensor, A: Tensor, B: Tensor) -> Tensor:
    M = A.T @ A
    N = B.T @ B
    if dA.ndim == 2:
        dA_t_dA = dA.T @ dA
        dB_t_dB = dB.T @ dB
        At_dA = A.T @ dA
        Bt_dB = B.T @ dB
        term1 = torch.sum(dA_t_dA * N)
        term2 = torch.sum(dB_t_dB * M)
        term3 = torch.sum(At_dA * Bt_dB.T)
        return term1 + term2 + 2.0 * term3
    if dA.ndim == 3:
        dA_t_dA = torch.matmul(dA.transpose(1, 2), dA)
        dB_t_dB = torch.matmul(dB.transpose(1, 2), dB)
        At_dA = torch.matmul(A.T, dA)
        Bt_dB = torch.matmul(B.T, dB)
        term1 = torch.einsum('bij,ij->b', dA_t_dA, N)
        term2 = torch.einsum('bij,ij->b', dB_t_dB, M)
        term3 = torch.einsum('bij,bji->b', At_dA, Bt_dB)
        return term1 + term2 + 2.0 * term3
    raise ValueError(f'dA must be 2D or 3D, got {dA.ndim}D')


def _tangent_fro_inner_product(
    dA: Tensor,
    dB: Tensor,
    eA: Tensor,
    eB: Tensor,
    A: Tensor,
    B: Tensor,
) -> Tensor:
    """Inner product of two induced tangent matrices without densifying them."""

    if any(value.ndim != 2 for value in (dA, dB, eA, eB)):
        raise ValueError('_tangent_fro_inner_product expects four 2D lifts')
    M = A.T @ A
    N = B.T @ B
    term_aa = torch.sum((dA.T @ eA) * N)
    term_bb = torch.sum((dB.T @ eB) * M)
    term_ab = torch.sum((A.T @ dA) * (B.T @ eB).T)
    term_ba = torch.sum((A.T @ eA) * (B.T @ dB).T)
    return term_aa + term_bb + term_ab + term_ba


def _cosine_from_inner(inner: float, first_norm_sq: float, second_norm_sq: float) -> float:
    denominator = math.sqrt(max(first_norm_sq, 0.0) * max(second_norm_sq, 0.0))
    if denominator <= 1e-30:
        return 0.0
    # Roundoff in low-rank Gram products can put a mathematically valid
    # cosine a few ulps outside [-1, 1].
    return max(-1.0, min(1.0, float(inner) / denominator))


def _z_fro_norm_sq(A: Tensor, B: Tensor) -> Tensor:
    M = A.T @ A
    N = B.T @ B
    return torch.sum(M * N)


def _factorized_delta_fro_norm_sq(
    A_new: Tensor,
    B_new: Tensor,
    A_old: Tensor,
    B_old: Tensor,
) -> Tensor:
    """Return ``||A_new B_new^T - A_old B_old^T||_F^2`` without densifying.

    LoRA matrices can be very wide, so materializing the full weight update just
    for telemetry would be prohibitively expensive.  Writing the difference as
    ``U V^T`` with at most ``2r`` columns keeps this calculation rank-sized.
    """

    U = torch.cat((A_new.float(), A_old.float()), dim=1)
    V = torch.cat((B_new.float(), -B_old.float()), dim=1)
    return torch.sum((U.T @ U) * (V.T @ V))

def _sylvester_symmetric_solve(M: Tensor, N: Tensor, C: Tensor, eps: float=1e-12) -> Tensor:
    Mf = _sym(M.float())
    Nf = _sym(N.float())
    Cf = C.float()
    wm, Um = _psd_eigh(Mf)
    wn, Vn = _psd_eigh(Nf)
    denom = wm[:, None] + wn[None, :]
    denom = torch.clamp(denom, min=float(eps))
    C_tilde = Um.T @ Cf @ Vn
    X_tilde = C_tilde / denom
    X = Um @ X_tilde @ Vn.T
    return X.to(device=C.device, dtype=C.dtype)

def _horizontalize_lift(dA: Tensor, dB: Tensor, A: Tensor, B: Tensor, eps: float=1e-12) -> Tuple[Tensor, Tensor]:
    if dA.ndim != 2 or dB.ndim != 2:
        raise ValueError('_horizontalize_lift currently expects 2D dA,dB')
    M = _sym(A.float().T @ A.float())
    N = _sym(B.float().T @ B.float())
    rhs = dB.float().T @ B.float() - A.float().T @ dA.float()
    Omega = _sylvester_symmetric_solve(M, N, rhs, eps=eps).to(device=dA.device, dtype=dA.dtype)
    return (dA + A.to(dtype=dA.dtype) @ Omega, dB - B.to(dtype=dB.dtype) @ Omega.T)

@dataclass
class _DPAccumState:
    max_grad_norm: float
    expected_batch_size: float
    noise_multiplier: float
    total_samples: int = 0
    microbatches: int = 0
    clipped_samples: int = 0
    coef_sum: float = 0.0
    coef_min: float = 1.0
    slack_sum: Optional[Tensor] = None
    slack_lambda: float = 0.0
    raw_norms: List[Tensor] = field(default_factory=list)
    last_micro_stats: Dict[str, float] = field(default_factory=dict)

class PRISM(torch.optim.Optimizer):

    def __init__(self, params: Iterable[torch.nn.Parameter], lr: float=0.0003, betas: Tuple[float, float]=(0.9, 0.999), eps: float=1e-08, weight_decay: float=0.0, rcond: float=1e-06, lora_l_dim: int=0, lora_r_dim: int=-1, use_adaptive: bool=True, dp_precond_floor_factor: float=1.0, dp_floor_mode: str='geometry', precond_cond_max: Optional[float]=10000.0, precond_cond_strategy: str='raise_small', precond_update_mode: str='current', lift_gauge_fix: str='none', gauge_fix_eps: float=1e-12, max_update_norm: float=0.0, use_trust_ratio: bool=False, trust_clip: Tuple[float, float]=(0.0, 10.0), trust_eps: float=1e-12, dp_debias_second_moment: bool=True, log_every: int=1, clipping_method: str='baseline', slaclip_num_slots: int=0, slaclip_eta: float=0.5, slaclip_beta: Optional[float]=None, slaclip_target_non_small_clip_fraction: Optional[float]=None, slaclip_target_clip_fraction: Optional[float]=None, slaclip_c_min: float=0.1, slaclip_c_max: float=50.0, telemetry_mode: str='dp_safe', raw_hist_bins: int=32, raw_hist_max: float=0.0):
        if lr <= 0:
            raise ValueError('lr must be positive')
        beta1, beta2 = betas
        if not (0.0 <= beta1 < 1.0 and 0.0 <= beta2 < 1.0):
            raise ValueError('betas must be in [0,1)')
        if eps <= 0:
            raise ValueError('eps must be positive')
        if not 0.0 <= float(rcond) < 1.0:
            raise ValueError('rcond must be in [0, 1)')
        if weight_decay < 0:
            raise ValueError('weight_decay must be >= 0')
        if trust_eps <= 0:
            raise ValueError('trust_eps must be positive')
        if trust_clip[1] < trust_clip[0]:
            raise ValueError('trust_clip max must be >= min')
        if trust_clip[0] < 0:
            raise ValueError('trust_clip min must be >= 0')
        valid_floor_modes = {'geometry', 'scalar', 'none'}
        if str(dp_floor_mode).lower() not in valid_floor_modes:
            raise ValueError(f'dp_floor_mode must be one of {sorted(valid_floor_modes)}, got {dp_floor_mode!r}')
        valid_precond_modes = {'current', 'delayed'}
        if str(precond_update_mode).lower() not in valid_precond_modes:
            raise ValueError(f'precond_update_mode must be one of {sorted(valid_precond_modes)}, got {precond_update_mode!r}')
        valid_gauge_fix_modes = {'none', 'pre_moment', 'pre_retract', 'both'}
        if str(lift_gauge_fix).lower() not in valid_gauge_fix_modes:
            raise ValueError(f'lift_gauge_fix must be one of {sorted(valid_gauge_fix_modes)}, got {lift_gauge_fix!r}')
        if gauge_fix_eps <= 0:
            raise ValueError('gauge_fix_eps must be positive')
        clipping_method = str(clipping_method).lower()
        if clipping_method not in {'baseline', 'slaclip', 'slaclip_q'}:
            raise ValueError("clipping_method must be 'baseline', 'slaclip', or 'slaclip_q'")
        telemetry_mode = str(telemetry_mode).lower()
        if telemetry_mode not in {'dp_safe', 'research_raw'}:
            raise ValueError("telemetry_mode must be 'dp_safe' or 'research_raw'")
        if int(slaclip_num_slots) < 0:
            raise ValueError('slaclip_num_slots must be >= 0 (0 selects it automatically)')
        if float(slaclip_eta) < 0:
            raise ValueError('slaclip_eta must be non-negative')
        target_non_small_clip_fraction = resolve_full_slaclip_target(
            slaclip_target_non_small_clip_fraction,
            beta=slaclip_beta,
        )
        if slaclip_target_clip_fraction is None:
            slaclip_target_clip_fraction = 0.99
        if not 0.0 <= float(slaclip_target_clip_fraction) <= 1.0:
            raise ValueError('slaclip_target_clip_fraction must be in [0, 1]')
        if float(slaclip_c_min) <= 0 or float(slaclip_c_max) < float(slaclip_c_min):
            raise ValueError('require 0 < slaclip_c_min <= slaclip_c_max')
        if int(raw_hist_bins) <= 0:
            raise ValueError('raw_hist_bins must be positive')
        defaults = dict(lr=lr)
        super().__init__(params, defaults)
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.eps = float(eps)
        self.weight_decay = float(weight_decay)
        self.rcond = float(rcond)
        self.lora_l_dim = int(lora_l_dim)
        self.lora_r_dim = int(lora_r_dim)
        self.use_adaptive = bool(use_adaptive)
        self.dp_precond_floor_factor = float(dp_precond_floor_factor)
        self.dp_floor_mode = str(dp_floor_mode).lower()
        self.precond_cond_max = float(precond_cond_max) if precond_cond_max is not None else None
        self.precond_cond_strategy = str(precond_cond_strategy)
        self.precond_update_mode = str(precond_update_mode).lower()
        self.lift_gauge_fix = str(lift_gauge_fix).lower()
        self.gauge_fix_eps = float(gauge_fix_eps)
        self.max_update_norm = float(max_update_norm)
        self.log_every = int(log_every)
        self.use_trust_ratio = bool(use_trust_ratio)
        self.trust_clip = (float(trust_clip[0]), float(trust_clip[1]))
        self.trust_eps = float(trust_eps)
        self.dp_debias_second_moment = bool(dp_debias_second_moment)
        self.clipping_method = clipping_method
        self.slaclip_num_slots = int(slaclip_num_slots)
        self.slaclip_eta = float(slaclip_eta)
        self.slaclip_target_non_small_clip_fraction = float(
            target_non_small_clip_fraction
        )
        # Keep the historical attribute/checkpoint key readable by existing
        # analysis code. It is an alias for rho, not a feedback gain.
        self.slaclip_beta = self.slaclip_target_non_small_clip_fraction
        self.slaclip_target_clip_fraction = float(slaclip_target_clip_fraction)
        self.slaclip_c_min = float(slaclip_c_min)
        self.slaclip_c_max = float(slaclip_c_max)
        self.telemetry_mode = telemetry_mode
        self.raw_hist_bins = int(raw_hist_bins)
        self.raw_hist_max = float(raw_hist_max)
        self.current_clip: Optional[float] = None
        self._helper = _LoraShapeHelper()
        self._dp_state: Optional[_DPAccumState] = None
        self._state_initialized = False
        self._dp_noise_generator: Optional[torch.Generator] = None
        self._dp_secure_mode = False
        self.last_log: Dict[str, Any] = {}
        self.last_raw_log: Dict[str, Any] = {}

    def _pair_iter(self):
        for group in self.param_groups:
            params = group['params']
            for p1, p2 in list(zip(params, params[1:]))[::2]:
                yield (group, p1, p2)

    def state_dict(self):
        state = super().state_dict()
        state['_prism_runtime'] = {
            'clipping_method': self.clipping_method,
            'current_clip': self.current_clip,
            'slaclip_num_slots': self.slaclip_num_slots,
            'slaclip_eta': self.slaclip_eta,
            'slaclip_target_non_small_clip_fraction': (
                self.slaclip_target_non_small_clip_fraction
            ),
            'slaclip_beta': self.slaclip_beta,
            'slaclip_target_clip_fraction': self.slaclip_target_clip_fraction,
            'slaclip_c_min': self.slaclip_c_min,
            'slaclip_c_max': self.slaclip_c_max,
            'telemetry_mode': self.telemetry_mode,
            'raw_hist_max': self.raw_hist_max,
        }
        return state

    def load_state_dict(self, state_dict):
        state_copy = dict(state_dict)
        runtime = state_copy.pop('_prism_runtime', None)
        result = super().load_state_dict(state_copy)
        # A valid step-zero checkpoint can precede the first dp_begin(), in
        # which case torch's optimizer state is empty. Recreate only missing
        # PRISM slots after loading while preserving any restored moments.
        self._state_initialized = False
        self._ensure_state()
        if isinstance(runtime, dict):
            saved_method = runtime.get('clipping_method')
            if saved_method is not None and saved_method != self.clipping_method:
                raise ValueError(
                    f"checkpoint clipping_method={saved_method!r} does not match current method={self.clipping_method!r}"
                )
            saved_non_small_target = runtime.get(
                'slaclip_target_non_small_clip_fraction',
                runtime.get('slaclip_beta'),
            )
            if saved_non_small_target is not None and not math.isclose(
                float(saved_non_small_target),
                self.slaclip_target_non_small_clip_fraction,
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                raise ValueError(
                    'checkpoint slaclip_target_non_small_clip_fraction='
                    f'{saved_non_small_target!r} does not match current value='
                    f'{self.slaclip_target_non_small_clip_fraction!r}'
                )
            saved_legacy_target = runtime.get('slaclip_beta')
            saved_canonical_target = runtime.get(
                'slaclip_target_non_small_clip_fraction'
            )
            if (
                saved_legacy_target is not None
                and saved_canonical_target is not None
                and not math.isclose(
                    float(saved_legacy_target),
                    float(saved_canonical_target),
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                )
            ):
                raise ValueError(
                    'checkpoint has conflicting full-SlaClip target aliases'
                )
            for key, current in (
                ('slaclip_eta', self.slaclip_eta),
                ('slaclip_target_clip_fraction', self.slaclip_target_clip_fraction),
                ('slaclip_c_min', self.slaclip_c_min),
                ('slaclip_c_max', self.slaclip_c_max),
            ):
                saved = runtime.get(key)
                if saved is not None and not math.isclose(
                    float(saved), float(current), rel_tol=1e-12, abs_tol=1e-12
                ):
                    raise ValueError(
                        f'checkpoint {key}={saved!r} does not match current value={current!r}'
                    )
            saved_telemetry = runtime.get('telemetry_mode')
            if saved_telemetry is not None and saved_telemetry != self.telemetry_mode:
                raise ValueError(
                    f'checkpoint telemetry_mode={saved_telemetry!r} does not match '
                    f'current mode={self.telemetry_mode!r}'
                )
            saved_clip = runtime.get('current_clip')
            self.current_clip = None if saved_clip is None else float(saved_clip)
            saved_slots = int(runtime.get('slaclip_num_slots', self.slaclip_num_slots))
            if self.slaclip_num_slots not in {0, saved_slots}:
                raise ValueError(
                    f'checkpoint slaclip_num_slots={saved_slots} does not match '
                    f'current value={self.slaclip_num_slots}'
                )
            self.slaclip_num_slots = saved_slots
            self.raw_hist_max = float(runtime.get('raw_hist_max', self.raw_hist_max))
        # torch.optim recursively moves tensors in dictionaries/lists, but not
        # tensors stored in SimpleNamespace.  Checkpoints are taken between
        # logical steps, so only persistent moments should be populated here.
        for _, pA, _ in self._pair_iter():
            st = self.state.get(pA, {}).get('prism')
            if st is None:
                continue
            for name in ('mA', 'mB', 'vA', 'vB'):
                value = getattr(st, name, None)
                if isinstance(value, Tensor):
                    setattr(st, name, value.to(device=pA.device))
            for transient in ('dp_accum_A', 'dp_accum_B', 'dp_raw_accum_A', 'dp_raw_accum_B'):
                value = getattr(st, transient, None)
                if value is not None:
                    raise ValueError(
                        f'checkpoint unexpectedly contains in-progress state {transient}; '
                        'only logical-step checkpoints are supported'
                    )
        return result

    def configure_dp_noise(self, *, generator=None, secure_mode: bool = False) -> None:
        """Use the same noise source configured by Opacus's PrivacyEngine."""

        self._dp_noise_generator = generator
        self._dp_secure_mode = bool(secure_mode)

    def get_dp_noise_generator_state(self) -> Optional[Tensor]:
        generator = self._dp_noise_generator
        if generator is None or not hasattr(generator, 'get_state'):
            return None
        return generator.get_state().detach().cpu()

    def set_dp_noise_generator_state(self, state: Optional[Tensor]) -> None:
        if state is None:
            return
        generator = self._dp_noise_generator
        if generator is None or not hasattr(generator, 'set_state'):
            raise RuntimeError('checkpoint contains a DP noise RNG state, but no compatible generator is configured')
        generator.set_state(state)

    def _generate_dp_noise(self, reference: Tensor, std: float) -> Tensor:
        if float(std) == 0.0:
            return torch.zeros_like(reference)
        try:
            # Reuse Opacus's hardened four-sample construction when secure mode
            # is enabled; in ordinary research mode this also preserves the
            # PrivacyEngine generator semantics.
            from opacus.optimizers.optimizer import _generate_noise

            return _generate_noise(
                std=float(std),
                reference=reference,
                generator=self._dp_noise_generator,
                secure_mode=self._dp_secure_mode,
            )
        except ImportError:
            if self._dp_secure_mode:
                raise RuntimeError('secure DP noise requires an Opacus version exposing _generate_noise')
            return torch.normal(
                mean=0.0,
                std=float(std),
                size=tuple(reference.shape),
                generator=self._dp_noise_generator,
                device=reference.device,
                dtype=reference.dtype,
            )

    def _get_factors(self, pA: torch.nn.Parameter, pB: torch.nn.Parameter) -> Tuple[Tensor, Tensor]:
        B, _ = self._helper.move_lora_dim_to_last(pA.data, self.lora_l_dim)
        A, _ = self._helper.move_lora_dim_to_last(pB.data, self.lora_r_dim)
        return (A, B)

    def _set_factors(self, pA: torch.nn.Parameter, pB: torch.nn.Parameter, A: Tensor, B: Tensor) -> None:
        pB.data.copy_(self._helper.restore_param_shape(A, pB.data, self.lora_r_dim))
        pA.data.copy_(self._helper.restore_param_shape(B, pA.data, self.lora_l_dim))

    def _ensure_state(self):
        if self._state_initialized:
            return
        for _, pA, pB in self._pair_iter():
            if self.state[pA].get('prism') is not None:
                continue
            A, B = self._get_factors(pA, pB)
            r = A.shape[1]
            self.state[pA]['prism'] = types.SimpleNamespace(step=0, mA=torch.zeros_like(A, dtype=torch.float32), mB=torch.zeros_like(B, dtype=torch.float32), vA=torch.zeros((r, r), device=A.device, dtype=torch.float32), vB=torch.zeros((r, r), device=B.device, dtype=torch.float32), dp_accum_A=None, dp_accum_B=None, dp_raw_accum_A=None, dp_raw_accum_B=None, dp_cache=None, last_spec_A={}, last_spec_B={})
        self._state_initialized = True

    def _compute_tangent_grad(self, A: Tensor, B: Tensor, gA: Tensor, gB: Tensor, M_pinv: Optional[Tensor]=None, N_pinv: Optional[Tensor]=None) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        M = A.T @ A
        N = B.T @ B
        if M_pinv is None:
            M_pinv = _psd_pinv(M, rcond=self.rcond)
        if N_pinv is None:
            N_pinv = _psd_pinv(N, rcond=self.rcond)
        QA, _ = torch.linalg.qr(A.float(), mode='reduced')
        QB, _ = torch.linalg.qr(B.float(), mode='reduced')
        YA = gA @ N_pinv
        proj_YA = _apply_projector_Q(QA, YA)
        dA = YA - 0.5 * proj_YA
        YB = gB @ M_pinv
        proj_YB = _apply_projector_Q(QB, YB)
        dB = YB - 0.5 * proj_YB
        return (dA, dB, M, N, M_pinv, N_pinv)

    def _adaptive_direction(self, st: types.SimpleNamespace, gradA: Tensor, gradB: Tensor, *, noise_floor: float=0.0, noise_floor_A: Optional[float]=None, noise_floor_B: Optional[float]=None, debias_vA: Optional[Tensor]=None, debias_vB: Optional[Tensor]=None) -> Tuple[Tensor, Tensor]:
        if not self.use_adaptive:
            return (gradA, gradB)
        st.step += 1
        t = st.step
        b1, b2 = (self.beta1, self.beta2)
        st.mA.mul_(b1).add_(gradA, alpha=1.0 - b1)
        st.mB.mul_(b1).add_(gradB, alpha=1.0 - b1)
        m_dim = float(gradA.shape[0])
        n_dim = float(gradB.shape[0])
        vA_new = gradA.T @ gradA / max(1.0, m_dim)
        vB_new = gradB.T @ gradB / max(1.0, n_dim)
        bc1 = 1.0 - b1 ** t
        mA_hat = st.mA / bc1
        mB_hat = st.mB / bc1
        use_delayed = self.precond_update_mode == 'delayed' and t > 1
        if use_delayed:
            bc2_prev = 1.0 - b2 ** (t - 1)
            vA_hat = st.vA / max(bc2_prev, 1e-12)
            vB_hat = st.vB / max(bc2_prev, 1e-12)
            st.vA.mul_(b2).add_(vA_new, alpha=1.0 - b2)
            st.vB.mul_(b2).add_(vB_new, alpha=1.0 - b2)
        else:
            st.vA.mul_(b2).add_(vA_new, alpha=1.0 - b2)
            st.vB.mul_(b2).add_(vB_new, alpha=1.0 - b2)
            bc2 = 1.0 - b2 ** t
            vA_hat = st.vA / bc2
            vB_hat = st.vB / bc2
        if debias_vA is not None:
            vA_hat = _sym(vA_hat - debias_vA.to(device=vA_hat.device, dtype=vA_hat.dtype))
        if debias_vB is not None:
            vB_hat = _sym(vB_hat - debias_vB.to(device=vB_hat.device, dtype=vB_hat.dtype))
        nfA = float(noise_floor if noise_floor_A is None else noise_floor_A)
        nfB = float(noise_floor if noise_floor_B is None else noise_floor_B)
        PA, specA = _psd_invsqrt_clamped(vA_hat, eps=self.eps, floor=nfA, cond_max=self.precond_cond_max, cond_strategy=self.precond_cond_strategy)
        PB, specB = _psd_invsqrt_clamped(vB_hat, eps=self.eps, floor=nfB, cond_max=self.precond_cond_max, cond_strategy=self.precond_cond_strategy)
        st.last_spec_A = specA
        st.last_spec_B = specB
        uA = mA_hat @ PA
        uB = mB_hat @ PB
        return (uA, uB)

    def _maybe_clip_update(self, dA: Tensor, dB: Tensor, A: Tensor, B: Tensor) -> Tuple[Tensor, Tensor, float]:
        cap = float(self.max_update_norm)
        if cap <= 0:
            return (dA, dB, 1.0)
        n2 = _tangent_fro_norm_sq(dA, dB, A, B)
        n = torch.sqrt(torch.clamp(n2, min=0.0) + 1e-12)
        coef = float(torch.clamp(torch.tensor(cap, device=n.device, dtype=n.dtype) / n, max=1.0).item())
        return (dA * coef, dB * coef, coef)

    def _compute_trust_ratio(self, A: Tensor, B: Tensor, dA: Tensor, dB: Tensor) -> float:
        if not self.use_trust_ratio:
            return 1.0
        z2 = _z_fro_norm_sq(A, B)
        u2 = _tangent_fro_norm_sq(dA, dB, A, B)
        z = torch.sqrt(torch.clamp(z2, min=0.0) + 1e-12)
        u = torch.sqrt(torch.clamp(u2, min=0.0) + 1e-12)
        if float(z.item()) <= float(self.trust_eps):
            return 1.0
        trust = z / (u + float(self.trust_eps))
        tmin, tmax = self.trust_clip
        trust = torch.clamp(trust, min=float(tmin), max=float(tmax))
        return float(trust.item())

    @torch.no_grad()
    def step(self, closure=None):
        self._ensure_state()
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        step_logs = {}
        for group, pA, pB in self._pair_iter():
            lr: float = float(group['lr'])
            st = self.state[pA]['prism']
            A, B = self._get_factors(pA, pB)
            gB_raw = self._helper.get_grad_tensor(pA)
            gA_raw = self._helper.get_grad_tensor(pB)
            gB, _ = self._helper.move_lora_dim_to_last(gB_raw, self.lora_l_dim)
            gA, _ = self._helper.move_lora_dim_to_last(gA_raw, self.lora_r_dim)
            A32, B32 = (A.float(), B.float())
            gA32, gB32 = (gA.float(), gB.float())
            dA_grad, dB_grad, _, _, _, _ = self._compute_tangent_grad(A32, B32, gA32, gB32)
            if self.lift_gauge_fix in {'pre_moment', 'both'}:
                dA_grad, dB_grad = _horizontalize_lift(dA_grad, dB_grad, A32, B32, eps=self.gauge_fix_eps)
            uA, uB = self._adaptive_direction(st, dA_grad, dB_grad, noise_floor=0.0)
            if self.weight_decay > 0:
                uA = uA + self.weight_decay * A32
                uB = uB + self.weight_decay * B32
            dA_dir = -uA
            dB_dir = -uB
            if self.lift_gauge_fix in {'pre_retract', 'both'}:
                dA_dir, dB_dir = _horizontalize_lift(dA_dir, dB_dir, A32, B32, eps=self.gauge_fix_eps)
            trust = self._compute_trust_ratio(A32, B32, dA_dir, dB_dir)
            dA_dir = dA_dir * trust
            dB_dir = dB_dir * trust
            dA_dir, dB_dir, upd_coef = self._maybe_clip_update(dA_dir, dB_dir, A32, B32)
            r = A32.shape[1]
            A_new, B_new = _retract_rank_r(A32, B32, dA_dir, dB_dir, eta=lr, r=r, align_to=(A32, B32))
            self._set_factors(pA, pB, A_new.to(A.dtype), B_new.to(B.dtype))
            step_logs.update({'nondp_update_clip_coef': float(upd_coef), 'nondp_trust_ratio': float(trust), 'nondp_precond_eigA_min': float(st.last_spec_A.get('eig_min_clamped', 0.0) or 0.0), 'nondp_precond_eigA_max': float(st.last_spec_A.get('eig_max_clamped', 0.0) or 0.0)})
        self.last_log = step_logs
        return loss

    @torch.no_grad()
    def dp_begin(self, max_grad_norm: float, *, expected_batch_size: float, noise_multiplier: float) -> None:
        if max_grad_norm <= 0:
            raise ValueError('max_grad_norm must be positive')
        if expected_batch_size <= 0:
            raise ValueError('expected_batch_size must be positive')
        if noise_multiplier <= 0:
            raise ValueError('noise_multiplier must be positive')
        self._ensure_state()
        if self.current_clip is None:
            self.current_clip = float(max_grad_norm)
        if self.raw_hist_max <= 0:
            # Freeze one absolute range for the whole run so histograms remain comparable.
            self.raw_hist_max = 4.0 * float(max_grad_norm)
        uses_slack = self.clipping_method in {'slaclip', 'slaclip_q'}
        clip_threshold = float(self.current_clip if uses_slack else max_grad_norm)
        if uses_slack and self.slaclip_num_slots == 0:
            self.slaclip_num_slots = automatic_num_slots(expected_batch_size, noise_multiplier)
        self._dp_state = _DPAccumState(
            max_grad_norm=clip_threshold,
            expected_batch_size=float(expected_batch_size),
            noise_multiplier=float(noise_multiplier),
        )
        if uses_slack:
            # Lambda is defined by C and K, not by the realized batch.  Set it
            # before accumulation so an empty Poisson batch still performs the
            # noisy Slack-Indicator release and controller update.
            self._dp_state.slack_lambda = clip_threshold / math.sqrt(
                int(self.slaclip_num_slots)
            )
        self.last_log = {}
        self.last_raw_log = {}
        for _, pA, pB in self._pair_iter():
            st = self.state[pA]['prism']
            A, B = self._get_factors(pA, pB)
            A32 = A.float()
            B32 = B.float()
            # The exact rank-r mechanism lives on the smooth fixed-rank
            # manifold.  Check this before accumulating any per-record
            # gradient or making/accounting a release, and cache a stable QR
            # chart used by both the query and the Gaussian sampler.
            QA, RA, rank_stats_A = _full_column_rank_qr(
                A32,
                gram_rcond=self.rcond,
                factor_name='LoRA factor A',
            )
            _QB, RB, rank_stats_B = _full_column_rank_qr(
                B32,
                gram_rcond=self.rcond,
                factor_name='LoRA factor B',
            )
            M = A32.T @ A32
            N = B32.T @ B32
            M_inv = _gram_inverse_from_qr(RA)
            N_inv = _gram_inverse_from_qr(RB)
            st.dp_cache = {
                'A': A32,
                'B': B32,
                'QA': QA,
                'RA': RA,
                'RB': RB,
                'M': M,
                'N': N,
                'M_inv': M_inv,
                'N_inv': N_inv,
                'rank_stats_A': rank_stats_A,
                'rank_stats_B': rank_stats_B,
            }
            st.dp_accum_A = torch.zeros_like(A32)
            st.dp_accum_B = torch.zeros_like(B32)
            if uses_slack and self._dp_state.slack_sum is None:
                self._dp_state.slack_sum = torch.zeros(
                    int(self.slaclip_num_slots),
                    device=A32.device,
                    dtype=torch.float32,
                )
            if self.telemetry_mode == 'research_raw':
                st.dp_raw_accum_A = torch.zeros_like(A32)
                st.dp_raw_accum_B = torch.zeros_like(B32)
            else:
                st.dp_raw_accum_A = None
                st.dp_raw_accum_B = None

    @torch.no_grad()
    def dp_accumulate(self) -> int:
        if self._dp_state is None:
            raise RuntimeError('dp_begin() must be called before dp_accumulate().')
        max_norm = float(self._dp_state.max_grad_norm)
        total_in_micro: Optional[int] = None
        global_norm_sq: Optional[Tensor] = None
        for _, pA, pB in self._pair_iter():
            st = self.state[pA]['prism']
            cache = st.dp_cache
            if cache is None:
                raise RuntimeError('dp_cache missing; did you call dp_begin()?')
            gsA_raw = self._helper.get_grad_sample_tensor(pB)
            gsB_raw = self._helper.get_grad_sample_tensor(pA)
            if (gsA_raw is None) != (gsB_raw is None):
                raise RuntimeError('paired LoRA factors must either both have grad_sample or both be unused')
            if gsA_raw is None:
                continue
            gA = self._helper.move_lora_dim_to_last_grad_sample(gsA_raw, self.lora_r_dim).float()
            gB = self._helper.move_lora_dim_to_last_grad_sample(gsB_raw, self.lora_l_dim).float()
            bs = int(gA.shape[0])
            if int(gB.shape[0]) != bs:
                raise RuntimeError(
                    f'paired LoRA grad_sample batch mismatch: A={bs}, B={int(gB.shape[0])}'
                )
            if total_in_micro is None:
                total_in_micro = bs
            elif total_in_micro != bs:
                raise RuntimeError(
                    f'inconsistent grad_sample batch dimension across LoRA modules: '
                    f'expected {total_in_micro}, got {bs}'
                )
            A = cache['A']
            B = cache['B']
            RA = cache['RA']
            RB = cache['RB']
            QA = cache['QA']
            dA, dB = _dp_isometric_chart_lift(gA, gB, QA, RA, RB)
            n2 = _tangent_fro_norm_sq(dA, dB, A, B)
            if global_norm_sq is None:
                global_norm_sq = n2
            else:
                global_norm_sq = global_norm_sq + n2
        if total_in_micro is None or global_norm_sq is None:
            return 0
        global_norm = torch.sqrt(torch.clamp(global_norm_sq, min=0.0) + 1e-12)
        coef = (max_norm / global_norm).clamp(max=1.0)
        clipped_frac = float((coef < 1.0).float().mean().item())
        coef_mean = float(coef.mean().item())
        coef_min = float(coef.min().item())
        self._dp_state.microbatches += 1
        self._dp_state.clipped_samples += int((coef < 1.0).sum().item())
        self._dp_state.coef_sum += float(coef.sum().item())
        self._dp_state.coef_min = min(self._dp_state.coef_min, coef_min)
        self._dp_state.last_micro_stats = {'micro_clipped_frac': clipped_frac, 'micro_coef_mean': coef_mean, 'micro_coef_min': coef_min, 'micro_global_norm_mean': float(global_norm.mean().item()), 'micro_global_norm_p95': float(torch.quantile(global_norm, 0.95).item())}
        if self.clipping_method in {'slaclip', 'slaclip_q'}:
            slack_vectors, lambda_t = build_slack_vectors(
                global_norm,
                max_norm,
                self.slaclip_num_slots,
            )
            if not math.isclose(
                self._dp_state.slack_lambda,
                lambda_t,
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                raise RuntimeError('SlaClip threshold changed inside one logical batch')
            slack_sum = slack_vectors.sum(dim=0)
            if self._dp_state.slack_sum is None:
                raise RuntimeError('SlaClip slack accumulator is missing')
            self._dp_state.slack_sum.add_(
                slack_sum.to(self._dp_state.slack_sum.device)
            )
        if self.telemetry_mode == 'research_raw':
            self._dp_state.raw_norms.append(global_norm.detach().to('cpu', dtype=torch.float32))
        coef_view = coef.view(-1, 1, 1)
        for _, pA, pB in self._pair_iter():
            st = self.state[pA]['prism']
            cache = st.dp_cache
            if cache is None:
                raise RuntimeError('dp_cache missing; did you call dp_begin()?')
            gsA_raw = self._helper.get_grad_sample_tensor(pB)
            gsB_raw = self._helper.get_grad_sample_tensor(pA)
            if (gsA_raw is None) != (gsB_raw is None):
                raise RuntimeError('paired LoRA factors must either both have grad_sample or both be unused')
            if gsA_raw is None:
                continue
            gA = self._helper.move_lora_dim_to_last_grad_sample(gsA_raw, self.lora_r_dim).float()
            gB = self._helper.move_lora_dim_to_last_grad_sample(gsB_raw, self.lora_l_dim).float()
            if int(gA.shape[0]) != total_in_micro or int(gB.shape[0]) != total_in_micro:
                raise RuntimeError(
                    'grad_sample batch dimension changed between norm computation and accumulation'
                )
            A = cache['A']
            B = cache['B']
            RA = cache['RA']
            RB = cache['RB']
            QA = cache['QA']
            dA, dB = _dp_isometric_chart_lift(gA, gB, QA, RA, RB)
            if self.telemetry_mode == 'research_raw':
                st.dp_raw_accum_A.add_(dA.sum(dim=0))
                st.dp_raw_accum_B.add_(dB.sum(dim=0))
            st.dp_accum_A.add_((dA * coef_view).sum(dim=0))
            st.dp_accum_B.add_((dB * coef_view).sum(dim=0))
            self._helper.clear_grad_sample(pA)
            self._helper.clear_grad_sample(pB)
        self._dp_state.total_samples += int(total_in_micro)
        return int(total_in_micro)

    @torch.no_grad()
    def dp_finalize(self, noise_multiplier: float) -> int:
        if self._dp_state is None:
            raise RuntimeError('dp_begin() must be called before dp_finalize().')
        total = int(self._dp_state.total_samples)
        C = float(self._dp_state.max_grad_norm)
        sigma = float(noise_multiplier)
        if not math.isclose(sigma, float(self._dp_state.noise_multiplier), rel_tol=1e-12):
            raise ValueError('noise_multiplier changed inside one logical batch')
        release_denom = float(self._dp_state.expected_batch_size)
        std = sigma * C / release_denom
        slack_indicator: Optional[Tensor] = None
        raw_slack_indicator: Optional[Tensor] = None
        if self._dp_state.slack_sum is not None and self._dp_state.slack_lambda > 0:
            if self.telemetry_mode == 'research_raw':
                # Exact, non-DP oracle corresponding to the same normalization
                # used by the released/noised Slack Indicator.
                raw_slack_indicator = self._dp_state.slack_sum / (
                    float(self._dp_state.slack_lambda) * release_denom
                )
            slack_noise = self._generate_dp_noise(self._dp_state.slack_sum, sigma * C)
            slack_indicator = (
                self._dp_state.slack_sum + slack_noise
            ) / (float(self._dp_state.slack_lambda) * release_denom)
        signal_norm_sq = 0.0
        raw_unclipped_norm_sq = 0.0
        clipping_bias_norm_sq = 0.0
        noise_norm_sq = 0.0
        raw_unclipped_clipped_inner = 0.0
        raw_clipped_noisy_inner = 0.0
        noisy_gradient_norm_sq = 0.0
        effective_update_norm_sq = 0.0
        update_clip_coef_min = 1.0
        trust_ratio_min = float('inf')
        trust_ratio_max = 0.0
        precond_eigA_min = float('inf')
        precond_eigA_max = 0.0
        precond_eigB_min = float('inf')
        precond_eigB_max = 0.0
        floorA_min = float('inf')
        floorA_max = 0.0
        floorB_min = float('inf')
        floorB_max = 0.0
        factor_relative_gram_eigenvalue_min = float('inf')
        for group, pA, pB in self._pair_iter():
            lr: float = float(group['lr'])
            st = self.state[pA]['prism']
            cache = st.dp_cache
            if cache is None:
                continue
            A = cache['A']
            B = cache['B']
            M_inv = cache['M_inv']
            N_inv = cache['N_inv']
            RA = cache['RA']
            RB = cache['RB']
            QA = cache.get('QA')
            if QA is None or RA is None or RB is None:
                raise RuntimeError('exact full-rank QR cache missing during PRISM DP release')
            factor_relative_gram_eigenvalue_min = min(
                factor_relative_gram_eigenvalue_min,
                float(cache['rank_stats_A']['relative_gram_eigenvalue']),
                float(cache['rank_stats_B']['relative_gram_eigenvalue']),
            )
            gradA = st.dp_accum_A / release_denom
            gradB = st.dp_accum_B / release_denom
            if std > 0:
                U = self._generate_dp_noise(A, 1.0)
                V = self._generate_dp_noise(B, 1.0)
                unit_noise_A, unit_noise_B = _factorized_isotropic_tangent_noise(
                    U,
                    V,
                    QA,
                    RA,
                    RB,
                )
                noise_A = unit_noise_A * std
                noise_B = unit_noise_B * std
            else:
                noise_A = torch.zeros_like(A)
                noise_B = torch.zeros_like(B)
            gradA_noisy = gradA + noise_A
            gradB_noisy = gradB + noise_B
            noisy_n2 = _tangent_fro_norm_sq(gradA_noisy, gradB_noisy, A, B)
            noisy_gradient_norm_sq += float(noisy_n2.item())
            if self.lift_gauge_fix in {'pre_moment', 'both'}:
                gradA_noisy, gradB_noisy = _horizontalize_lift(gradA_noisy, gradB_noisy, A, B, eps=self.gauge_fix_eps)
            noi_n2 = _tangent_fro_norm_sq(noise_A, noise_B, A, B)
            if self.telemetry_mode == 'research_raw':
                sig_n2 = _tangent_fro_norm_sq(gradA, gradB, A, B)
                signal_norm_sq += float(sig_n2.item())
                if st.dp_raw_accum_A is None or st.dp_raw_accum_B is None:
                    raise RuntimeError('raw telemetry accumulator is missing')
                raw_gradA = st.dp_raw_accum_A / release_denom
                raw_gradB = st.dp_raw_accum_B / release_denom
                raw_n2 = _tangent_fro_norm_sq(raw_gradA, raw_gradB, A, B)
                bias_n2 = _tangent_fro_norm_sq(
                    raw_gradA - gradA,
                    raw_gradB - gradB,
                    A,
                    B,
                )
                raw_unclipped_norm_sq += float(raw_n2.item())
                clipping_bias_norm_sq += float(bias_n2.item())
                raw_unclipped_clipped_inner += float(
                    _tangent_fro_inner_product(
                        raw_gradA,
                        raw_gradB,
                        gradA,
                        gradB,
                        A,
                        B,
                    ).item()
                )
                raw_clipped_noisy_inner += float(
                    _tangent_fro_inner_product(
                        gradA,
                        gradB,
                        gradA_noisy,
                        gradB_noisy,
                        A,
                        B,
                    ).item()
                )
            noise_norm_sq += float(noi_n2.item())
            base_floor = float(self.dp_precond_floor_factor) * float(std * std)
            if self.dp_floor_mode == 'none':
                noise_floor_A = 0.0
                noise_floor_B = 0.0
            else:
                noise_floor_A = base_floor
                noise_floor_B = base_floor
            debias_vA = None
            debias_vB = None
            if std > 0:
                m_dim = float(A.shape[0])
                r_dim = float(A.shape[1])
                coefA = max(m_dim - r_dim, 0.0) / max(m_dim, 1.0)
                if self.dp_floor_mode == 'geometry':
                    trN_inv = float(torch.trace(N_inv).item()) / max(r_dim, 1.0)
                    trM_inv = float(torch.trace(M_inv).item()) / max(r_dim, 1.0)
                    noise_floor_A = max(base_floor, float(self.dp_precond_floor_factor) * float(std * std) * float(coefA) * trN_inv)
                    noise_floor_B = max(base_floor, float(self.dp_precond_floor_factor) * float(std * std) * trM_inv)
                if self.dp_debias_second_moment:
                    debias_vA = std * std * float(coefA) * N_inv
                    debias_vB = std * std * M_inv
            floorA_min = min(floorA_min, float(noise_floor_A))
            floorA_max = max(floorA_max, float(noise_floor_A))
            floorB_min = min(floorB_min, float(noise_floor_B))
            floorB_max = max(floorB_max, float(noise_floor_B))
            uA, uB = self._adaptive_direction(st, gradA_noisy, gradB_noisy, noise_floor=base_floor, noise_floor_A=noise_floor_A, noise_floor_B=noise_floor_B, debias_vA=debias_vA, debias_vB=debias_vB)
            precond_eigA_min = min(precond_eigA_min, float(st.last_spec_A.get('eig_min_clamped', 0.0) or 0.0))
            precond_eigA_max = max(precond_eigA_max, float(st.last_spec_A.get('eig_max_clamped', 0.0) or 0.0))
            precond_eigB_min = min(precond_eigB_min, float(st.last_spec_B.get('eig_min_clamped', 0.0) or 0.0))
            precond_eigB_max = max(precond_eigB_max, float(st.last_spec_B.get('eig_max_clamped', 0.0) or 0.0))
            if self.weight_decay > 0:
                uA = uA + self.weight_decay * A
                uB = uB + self.weight_decay * B
            dA_dir = -uA
            dB_dir = -uB
            if self.lift_gauge_fix in {'pre_retract', 'both'}:
                dA_dir, dB_dir = _horizontalize_lift(dA_dir, dB_dir, A, B, eps=self.gauge_fix_eps)
            trust = self._compute_trust_ratio(A, B, dA_dir, dB_dir)
            dA_dir = dA_dir * trust
            dB_dir = dB_dir * trust
            trust_ratio_min = min(trust_ratio_min, float(trust))
            trust_ratio_max = max(trust_ratio_max, float(trust))
            dA_dir, dB_dir, upd_coef = self._maybe_clip_update(dA_dir, dB_dir, A, B)
            update_clip_coef_min = min(update_clip_coef_min, float(upd_coef))
            r = A.shape[1]
            A_new, B_new = _retract_rank_r(A, B, dA_dir, dB_dir, eta=lr, r=r, align_to=(A, B))
            upd_n2 = _factorized_delta_fro_norm_sq(A_new, B_new, A, B)
            effective_update_norm_sq += float(upd_n2.item())
            self._set_factors(pA, pB, A_new.to(pB.data.dtype), B_new.to(pA.data.dtype))
            st.dp_cache = None
            st.dp_accum_A = None
            st.dp_accum_B = None
            st.dp_raw_accum_A = None
            st.dp_raw_accum_B = None
        clip_frac = float(self._dp_state.clipped_samples) / max(1, total)
        coef_mean = float(self._dp_state.coef_sum) / max(1, total)
        coef_min = float(self._dp_state.coef_min)
        noi = math.sqrt(max(noise_norm_sq, 0.0))
        noisy_grad_norm = math.sqrt(max(noisy_gradient_norm_sq, 0.0))
        effective_update_norm = math.sqrt(max(effective_update_norm_sq, 0.0))
        c_next = C
        threshold_update = None
        if self.clipping_method == 'slaclip' and slack_indicator is not None:
            threshold_update = full_slaclip_threshold_update(
                C,
                slack_indicator,
                eta=self.slaclip_eta,
                c_min=self.slaclip_c_min,
                c_max=self.slaclip_c_max,
                target_non_small_clip_fraction=(
                    self.slaclip_target_non_small_clip_fraction
                ),
            )
            c_next = threshold_update.next_clip
            self.current_clip = float(c_next)
        elif self.clipping_method == 'slaclip_q' and slack_indicator is not None:
            threshold_update = slaclip_q_threshold_update(
                C,
                slack_indicator,
                eta=self.slaclip_eta,
                target_clip_fraction=self.slaclip_target_clip_fraction,
                c_min=self.slaclip_c_min,
                c_max=self.slaclip_c_max,
            )
            c_next = threshold_update.next_clip
            self.current_clip = float(c_next)
        elif self.clipping_method == 'baseline':
            self.current_clip = C

        self.last_log = {
            'telemetry_mode': self.telemetry_mode,
            'dp_expected_batch_size': float(release_denom),
            'dp_noise_multiplier': float(noise_multiplier),
            'dp_clip_threshold': float(C),
            'dp_next_clip_threshold': float(c_next),
            'dp_std_per_factor': float(std),
            'dp_tangent_noise_sampler': 'qr_exact_full_rank',
            'dp_tangent_query_chart': 'qr_asymmetric_isometric',
            'dp_factor_rank_rcond': float(self.rcond),
            'dp_factor_relative_gram_eigenvalue_min': float(
                0.0
                if factor_relative_gram_eigenvalue_min == float('inf')
                else factor_relative_gram_eigenvalue_min
            ),
            # Both are post-processing of the DP release/model update.  The
            # realized noise itself is intentionally kept out of this log.
            'dp_noisy_tangent_gradient_norm': float(noisy_grad_norm),
            'dp_factor_product_update_norm': float(effective_update_norm),
            'dp_floor_mode_id': float({'none': 0, 'scalar': 1, 'geometry': 2}.get(self.dp_floor_mode, -1)),
            'dp_precond_update_mode_id': float({'current': 0, 'delayed': 1}.get(self.precond_update_mode, -1)),
            'dp_lift_gauge_fix_id': float({'none': 0, 'pre_moment': 1, 'pre_retract': 2, 'both': 3}.get(self.lift_gauge_fix, -1)),
            'dp_floorA_min': float(0.0 if floorA_min == float('inf') else floorA_min),
            'dp_floorA_max': float(floorA_max),
            'dp_floorB_min': float(0.0 if floorB_min == float('inf') else floorB_min),
            'dp_floorB_max': float(floorB_max),
            'dp_update_clip_coef_min': float(update_clip_coef_min),
            'dp_trust_ratio_min': float(0.0 if trust_ratio_min == float('inf') else trust_ratio_min),
            'dp_trust_ratio_max': float(trust_ratio_max),
            'dp_precond_eigA_min': float(0.0 if precond_eigA_min == float('inf') else precond_eigA_min),
            'dp_precond_eigA_max': float(precond_eigA_max),
            'dp_precond_eigB_min': float(0.0 if precond_eigB_min == float('inf') else precond_eigB_min),
            'dp_precond_eigB_max': float(precond_eigB_max),
        }
        if slack_indicator is not None:
            slack_cpu = slack_indicator.detach().to('cpu', dtype=torch.float32)
            self.last_log['slaclip_num_slots'] = int(self.slaclip_num_slots)
            self.last_log['slaclip_controller'] = str(self.clipping_method)
            self.last_log['slack_indicator'] = [float(x) for x in slack_cpu.tolist()]
            self.last_log['slack_unclipped_proxy'] = float(slack_cpu[0].item())
            self.last_log['slack_clipped_proxy'] = float(1.0 - slack_cpu[0].item())
            self.last_log['slack_indicator_noise_std'] = slack_indicator_noise_std(
                sigma,
                self.slaclip_num_slots,
                release_denom,
            )
        if threshold_update is not None:
            self.last_log['slaclip_target_unclipped_proxy'] = float(
                threshold_update.target_unclipped_proxy
            )
            self.last_log[
                'slaclip_target_unclipped_proxy_preprojection'
            ] = float(threshold_update.target_unclipped_proxy_preprojection)
            self.last_log['slaclip_target_clipped_proxy'] = float(
                1.0 - threshold_update.target_unclipped_proxy
            )
            self.last_log['slaclip_observed_unclipped_proxy'] = float(
                threshold_update.observed_unclipped_proxy
            )
            self.last_log['slaclip_controller_error'] = float(
                threshold_update.controller_error
            )
            self.last_log['slaclip_c_next_unbounded'] = float(
                threshold_update.unbounded_next_clip
            )
            self.last_log['slaclip_c_hit_min'] = bool(threshold_update.hit_lower_bound)
            self.last_log['slaclip_c_hit_max'] = bool(threshold_update.hit_upper_bound)
            self.last_log['slaclip_eta'] = float(self.slaclip_eta)
            self.last_log['slaclip_c_min'] = float(self.slaclip_c_min)
            self.last_log['slaclip_c_max'] = float(self.slaclip_c_max)
            if self.clipping_method == 'slaclip':
                self.last_log['slaclip_gamma_t'] = float(
                    threshold_update.target_unclipped_proxy
                )
                self.last_log[
                    'slaclip_target_non_small_clip_fraction'
                ] = float(self.slaclip_target_non_small_clip_fraction)
                self.last_log['slaclip_small_gradient_proxy_noisy'] = float(
                    threshold_update.small_gradient_proxy_noisy
                )
                self.last_log['slaclip_remaining_mass_proxy_noisy'] = float(
                    threshold_update.remaining_mass_proxy_noisy
                )
                # Deprecated telemetry alias retained for old campaign tools.
                self.last_log['slaclip_beta'] = float(self.slaclip_beta)
            else:
                self.last_log['slaclip_target_clip_fraction'] = float(
                    self.slaclip_target_clip_fraction
                )

        if self.telemetry_mode == 'research_raw':
            sig = math.sqrt(max(signal_norm_sq, 0.0))
            raw_unclipped = math.sqrt(max(raw_unclipped_norm_sq, 0.0))
            clipping_bias = math.sqrt(max(clipping_bias_norm_sq, 0.0))
            unclipped_clipped_cosine = _cosine_from_inner(
                raw_unclipped_clipped_inner,
                raw_unclipped_norm_sq,
                signal_norm_sq,
            )
            clipped_noisy_cosine = _cosine_from_inner(
                raw_clipped_noisy_inner,
                signal_norm_sq,
                noisy_gradient_norm_sq,
            )
            raw: Dict[str, Any] = {
                'NON_PRIVATE_TELEMETRY': True,
                'raw_realized_batch_size': int(total),
                'raw_clip_fraction': float(clip_frac),
                'raw_clip_coefficient_mean': float(coef_mean),
                'raw_clip_coefficient_min': float(coef_min),
                'raw_clipped_signal_norm': float(sig),
                'raw_unclipped_signal_norm': float(raw_unclipped),
                'raw_clipping_bias_norm': float(clipping_bias),
                'raw_realized_noise_norm': float(noi),
                'raw_signal_to_noise_ratio': float(sig / (noi + 1e-12)),
                'raw_unclipped_clipped_cosine': float(unclipped_clipped_cosine),
                'raw_clipped_noisy_cosine': float(clipped_noisy_cosine),
                'raw_clipping_bias_to_noise_ratio': float(
                    clipping_bias / (noi + 1e-12)
                ),
                'raw_bias_noise_squared_error_proxy': float(
                    clipping_bias_norm_sq + noise_norm_sq
                ),
            }
            if raw_slack_indicator is not None and slack_indicator is not None:
                raw_slack_cpu = raw_slack_indicator.detach().to(
                    'cpu', dtype=torch.float32
                )
                residual = (
                    slack_indicator.detach().to('cpu', dtype=torch.float32)
                    - raw_slack_cpu
                )
                raw['raw_slack_indicator'] = [
                    float(value) for value in raw_slack_cpu.tolist()
                ]
                raw['raw_slack_indicator_noise_residual'] = [
                    float(value) for value in residual.tolist()
                ]
                raw['raw_slack_indicator_noise_residual_l2'] = float(
                    torch.linalg.vector_norm(residual).item()
                )
                raw['raw_slack_indicator_noise_residual_rmse'] = float(
                    torch.sqrt(torch.mean(residual.square())).item()
                )
                raw['raw_slack_indicator_noise_residual_first_coordinate'] = float(
                    residual[0].item()
                )
            if self._dp_state.raw_norms:
                norms = torch.cat(self._dp_state.raw_norms).float()
                hist_max = float(self.raw_hist_max if self.raw_hist_max > 0 else 4.0 * C)
                in_range = norms[norms <= hist_max]
                counts = torch.histc(in_range, bins=self.raw_hist_bins, min=0.0, max=hist_max)
                edges = torch.linspace(0.0, hist_max, self.raw_hist_bins + 1)
                raw.update({
                    'raw_global_norm_mean': float(norms.mean().item()),
                    'raw_global_norm_std': float(norms.std(unbiased=False).item()),
                    'raw_global_norm_min': float(norms.min().item()),
                    'raw_global_norm_max': float(norms.max().item()),
                    'raw_global_norm_quantiles': {
                        str(q): float(torch.quantile(norms, q).item())
                        for q in (0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99)
                    },
                    'raw_global_norm_hist_counts': [int(x) for x in counts.tolist()],
                    'raw_global_norm_hist_edges': [float(x) for x in edges.tolist()],
                    'raw_global_norm_hist_overflow': int((norms > hist_max).sum().item()),
                })
                # Counterfactual full-SlaClip calibration for fixed-C scans.
                # This is exact, access-controlled research telemetry only: it
                # does not add slack coordinates to the DP query, alter the
                # clipped gradient, consume controller output, or change the
                # Gaussian release.  It mirrors the implemented full-SlaClip
                # small-gradient proxy so a fixed run's whole-batch clipping
                # rate is not incorrectly reused as the conditional rho.
                reference_slots = int(self.slaclip_num_slots)
                if reference_slots > 0:
                    reference_slack, reference_lambda = build_slack_vectors(
                        norms,
                        float(C),
                        reference_slots,
                    )
                    reference_indicator_last = float(
                        (
                            reference_slack[:, -1].sum()
                            / (
                                float(reference_lambda)
                                * float(release_denom)
                            )
                        ).item()
                    )
                    reference_small_proxy = reference_indicator_last / (
                        float(C) + 1e-6
                    )
                    reference_remaining_proxy = 1.0 - reference_small_proxy
                    reference_valid = bool(
                        math.isfinite(reference_remaining_proxy)
                        and reference_remaining_proxy > 1e-12
                    )
                    reference_conditional = (
                        float(clip_frac) / reference_remaining_proxy
                        if reference_valid
                        else None
                    )
                    raw.update({
                        'raw_reference_slaclip_num_slots': reference_slots,
                        'raw_reference_expected_batch_size_normalization': (
                            float(release_denom)
                        ),
                        'raw_reference_slack_indicator_last': (
                            reference_indicator_last
                        ),
                        'raw_reference_small_gradient_proxy': (
                            reference_small_proxy
                        ),
                        'raw_reference_remaining_mass_proxy': (
                            reference_remaining_proxy
                        ),
                        'raw_reference_conditional_clip_fraction': (
                            reference_conditional
                        ),
                        'raw_reference_conditional_clip_fraction_valid': (
                            reference_valid
                        ),
                    })
            self.last_raw_log = raw
        self._dp_state = None
        return total

def get_paired_lora_parameters(model: torch.nn.Module, *, adapter_name: str='default', require_grad_only: bool=True) -> List[torch.nn.Parameter]:
    params: List[torch.nn.Parameter] = []
    for _, mod in model.named_modules():
        if not (hasattr(mod, 'lora_A') and hasattr(mod, 'lora_B')):
            continue
        if adapter_name not in getattr(mod, 'lora_A') or adapter_name not in getattr(mod, 'lora_B'):
            continue
        lora_A = mod.lora_A[adapter_name]
        lora_B = mod.lora_B[adapter_name]
        if not (hasattr(lora_A, 'weight') and hasattr(lora_B, 'weight')):
            continue
        pA = lora_A.weight
        pB = lora_B.weight
        if require_grad_only and (not pA.requires_grad or not pB.requires_grad):
            continue
        params.extend([pA, pB])
    if len(params) == 0:
        raise ValueError('No LoRA parameters found. Check adapter_name/target_modules.')
    if len(params) % 2 != 0:
        raise ValueError('Internal error: expected an even number of LoRA parameters.')
    return params

@torch.no_grad()
def balanced_full_rank_lora_init_peft_model(model: torch.nn.Module, *, adapter_name: str='default', factor_scale: float=0.02, seed: Optional[int]=None, verbose: bool=True) -> Dict[str, float]:
    if factor_scale <= 0:
        raise ValueError('factor_scale must be positive')
    gen = None
    if seed is not None:
        gen = torch.Generator(device='cpu')
        gen.manual_seed(int(seed))
    touched = 0
    z_energy = 0.0
    for mod_name, mod in model.named_modules():
        if not (hasattr(mod, 'lora_A') and hasattr(mod, 'lora_B')):
            continue
        if adapter_name not in getattr(mod, 'lora_A') or adapter_name not in getattr(mod, 'lora_B'):
            continue
        lora_A = mod.lora_A[adapter_name]
        lora_B = mod.lora_B[adapter_name]
        if not (hasattr(lora_A, 'weight') and hasattr(lora_B, 'weight')):
            continue
        # Multimodal backbones such as Gemma 3 may receive suffix-matched LoRA
        # modules inside their frozen vision tower.  PRISM is text-only here;
        # never alter a frozen base weight during spectral residual init.
        if not (lora_A.weight.requires_grad and lora_B.weight.requires_grad):
            continue
        r = int(lora_A.weight.shape[0])
        in_features = int(lora_A.weight.shape[1])
        out_features = int(lora_B.weight.shape[0])
        if lora_B.weight.shape[1] != r:
            raise ValueError(f'LoRA shape mismatch in {mod_name}')
        G_in = torch.randn((in_features, r), generator=gen, dtype=torch.float32)
        G_out = torch.randn((out_features, r), generator=gen, dtype=torch.float32)
        Q_in, _ = torch.linalg.qr(G_in, mode='reduced')
        Q_out, _ = torch.linalg.qr(G_out, mode='reduced')
        B_T = (float(factor_scale) * Q_in.T).to(device=lora_A.weight.device, dtype=lora_A.weight.dtype)
        A = (float(factor_scale) * Q_out).to(device=lora_B.weight.device, dtype=lora_B.weight.dtype)
        lora_A.weight.data.copy_(B_T)
        lora_B.weight.data.copy_(A)
        z_energy += float(r) * float(factor_scale) ** 4
        touched += 1
        if verbose and touched <= 3:
            print(f'[FullRankInit] {mod_name}: r={r} factor_scale={factor_scale:g}')
    stats = {'full_rank_init_modules': float(touched), 'full_rank_init_factor_scale': float(factor_scale), 'full_rank_init_z_energy_unscaled': float(z_energy)}
    if verbose:
        print(f'[FullRankInit] touched_modules={touched}, factor_scale={factor_scale:g}')
    return stats

@torch.no_grad()
def randomized_svd(W: Tensor, rank: int, oversample: int=8, n_iter: int=2) -> Tuple[Tensor, Tensor, Tensor]:
    if W.ndim != 2:
        raise ValueError('W must be 2D')
    m, n = W.shape
    r = int(rank)
    l = min(n, r + int(oversample))
    Wf = W.float()
    Omega = torch.randn((n, l), device=W.device, dtype=Wf.dtype)
    Y = Wf @ Omega
    for _ in range(max(0, int(n_iter))):
        Y = Wf @ (Wf.T @ Y)
    Q, _ = torch.linalg.qr(Y, mode='reduced')
    B = Q.T @ Wf
    Ub, S, Vh = torch.linalg.svd(B, full_matrices=False)
    U = Q @ Ub
    U_r = U[:, :r]
    S_r = S[:r]
    Vh_r = Vh[:r, :]
    return (U_r.to(W.dtype), S_r.to(W.dtype), Vh_r.to(W.dtype))

@torch.no_grad()
def spectral_init_peft_model(model: torch.nn.Module, *, adapter_name: str='default', svd_rank: Optional[int]=None, svd_oversample: int=8, svd_n_iter: int=2, svd_device: str='cpu', verbose: bool=True, store_cache: bool=True, cache_dtype: torch.dtype=torch.float32) -> Dict[str, float]:
    cache: Dict[str, Dict[str, Tensor]] = {}
    touched = 0
    total_energy = 0.0
    kept_energy = 0.0
    for mod_name, mod in model.named_modules():
        if not (hasattr(mod, 'lora_A') and hasattr(mod, 'lora_B') and hasattr(mod, 'weight')):
            continue
        if adapter_name not in getattr(mod, 'lora_A'):
            continue
        lora_A = mod.lora_A[adapter_name]
        lora_B = mod.lora_B[adapter_name]
        if not (hasattr(lora_A, 'weight') and hasattr(lora_B, 'weight')):
            continue
        # Suffix matching can inject adapters into a multimodal vision tower.
        # Frozen adapters are outside the text-only mechanism and their base
        # weights must not be rewritten by the residual initialization.
        if not (lora_A.weight.requires_grad and lora_B.weight.requires_grad):
            continue
        r = int(lora_A.weight.shape[0])
        if svd_rank is not None:
            r = min(r, int(svd_rank))
        scaling = getattr(mod, 'scaling', None)
        if isinstance(scaling, dict):
            scale = float(scaling.get(adapter_name, 1.0))
        else:
            scale = float(scaling) if scaling is not None else 1.0
        if scale <= 0:
            scale = 1.0
        W = mod.weight.data
        if svd_device == 'cpu':
            W_svd = W.detach().to('cpu', dtype=torch.float32)
            U, S, Vh = randomized_svd(W_svd, rank=r, oversample=svd_oversample, n_iter=svd_n_iter)
            U = U.to(W.device, dtype=torch.float32)
            S = S.to(W.device, dtype=torch.float32)
            Vh = Vh.to(W.device, dtype=torch.float32)
        else:
            W_svd = W.detach().to(dtype=torch.float32)
            U, S, Vh = randomized_svd(W_svd, rank=r, oversample=svd_oversample, n_iter=svd_n_iter)
            U = U.to(dtype=torch.float32)
            S = S.to(dtype=torch.float32)
            Vh = Vh.to(dtype=torch.float32)
        Wr = U * S.unsqueeze(0) @ Vh
        total_energy += float((W.float() ** 2).sum().item())
        kept_energy += float((Wr ** 2).sum().item())
        mod.weight.data = (W.float() - Wr).to(W.dtype)
        S_scaled = S / float(scale)
        s_sqrt = torch.sqrt(torch.clamp(S_scaled, min=0.0))
        A_init = (U * s_sqrt.unsqueeze(0)).to(lora_B.weight.dtype)
        B_init_T = (Vh * s_sqrt.unsqueeze(1)).to(lora_A.weight.dtype)
        if lora_B.weight.data.shape != A_init.shape:
            raise ValueError(f'spectral residual shape mismatch for {mod_name}: lora_B {tuple(lora_B.weight.shape)} vs A_init {tuple(A_init.shape)}')
        if lora_A.weight.data.shape != B_init_T.shape:
            raise ValueError(f'spectral residual shape mismatch for {mod_name}: lora_A {tuple(lora_A.weight.shape)} vs B_init_T {tuple(B_init_T.shape)}')
        lora_B.weight.data.copy_(A_init)
        lora_A.weight.data.copy_(B_init_T)
        if store_cache:
            cache[mod_name] = {'lora_B_init': A_init.detach().to('cpu', dtype=cache_dtype).contiguous(), 'lora_A_init': B_init_T.detach().to('cpu', dtype=cache_dtype).contiguous(), 'scale': float(scale)}
        touched += 1
        if verbose and touched <= 3:
            print(f'[spectral residual] init {mod_name}: r={r} scale={scale:.6g} ||Wr||F={Wr.norm().item():.4g}')
    stats = {'spectral_modules': float(touched), 'spectral_energy_total': float(total_energy), 'spectral_energy_kept': float(kept_energy), 'spectral_energy_ratio': float(kept_energy / (total_energy + 1e-12))}
    if verbose:
        print(f"[spectral residual] touched_modules={touched} kept_energy_ratio={stats['spectral_energy_ratio']:.4f}")
    if store_cache:
        setattr(model, '_spectral_cache', cache)
        setattr(model, '_spectral_adapter_name', adapter_name)
    return stats

@torch.no_grad()
def spectral_rebase_adapter_inplace(model: torch.nn.Module, *, adapter_name: str='default', rank_multiplier: int=2, verbose: bool=True) -> Dict[str, float]:
    cache = getattr(model, '_spectral_cache', None)
    if not isinstance(cache, dict) or len(cache) == 0:
        raise ValueError('No spectral residual cache found on model. Run spectral_init_peft_model(..., store_cache=True) first.')
    if rank_multiplier < 1:
        raise ValueError('rank_multiplier must be >= 1')
    if rank_multiplier == 1:
        return {'spectral_rebase': 0.0, 'spectral_rebase_modules': 0.0}
    touched = 0
    new_r_val: Optional[int] = None
    scale_val: Optional[float] = None
    for mod_name, mod in model.named_modules():
        if mod_name not in cache:
            continue
        if not (hasattr(mod, 'lora_A') and hasattr(mod, 'lora_B')):
            continue
        if adapter_name not in getattr(mod, 'lora_A'):
            continue
        lora_A = mod.lora_A[adapter_name]
        lora_B = mod.lora_B[adapter_name]
        if not (hasattr(lora_A, 'weight') and hasattr(lora_B, 'weight')):
            continue
        A0_cpu = cache[mod_name]['lora_B_init']
        B0T_cpu = cache[mod_name]['lora_A_init']
        scale = float(cache[mod_name].get('scale', 1.0))
        B_curr_T = lora_A.weight.data
        A_curr = lora_B.weight.data
        r = int(B_curr_T.shape[0])
        if A_curr.shape[1] != r:
            raise ValueError(f'Unexpected LoRA shapes for {mod_name}: lora_A {tuple(B_curr_T.shape)}, lora_B {tuple(A_curr.shape)}')
        A0 = A0_cpu.to(device=A_curr.device, dtype=A_curr.dtype)
        B0T = B0T_cpu.to(device=B_curr_T.device, dtype=B_curr_T.dtype)
        if A0.shape != A_curr.shape or B0T.shape != B_curr_T.shape:
            raise ValueError(f'spectral residual cache shape mismatch for {mod_name}: A0 {tuple(A0.shape)} vs A {tuple(A_curr.shape)}, B0T {tuple(B0T.shape)} vs B {tuple(B_curr_T.shape)}')
        new_r = int(r * rank_multiplier)
        if new_r_val is None:
            new_r_val = new_r
            scale_val = scale
        elif new_r != new_r_val:
            raise ValueError('Rank mismatch across modules during spectral residual rebase. This helper expects a uniform LoRA rank.')
        if rank_multiplier != 2:
            raise NotImplementedError('Only rank_multiplier=2 is currently supported.')
        B_new_T = torch.cat([B_curr_T, B0T], dim=0)
        A_new = torch.cat([A_curr, -A0], dim=1)
        in_features = int(getattr(lora_A, 'in_features'))
        out_features = int(getattr(lora_B, 'out_features'))
        new_lora_A = torch.nn.Linear(in_features, new_r, bias=False).to(device=B_curr_T.device, dtype=B_curr_T.dtype)
        new_lora_B = torch.nn.Linear(new_r, out_features, bias=False).to(device=A_curr.device, dtype=A_curr.dtype)
        new_lora_A.weight.data.copy_(B_new_T)
        new_lora_B.weight.data.copy_(A_new)
        mod.lora_A[adapter_name] = new_lora_A
        mod.lora_B[adapter_name] = new_lora_B
        if hasattr(mod, 'r'):
            if isinstance(mod.r, dict):
                mod.r[adapter_name] = new_r
            else:
                mod.r = new_r
        if hasattr(mod, 'lora_alpha'):
            alpha_new = float(scale) * float(new_r)
            if abs(alpha_new - round(alpha_new)) < 1e-06:
                alpha_new = int(round(alpha_new))
            if isinstance(mod.lora_alpha, dict):
                mod.lora_alpha[adapter_name] = alpha_new
            else:
                mod.lora_alpha = alpha_new
        if hasattr(mod, 'scaling'):
            if isinstance(mod.scaling, dict):
                mod.scaling[adapter_name] = float(scale)
            else:
                mod.scaling = float(scale)
        touched += 1
    cfg = getattr(model, 'peft_config', None)
    if isinstance(cfg, dict) and adapter_name in cfg:
        cfg_obj = cfg[adapter_name]
        if hasattr(cfg_obj, 'rank_pattern'):
            cfg_obj.rank_pattern = {}
        if hasattr(cfg_obj, 'alpha_pattern'):
            cfg_obj.alpha_pattern = {}
        if hasattr(cfg_obj, 'r') and new_r_val is not None:
            cfg_obj.r = int(new_r_val)
        if hasattr(cfg_obj, 'lora_alpha') and new_r_val is not None and (scale_val is not None):
            alpha_new = float(scale_val) * float(new_r_val)
            if abs(alpha_new - round(alpha_new)) < 1e-06:
                alpha_new = int(round(alpha_new))
            cfg_obj.lora_alpha = alpha_new
    if verbose:
        print(f'[spectral residual] rebased adapter for saving/loading: touched_modules={touched}, new_rank={new_r_val}, scale={scale_val}')
    return {'spectral_rebase_modules': float(touched), 'spectral_rebase_new_rank': float(0.0 if new_r_val is None else new_r_val), 'spectral_rebase_scale': float(1.0 if scale_val is None else scale_val)}
