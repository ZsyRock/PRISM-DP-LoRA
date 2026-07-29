"""Pure SlaClip helpers used by the PRISM tangent-space optimizer.

The slack coordinates are part of the same Gaussian release as the clipped
tangent update.  They must never be computed as a second, unaccounted query.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import torch
from torch import Tensor


Z_0995 = 2.5758293035489004
SMALL_BATCH_SLOT_THRESHOLD = 128.0
SMALL_BATCH_NUM_SLOTS = 15


@dataclass(frozen=True)
class ThresholdUpdate:
    """Diagnostics for one post-processing-only clipping-threshold update."""

    next_clip: float
    unbounded_next_clip: float
    target_unclipped_proxy: float
    hit_lower_bound: bool
    hit_upper_bound: bool


def paper_bound_num_slots(
    expected_batch_size: float,
    noise_multiplier: float,
    *,
    confidence_z: float = Z_0995,
) -> int:
    """Choose the largest integer K satisfying the paper's monotonicity bound."""
    if expected_batch_size <= 0:
        raise ValueError("expected_batch_size must be positive")
    if noise_multiplier <= 0:
        raise ValueError("noise_multiplier must be positive")
    if confidence_z <= 0:
        raise ValueError("confidence_z must be positive")
    upper = (float(expected_batch_size) / (2.0 * confidence_z * float(noise_multiplier))) ** (2.0 / 3.0)
    return max(1, int(math.floor(upper)))


def automatic_num_slots(
    expected_batch_size: float,
    noise_multiplier: float,
    *,
    confidence_z: float = Z_0995,
) -> int:
    """Select K using the journal-extension experimental policy.

    Expected batches below 128 use K=15.  Batches of 128 or more use the
    original paper monotonicity-bound formula.  The strict boundary keeps the
    policy faithful to the declared ``B < 128`` rule.

    K=15 can exceed the paper's high-probability monotonicity bound for small
    batches.  It remains a valid joint Gaussian DP query, but must be reported
    as the small-batch journal policy rather than a paper-bound choice.
    """
    # Run the shared validation even on the fixed-K branch.
    paper_k = paper_bound_num_slots(
        expected_batch_size,
        noise_multiplier,
        confidence_z=confidence_z,
    )
    if float(expected_batch_size) < SMALL_BATCH_SLOT_THRESHOLD:
        return SMALL_BATCH_NUM_SLOTS
    return paper_k


def build_slack_vectors(norms: Tensor, clip_threshold: float, num_slots: int) -> Tuple[Tensor, float]:
    """Encode SlaClip slack for each norm using the official K-slot construction.

    For every row, the squared norm of ``[clipped_gradient, slack_vector]``
    remains at most ``clip_threshold**2`` when ``norms`` is the norm of the
    corresponding unclipped gradient.
    """
    if norms.ndim != 1:
        raise ValueError(f"norms must be one-dimensional, got shape={tuple(norms.shape)}")
    if clip_threshold <= 0:
        raise ValueError("clip_threshold must be positive")
    if num_slots <= 0:
        raise ValueError("num_slots must be positive")

    k = int(num_slots)
    c_t = float(clip_threshold)
    lambda_t = c_t / math.sqrt(k)
    slack_amount = torch.clamp(c_t - norms.float(), min=0.0)
    total_slack = slack_amount * math.sqrt(k)

    full_slots = torch.floor(total_slack / lambda_t).to(torch.int64)
    full_slots = torch.clamp(full_slots, min=0, max=k)
    remainder = total_slack - full_slots.to(total_slack.dtype) * lambda_t
    remainder = torch.where(full_slots >= k, torch.zeros_like(remainder), remainder)

    slot_index = torch.arange(k, device=norms.device).view(1, k)
    vectors = (slot_index < full_slots.view(-1, 1)).to(torch.float32) * lambda_t
    has_remainder = full_slots < k
    if has_remainder.any():
        row_index = torch.arange(norms.shape[0], device=norms.device)[has_remainder]
        vectors[row_index, full_slots[has_remainder]] = remainder[has_remainder]
    return vectors, float(lambda_t)


def slack_indicator_noise_std(
    noise_multiplier: float,
    num_slots: int,
    expected_batch_size: float,
) -> float:
    """Return the public per-coordinate noise s.d. of the normalized indicator.

    The unnormalized joint Gaussian release uses standard deviation ``sigma*C``.
    Dividing a slack coordinate by ``lambda*B`` with ``lambda=C/sqrt(K)``
    cancels ``C``, leaving ``sigma*sqrt(K)/B``.
    """
    if noise_multiplier <= 0:
        raise ValueError("noise_multiplier must be positive")
    if num_slots <= 0:
        raise ValueError("num_slots must be positive")
    if expected_batch_size <= 0:
        raise ValueError("expected_batch_size must be positive")
    return float(noise_multiplier) * math.sqrt(int(num_slots)) / float(expected_batch_size)


def _bounded_exponential_update(
    clip_threshold: float,
    *,
    eta: float,
    error: float,
    target_unclipped_proxy: float,
    c_min: float,
    c_max: float,
) -> ThresholdUpdate:
    # A DP Gaussian coordinate is unbounded.  Clamping the exponent prevents an
    # overflow without inspecting any non-private quantity and is therefore
    # ordinary post-processing of the same release.
    exponent = max(-50.0, min(50.0, float(eta) * float(error)))
    unbounded = float(clip_threshold) * math.exp(exponent)
    bounded = max(float(c_min), min(float(c_max), unbounded))
    return ThresholdUpdate(
        next_clip=float(bounded),
        unbounded_next_clip=float(unbounded),
        target_unclipped_proxy=float(target_unclipped_proxy),
        hit_lower_bound=bool(unbounded < float(c_min)),
        hit_upper_bound=bool(unbounded > float(c_max)),
    )


def full_slaclip_threshold_update(
    clip_threshold: float,
    slack_indicator: Tensor,
    *,
    eta: float,
    beta: float,
    c_min: float,
    c_max: float,
) -> ThresholdUpdate:
    """Apply the camera-ready full-SlaClip feedback controller.

    The first indicator coordinate is a noisy, bin-averaged surrogate for the
    *unclipped* CDF near ``C_t``.  Full SlaClip constructs a dynamic target from
    the last (near-zero) coordinate; it does not expose a fixed clipping-rate
    target.
    """
    if clip_threshold <= 0:
        raise ValueError("clip_threshold must be positive")
    if slack_indicator.ndim != 1 or slack_indicator.numel() == 0:
        raise ValueError("slack_indicator must be a non-empty vector")
    if eta < 0:
        raise ValueError("eta must be non-negative")
    if not 0.0 <= beta <= 1.0:
        raise ValueError("beta must be in [0, 1]")
    if c_min <= 0 or c_max < c_min:
        raise ValueError("require 0 < c_min <= c_max")

    c_t = float(clip_threshold)
    near_threshold = float(slack_indicator[0].item())
    near_zero = float(slack_indicator[-1].item())
    gamma_t = 1.0 - float(beta) * (1.0 - near_zero / (c_t + 1e-6))
    gamma_t = max(0.0, min(1.0, gamma_t))
    return _bounded_exponential_update(
        c_t,
        eta=eta,
        error=gamma_t - near_threshold,
        target_unclipped_proxy=gamma_t,
        c_min=c_min,
        c_max=c_max,
    )


def slaclip_q_threshold_update(
    clip_threshold: float,
    slack_indicator: Tensor,
    *,
    eta: float,
    target_clip_fraction: float,
    c_min: float,
    c_max: float,
) -> ThresholdUpdate:
    """Apply SlaClip-Q with an explicit requested clipped fraction.

    SlaClip-Q's paper parameter ``gamma`` is the desired *unclipped* CDF level.
    This interface accepts the complement because it prevents the common and
    consequential inversion error when an experiment is specified as, e.g.,
    "99% clipped".  The controller tracks a noisy, smoothed CDF proxy, so the
    requested fraction is not a guarantee about the exact raw clipping rate.
    """
    if clip_threshold <= 0:
        raise ValueError("clip_threshold must be positive")
    if slack_indicator.ndim != 1 or slack_indicator.numel() == 0:
        raise ValueError("slack_indicator must be a non-empty vector")
    if eta < 0:
        raise ValueError("eta must be non-negative")
    if not 0.0 <= target_clip_fraction <= 1.0:
        raise ValueError("target_clip_fraction must be in [0, 1]")
    if c_min <= 0 or c_max < c_min:
        raise ValueError("require 0 < c_min <= c_max")

    target_unclipped = 1.0 - float(target_clip_fraction)
    near_threshold = float(slack_indicator[0].item())
    return _bounded_exponential_update(
        float(clip_threshold),
        eta=eta,
        error=target_unclipped - near_threshold,
        target_unclipped_proxy=target_unclipped,
        c_min=c_min,
        c_max=c_max,
    )


def update_slaclip_threshold(
    clip_threshold: float,
    slack_indicator: Tensor,
    *,
    eta: float,
    beta: float,
    c_min: float,
    c_max: float,
) -> Tuple[float, float]:
    """Compatibility wrapper returning ``(C_next, gamma_t)`` for full SlaClip."""
    update = full_slaclip_threshold_update(
        clip_threshold,
        slack_indicator,
        eta=eta,
        beta=beta,
        c_min=c_min,
        c_max=c_max,
    )
    return update.next_clip, update.target_unclipped_proxy
