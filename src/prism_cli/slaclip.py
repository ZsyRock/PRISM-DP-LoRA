"""Pure SlaClip helpers used by the PRISM tangent-space optimizer.

The slack coordinates are part of the same Gaussian release as the clipped
tangent update.  They must never be computed as a second, unaccounted query.
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
from torch import Tensor


Z_0995 = 2.5758293035489004


def automatic_num_slots(
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


def update_slaclip_threshold(
    clip_threshold: float,
    slack_indicator: Tensor,
    *,
    eta: float,
    beta: float,
    c_min: float,
    c_max: float,
) -> Tuple[float, float]:
    """Apply the full SlaClip controller (not the SlaClip-Q ablation)."""
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
    exponent = max(-50.0, min(50.0, float(eta) * (gamma_t - near_threshold)))
    c_next = c_t * math.exp(exponent)
    c_next = max(float(c_min), min(float(c_max), c_next))
    return float(c_next), float(gamma_t)
