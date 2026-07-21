from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F
from torch import Tensor


def causal_lm_per_example_loss(
    logits: Tensor,
    labels: Tensor,
    *,
    ignore_index: int = -100,
) -> Tuple[Tensor, Tensor]:
    """Compute one normalized causal-LM loss per training record.

    Hugging Face causal language models normally return a single mean over all
    non-ignored tokens in the physical batch.  Opacus's ``loss_reduction='mean'``
    contract, however, assumes a mean over records and multiplies gradients by
    the physical batch size when constructing ``grad_sample``.  Mixing these two
    reductions makes per-record gradients depend on how a logical batch is split
    into physical microbatches.

    This function first averages next-token loss within each record and leaves
    the caller to take the mean over records.  The resulting per-record gradients
    and clipping decisions are invariant to physical microbatch boundaries.
    """

    if logits.ndim != 3:
        raise ValueError(f'logits must have shape [batch, sequence, vocab], got {tuple(logits.shape)}')
    if labels.ndim != 2:
        raise ValueError(f'labels must have shape [batch, sequence], got {tuple(labels.shape)}')
    if tuple(logits.shape[:2]) != tuple(labels.shape):
        raise ValueError(
            f'logits/labels shape mismatch: logits={tuple(logits.shape)}, labels={tuple(labels.shape)}'
        )
    if labels.shape[1] < 2:
        # Keep a differentiable zero for callers performing backward.
        zeros = logits[:, :1, :].sum(dim=(1, 2)) * 0.0
        return zeros, torch.zeros(labels.shape[0], device=labels.device, dtype=torch.long)

    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    token_losses = F.cross_entropy(
        shift_logits.float().view(-1, shift_logits.shape[-1]),
        shift_labels.view(-1),
        reduction='none',
        ignore_index=int(ignore_index),
    ).view_as(shift_labels)
    valid = shift_labels.ne(int(ignore_index))
    token_counts = valid.sum(dim=1)
    per_example = (token_losses * valid).sum(dim=1) / token_counts.clamp_min(1)
    return per_example, token_counts


def forward_causal_lm_per_example_loss(model, batch: dict) -> Tuple[Tensor, Tensor]:
    """Run a causal LM without its batch-level loss and return record losses."""

    if 'labels' not in batch:
        raise KeyError("training batch must contain 'labels'")
    model_inputs = {key: value for key, value in batch.items() if key != 'labels'}
    outputs = model(**model_inputs)
    logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]
    return causal_lm_per_example_loss(logits, batch['labels'])
