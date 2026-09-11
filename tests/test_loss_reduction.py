from __future__ import annotations

import copy

import pytest
import torch

from prism_cli.losses import causal_lm_per_example_loss


class _TinyCausalLM(torch.nn.Module):
    def __init__(self, vocab_size: int = 13, hidden_size: int = 7) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, hidden_size)
        self.projection = torch.nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.projection(self.embedding(input_ids))


def _grad_samples(state, input_ids, labels, slices):
    opacus = pytest.importorskip('opacus')
    model = _TinyCausalLM()
    model.load_state_dict(copy.deepcopy(state))
    wrapped = opacus.GradSampleModule(model, loss_reduction='mean')
    collected = {name: [] for name, parameter in wrapped.named_parameters() if parameter.requires_grad}
    for start, end in slices:
        wrapped.zero_grad(set_to_none=True)
        logits = wrapped(input_ids[start:end])
        per_example, _ = causal_lm_per_example_loss(logits, labels[start:end])
        per_example.mean().backward()
        for name, parameter in wrapped.named_parameters():
            if parameter.requires_grad:
                collected[name].append(parameter.grad_sample.detach().clone())
    return {name: torch.cat(parts, dim=0) for name, parts in collected.items()}


def test_per_record_causal_loss_is_microbatch_invariant_with_variable_lengths() -> None:
    torch.manual_seed(91)
    input_ids = torch.randint(0, 13, (4, 7))
    labels = input_ids.clone()
    # After the causal shift these records contain 1, 2, 4, and 6 supervised
    # tokens. This is the case where a global token mean violates Opacus's
    # batch-mean loss contract.
    for row, count in enumerate((1, 2, 4, 6)):
        labels[row, 1 + count :] = -100
    base = _TinyCausalLM()
    state = copy.deepcopy(base.state_dict())

    full = _grad_samples(state, input_ids, labels, [(0, 4)])
    split = _grad_samples(state, input_ids, labels, [(0, 2), (2, 4)])

    assert full.keys() == split.keys()
    for name in full:
        assert torch.allclose(full[name], split[name], atol=1e-6, rtol=1e-6), name


def test_per_record_loss_reports_supervised_next_token_counts() -> None:
    logits = torch.zeros(2, 4, 5, requires_grad=True)
    labels = torch.tensor([[1, 2, -100, -100], [1, 2, 3, 4]])
    losses, counts = causal_lm_per_example_loss(logits, labels)
    assert counts.tolist() == [1, 3]
    assert losses.shape == (2,)
    assert torch.isfinite(losses).all()
