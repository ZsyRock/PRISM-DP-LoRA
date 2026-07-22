from __future__ import annotations

import pytest
import torch

from prism_cli.utils import ensure_text_only_token_type_ids, iter_microbatches


def test_adds_zero_token_type_ids_for_text_only_multimodal_batch() -> None:
    original = {'input_ids': torch.tensor([[4, 5, 6], [0, 7, 8]])}

    prepared = ensure_text_only_token_type_ids(original, required=True)

    assert 'token_type_ids' not in original
    assert prepared is not original
    assert prepared['token_type_ids'].shape == original['input_ids'].shape
    assert prepared['token_type_ids'].dtype == original['input_ids'].dtype
    assert prepared['token_type_ids'].device == original['input_ids'].device
    assert torch.count_nonzero(prepared['token_type_ids']).item() == 0


def test_does_not_add_token_type_ids_when_not_required() -> None:
    original = {'input_ids': torch.tensor([[1, 2]])}

    prepared = ensure_text_only_token_type_ids(original, required=False)

    assert prepared is original
    assert 'token_type_ids' not in prepared


def test_preserves_existing_token_type_ids() -> None:
    existing = torch.tensor([[0, 1, 0]])
    original = {
        'input_ids': torch.tensor([[1, 2, 3]]),
        'token_type_ids': existing,
    }

    prepared = ensure_text_only_token_type_ids(original, required=True)

    assert prepared is original
    assert prepared['token_type_ids'] is existing


def test_requires_input_ids_tensor_when_synthesis_is_needed() -> None:
    with pytest.raises(KeyError, match='input_ids tensor'):
        ensure_text_only_token_type_ids({}, required=True)


def test_microbatching_preserves_synthesized_token_type_ids() -> None:
    batch = ensure_text_only_token_type_ids(
        {'input_ids': torch.arange(12).reshape(4, 3)},
        required=True,
    )

    pieces = list(iter_microbatches(batch, micro_bs=3))

    assert torch.equal(
        torch.cat([piece['token_type_ids'] for piece in pieces]),
        batch['token_type_ids'],
    )
