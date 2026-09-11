from __future__ import annotations

import types

import pytest
import torch

from prism_cli.modeling import (
    is_multimodal_causal_lm_config,
    resolve_text_lora_target_modules,
    resolved_model_revision,
)


def test_gemma3_with_vision_config_uses_multimodal_loader() -> None:
    config = types.SimpleNamespace(model_type='gemma3', vision_config=object())
    assert is_multimodal_causal_lm_config(config)


def test_text_only_config_does_not_use_multimodal_loader() -> None:
    assert not is_multimodal_causal_lm_config(
        types.SimpleNamespace(model_type='gemma3_text', vision_config=None)
    )
    assert not is_multimodal_causal_lm_config(
        types.SimpleNamespace(model_type='llama', vision_config=object())
    )


def test_resolved_model_revision_reads_hugging_face_commit() -> None:
    model = types.SimpleNamespace(config=types.SimpleNamespace(_commit_hash='abc123'))
    assert resolved_model_revision(model) == 'abc123'


class _TinyMultimodal(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.language_model = torch.nn.ModuleDict(
            {
                'q_proj': torch.nn.Linear(2, 2),
                'down_proj': torch.nn.Linear(2, 2),
            }
        )
        self.vision_tower = torch.nn.ModuleDict(
            {
                'q_proj': torch.nn.Linear(2, 2),
                'down_proj': torch.nn.Linear(2, 2),
            }
        )


def test_lora_targets_are_resolved_to_exact_text_modules() -> None:
    targets = resolve_text_lora_target_modules(
        _TinyMultimodal(),
        ['q_proj', 'down_proj'],
    )
    assert targets == ['language_model.q_proj', 'language_model.down_proj']


def test_lora_target_resolution_rejects_missing_text_suffix() -> None:
    with pytest.raises(ValueError, match='v_proj'):
        resolve_text_lora_target_modules(_TinyMultimodal(), ['v_proj'])
