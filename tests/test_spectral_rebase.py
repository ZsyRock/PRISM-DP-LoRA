from __future__ import annotations

from types import SimpleNamespace

import torch

from prism_cli.optim.prism import (
    spectral_init_peft_model,
    spectral_rebase_adapter_inplace,
)


class _FakePeftLinear(torch.nn.Module):
    def __init__(self, in_features: int = 5, out_features: int = 4, rank: int = 2) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(out_features, in_features))
        self.lora_A = torch.nn.ModuleDict(
            {'default': torch.nn.Linear(in_features, rank, bias=False)}
        )
        self.lora_B = torch.nn.ModuleDict(
            {'default': torch.nn.Linear(rank, out_features, bias=False)}
        )
        self.scaling = {'default': 1.0}
        self.r = {'default': rank}
        self.lora_alpha = {'default': rank}


class _FakePeftModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = _FakePeftLinear()
        self.peft_config = {
            'default': SimpleNamespace(
                r=2,
                lora_alpha=2,
                rank_pattern={},
                alpha_pattern={},
            )
        }


def _effective_weight(module: _FakePeftLinear) -> torch.Tensor:
    scale = float(module.scaling['default'])
    return module.weight + scale * (
        module.lora_B['default'].weight @ module.lora_A['default'].weight
    )


def test_spectral_rebase_preserves_update_against_original_base() -> None:
    torch.manual_seed(211)
    model = _FakePeftModel()
    original_base = model.proj.weight.detach().clone()
    spectral_init_peft_model(
        model,
        svd_device='cpu',
        svd_oversample=2,
        svd_n_iter=1,
        verbose=False,
    )
    # Spectral residual initialization itself is function-preserving.
    assert torch.allclose(_effective_weight(model.proj), original_base, atol=1e-5, rtol=1e-5)

    with torch.no_grad():
        model.proj.lora_A['default'].weight.add_(0.03)
        model.proj.lora_B['default'].weight.sub_(0.02)
    trained_effective = _effective_weight(model.proj).detach().clone()

    stats = spectral_rebase_adapter_inplace(model, verbose=False)
    rebased_adapter = (
        model.proj.lora_B['default'].weight @ model.proj.lora_A['default'].weight
    )
    loaded_against_original = original_base + float(model.proj.scaling['default']) * rebased_adapter

    assert stats['spectral_rebase_modules'] == 1.0
    assert model.proj.lora_A['default'].out_features == 4
    assert model.proj.lora_B['default'].in_features == 4
    assert model.peft_config['default'].r == 4
    assert torch.allclose(loaded_against_original, trained_effective, atol=1e-5, rtol=1e-5)
