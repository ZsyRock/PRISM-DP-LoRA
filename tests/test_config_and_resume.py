from __future__ import annotations

from pathlib import Path

import pytest
import torch

from prism_cli.trainers import RunConfig
from prism_cli.utils import load_trainable_state_dict, trainable_state_dict_cpu


def test_research_raw_requires_explicit_acknowledgement(tmp_path: Path) -> None:
    cfg = RunConfig(
        dataset='math10k',
        method='baseline',
        privacy='dp',
        root=tmp_path,
        telemetry_mode='research_raw',
    )
    with pytest.raises(ValueError, match='allow_non_private_telemetry'):
        cfg.finalize()


def test_only_baseline_and_full_slaclip_are_public_methods(tmp_path: Path) -> None:
    assert RunConfig(dataset='math10k', method='prism', privacy='dp', root=tmp_path).finalize().method == 'baseline'
    assert RunConfig(dataset='math10k', method='slaclip', privacy='dp', root=tmp_path).finalize().method == 'slaclip'
    with pytest.raises(ValueError, match='baseline or slaclip'):
        RunConfig(dataset='math10k', method='slaclip_q', privacy='dp', root=tmp_path).finalize()


def test_checkpoint_loader_accepts_old_opacus_module_prefix() -> None:
    source = torch.nn.Linear(3, 2, bias=False)
    saved = {f'_module.{name}': value for name, value in trainable_state_dict_cpu(source).items()}
    target = torch.nn.Linear(3, 2, bias=False)
    with torch.no_grad():
        target.weight.zero_()
    assert load_trainable_state_dict(target, saved) == 1
    assert torch.equal(source.weight, target.weight)
