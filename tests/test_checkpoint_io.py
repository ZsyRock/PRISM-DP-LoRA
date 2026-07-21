from __future__ import annotations

import json
import os
import random
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from prism_cli.optim.prism import PRISM
from prism_cli.utils import (
    checkpoint_file,
    get_rng_state,
    load_resume_checkpoint_if_available,
    save_resume_checkpoint,
    set_rng_state,
    truncate_jsonl_to_step,
)


def _model_and_optimizer() -> tuple[torch.nn.Module, PRISM]:
    model = torch.nn.Sequential(
        torch.nn.Linear(3, 2, bias=False),
        torch.nn.Linear(2, 1, bias=False),
    )
    optimizer = PRISM(
        [model[0].weight, model[1].weight],
        lr=1e-3,
        clipping_method='baseline',
        use_adaptive=False,
    )
    optimizer._ensure_state()
    optimizer.current_clip = 0.75
    optimizer.state[model[0].weight]['prism'].step = 3
    return model, optimizer


def test_checkpoint_real_file_round_trip_handles_prism_state(tmp_path: Path) -> None:
    model, optimizer = _model_and_optimizer()
    expected_weight = model[0].weight.detach().clone()

    save_resume_checkpoint(
        tmp_path,
        model,
        optimizer,
        update_steps=7,
        extra={'marker': SimpleNamespace(value='trusted-local-state')},
    )

    path = checkpoint_file(tmp_path)
    assert path.is_file()
    assert not list(tmp_path.glob(f'.{path.name}.*.tmp'))

    checkpoint = load_resume_checkpoint_if_available(tmp_path)
    assert checkpoint is not None
    assert checkpoint['update_steps'] == 7
    assert checkpoint['extra']['marker'].value == 'trusted-local-state'
    assert torch.equal(checkpoint['trainable_state']['0.weight'], expected_weight)
    prism_states = [
        value['prism']
        for value in checkpoint['optimizer_state']['state'].values()
        if 'prism' in value
    ]
    assert prism_states
    assert isinstance(prism_states[0], SimpleNamespace)
    assert prism_states[0].step == 3


def test_step_zero_optimizer_checkpoint_initializes_missing_prism_state() -> None:
    source_a = torch.nn.Parameter(torch.randn(2, 3))
    source_b = torch.nn.Parameter(torch.randn(4, 2))
    source = PRISM([source_a, source_b], lr=1e-3, clipping_method='slaclip')
    saved = source.state_dict()
    assert saved['state'] == {}

    target_a = torch.nn.Parameter(source_a.detach().clone())
    target_b = torch.nn.Parameter(source_b.detach().clone())
    target = PRISM([target_a, target_b], lr=1e-3, clipping_method='slaclip')
    target._ensure_state()
    target.load_state_dict(saved)

    assert target.state[target_a].get('prism') is not None
    target.dp_begin(max_grad_norm=1.0, expected_batch_size=4.0, noise_multiplier=0.8)


def test_checkpoint_interrupted_write_preserves_previous_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, optimizer = _model_and_optimizer()
    save_resume_checkpoint(tmp_path, model, optimizer, update_steps=2)
    path = checkpoint_file(tmp_path)
    original = path.read_bytes()

    def interrupted_save(_payload, temporary_path) -> None:
        Path(temporary_path).write_bytes(b'partial checkpoint')
        raise RuntimeError('simulated interruption')

    monkeypatch.setattr(torch, 'save', interrupted_save)
    with pytest.raises(RuntimeError, match='simulated interruption'):
        save_resume_checkpoint(tmp_path, model, optimizer, update_steps=3)

    assert path.read_bytes() == original
    assert not list(tmp_path.glob(f'.{path.name}.*.tmp'))


def test_rng_state_round_trip_covers_python_numpy_and_torch() -> None:
    np = pytest.importorskip('numpy')
    random.seed(1234)
    np.random.seed(5678)
    torch.manual_seed(9012)
    state = get_rng_state()

    expected_python = random.random()
    expected_numpy = float(np.random.random())
    expected_torch = torch.rand(4)

    random.random()
    np.random.random()
    torch.rand(4)
    set_rng_state(state)

    assert random.random() == expected_python
    assert float(np.random.random()) == expected_numpy
    assert torch.equal(torch.rand(4), expected_torch)


def test_truncate_jsonl_to_step_removes_future_and_partial_tail(tmp_path: Path) -> None:
    path = tmp_path / 'train_log.jsonl'
    records = [
        {'step': 1, 'loss': 3.0},
        {'step': 2, 'loss': 2.0},
        {'step': 4, 'loss': 1.0},
    ]
    path.write_text(
        ''.join(json.dumps(record) + '\n' for record in records) + '{"step": 5',
        encoding='utf-8',
    )

    assert truncate_jsonl_to_step(path, 2) == 2
    restored = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
    assert restored == records[:2]
    assert not list(tmp_path.glob(f'.{path.name}.*.tmp'))


def test_truncate_jsonl_is_atomic_if_replace_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / 'train_log.jsonl'
    original = '{"step": 1}\n{"step": 2}\n'
    path.write_text(original, encoding='utf-8')

    def failed_replace(_source, _target) -> None:
        raise OSError('simulated replace failure')

    monkeypatch.setattr(os, 'replace', failed_replace)
    with pytest.raises(OSError, match='simulated replace failure'):
        truncate_jsonl_to_step(path, 1)

    assert path.read_text(encoding='utf-8') == original
    assert not list(tmp_path.glob(f'.{path.name}.*.tmp'))
