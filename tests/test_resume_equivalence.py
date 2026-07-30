from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from prism_cli.optim.prism import PRISM
from prism_cli.trainers import (
    RunConfig,
    _checkpoint_extra,
    _decouple_loader_worker_rng,
    _restore_if_possible,
    _restore_runtime_state,
    _step_accountant,
)
from prism_cli.utils import save_resume_checkpoint, set_seed


class _ToyLoRA(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.base = nn.Linear(4, 3, bias=False)
        self.lora_A = nn.Linear(4, 2, bias=False)
        self.lora_B = nn.Linear(2, 3, bias=False)
        self.base.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.lora_B(self.lora_A(x))


@dataclass
class _Run:
    model: nn.Module
    optimizer: PRISM
    loader: DataLoader
    iterator: object
    engine: object
    noise_multiplier: float
    sample_rate: float
    expected_batch_size: float
    step: int


def _config(tmp_path: Path, *, method: str = 'slaclip') -> RunConfig:
    tmp_path.mkdir(parents=True, exist_ok=True)
    data_path = tmp_path / 'data.json'
    data_path.write_text('[{"instruction":"x","input":"","output":"y"}]\n', encoding='utf-8')
    return RunConfig(
        dataset='math10k',
        method=method,
        privacy='dp',
        root=tmp_path,
        data_path=data_path,
        output_dir=tmp_path / 'model',
        result_dir=tmp_path / 'result',
        total_update_steps=3,
        batch_size=6,
        micro_batch_size=3,
        dp_accountant='rdp',
        telemetry_mode='research_raw',
        allow_non_private_telemetry=True,
        slaclip_num_slots=3,
        slaclip_target_non_small_clip_fraction=(
            0.8 if method == 'slaclip' else None
        ),
        slaclip_target_clip_fraction=(
            0.99 if method == 'slaclip_q' else None
        ),
        slaclip_c_min=0.1,
        slaclip_c_max=15.0 if method == 'slaclip_q' else 50.0,
    ).finalize()


def _build(cfg: RunConfig, *, resume: bool) -> _Run:
    opacus = pytest.importorskip('opacus')
    set_seed(cfg.seed)
    x = torch.arange(48, dtype=torch.float32).reshape(12, 4) / 17.0
    y = torch.arange(36, dtype=torch.float32).reshape(12, 3) / 19.0
    sampling_generator = torch.Generator().manual_seed(cfg.seed)
    loader = DataLoader(
        TensorDataset(x, y),
        batch_size=6,
        shuffle=True,
        generator=sampling_generator,
        num_workers=0,
    )
    model = _ToyLoRA()
    optimizer = PRISM(
        [model.lora_A.weight, model.lora_B.weight],
        lr=1e-2,
        clipping_method=cfg.method,
        slaclip_num_slots=3,
        slaclip_target_non_small_clip_fraction=(
            cfg.slaclip_target_non_small_clip_fraction
        ),
        slaclip_target_clip_fraction=cfg.slaclip_target_clip_fraction,
        slaclip_c_min=cfg.slaclip_c_min,
        slaclip_c_max=cfg.slaclip_c_max,
        telemetry_mode='research_raw',
        raw_hist_bins=4,
    )
    checkpoint = _restore_if_possible(cfg, model, optimizer) if resume else None
    engine = opacus.PrivacyEngine(accountant='rdp')
    noise_multiplier = 0.8
    private_model, dp_optimizer, private_loader = engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=loader,
        noise_multiplier=noise_multiplier,
        max_grad_norm=cfg.dp_max_grad_norm,
        poisson_sampling=True,
        grad_sample_mode='hooks',
    )
    optimizer = getattr(dp_optimizer, 'original_optimizer', optimizer)
    optimizer.configure_dp_noise(generator=getattr(dp_optimizer, 'generator', None), secure_mode=False)
    sample_rate = 1.0 / len(private_loader)
    expected_batch_size = float(dp_optimizer.expected_batch_size)
    _decouple_loader_worker_rng(private_loader, cfg.seed)
    step = _restore_runtime_state(
        cfg,
        checkpoint,
        train_loader=private_loader,
        optimizer=optimizer,
        privacy_engine=engine,
        noise_multiplier=noise_multiplier,
        sample_rate=sample_rate,
        expected_batch_size=expected_batch_size,
    )
    return _Run(
        model=private_model,
        optimizer=optimizer,
        loader=private_loader,
        iterator=iter(private_loader),
        engine=engine,
        noise_multiplier=noise_multiplier,
        sample_rate=sample_rate,
        expected_batch_size=expected_batch_size,
        step=step,
    )


def _one_step(run: _Run):
    x, y = next(run.iterator)
    run.optimizer.dp_begin(
        max_grad_norm=1.0,
        expected_batch_size=run.expected_batch_size,
        noise_multiplier=run.noise_multiplier,
    )
    for start in range(0, x.shape[0], 3):
        end = min(start + 3, x.shape[0])
        run.optimizer.zero_grad(set_to_none=True)
        predictions = run.model(x[start:end])
        per_example = (predictions - y[start:end]).square().mean(dim=1)
        per_example.mean().backward()
        assert run.optimizer.dp_accumulate() == end - start
    run.optimizer.dp_finalize(noise_multiplier=run.noise_multiplier)
    _step_accountant(run.engine, run.noise_multiplier, run.sample_rate)
    run.step += 1
    return x.detach().clone(), y.detach().clone()


def _trainable_parameters(run: _Run):
    raw = getattr(run.model, '_module', run.model)
    return {
        name: parameter.detach().clone()
        for name, parameter in raw.named_parameters()
        if parameter.requires_grad
    }


@pytest.mark.parametrize('method', ['slaclip', 'slaclip_q'])
def test_checkpoint_resume_matches_uninterrupted_sampling_noise_and_accounting(
    tmp_path: Path,
    method: str,
) -> None:
    cfg = _config(tmp_path / method, method=method)
    uninterrupted = _build(cfg, resume=False)
    _one_step(uninterrupted)
    save_resume_checkpoint(
        Path(cfg.output_dir),
        uninterrupted.model,
        uninterrupted.optimizer,
        uninterrupted.step,
        extra=_checkpoint_extra(
            cfg,
            train_loader=uninterrupted.loader,
            optimizer=uninterrupted.optimizer,
            privacy_engine=uninterrupted.engine,
            noise_multiplier=uninterrupted.noise_multiplier,
            sample_rate=uninterrupted.sample_rate,
            expected_batch_size=uninterrupted.expected_batch_size,
        ),
    )
    ref_x, ref_y = _one_step(uninterrupted)
    ref_parameters = _trainable_parameters(uninterrupted)
    ref_epsilon = uninterrupted.engine.get_epsilon(delta=1e-5)
    ref_clip = uninterrupted.optimizer.current_clip

    resumed = _build(cfg, resume=True)
    assert resumed.step == 1
    got_x, got_y = _one_step(resumed)

    assert torch.equal(got_x, ref_x)
    assert torch.equal(got_y, ref_y)
    for name, expected in ref_parameters.items():
        assert torch.allclose(_trainable_parameters(resumed)[name], expected, atol=1e-7, rtol=1e-7)
    assert resumed.optimizer.current_clip == pytest.approx(ref_clip, rel=1e-7, abs=1e-7)
    assert resumed.engine.get_epsilon(delta=1e-5) == pytest.approx(ref_epsilon, rel=1e-12)


def test_step_zero_checkpoint_resumes_first_update_exactly(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    uninterrupted = _build(cfg, resume=False)
    save_resume_checkpoint(
        Path(cfg.output_dir),
        uninterrupted.model,
        uninterrupted.optimizer,
        0,
        extra=_checkpoint_extra(
            cfg,
            train_loader=uninterrupted.loader,
            optimizer=uninterrupted.optimizer,
            privacy_engine=uninterrupted.engine,
            noise_multiplier=uninterrupted.noise_multiplier,
            sample_rate=uninterrupted.sample_rate,
            expected_batch_size=uninterrupted.expected_batch_size,
        ),
    )
    ref_x, ref_y = _one_step(uninterrupted)
    ref_parameters = _trainable_parameters(uninterrupted)
    ref_clip = uninterrupted.optimizer.current_clip

    resumed = _build(cfg, resume=True)
    assert resumed.step == 0
    got_x, got_y = _one_step(resumed)

    assert torch.equal(got_x, ref_x)
    assert torch.equal(got_y, ref_y)
    for name, expected in ref_parameters.items():
        assert torch.allclose(_trainable_parameters(resumed)[name], expected, atol=1e-7, rtol=1e-7)
    assert resumed.optimizer.current_clip == pytest.approx(ref_clip, rel=1e-7, abs=1e-7)
