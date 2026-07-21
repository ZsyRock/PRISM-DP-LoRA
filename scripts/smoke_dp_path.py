#!/usr/bin/env python3
"""CPU/GPU smoke test for the Opacus -> PRISM baseline/SlaClip DP path.

This uses synthetic data and a tiny pair of LoRA-shaped linear factors. It does
not download a model or claim to replace the required Gemma GPU smoke test.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

try:
    import torch
    import torch.nn.functional as F
    from opacus import PrivacyEngine
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError as exc:  # pragma: no cover - exercised on incomplete environments
    raise SystemExit(f"Missing smoke-test dependency: {exc}. Install requirements.txt first.") from exc

from prism_cli.optim.prism import PRISM


class ToyLoRAModel(nn.Module):
    def __init__(self, input_dim: int = 5, rank: int = 2, output_dim: int = 3) -> None:
        super().__init__()
        self.base = nn.Linear(input_dim, output_dim, bias=False)
        self.lora_A = nn.Linear(input_dim, rank, bias=False)
        self.lora_B = nn.Linear(rank, output_dim, bias=False)
        self.base.requires_grad_(False)
        nn.init.normal_(self.lora_A.weight, mean=0.0, std=0.25)
        nn.init.normal_(self.lora_B.weight, mean=0.0, std=0.25)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.base(inputs) + self.lora_B(self.lora_A(inputs))


def select_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("--device cuda was requested, but CUDA is unavailable")
        return torch.device("cuda:0")
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def accountant_step(engine: PrivacyEngine, noise_multiplier: float, sample_rate: float) -> None:
    try:
        engine.accountant.step(noise_multiplier=noise_multiplier, sample_rate=sample_rate)
    except TypeError:
        engine.accountant.step(noise_multiplier, sample_rate)


def run_method(
    method: str,
    *,
    device: torch.device,
    seed: int,
    steps: int,
    clip_norm: float,
    noise_multiplier: float,
    slaclip_beta: float,
) -> dict:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    inputs = torch.randn(8, 5)
    targets = torch.randn(8, 3)
    loader = DataLoader(TensorDataset(inputs, targets), batch_size=8, shuffle=False)
    model = ToyLoRAModel().to(device)
    optimizer = PRISM(
        [model.lora_A.weight, model.lora_B.weight],
        lr=1e-2,
        use_adaptive=False,
        clipping_method=method,
        slaclip_num_slots=3,
        slaclip_eta=0.2,
        slaclip_beta=slaclip_beta,
        slaclip_c_min=0.05,
        slaclip_c_max=5.0,
        telemetry_mode="research_raw",
        raw_hist_bins=8,
    )
    engine = PrivacyEngine(accountant="rdp")
    private_model, private_optimizer, private_loader = engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=loader,
        noise_multiplier=noise_multiplier,
        max_grad_norm=clip_norm,
        poisson_sampling=True,
        grad_sample_mode="hooks",
    )
    base_optimizer = getattr(private_optimizer, "original_optimizer", optimizer)
    expected_batch_size = float(getattr(private_optimizer, "expected_batch_size", 8.0))
    sample_rate = 1.0 / float(len(private_loader))
    trajectory = []
    for step in range(1, steps + 1):
        batch_inputs, batch_targets = next(iter(private_loader))
        batch_inputs = batch_inputs.to(device)
        batch_targets = batch_targets.to(device)
        base_optimizer.dp_begin(
            max_grad_norm=clip_norm,
            expected_batch_size=expected_batch_size,
            noise_multiplier=noise_multiplier,
        )
        seen = 0
        for start in range(0, int(batch_inputs.shape[0]), 4):
            end = min(start + 4, int(batch_inputs.shape[0]))
            base_optimizer.zero_grad(set_to_none=True)
            predictions = private_model(batch_inputs[start:end])
            loss = F.mse_loss(predictions, batch_targets[start:end], reduction="mean")
            loss.backward()
            seen += int(base_optimizer.dp_accumulate())
        finalized = int(base_optimizer.dp_finalize(noise_multiplier=noise_multiplier))
        if finalized != seen or finalized <= 0:
            raise RuntimeError(f"Unexpected sample accounting: accumulated={seen}, finalized={finalized}")
        accountant_step(engine, noise_multiplier, sample_rate)
        safe_log = dict(base_optimizer.last_log)
        raw_log = dict(base_optimizer.last_raw_log)
        required_safe = {"dp_clip_threshold", "dp_next_clip_threshold"}
        if method == "slaclip":
            required_safe.add("slack_indicator")
        required_raw = {
            "NON_PRIVATE_TELEMETRY",
            "raw_clip_fraction",
            "raw_global_norm_quantiles",
            "raw_signal_to_noise_ratio",
        }
        if not required_safe.issubset(safe_log):
            raise RuntimeError(f"Missing DP-safe smoke fields: {sorted(required_safe - safe_log.keys())}")
        if method == "baseline" and "slack_indicator" in safe_log:
            raise RuntimeError("Baseline unexpectedly emitted a SlaClip Slack Indicator")
        if not required_raw.issubset(raw_log):
            raise RuntimeError(f"Missing raw smoke fields: {sorted(required_raw - raw_log.keys())}")
        current = float(safe_log["dp_clip_threshold"])
        next_clip = float(safe_log["dp_next_clip_threshold"])
        if not (math.isfinite(current) and math.isfinite(next_clip)):
            raise RuntimeError("Non-finite clipping threshold in smoke test")
        if method == "baseline" and not (
            math.isclose(current, clip_norm, rel_tol=0.0, abs_tol=1e-8)
            and math.isclose(next_clip, clip_norm, rel_tol=0.0, abs_tol=1e-8)
        ):
            raise RuntimeError(
                f"Baseline threshold changed: expected={clip_norm}, "
                f"current={current}, next={next_clip}"
            )
        trajectory.append(
            {
                "step": step,
                "clip": current,
                "next_clip": next_clip,
                "clip_fraction": float(raw_log["raw_clip_fraction"]),
                "snr": float(raw_log["raw_signal_to_noise_ratio"]),
                "batch_n": finalized,
            }
        )
    epsilon = float(engine.get_epsilon(delta=1e-5))
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise RuntimeError(f"Invalid epsilon from smoke accountant: {epsilon}")
    if method == "slaclip":
        for record in trajectory:
            if not 0.05 <= record["next_clip"] <= 5.0:
                raise RuntimeError(f"SlaClip threshold escaped configured bounds: {record}")
    return {
        "method": method,
        "device": str(device),
        "steps": steps,
        "noise_multiplier": noise_multiplier,
        "epsilon_at_delta_1e-5": epsilon,
        "trajectory": trajectory,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--clip-norm", type=float, default=1.0)
    parser.add_argument("--noise-multiplier", type=float, default=0.8)
    parser.add_argument("--slaclip-beta", type=float, default=0.5)
    parser.add_argument("--method", choices=["baseline", "slaclip", "both"], default="both")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.steps <= 0:
        raise SystemExit("--steps must be positive")
    if args.clip_norm <= 0 or args.noise_multiplier <= 0:
        raise SystemExit("--clip-norm and --noise-multiplier must be positive")
    if not 0 <= args.slaclip_beta <= 1:
        raise SystemExit("--slaclip-beta must be in [0, 1]")
    device = select_device(args.device)
    methods = ["baseline", "slaclip"] if args.method == "both" else [args.method]
    results = [
        run_method(
            method,
            device=device,
            seed=args.seed,
            steps=args.steps,
            clip_norm=args.clip_norm,
            noise_multiplier=args.noise_multiplier,
            slaclip_beta=args.slaclip_beta,
        )
        for method in methods
    ]
    print(json.dumps({"status": "ok", "synthetic_only": True, "results": results}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
