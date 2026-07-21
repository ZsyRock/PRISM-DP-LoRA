#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

common_args=(
  --dataset math10k
  --privacy dp
  --epsilon 6
  --delta 1e-5
  --dp_max_grad_norm 1.0
  --batch_size 64
  --micro_batch_size 4
  --seed 42
  --telemetry_mode dp_safe
)

python train_eval.py --method baseline "${common_args[@]}" "$@"
python train_eval.py --method slaclip "${common_args[@]}" "$@"
