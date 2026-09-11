#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

for arg in "$@"; do
  case "$arg" in
    --method|--method=*|--config|--config=*|--dataset|--dataset=*|--privacy|--privacy=*|--output_dir|--output_dir=*|--output-dir|--output-dir=*|--result_dir|--result_dir=*|--result-dir|--result-dir=*)
      echo "error: paired runs manage method/config/dataset/privacy/output paths; unsupported argument: $arg" >&2
      exit 2
      ;;
  esac
done

PRISM_PYTHON="${PRISM_PYTHON:-python}"
"$PRISM_PYTHON" train_eval.py --config configs/math10k_paper.json --method baseline "$@"
"$PRISM_PYTHON" train_eval.py --config configs/math10k_paper.json --method slaclip "$@"
