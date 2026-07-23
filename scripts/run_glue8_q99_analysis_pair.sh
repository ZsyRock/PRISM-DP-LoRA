#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

for arg in "$@"; do
  case "$arg" in
    --method|--method=*|--config|--config=*|--dataset|--dataset=*|--privacy|--privacy=*|--output_dir|--output_dir=*|--output-dir|--output-dir=*|--result_dir|--result_dir=*|--result-dir|--result-dir=*|--telemetry_mode|--telemetry_mode=*|--allow_non_private_telemetry|--no-allow_non_private_telemetry|--slaclip_target_clip_fraction|--slaclip_target_clip_fraction=*|--slaclip_c_min|--slaclip_c_min=*|--slaclip_c_max|--slaclip_c_max=*|--slaclip_eta|--slaclip_eta=*)
      echo "error: paired analysis runs manage method/config/dataset/privacy/output/telemetry; unsupported argument: $arg" >&2
      exit 2
      ;;
  esac
done

PRISM_PYTHON="${PRISM_PYTHON:-python}"
PAIR_ROOT="${PRISM_PAIR_RUN_ROOT:-}"
PAIR_RUN_ID="${PRISM_PAIR_RUN_ID:-}"
if [[ -n "${PAIR_ROOT}" || -n "${PAIR_RUN_ID}" ]]; then
  [[ "${PAIR_ROOT}" == /* && "${PAIR_RUN_ID}" =~ ^[A-Za-z0-9._-]+$ ]] || {
    echo "error: set an absolute PRISM_PAIR_RUN_ROOT and a safe PRISM_PAIR_RUN_ID" >&2
    exit 2
  }
fi
run_one() {
  local method="$1"
  shift
  local -a paths=()
  if [[ -n "${PAIR_ROOT}" ]]; then
    paths=(--output_dir "${PAIR_ROOT}/glue8/${PAIR_RUN_ID}/${method}/adapter" --result_dir "${PAIR_ROOT}/glue8/${PAIR_RUN_ID}/${method}/results")
  fi
  "$PRISM_PYTHON" train_eval.py --config configs/glue8_slaclip_q99.json --method "${method}" "${paths[@]}" "$@" --telemetry_mode research_raw --allow_non_private_telemetry
}
run_one baseline "$@"
run_one slaclip_q "$@"
