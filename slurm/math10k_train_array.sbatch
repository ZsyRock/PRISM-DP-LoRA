#!/usr/bin/env bash
#SBATCH --job-name=prism-math10k
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=48:00:00
#SBATCH --array=0-5
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=slurm-%x-%A_%a.out

# Pass site-specific --account and --partition flags to sbatch. This project is
# intentionally single-process: do not launch it with torchrun or >1 Slurm task.
set -euo pipefail

REPO_ROOT="${PRISM_REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
PYTHON_BIN="${PRISM_PYTHON:-python}"
RUN_ROOT="${PRISM_RUN_ROOT:-${SCRATCH:-}/prism-dp-lora}"

if [[ ! -f "${REPO_ROOT}/train_eval.py" ]]; then
  echo "PRISM_REPO_ROOT does not contain train_eval.py: ${REPO_ROOT}" >&2
  exit 2
fi
if [[ -z "${SCRATCH:-}" && -z "${PRISM_RUN_ROOT:-}" ]]; then
  echo "Set SCRATCH or PRISM_RUN_ROOT to persistent cluster storage." >&2
  exit 2
fi
if [[ "${SLURM_NTASKS:-1}" != "1" ]]; then
  echo "This trainer supports exactly one Slurm task; SLURM_NTASKS=${SLURM_NTASKS}" >&2
  exit 2
fi
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "Python executable not found: ${PYTHON_BIN}" >&2
  exit 2
fi

cd "${REPO_ROOT}"

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
SEEDS=(42 43 44)
METHODS=(baseline slaclip)
if (( TASK_ID < 0 || TASK_ID >= ${#SEEDS[@]} * ${#METHODS[@]} )); then
  echo "Array task ${TASK_ID} is outside the supported range 0-5." >&2
  exit 2
fi
SEED="${SEEDS[$((TASK_ID / ${#METHODS[@]}))]}"
METHOD="${METHODS[$((TASK_ID % ${#METHODS[@]}))]}"

EPSILON="${PRISM_EPSILON:-6}"
DELTA="${PRISM_DELTA:-1e-5}"
CLIP_NORM="${PRISM_CLIP_NORM:-1.0}"
BATCH_SIZE="${PRISM_BATCH_SIZE:-64}"
MICRO_BATCH_SIZE="${PRISM_MICRO_BATCH_SIZE:-4}"
STEPS="${PRISM_STEPS:-300}"
CHECKPOINT_EVERY="${PRISM_CHECKPOINT_EVERY:-10}"
BASE_MODEL="${PRISM_BASE_MODEL:-google/gemma-3-4b-pt}"
MODEL_REVISION="${PRISM_MODEL_REVISION:-}"
if [[ -z "${MODEL_REVISION}" || "${MODEL_REVISION}" == "main" ]]; then
  echo "Set PRISM_MODEL_REVISION to an immutable Hugging Face commit SHA for paired runs." >&2
  exit 2
fi
TELEMETRY_MODE="${PRISM_TELEMETRY_MODE:-research_raw}"
SLACLIP_BETA="${PRISM_SLACLIP_BETA:-0.5}"
SLACLIP_ETA="${PRISM_SLACLIP_ETA:-0.5}"
SLACLIP_C_MIN="${PRISM_SLACLIP_C_MIN:-0.1}"
SLACLIP_C_MAX="${PRISM_SLACLIP_C_MAX:-50.0}"
SLACLIP_NUM_SLOTS="${PRISM_SLACLIP_NUM_SLOTS:-0}"

BASE_RUN_ID="${PRISM_RUN_ID:-${SLURM_ARRAY_JOB_ID:-manual}}"
RUN_KEY="math10k_${METHOD}_eps${EPSILON}_C${CLIP_NORM}_seed${SEED}_${BASE_RUN_ID}_task${TASK_ID}"
if [[ "${METHOD}" == "slaclip" ]]; then
  RUN_KEY+="_beta${SLACLIP_BETA}_eta${SLACLIP_ETA}"
fi
OUTPUT_DIR="${RUN_ROOT}/trained_models/${RUN_KEY}"
RESULT_DIR="${RUN_ROOT}/experiment/${RUN_KEY}"

export HF_HOME="${HF_HOME:-${RUN_ROOT}/cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${RUN_ROOT}/cache/huggingface/datasets}"
export TMPDIR="${TMPDIR:-${RUN_ROOT}/tmp/${SLURM_JOB_ID:-manual}}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONUNBUFFERED=1
mkdir -p "${HF_HOME}" "${HF_DATASETS_CACHE}" "${TMPDIR}" "${OUTPUT_DIR}" "${RESULT_DIR}"

preflight=(
  "${PYTHON_BIN}" scripts/preflight_hpc.py
  --dataset math10k
  --output-root "${RUN_ROOT}"
  --min-free-gb "${PRISM_MIN_FREE_GB:-30}"
  --require-cuda
  --model-id "${BASE_MODEL}"
  --model-revision "${MODEL_REVISION}"
)
if [[ "${PRISM_CHECK_HF_ACCESS:-0}" == "1" ]]; then
  preflight+=(--check-hf-access)
fi
"${preflight[@]}"

if [[ "${PRISM_RUN_TOY_SMOKE:-0}" == "1" ]]; then
  "${PYTHON_BIN}" scripts/smoke_dp_path.py --device cuda --steps 1 --method both
fi

train_cmd=(
  "${PYTHON_BIN}" -u train_eval.py
  --dataset math10k
  --method "${METHOD}"
  --privacy dp
  --epsilon "${EPSILON}"
  --delta "${DELTA}"
  --base_model "${BASE_MODEL}"
  --model_revision "${MODEL_REVISION}"
  --require_cuda
  --seed "${SEED}"
  --steps "${STEPS}"
  --batch_size "${BATCH_SIZE}"
  --micro_batch_size "${MICRO_BATCH_SIZE}"
  --initial_clip_threshold "${CLIP_NORM}"
  --checkpoint_every "${CHECKPOINT_EVERY}"
  --output_dir "${OUTPUT_DIR}"
  --result_dir "${RESULT_DIR}"
  --telemetry_mode "${TELEMETRY_MODE}"
  --run_eval false
)

case "${TELEMETRY_MODE}" in
  research_raw)
    train_cmd+=(--allow_non_private_telemetry)
    ;;
  dp_safe)
    ;;
  *)
    echo "PRISM_TELEMETRY_MODE must be research_raw or dp_safe, got ${TELEMETRY_MODE}" >&2
    exit 2
    ;;
esac

if [[ "${METHOD}" == "slaclip" ]]; then
  train_cmd+=(
    --slaclip_beta "${SLACLIP_BETA}"
    --slaclip_eta "${SLACLIP_ETA}"
    --slaclip_c_min "${SLACLIP_C_MIN}"
    --slaclip_c_max "${SLACLIP_C_MAX}"
    --slaclip_num_slots "${SLACLIP_NUM_SLOTS}"
  )
fi

echo "Launching ${METHOD} seed=${SEED}; output=${OUTPUT_DIR}"
exec srun --nodes=1 --ntasks=1 "${train_cmd[@]}"
