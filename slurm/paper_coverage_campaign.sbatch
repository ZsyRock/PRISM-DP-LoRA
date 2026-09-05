#!/usr/bin/env bash

# One allocation containing all PRISM paper-coverage subexperiments. Scheduler
# options and portable paths are supplied by the submission wrapper.
set -Eeuo pipefail
umask 077

USER_NAME="${1:?missing user name}"
REPO_ROOT="${2:?missing immutable repository path}"
ENV_PREFIX="${3:?missing environment prefix}"
RUN_ROOT="${4:?missing run root}"
SHARED_HF_HOME="${5:?missing Hugging Face cache root}"
EXPECTED_REPO_SHA="${6:?missing repository SHA}"
CAMPAIGN_ID="${7:?missing campaign ID}"
SCRATCH_ROOT="${8:?missing scratch root}"
MODEL_4B_REVISION="${9:?missing Gemma-3-4B revision}"
MODEL_9B_REVISION="${10:?missing Gemma-2-9B revision}"
GLUE_EVAL_ROOT="${11:?missing pinned GLUE evaluation assets}"
COVERAGE_PROFILE="${12:?missing coverage profile}"
MODEL_12B_REVISION="${13:?missing Gemma-3-12B revision}"
STEP_GRES="${14:?missing per-lane GPU GRES}"
ALLOCATION_LANES="${15:?missing GPU lane count}"
STEP_MEMORY="${16:?missing per-lane host memory}"
EXPECTED_CPUS_PER_TASK="${17:?missing CPUs per task}"

PYTHON_BIN="${ENV_PREFIX}/bin/python"
CAMPAIGN_ROOT="${RUN_ROOT}/campaigns/${CAMPAIGN_ID}"
STATUS_ROOT="${CAMPAIGN_ROOT}/status"
JOB_ID="${SLURM_JOB_ID:-manual}"
JOB_TMP="${SLURM_TMPDIR:-${SCRATCH_ROOT}/tmp/prism-paper-coverage-${JOB_ID}}"
JOB_STATUS="${STATUS_ROOT}/job-${JOB_ID}.txt"
CURRENT_PHASE=bootstrap

write_status() {
  local state="$1" code="$2" temporary="${JOB_STATUS}.tmp.$$"
  mkdir -p "${STATUS_ROOT}"
  {
    echo "schema_version=1"
    echo "job_id=${JOB_ID}"
    echo "state=${state}"
    echo "phase=${CURRENT_PHASE}"
    echo "exit_code=${code}"
    echo "host=$(hostname)"
    echo "updated_at=$(date --iso-8601=seconds)"
    echo "campaign_root=${CAMPAIGN_ROOT}"
    echo "code_sha=${EXPECTED_REPO_SHA}"
  } >"${temporary}"
  mv -f -- "${temporary}" "${JOB_STATUS}"
}

finish() {
  local code=$? state=completed
  trap - EXIT TERM INT
  if [[ "${code}" == 130 || "${code}" == 143 ]]; then
    state=interrupted
  elif [[ "${code}" != 0 ]]; then
    state=failed
  fi
  write_status "${state}" "${code}" || true
  exit "${code}"
}
trap finish EXIT
trap 'exit 143' TERM INT

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "error: environment unavailable: ${ENV_PREFIX}" >&2
  exit 2
fi
if [[ "$(git -C "${REPO_ROOT}" rev-parse HEAD)" != "${EXPECTED_REPO_SHA}" ]] \
    || [[ -n "$(git -C "${REPO_ROOT}" status --porcelain)" ]]; then
  echo "error: immutable repository is dirty or at the wrong SHA" >&2
  exit 2
fi
if [[ "${ALLOCATION_LANES}" != 1 && "${ALLOCATION_LANES}" != 2 ]]; then
  echo "error: allocation lane count must be 1 or 2" >&2
  exit 2
fi
if [[ ( "${COVERAGE_PROFILE}" == baseline-gap-fill-cached \
      || "${COVERAGE_PROFILE}" == baseline-gap-fill-all-cached \
      || "${COVERAGE_PROFILE}" == baseline-gap-fill-math-only-cached \
      || "${COVERAGE_PROFILE}" == glue-target-baseline-screen \
      || "${COVERAGE_PROFILE}" == glue-weighted-target-screen ) \
    && "${ALLOCATION_LANES}" != 1 ]]; then
  echo "error: ${COVERAGE_PROFILE} requires one sequential GPU lane" >&2
  exit 2
fi
if [[ "${SLURM_NTASKS:-${ALLOCATION_LANES}}" != "${ALLOCATION_LANES}" \
    || "${SLURM_CPUS_PER_TASK:-${EXPECTED_CPUS_PER_TASK}}" != "${EXPECTED_CPUS_PER_TASK}" ]]; then
  echo "error: allocation task/CPU shape differs from the submission lock" >&2
  exit 2
fi
for required in \
  train_eval.py \
  configs/glue8_paper.json \
  configs/math10k_paper.json \
  scripts/build_paper_coverage_campaign.py \
  scripts/weighted_target_screen.py \
  scripts/prepare_glue_eval_assets.py \
  scripts/preflight_hpc.py \
  scripts/smoke_dp_path.py \
  scripts/summarize_telemetry.py \
  slurm/paper_coverage_lane.sh; do
  [[ -f "${REPO_ROOT}/${required}" ]] || { echo "error: missing ${required}" >&2; exit 2; }
done
if [[ "${COVERAGE_PROFILE}" != baseline-gap-fill-math-only-cached ]]; then
  "${PYTHON_BIN}" - "${GLUE_EVAL_ROOT}" <<'PY'
import sys
from pathlib import Path
from src.prism_cli.eval_glue import _load_glue_asset_manifest, _load_glue_dataset

root = Path(sys.argv[1])
manifest = _load_glue_asset_manifest(root)
for task in manifest['tasks']:
    _load_glue_dataset(task, root)
print(f"offline_glue_assets={root} revision={manifest['dataset_revision']} content={manifest['content_sha256']}")
PY
fi

cd -- "${REPO_ROOT}"
export PRISM_REPO_ROOT="${REPO_ROOT}"
export PRISM_RUN_ROOT="${RUN_ROOT}"
export HF_HOME="${SHARED_HF_HOME}"
export HF_HUB_CACHE="${SHARED_HF_HOME}/hub"
export HUGGINGFACE_HUB_CACHE="${SHARED_HF_HOME}/hub"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_XET=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
mkdir -p "${CAMPAIGN_ROOT}/artifacts" "${CAMPAIGN_ROOT}/plans" \
  "${CAMPAIGN_ROOT}/preflight" "${CAMPAIGN_ROOT}/selection" \
  "${CAMPAIGN_ROOT}/smoke" "${STATUS_ROOT}" "${JOB_TMP}"

CURRENT_PHASE=preflight
write_status running 0
echo "job=${JOB_ID} campaign=${CAMPAIGN_ID} host=$(hostname) code=${EXPECTED_REPO_SHA}"
nvidia-smi
"${PYTHON_BIN}" - "${ALLOCATION_LANES}" <<'PY'
import sys
import torch

torch.cuda.init()
expected = int(sys.argv[1])
if not torch.cuda.is_available() or torch.cuda.device_count() != expected:
    raise SystemExit(f"expected exactly {expected} visible CUDA GPU(s), got {torch.cuda.device_count()}")
for index in range(expected):
    x = torch.ones((16, 16), device=f"cuda:{index}")
    if float((x @ x).sum().item()) != 4096.0:
        raise SystemExit(f"CUDA checksum failed on device {index}")
    torch.cuda.synchronize(index)
print([torch.cuda.get_device_name(i) for i in range(expected)])
PY
"${PYTHON_BIN}" -m pip check
"${PYTHON_BIN}" -m compileall -q train_eval.py src scripts

# The parent allocation deliberately has two tasks so it can host two
# independent one-GPU lanes.  The repository preflight is single-process-aware
# and must therefore run inside an exact one-task Slurm step; otherwise the
# inherited parent SLURM_NTASKS=2 is correctly rejected as distributed
# execution before any training starts.
single_gpu_step() {
  srun --export=ALL --exclusive --exact --kill-on-bad-exit=1 \
    --nodes=1 --ntasks=1 --cpus-per-task="${EXPECTED_CPUS_PER_TASK}" \
    --mem="${STEP_MEMORY}" --gres="${STEP_GRES}" \
    "$@"
}

verify_plan_sidecar() {
  local plan="${1:?missing plan path}" sidecar
  sidecar="${plan}.sha256"
  if [[ ! -s "${plan}" || ! -s "${sidecar}" ]]; then
    echo "error: immutable plan or SHA sidecar is missing: ${plan}" >&2
    return 2
  fi
  (
    cd -- "$(dirname -- "${plan}")"
    sha256sum --check --status "$(basename -- "${sidecar}")"
  ) || {
    echo "error: immutable plan SHA check failed: ${plan}" >&2
    return 2
  }
}

if [[ "${COVERAGE_PROFILE}" != baseline-gap-fill-math-only-cached ]]; then
  single_gpu_step "${PYTHON_BIN}" scripts/preflight_hpc.py \
    --dataset glue8 \
    --data-path "${REPO_ROOT}/LLM-Adapters/ft-training_set/glue8_1250.json" \
    --expected-data-sha256 281bad3305e8be7a7660b64b22704a9462f983b90df5d6f0c1c4abb649d6d091 \
    --output-root "${CAMPAIGN_ROOT}/preflight/glue8" \
    --require-cuda --model-id google/gemma-3-4b-pt --model-revision "${MODEL_4B_REVISION}"
fi
if [[ "${COVERAGE_PROFILE}" != glue-slaclip-screen \
    && "${COVERAGE_PROFILE}" != glue-high-c-refinement \
    && "${COVERAGE_PROFILE}" != glue-r8-slack-screen \
    && "${COVERAGE_PROFILE}" != glue-target-baseline-screen \
    && "${COVERAGE_PROFILE}" != glue-weighted-target-screen ]]; then
  run_math_preflight() {
    local model_id="${1:?missing Math preflight model}" \
      model_revision="${2:?missing Math preflight revision}" \
      output_name="${3:?missing Math preflight output name}"
    single_gpu_step "${PYTHON_BIN}" scripts/preflight_hpc.py \
      --dataset math10k \
      --data-path "${REPO_ROOT}/LLM-Adapters/ft-training_set/math_10k.json" \
      --expected-data-sha256 0342d0d860ad8592b579329337c90e42eefd3d9f2898043140cbd120630418b8 \
      --output-root "${CAMPAIGN_ROOT}/preflight/${output_name}" \
      --require-cuda --model-id "${model_id}" \
      --model-revision "${model_revision}"
  }
  if [[ "${COVERAGE_PROFILE}" == baseline-gap-fill-cached ]]; then
    run_math_preflight \
      google/gemma-3-4b-pt "${MODEL_4B_REVISION}" math10k
  elif [[ "${COVERAGE_PROFILE}" == baseline-gap-fill-all-cached \
      || "${COVERAGE_PROFILE}" == baseline-gap-fill-math-only-cached ]]; then
    run_math_preflight \
      google/gemma-3-4b-pt "${MODEL_4B_REVISION}" math10k-4b
    run_math_preflight \
      google/gemma-3-12b-pt "${MODEL_12B_REVISION}" math10k-12b
  else
    run_math_preflight \
      google/gemma-2-9b "${MODEL_9B_REVISION}" math10k
  fi
fi

CURRENT_PHASE=synthetic-smoke
write_status running 0
single_gpu_step "${PYTHON_BIN}" scripts/smoke_dp_path.py \
  --device cuda --method both --steps 2 --clip-norm 1.0 \
  --slaclip-target-non-small-clip-fraction 0.9 --slaclip-eta 0.05 \
  --slaclip-num-slots 15 --slaclip-c-min 0.1 --slaclip-c-max 15 \
  >"${CAMPAIGN_ROOT}/smoke/synthetic-${JOB_ID}.json"

CURRENT_PHASE=plan-lock
write_status running 0
"${PYTHON_BIN}" scripts/build_paper_coverage_campaign.py prepare \
  --campaign-root "${CAMPAIGN_ROOT}" \
  --code-sha "${EXPECTED_REPO_SHA}" \
  --model-4b-revision "${MODEL_4B_REVISION}" \
  --model-9b-revision "${MODEL_9B_REVISION}" \
  --model-12b-revision "${MODEL_12B_REVISION}" \
  --profile "${COVERAGE_PROFILE}" \
  >"${CAMPAIGN_ROOT}/plans/prepare-${JOB_ID}.out" \
  2>"${CAMPAIGN_ROOT}/plans/prepare-${JOB_ID}.err"

CURRENT_PHASE=parallel-paper-coverage
write_status running 0
LANE_SCRIPT="${REPO_ROOT}/slurm/paper_coverage_lane.sh"
if [[ "${COVERAGE_PROFILE}" == glue-high-c-refinement \
    || "${COVERAGE_PROFILE}" == glue-r8-slack-screen \
    || "${COVERAGE_PROFILE}" == glue-weighted-target-screen ]]; then
  if [[ "${ALLOCATION_LANES}" != 1 ]]; then
    echo "error: ${COVERAGE_PROFILE} requires one sequential GPU lane" >&2
    exit 2
  fi
  lock_command=lock-high-c-refinement
  stage1_phase=stage1-high-c-fixed-screen
  stage2_plan=stage2-slaclip.tsv
  if [[ "${COVERAGE_PROFILE}" == glue-r8-slack-screen ]]; then
    lock_command=lock-r8-slack-screen
    stage1_phase=stage1-r8-fixed-screen
  fi
  if [[ "${COVERAGE_PROFILE}" == glue-weighted-target-screen ]]; then
    lock_command=lock-weighted-target-screen
    stage1_phase=stage1-weighted-fixed-boundary-screen
    stage2_plan=stage2-weighted.tsv
  fi
  CURRENT_PHASE="${stage1_phase}"
  write_status running 0
  verify_plan_sidecar "${CAMPAIGN_ROOT}/plans/stage1-fixed.tsv"
  srun --export=ALL --exclusive --exact --kill-on-bad-exit=1 \
    --nodes=1 --ntasks=1 --cpus-per-task="${EXPECTED_CPUS_PER_TASK}" \
    --mem="${STEP_MEMORY}" --gres="${STEP_GRES}" \
    "${LANE_SCRIPT}" 0 "${CAMPAIGN_ROOT}/plans/stage1-fixed.tsv" "${REPO_ROOT}" \
    "${ENV_PREFIX}" "${CAMPAIGN_ROOT}" "${EXPECTED_REPO_SHA}" "${JOB_TMP}" \
    "${GLUE_EVAL_ROOT}" "${COVERAGE_PROFILE}" "${MODEL_12B_REVISION}" \
    "${EXPECTED_CPUS_PER_TASK}" true

  CURRENT_PHASE=stage1-selection-lock
  write_status running 0
  "${PYTHON_BIN}" scripts/build_paper_coverage_campaign.py \
    "${lock_command}" --campaign-root "${CAMPAIGN_ROOT}" \
    >"${CAMPAIGN_ROOT}/selection/lock-${JOB_ID}.out" \
    2>"${CAMPAIGN_ROOT}/selection/lock-${JOB_ID}.err"

  CURRENT_PHASE=stage2-derived-full-slaclip-screen
  write_status running 0
  verify_plan_sidecar "${CAMPAIGN_ROOT}/plans/${stage2_plan}"
  srun --export=ALL --exclusive --exact --kill-on-bad-exit=1 \
    --nodes=1 --ntasks=1 --cpus-per-task="${EXPECTED_CPUS_PER_TASK}" \
    --mem="${STEP_MEMORY}" --gres="${STEP_GRES}" \
    "${LANE_SCRIPT}" 0 "${CAMPAIGN_ROOT}/plans/${stage2_plan}" "${REPO_ROOT}" \
    "${ENV_PREFIX}" "${CAMPAIGN_ROOT}" "${EXPECTED_REPO_SHA}" "${JOB_TMP}" \
    "${GLUE_EVAL_ROOT}" "${COVERAGE_PROFILE}" "${MODEL_12B_REVISION}" \
    "${EXPECTED_CPUS_PER_TASK}" false
elif [[ "${ALLOCATION_LANES}" == 1 ]]; then
  srun --export=ALL --exclusive --exact --kill-on-bad-exit=1 \
    --nodes=1 --ntasks=1 --cpus-per-task="${EXPECTED_CPUS_PER_TASK}" \
    --mem="${STEP_MEMORY}" --gres="${STEP_GRES}" \
    "${LANE_SCRIPT}" 0 "${CAMPAIGN_ROOT}/plans/sequential.tsv" "${REPO_ROOT}" \
    "${ENV_PREFIX}" "${CAMPAIGN_ROOT}" "${EXPECTED_REPO_SHA}" "${JOB_TMP}" \
    "${GLUE_EVAL_ROOT}" "${COVERAGE_PROFILE}" "${MODEL_12B_REVISION}" \
    "${EXPECTED_CPUS_PER_TASK}"
else
  srun --export=ALL --exclusive --exact --kill-on-bad-exit=1 \
    --nodes=1 --ntasks=1 --cpus-per-task="${EXPECTED_CPUS_PER_TASK}" \
    --mem="${STEP_MEMORY}" --gres="${STEP_GRES}" \
    "${LANE_SCRIPT}" 0 "${CAMPAIGN_ROOT}/plans/lane-0.tsv" "${REPO_ROOT}" \
    "${ENV_PREFIX}" "${CAMPAIGN_ROOT}" "${EXPECTED_REPO_SHA}" "${JOB_TMP}" \
    "${GLUE_EVAL_ROOT}" "${COVERAGE_PROFILE}" "${MODEL_12B_REVISION}" \
    "${EXPECTED_CPUS_PER_TASK}" &
  PID0=$!
  srun --export=ALL --exclusive --exact --kill-on-bad-exit=1 \
    --nodes=1 --ntasks=1 --cpus-per-task="${EXPECTED_CPUS_PER_TASK}" \
    --mem="${STEP_MEMORY}" --gres="${STEP_GRES}" \
    "${LANE_SCRIPT}" 1 "${CAMPAIGN_ROOT}/plans/lane-1.tsv" "${REPO_ROOT}" \
    "${ENV_PREFIX}" "${CAMPAIGN_ROOT}" "${EXPECTED_REPO_SHA}" "${JOB_TMP}" \
    "${GLUE_EVAL_ROOT}" "${COVERAGE_PROFILE}" "${MODEL_12B_REVISION}" \
    "${EXPECTED_CPUS_PER_TASK}" &
  PID1=$!
  LANE_CODE=0
  wait "${PID0}" || LANE_CODE=$?
  wait "${PID1}" || LANE_CODE=$?
  if [[ "${LANE_CODE}" != 0 ]]; then
    echo "error: at least one GPU lane failed with exit ${LANE_CODE}" >&2
    exit "${LANE_CODE}"
  fi
fi

CURRENT_PHASE=analysis
write_status running 0
"${PYTHON_BIN}" scripts/build_paper_coverage_campaign.py analyze \
  --campaign-root "${CAMPAIGN_ROOT}" \
  >"${CAMPAIGN_ROOT}/artifacts/analyze-${JOB_ID}.out" \
  2>"${CAMPAIGN_ROOT}/artifacts/analyze-${JOB_ID}.err"

CURRENT_PHASE=complete
write_status completed 0
echo "Paper-coverage campaign completed at $(date --iso-8601=seconds)."
