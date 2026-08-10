#!/usr/bin/env bash

# One-A100, one-allocation, sequential Math-10K low-target Full-SlaClip worker.
# Site-specific scheduler options and portable paths are supplied by the
# submission wrapper; this file deliberately contains no #SBATCH directives.
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
MODEL_ID="${9:?missing model ID}"
MODEL_REVISION="${10:?missing model revision}"
MODEL_SLUG="${11:?missing model slug}"

PYTHON_BIN="${ENV_PREFIX}/bin/python"
CAMPAIGN_ROOT="${RUN_ROOT}/campaigns/${CAMPAIGN_ID}"
STAGE1_PLAN="${CAMPAIGN_ROOT}/plans/stage1.tsv"
FINAL_PLAN="${CAMPAIGN_ROOT}/plans/final.tsv"
STATUS_ROOT="${CAMPAIGN_ROOT}/status"
JOB_ID="${SLURM_JOB_ID:-manual}"
JOB_TMP="${SLURM_TMPDIR:-${SCRATCH_ROOT}/tmp/prism-low-target-${JOB_ID}}"
JOB_STATUS="${STATUS_ROOT}/job-${JOB_ID}.txt"
CURRENT_PHASE="bootstrap"
ACTIVE_STATUS=""
ACTIVE_PHASE=""
ACTIVE_CANDIDATE=""
ACTIVE_SEED=""

write_job_status() {
  local state="$1" exit_code="$2" temporary="${JOB_STATUS}.tmp.$$"
  mkdir -p "${STATUS_ROOT}"
  {
    echo "schema_version=1"
    echo "job_id=${JOB_ID}"
    echo "state=${state}"
    echo "phase=${CURRENT_PHASE}"
    echo "exit_code=${exit_code}"
    echo "updated_at=$(date --iso-8601=seconds)"
    echo "host=$(hostname)"
    echo "campaign_root=${CAMPAIGN_ROOT}"
    echo "code_sha=${EXPECTED_REPO_SHA}"
  } >"${temporary}"
  mv -f -- "${temporary}" "${JOB_STATUS}"
}

write_arm_status() {
  local path="$1" state="$2" phase="$3" candidate="$4" seed="$5" exit_code="$6"
  local temporary="${path}.tmp.$$"
  mkdir -p "$(dirname -- "${path}")"
  {
    echo "schema_version=1"
    echo "job_id=${JOB_ID}"
    echo "state=${state}"
    echo "phase=${phase}"
    echo "candidate_id=${candidate}"
    echo "seed=${seed}"
    echo "exit_code=${exit_code}"
    echo "updated_at=$(date --iso-8601=seconds)"
    echo "code_sha=${EXPECTED_REPO_SHA}"
  } >"${temporary}"
  mv -f -- "${temporary}" "${path}"
}

finish() {
  local exit_code=$? state="completed"
  trap - EXIT TERM INT
  if [[ "${exit_code}" != "0" ]]; then
    state="failed"
    if [[ -n "${ACTIVE_STATUS}" ]]; then
      write_arm_status \
        "${ACTIVE_STATUS}" failed "${ACTIVE_PHASE}" \
        "${ACTIVE_CANDIDATE}" "${ACTIVE_SEED}" "${exit_code}" || true
    fi
  fi
  write_job_status "${state}" "${exit_code}" || true
  exit "${exit_code}"
}
trap finish EXIT
trap 'exit 143' TERM INT

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "error: Python environment is unavailable: ${PYTHON_BIN}" >&2
  exit 2
fi
if [[ ! "${EXPECTED_REPO_SHA}" =~ ^[0-9a-f]{40}$ ]]; then
  echo "error: expected repository SHA must be full lowercase hex" >&2
  exit 2
fi
if [[ "$(git -C "${REPO_ROOT}" rev-parse HEAD)" != "${EXPECTED_REPO_SHA}" ]]; then
  echo "error: immutable repository SHA changed after submission" >&2
  exit 2
fi
if [[ -n "$(git -C "${REPO_ROOT}" status --porcelain)" ]]; then
  echo "error: immutable repository is dirty: ${REPO_ROOT}" >&2
  exit 2
fi
if [[ "${SLURM_NTASKS:-1}" != "1" || "${SLURM_CPUS_PER_TASK:-8}" != "8" ]]; then
  echo "error: worker requires one task with eight CPUs" >&2
  exit 2
fi
for required in \
  train_eval.py \
  configs/math10k_paper.json \
  scripts/build_math10k_4b_low_target_campaign.py \
  scripts/preflight_hpc.py \
  scripts/smoke_dp_path.py \
  scripts/summarize_telemetry.py; do
  if [[ ! -f "${REPO_ROOT}/${required}" ]]; then
    echo "error: immutable source is missing ${required}" >&2
    exit 2
  fi
done
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
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export TMPDIR="${JOB_TMP}"
mkdir -p \
  "${CAMPAIGN_ROOT}/artifacts" \
  "${CAMPAIGN_ROOT}/plans" \
  "${CAMPAIGN_ROOT}/screen/runs" \
  "${CAMPAIGN_ROOT}/selection" \
  "${CAMPAIGN_ROOT}/final/${MODEL_SLUG}" \
  "${CAMPAIGN_ROOT}/preflight" \
  "${CAMPAIGN_ROOT}/smoke" \
  "${CAMPAIGN_ROOT}/cache" \
  "${STATUS_ROOT}" \
  "${JOB_TMP}"

run_training_spec() {
  local spec="$1"
  local kind phase candidate role seed method clip rho eta schedule arm_root
  IFS='|' read -r \
    kind phase candidate role seed method clip rho eta schedule arm_root <<<"${spec}"
  if [[ "${kind}" != "train" || "${schedule}" != "NA" ]]; then
    echo "error: invalid low-target training spec: ${spec}" >&2
    return 2
  fi
  if [[ "${method}" != "baseline" && "${method}" != "slaclip" ]]; then
    echo "error: campaign supports only fixed PRISM and Full SlaClip: ${spec}" >&2
    return 2
  fi
  local steps
  case "${phase}" in
    selection) steps=150 ;;
    final) steps=300 ;;
    *)
      echo "error: unknown campaign phase: ${phase}" >&2
      return 2
      ;;
  esac
  local adapter_dir="${arm_root}/adapter"
  local result_dir="${arm_root}/results"
  local status_file="${arm_root}/orchestration-status.txt"
  local arm_tmp="${JOB_TMP}/${phase}-${candidate}-seed${seed}"
  mkdir -p \
    "${adapter_dir}" "${result_dir}" "${arm_root}/logs" \
    "${arm_root}/cache/datasets" "${arm_root}/cache/triton" \
    "${arm_root}/cache/torch-extensions" "${arm_tmp}"
  ACTIVE_STATUS="${status_file}"
  ACTIVE_PHASE="${phase}"
  ACTIVE_CANDIDATE="${candidate}"
  ACTIVE_SEED="${seed}"
  write_arm_status "${status_file}" running "${phase}" "${candidate}" "${seed}" 0
  {
    echo "schema_version=1"
    echo "phase=${phase}"
    echo "candidate_id=${candidate}"
    echo "role=${role}"
    echo "seed=${seed}"
    echo "method=${method}"
    echo "fixed_or_initial_C=${clip}"
    echo "full_slaclip_conditional_rho=${rho}"
    echo "full_slaclip_eta=${eta}"
    echo "dynamic_global_target=p_star_t=rho*(1-z_t)"
    echo "rho_is_not_a_fixed_global_clipping_fraction=true"
    echo "K=15"
    echo "C_bounds=0.1,15"
    echo "code_sha=${EXPECTED_REPO_SHA}"
    echo "model=${MODEL_ID}@${MODEL_REVISION}"
    echo "raw_training_telemetry=NONPRIVATE"
  } >"${arm_root}/arm-manifest.txt"

  local -a args=(
    "${PYTHON_BIN}" -u train_eval.py
    --config configs/math10k_paper.json
    --dataset math10k
    --method "${method}"
    --privacy dp
    --base_model "${MODEL_ID}"
    --model_revision "${MODEL_REVISION}"
    --seed "${seed}"
    --run_name "${phase}-${MODEL_SLUG}-${role}-${candidate}-seed${seed}"
    --steps "${steps}"
    --batch_size 64
    --micro_batch_size 4
    --lr 0.0003
    --cutoff_len 256
    --train_on_inputs true
    --epsilon 6.0
    --delta 1e-5
    --initial_clip_threshold "${clip}"
    --slaclip_num_slots 15
    --telemetry_mode research_raw
    --allow_non_private_telemetry
    --raw_hist_bins 128
    --raw_hist_max 30.0
    --require_cuda
    --resume
    --checkpoint_every 25
    --run_train true
    --output_dir "${adapter_dir}"
    --result_dir "${result_dir}"
  )
  if [[ "${method}" == "slaclip" ]]; then
    if [[ "${rho}" == "NA" || "${eta}" == "NA" ]]; then
      echo "error: Full SlaClip spec is missing rho or eta" >&2
      return 2
    fi
    args+=(
      --slaclip_target_non_small_clip_fraction "${rho}"
      --slaclip_eta "${eta}"
      --slaclip_c_min 0.1
      --slaclip_c_max 15.0
    )
  elif [[ "${rho}" != "NA" || "${eta}" != "NA" ]]; then
    echo "error: fixed candidate unexpectedly carries SlaClip parameters" >&2
    return 2
  fi
  case "${phase}" in
    selection)
      args+=(
        --val_set_size 500
        --validation_seed 1729
        --validation_batch_size 8
        --validation_eval_interval 50
        --validation_generate_numeric
        --validation_num_beams 1
        --validation_max_new_tokens 128
        --validation_max_input_length 512
        --validation_data_is_public
        --protocol_stage selection
        --run_eval false
      )
      ;;
    final)
      args+=(
        --val_set_size 0
        --validation_eval_interval 0
        --protocol_stage final
        --run_eval true
        --eval_batch_size 8
        --num_beams 4
        --max_new_tokens 256
        --max_input_length 1024
      )
      ;;
  esac

  (
    export HF_DATASETS_CACHE="${arm_root}/cache/datasets"
    export TRITON_CACHE_DIR="${arm_root}/cache/triton"
    export TORCH_EXTENSIONS_DIR="${arm_root}/cache/torch-extensions"
    export PYTHONPYCACHEPREFIX="${arm_root}/cache/pycache"
    export TMPDIR="${arm_tmp}"
    "${args[@]}"
  ) >"${arm_root}/logs/train-${JOB_ID}.out" \
    2>"${arm_root}/logs/train-${JOB_ID}.err"

  local raw_log="${result_dir}/research_raw/NON_PRIVATE_train_log.jsonl"
  local safe_log="${adapter_dir}/train_log.jsonl"
  if [[ ! -s "${raw_log}" || ! -s "${safe_log}" ]]; then
    echo "error: completed arm lacks telemetry: ${arm_root}" >&2
    return 3
  fi
  "${PYTHON_BIN}" scripts/summarize_telemetry.py \
    "${raw_log}" \
    --safe-log "${safe_log}" \
    --format both \
    --csv-out "${result_dir}/research_raw/telemetry_steps.csv" \
    --json-out "${result_dir}/research_raw/telemetry_summary.json" \
    >"${arm_root}/logs/summarize-${JOB_ID}.out" \
    2>"${arm_root}/logs/summarize-${JOB_ID}.err"
  write_arm_status "${status_file}" completed "${phase}" "${candidate}" "${seed}" 0
  ACTIVE_STATUS=""
  ACTIVE_PHASE=""
  ACTIVE_CANDIDATE=""
  ACTIVE_SEED=""
}

run_plan() {
  local plan="$1" expected_min="$2" expected_max="$3"
  if [[ ! -s "${plan}" ]]; then
    echo "error: campaign plan is missing: ${plan}" >&2
    return 2
  fi
  local count
  count="$(awk 'NF {count++} END {print count+0}' "${plan}")"
  if (( count < expected_min || count > expected_max )); then
    echo "error: plan ${plan} has ${count} arms; expected ${expected_min}..${expected_max}" >&2
    return 2
  fi
  local spec
  while IFS= read -r spec; do
    [[ -n "${spec}" ]] || continue
    run_training_spec "${spec}"
  done <"${plan}"
}

run_real_model_smoke() {
  local method="$1"
  local arm_root="${CAMPAIGN_ROOT}/smoke/real/${method}"
  local adapter_dir="${arm_root}/adapter"
  local result_dir="${arm_root}/results"
  mkdir -p "${adapter_dir}" "${result_dir}" "${arm_root}/cache/datasets"
  local -a args=(
    "${PYTHON_BIN}" -u train_eval.py
    --config configs/math10k_paper.json
    --dataset math10k
    --method "${method}"
    --privacy dp
    --base_model "${MODEL_ID}"
    --model_revision "${MODEL_REVISION}"
    --seed 42
    --run_name "low-target-smoke-${MODEL_SLUG}-${method}"
    --steps 2
    --batch_size 64
    --micro_batch_size 4
    --initial_clip_threshold 2.0
    --val_set_size 8
    --validation_seed 1729
    --validation_batch_size 4
    --validation_eval_interval 1
    --validation_generate_numeric
    --validation_num_beams 1
    --validation_max_new_tokens 32
    --validation_max_input_length 512
    --validation_data_is_public
    --protocol_stage selection
    --telemetry_mode dp_safe
    --slaclip_num_slots 15
    --require_cuda
    --resume
    --checkpoint_every 1
    --run_train true
    --run_eval false
    --output_dir "${adapter_dir}"
    --result_dir "${result_dir}"
  )
  if [[ "${method}" == "slaclip" ]]; then
    args+=(
      --slaclip_target_non_small_clip_fraction 0.8
      --slaclip_eta 0.1
      --slaclip_c_min 0.1
      --slaclip_c_max 15.0
    )
  elif [[ "${method}" != "baseline" ]]; then
    echo "error: unsupported real-model smoke method=${method}" >&2
    return 2
  fi
  (
    export HF_DATASETS_CACHE="${arm_root}/cache/datasets"
    export TRITON_CACHE_DIR="${arm_root}/cache/triton"
    export TORCH_EXTENSIONS_DIR="${arm_root}/cache/torch-extensions"
    export PYTHONPYCACHEPREFIX="${arm_root}/cache/pycache"
    export TMPDIR="${JOB_TMP}/real-smoke-${method}"
    mkdir -p "${TMPDIR}" "${TRITON_CACHE_DIR}" "${TORCH_EXTENSIONS_DIR}"
    "${args[@]}"
  ) >"${arm_root}/train-${JOB_ID}.out" 2>"${arm_root}/train-${JOB_ID}.err"
  if [[ ! -s "${adapter_dir}/adapter_model.safetensors" \
      || ! -s "${result_dir}/validation/validation_metrics.json" ]]; then
    echo "error: ${method} real-model smoke did not create expected artifacts" >&2
    return 3
  fi
}

CURRENT_PHASE="preflight"
write_job_status running 0
echo "job=${JOB_ID} campaign=${CAMPAIGN_ID} host=$(hostname) code=${EXPECTED_REPO_SHA}"
nvidia-smi
"${PYTHON_BIN}" - <<'PY'
import torch

torch.cuda.init()
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise SystemExit(f"expected exactly one visible CUDA GPU, got {torch.cuda.device_count()}")
x = torch.ones((16, 16), device="cuda:0")
if float((x @ x).sum().item()) != 4096.0:
    raise SystemExit("CUDA tensor checksum failed")
torch.cuda.synchronize(0)
print(f"gpu={torch.cuda.get_device_name(0)} torch={torch.__version__} cuda={torch.version.cuda}")
PY
"${PYTHON_BIN}" -m pip check
"${PYTHON_BIN}" -m compileall -q train_eval.py src scripts
"${PYTHON_BIN}" scripts/preflight_hpc.py \
  --dataset math10k \
  --data-path "${REPO_ROOT}/LLM-Adapters/ft-training_set/math_10k.json" \
  --expected-data-sha256 0342d0d860ad8592b579329337c90e42eefd3d9f2898043140cbd120630418b8 \
  --output-root "${CAMPAIGN_ROOT}/preflight" \
  --require-cuda \
  --model-id "${MODEL_ID}" \
  --model-revision "${MODEL_REVISION}"

CURRENT_PHASE="synthetic-smoke"
write_job_status running 0
"${PYTHON_BIN}" scripts/smoke_dp_path.py \
  --device cuda \
  --method both \
  --steps 2 \
  --clip-norm 2.5 \
  --slaclip-target-non-small-clip-fraction 0.8 \
  --slaclip-eta 0.2 \
  --slaclip-num-slots 15 \
  --slaclip-c-min 0.1 \
  --slaclip-c-max 15 \
  >"${CAMPAIGN_ROOT}/smoke/synthetic-${JOB_ID}.json"

CURRENT_PHASE="real-model-smoke"
write_job_status running 0
run_real_model_smoke baseline
run_real_model_smoke slaclip

CURRENT_PHASE="registry-lock"
write_job_status running 0
"${PYTHON_BIN}" scripts/build_math10k_4b_low_target_campaign.py prepare \
  --campaign-root "${CAMPAIGN_ROOT}" \
  --code-sha "${EXPECTED_REPO_SHA}" \
  --model-id "${MODEL_ID}" \
  --model-revision "${MODEL_REVISION}" \
  >"${CAMPAIGN_ROOT}/selection/prepare-${JOB_ID}.out" \
  2>"${CAMPAIGN_ROOT}/selection/prepare-${JOB_ID}.err"

CURRENT_PHASE="public-validation-screen"
write_job_status running 0
run_plan "${STAGE1_PLAN}" 4 4

CURRENT_PHASE="selection-lock"
write_job_status running 0
"${PYTHON_BIN}" scripts/build_math10k_4b_low_target_campaign.py lock \
  --campaign-root "${CAMPAIGN_ROOT}" \
  >"${CAMPAIGN_ROOT}/selection/lock-${JOB_ID}.out" \
  2>"${CAMPAIGN_ROOT}/selection/lock-${JOB_ID}.err"

CURRENT_PHASE="same-a100-fresh-seed-paired-confirmation"
write_job_status running 0
run_plan "${FINAL_PLAN}" 15 15

CURRENT_PHASE="analysis"
write_job_status running 0
"${PYTHON_BIN}" scripts/build_math10k_4b_low_target_campaign.py analyze \
  --campaign-root "${CAMPAIGN_ROOT}" \
  >"${CAMPAIGN_ROOT}/artifacts/analyze-${JOB_ID}.out" \
  2>"${CAMPAIGN_ROOT}/artifacts/analyze-${JOB_ID}.err"

CURRENT_PHASE="complete"
write_job_status completed 0
echo "Low-target Full-SlaClip campaign completed at $(date --iso-8601=seconds)."
