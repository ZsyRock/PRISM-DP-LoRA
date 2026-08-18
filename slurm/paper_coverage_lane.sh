#!/usr/bin/env bash

# Run one GPU lane of the paper-coverage breadth screen.  This script is
# launched only through an exclusive srun step inside the parent allocation.
set -Eeuo pipefail
umask 077

LANE="${1:?missing lane}"
PLAN="${2:?missing lane plan}"
REPO_ROOT="${3:?missing repository root}"
ENV_PREFIX="${4:?missing environment prefix}"
CAMPAIGN_ROOT="${5:?missing campaign root}"
EXPECTED_REPO_SHA="${6:?missing repository SHA}"
JOB_TMP_ROOT="${7:?missing job temporary root}"

PYTHON_BIN="${ENV_PREFIX}/bin/python"
JOB_ID="${SLURM_JOB_ID:-manual}"
LANE_TMP="${JOB_TMP_ROOT}/lane-${LANE}"
LANE_STATUS="${CAMPAIGN_ROOT}/status/lane-${LANE}.txt"
CURRENT_ARM="bootstrap"

write_lane_status() {
  local state="$1" exit_code="$2" temporary="${LANE_STATUS}.tmp.$$"
  mkdir -p "$(dirname -- "${LANE_STATUS}")"
  {
    echo "schema_version=1"
    echo "job_id=${JOB_ID}"
    echo "lane=${LANE}"
    echo "state=${state}"
    echo "current_arm=${CURRENT_ARM}"
    echo "exit_code=${exit_code}"
    echo "updated_at=$(date --iso-8601=seconds)"
  } >"${temporary}"
  mv -f -- "${temporary}" "${LANE_STATUS}"
}

finish() {
  local code=$? state=completed
  trap - EXIT TERM INT
  if [[ "${code}" != 0 ]]; then state=failed; fi
  write_lane_status "${state}" "${code}" || true
  exit "${code}"
}
trap finish EXIT
trap 'exit 143' TERM INT

if [[ "${LANE}" != 0 && "${LANE}" != 1 ]]; then
  echo "error: lane must be 0 or 1" >&2
  exit 2
fi
if [[ ! -x "${PYTHON_BIN}" || ! -s "${PLAN}" ]]; then
  echo "error: lane Python or plan is unavailable" >&2
  exit 2
fi
if [[ "$(git -C "${REPO_ROOT}" rev-parse HEAD)" != "${EXPECTED_REPO_SHA}" ]] \
    || [[ -n "$(git -C "${REPO_ROOT}" status --porcelain)" ]]; then
  echo "error: staged source is dirty or at the wrong SHA" >&2
  exit 2
fi
if [[ "${SLURM_CPUS_PER_TASK:-12}" != 12 ]]; then
  echo "error: each lane requires 12 CPUs" >&2
  exit 2
fi

cd -- "${REPO_ROOT}"
mkdir -p "${LANE_TMP}" "${CAMPAIGN_ROOT}/runs" "${CAMPAIGN_ROOT}/smoke/lane-${LANE}"
export TMPDIR="${LANE_TMP}/tmp"
export HF_DATASETS_CACHE="${LANE_TMP}/datasets"
export TRITON_CACHE_DIR="${LANE_TMP}/triton"
export TORCH_EXTENSIONS_DIR="${LANE_TMP}/torch-extensions"
export PYTHONPYCACHEPREFIX="${LANE_TMP}/pycache"
export OMP_NUM_THREADS=12
export MKL_NUM_THREADS=12
mkdir -p "${TMPDIR}" "${HF_DATASETS_CACHE}" "${TRITON_CACHE_DIR}" "${TORCH_EXTENSIONS_DIR}"

run_training() {
  local spec="$1"
  local plan_lane arm_id setting_id dataset model_slug model_id model_revision
  local epsilon lora_r method initial_c rho eta steps lr cutoff train_inputs seed relative_root
  IFS='|' read -r plan_lane arm_id setting_id dataset model_slug model_id model_revision \
    epsilon lora_r method initial_c rho eta steps lr cutoff train_inputs seed relative_root <<<"${spec}"
  if [[ "${plan_lane}" != "${LANE}" || -z "${arm_id}" || -z "${relative_root}" ]]; then
    echo "error: invalid or cross-lane plan row: ${spec}" >&2
    return 2
  fi
  if [[ "${relative_root}" == /* || "${relative_root}" == *..* ]]; then
    echo "error: unsafe relative run path: ${relative_root}" >&2
    return 2
  fi
  if [[ "${method}" != baseline && "${method}" != slaclip ]]; then
    echo "error: unsupported campaign method: ${method}" >&2
    return 2
  fi
  local config="configs/math10k_paper.json"
  if [[ "${dataset}" == glue8 ]]; then config="configs/glue8_paper.json"; fi
  local arm_root="${CAMPAIGN_ROOT}/${relative_root}"
  local adapter_dir="${arm_root}/adapter" result_dir="${arm_root}/results"
  local status_file="${arm_root}/orchestration-status.txt"
  mkdir -p "${adapter_dir}" "${result_dir}/research_raw" "${arm_root}/logs"
  CURRENT_ARM="${arm_id}"
  write_lane_status running 0
  {
    echo "schema_version=1"
    echo "arm_id=${arm_id}"
    echo "setting_id=${setting_id}"
    echo "paper_fixed_or_initial_C=${initial_c}"
    echo "full_slaclip_conditional_rho=${rho}"
    echo "full_slaclip_eta=${eta}"
    echo "dynamic_global_target=p_star_t=rho*(1-z_t)"
    echo "K=15"
    echo "C_bounds=0.1,15"
    echo "seed=${seed}"
    echo "code_sha=${EXPECTED_REPO_SHA}"
    echo "model=${model_id}@${model_revision}"
    echo "raw_training_telemetry=NONPRIVATE"
  } >"${arm_root}/arm-manifest.txt"

  local -a args=(
    "${PYTHON_BIN}" -u train_eval.py
    --config "${config}"
    --dataset "${dataset}"
    --method "${method}"
    --privacy dp
    --base_model "${model_id}"
    --model_revision "${model_revision}"
    --seed "${seed}"
    --run_name "paper-coverage-${setting_id}-${method}-seed${seed}"
    --steps "${steps}"
    --batch_size 64
    --micro_batch_size 4
    --lr "${lr}"
    --cutoff_len "${cutoff}"
    --train_on_inputs "${train_inputs}"
    --lora_r "${lora_r}"
    --lora_alpha 16
    --epsilon "${epsilon}"
    --delta 1e-5
    --initial_clip_threshold "${initial_c}"
    --slaclip_num_slots 15
    --telemetry_mode research_raw
    --allow_non_private_telemetry
    --raw_hist_bins 128
    --raw_hist_max 30.0
    --protocol_stage final
    --val_set_size 0
    --validation_eval_interval 0
    --require_cuda
    --resume
    --checkpoint_every 25
    --run_train true
    --run_eval true
    --output_dir "${adapter_dir}"
    --result_dir "${result_dir}"
  )
  if [[ "${method}" == slaclip ]]; then
    if [[ "${rho}" == NA || "${eta}" == NA ]]; then
      echo "error: Full SlaClip arm lacks rho or eta" >&2
      return 2
    fi
    args+=(
      --slaclip_target_non_small_clip_fraction "${rho}"
      --slaclip_eta "${eta}"
      --slaclip_c_min 0.1
      --slaclip_c_max 15.0
    )
  elif [[ "${rho}" != NA || "${eta}" != NA ]]; then
    echo "error: fixed arm unexpectedly includes SlaClip parameters" >&2
    return 2
  fi
  if [[ "${dataset}" == glue8 ]]; then
    args+=(--eval_batch_size 64 --num_beams 1 --max_new_tokens 8 --max_input_length 384)
  else
    args+=(--eval_batch_size 8 --num_beams 4 --max_new_tokens 256 --max_input_length 1024)
  fi

  echo "state=running" >"${status_file}"
  "${args[@]}" >"${arm_root}/logs/train-${JOB_ID}.out" 2>"${arm_root}/logs/train-${JOB_ID}.err"
  if [[ ! -s "${adapter_dir}/run_status.json" \
      || ! -s "${adapter_dir}/adapter_model.safetensors" \
      || ! -s "${result_dir}/summary.csv" \
      || ! -s "${result_dir}/research_raw/NON_PRIVATE_train_log.jsonl" ]]; then
    echo "error: arm completed without required artifacts: ${arm_id}" >&2
    return 3
  fi
  "${PYTHON_BIN}" scripts/summarize_telemetry.py \
    "${result_dir}/research_raw/NON_PRIVATE_train_log.jsonl" \
    --safe-log "${adapter_dir}/train_log.jsonl" \
    --format both \
    --csv-out "${result_dir}/research_raw/telemetry_steps.csv" \
    --json-out "${result_dir}/research_raw/telemetry_summary.json" \
    >"${arm_root}/logs/summarize-${JOB_ID}.out" \
    2>"${arm_root}/logs/summarize-${JOB_ID}.err"
  echo "state=completed" >"${status_file}"
}

run_real_smoke() {
  local dataset model_id model_revision config
  if [[ "${LANE}" == 0 ]]; then
    dataset=math10k
    model_id=google/gemma-2-9b
    model_revision=33c193028431c2fde6c6e51f29e6f17b60cbfac6
    config=configs/math10k_paper.json
  else
    dataset=glue8
    model_id=google/gemma-3-4b-pt
    model_revision=cc012e0a6d0787b4adcc0fa2c4da74402494554d
    config=configs/glue8_paper.json
  fi
  local method
  for method in baseline slaclip; do
    local root="${CAMPAIGN_ROOT}/smoke/lane-${LANE}/${dataset}-${method}"
    mkdir -p "${root}/adapter" "${root}/results"
    local -a args=(
      "${PYTHON_BIN}" -u train_eval.py --config "${config}"
      --dataset "${dataset}" --method "${method}" --privacy dp
      --base_model "${model_id}" --model_revision "${model_revision}"
      --seed 42 --steps 2 --batch_size 64 --micro_batch_size 4
      --initial_clip_threshold 1.0 --slaclip_num_slots 15
      --telemetry_mode dp_safe --protocol_stage pilot --val_set_size 0
      --require_cuda --resume --checkpoint_every 1
      --run_train true --run_eval false
      --output_dir "${root}/adapter" --result_dir "${root}/results"
    )
    if [[ "${method}" == slaclip ]]; then
      args+=(--slaclip_target_non_small_clip_fraction 0.9 --slaclip_eta 0.05 --slaclip_c_min 0.1 --slaclip_c_max 15)
    fi
    "${args[@]}" >"${root}/train-${JOB_ID}.out" 2>"${root}/train-${JOB_ID}.err"
    [[ -s "${root}/adapter/adapter_model.safetensors" ]] || return 3
  done
}

CURRENT_ARM="real-model-smoke"
write_lane_status running 0
run_real_smoke

expected=6
if [[ "${LANE}" == 1 ]]; then expected=9; fi
actual="$(awk 'NF {n++} END {print n+0}' "${PLAN}")"
if [[ "${actual}" != "${expected}" ]]; then
  echo "error: lane ${LANE} expected ${expected} arms, found ${actual}" >&2
  exit 2
fi
while IFS= read -r spec; do
  [[ -n "${spec}" ]] || continue
  run_training "${spec}"
done <"${PLAN}"

CURRENT_ARM="complete"
write_lane_status completed 0
