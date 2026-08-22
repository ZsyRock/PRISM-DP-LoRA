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
GLUE_EVAL_ROOT="${8:?missing pinned GLUE evaluation assets}"
COVERAGE_PROFILE="${9:?missing coverage profile}"
MODEL_12B_REVISION="${10:?missing Gemma-3-12B revision}"
EXPECTED_CPUS_PER_TASK="${11:?missing CPUs per task}"
RUN_REAL_SMOKE="${12:-true}"

PYTHON_BIN="${ENV_PREFIX}/bin/python"
JOB_ID="${SLURM_JOB_ID:-manual}"
LANE_TMP="${JOB_TMP_ROOT}/lane-${LANE}"
LANE_STATUS="${CAMPAIGN_ROOT}/status/lane-${LANE}.txt"
CURRENT_ARM="bootstrap"
CURRENT_ARM_STATUS=""

is_focused_glue_profile() {
  [[ "${COVERAGE_PROFILE}" == glue-slaclip-screen \
      || "${COVERAGE_PROFILE}" == glue-high-c-refinement ]]
}

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

write_arm_status() {
  local state="$1" exit_code="$2" temporary
  [[ -n "${CURRENT_ARM_STATUS}" ]] || return 0
  temporary="${CURRENT_ARM_STATUS}.tmp.$$"
  mkdir -p "$(dirname -- "${CURRENT_ARM_STATUS}")"
  {
    echo "schema_version=1"
    echo "state=${state}"
    echo "job_id=${JOB_ID}"
    echo "arm_id=${CURRENT_ARM}"
    echo "exit_code=${exit_code}"
    echo "updated_at=$(date --iso-8601=seconds)"
  } >"${temporary}"
  mv -f -- "${temporary}" "${CURRENT_ARM_STATUS}"
}

finish() {
  local code=$? state=completed
  trap - EXIT TERM INT
  if [[ "${code}" == 130 || "${code}" == 143 ]]; then
    state=interrupted
  elif [[ "${code}" != 0 ]]; then
    state=failed
  fi
  write_arm_status "${state}" "${code}" || true
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
if [[ "${SLURM_CPUS_PER_TASK:-${EXPECTED_CPUS_PER_TASK}}" != "${EXPECTED_CPUS_PER_TASK}" ]]; then
  echo "error: lane CPU allocation differs from its submission lock" >&2
  exit 2
fi

cd -- "${REPO_ROOT}"
mkdir -p "${LANE_TMP}" "${CAMPAIGN_ROOT}/runs" "${CAMPAIGN_ROOT}/smoke/lane-${LANE}"
export TMPDIR="${LANE_TMP}/tmp"
export HF_DATASETS_CACHE="${LANE_TMP}/datasets"
export TRITON_CACHE_DIR="${LANE_TMP}/triton"
export TORCH_EXTENSIONS_DIR="${LANE_TMP}/torch-extensions"
export PYTHONPYCACHEPREFIX="${LANE_TMP}/pycache"
export OMP_NUM_THREADS="${EXPECTED_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${EXPECTED_CPUS_PER_TASK}"
mkdir -p "${TMPDIR}" "${HF_DATASETS_CACHE}" "${TRITON_CACHE_DIR}" "${TORCH_EXTENSIONS_DIR}"

run_training() {
  local spec="$1"
  local plan_lane arm_id setting_id dataset model_slug model_id model_revision
  local epsilon lora_r method initial_c rho eta steps lr cutoff train_inputs seed eval_limit relative_root
  IFS='|' read -r plan_lane arm_id setting_id dataset model_slug model_id model_revision \
    epsilon lora_r method initial_c rho eta steps lr cutoff train_inputs seed eval_limit relative_root <<<"${spec}"
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
  local protocol_stage=final validation_rows=0 validation_interval=0 run_eval=true
  if is_focused_glue_profile; then
    protocol_stage=selection
    validation_rows=800
    validation_interval=50
    run_eval=false
  fi
  mkdir -p "${adapter_dir}" "${result_dir}/research_raw" "${arm_root}/logs"
  CURRENT_ARM="${arm_id}"
  CURRENT_ARM_STATUS="${status_file}"
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
    --protocol_stage "${protocol_stage}"
    --val_set_size "${validation_rows}"
    --validation_seed 1729
    --validation_eval_interval "${validation_interval}"
    --require_cuda
    --resume
    --checkpoint_every 25
    --run_train true
    --run_eval "${run_eval}"
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
  if is_focused_glue_profile; then
    args+=(--validation_data_is_public)
  fi
  if [[ "${dataset}" == glue8 ]]; then
    args+=(
      --eval_batch_size 64 --num_beams 1 --max_new_tokens 8 --max_input_length 384
      --glue_eval_data_root "${GLUE_EVAL_ROOT}"
    )
  else
    args+=(--eval_batch_size 8 --num_beams 4 --max_new_tokens 256 --max_input_length 1024)
  fi
  if [[ "${eval_limit}" =~ ^[0-9]+$ ]] && (( eval_limit > 0 )); then
    args+=(--fast_dev_run "${eval_limit}")
  fi

  write_arm_status running 0
  "${args[@]}" >"${arm_root}/logs/train-${JOB_ID}.out" 2>"${arm_root}/logs/train-${JOB_ID}.err"
  if [[ ! -s "${adapter_dir}/run_status.json" \
      || ! -s "${adapter_dir}/adapter_model.safetensors" \
      || ! -s "${result_dir}/research_raw/NON_PRIVATE_train_log.jsonl" ]]; then
    echo "error: arm completed without required artifacts: ${arm_id}" >&2
    return 3
  fi
  if ! is_focused_glue_profile && ! -s "${result_dir}/summary.csv"; then
    echo "error: evaluated arm completed without a summary: ${arm_id}" >&2
    return 3
  fi
  if is_focused_glue_profile; then
    for validation_artifact in \
      "${result_dir}/validation/split_manifest.json" \
      "${result_dir}/validation/validation_metrics.json" \
      "${result_dir}/validation/validation_curve.jsonl"; do
      if [[ ! -s "${validation_artifact}" ]]; then
        echo "error: focused arm lacks validation artifact: ${validation_artifact}" >&2
        return 3
      fi
    done
    "${PYTHON_BIN}" - \
      "${adapter_dir}/run_status.json" \
      "${result_dir}/validation/split_manifest.json" \
      "${result_dir}/validation/validation_metrics.json" \
      "${result_dir}/validation/validation_curve.jsonl" \
      "${arm_id}" "${steps}" <<'PY'
import json
import math
import sys

status_path, split_path, metrics_path, curve_path, arm_id, planned_steps = sys.argv[1:]
planned_steps = int(planned_steps)
status = json.load(open(status_path, encoding="utf-8"))
split = json.load(open(split_path, encoding="utf-8"))
metrics = json.load(open(metrics_path, encoding="utf-8"))
expected_indices = "34a59e5cf4d98300f3d484d9c82b37172b850939bc19e002c22428be22a12805"
expected_records = "43f7a3d0db422b2331a59d3611e777faf99d0bf434a405ef98a8ea7b1c582fad"
for label, payload in (("status split", status.get("data_split")), ("split", split), ("metrics", metrics)):
    if not isinstance(payload, dict):
        raise SystemExit(f"{arm_id}: missing {label}")
    if (
        payload.get("protocol_stage") != "selection"
        or payload.get("validation_data_is_public") is not True
        or payload.get("validation_rows") != 800
        or payload.get("seed") != 1729
        or payload.get("validation_indices_sha256") != expected_indices
        or payload.get("validation_record_hashes_sha256") != expected_records
    ):
        raise SystemExit(f"{arm_id}: invalid {label} identity")
if status.get("validation") != metrics:
    raise SystemExit(f"{arm_id}: status and validation metrics differ")
if split.get("manifest_sha256") != metrics.get("manifest_sha256"):
    raise SystemExit(f"{arm_id}: validation manifest mismatch")
curves = [json.loads(line) for line in open(curve_path, encoding="utf-8") if line.strip()]
expected_curve_steps = list(range(0, planned_steps + 1, 50))
if expected_curve_steps[-1] != planned_steps:
    expected_curve_steps.append(planned_steps)
if [row.get("step") for row in curves] != expected_curve_steps:
    raise SystemExit(f"{arm_id}: validation curve steps are incomplete")
if any(
    row.get("run_id") != status.get("run_id")
    or row.get("config_fingerprint") != status.get("config_fingerprint")
    or row.get("manifest_sha256") != split.get("manifest_sha256")
    for row in curves
):
    raise SystemExit(f"{arm_id}: validation curve identity mismatch")
if not math.isclose(
    float(curves[-1]["loss_mean"]),
    float(metrics["loss_mean"]),
    rel_tol=1e-9,
    abs_tol=1e-8,
):
    raise SystemExit(f"{arm_id}: validation endpoint mismatch")
PY
  fi
  "${PYTHON_BIN}" scripts/summarize_telemetry.py \
    "${result_dir}/research_raw/NON_PRIVATE_train_log.jsonl" \
    --safe-log "${adapter_dir}/train_log.jsonl" \
    --format both \
    --csv-out "${result_dir}/research_raw/telemetry_steps.csv" \
    --json-out "${result_dir}/research_raw/telemetry_summary.json" \
    >"${arm_root}/logs/summarize-${JOB_ID}.out" \
    2>"${arm_root}/logs/summarize-${JOB_ID}.err"
  write_arm_status completed 0
  CURRENT_ARM_STATUS=""
}

run_one_real_smoke() {
  local dataset="$1" model_slug="$2" model_id="$3" model_revision="$4" config="$5"
  local -a methods=(baseline slaclip)
  if [[ "${COVERAGE_PROFILE}" == baseline-reproduction* ]]; then
    methods=(baseline)
  fi
  local method
  for method in "${methods[@]}"; do
    local root="${CAMPAIGN_ROOT}/smoke/lane-${LANE}/${dataset}-${model_slug}-${method}"
    mkdir -p "${root}/adapter" "${root}/results"
    local smoke_stage=pilot smoke_validation_rows=0 smoke_validation_interval=0
    local -a smoke_public_args=()
    if is_focused_glue_profile; then
      smoke_stage=selection
      smoke_validation_rows=8
      smoke_validation_interval=1
      smoke_public_args+=(--validation_data_is_public)
    fi
    local -a args=(
      "${PYTHON_BIN}" -u train_eval.py --config "${config}"
      --dataset "${dataset}" --method "${method}" --privacy dp
      --base_model "${model_id}" --model_revision "${model_revision}"
      --seed 42 --steps 2 --batch_size 64 --micro_batch_size 4
      --initial_clip_threshold 1.0 --slaclip_num_slots 15
      --telemetry_mode dp_safe --protocol_stage "${smoke_stage}"
      --val_set_size "${smoke_validation_rows}" --validation_seed 1729
      --validation_eval_interval "${smoke_validation_interval}"
      --require_cuda --resume --checkpoint_every 1
      --run_train true --run_eval false
      --output_dir "${root}/adapter" --result_dir "${root}/results"
      "${smoke_public_args[@]}"
    )
    if [[ "${method}" == slaclip ]]; then
      args+=(--slaclip_target_non_small_clip_fraction 0.9 --slaclip_eta 0.05 --slaclip_c_min 0.1 --slaclip_c_max 15)
    fi
    if [[ "${dataset}" == glue8 ]] && ! is_focused_glue_profile; then
      args+=(
        --run_eval true --fast_dev_run 2 --eval_batch_size 2 --num_beams 1
        --max_new_tokens 8 --max_input_length 384
        --glue_eval_data_root "${GLUE_EVAL_ROOT}"
      )
    fi
    "${args[@]}" >"${root}/train-${JOB_ID}.out" 2>"${root}/train-${JOB_ID}.err"
    [[ -s "${root}/adapter/adapter_model.safetensors" ]] || return 3
    if is_focused_glue_profile; then
      "${PYTHON_BIN}" - "${root}/adapter/run_status.json" \
        "${root}/results/validation/validation_curve.jsonl" <<'PY'
import json
import sys

status = json.load(open(sys.argv[1], encoding="utf-8"))
validation = status.get("validation", {})
curve = [json.loads(line) for line in open(sys.argv[2], encoding="utf-8") if line.strip()]
if (
    status.get("state") != "completed"
    or validation.get("PUBLIC_VALIDATION_DATA") is not True
    or validation.get("validation_data_is_public") is not True
    or validation.get("protocol_stage") != "selection"
    or validation.get("records") != 8
    or [row.get("step") for row in curve] != [0, 1, 2]
):
    raise SystemExit("focused real-model smoke did not exercise the public validation path")
PY
    fi
  done
}

run_real_smoke() {
  if is_focused_glue_profile; then
    run_one_real_smoke \
      glue8 gemma-3-4b-pt google/gemma-3-4b-pt \
      cc012e0a6d0787b4adcc0fa2c4da74402494554d configs/glue8_paper.json
    return
  fi
  if [[ "$(basename -- "${PLAN}")" == sequential.tsv ]]; then
    run_one_real_smoke \
      glue8 gemma-3-4b-pt google/gemma-3-4b-pt \
      cc012e0a6d0787b4adcc0fa2c4da74402494554d configs/glue8_paper.json
    run_one_real_smoke \
      math10k gemma-2-9b google/gemma-2-9b \
      33c193028431c2fde6c6e51f29e6f17b60cbfac6 configs/math10k_paper.json
    if [[ "${COVERAGE_PROFILE}" == baseline-reproduction ]]; then
      run_one_real_smoke \
        math10k gemma-3-12b-pt google/gemma-3-12b-pt \
        "${MODEL_12B_REVISION}" configs/math10k_paper.json
    fi
  elif [[ "${LANE}" == 0 ]]; then
    run_one_real_smoke \
      math10k gemma-2-9b google/gemma-2-9b \
      33c193028431c2fde6c6e51f29e6f17b60cbfac6 configs/math10k_paper.json
  else
    run_one_real_smoke \
      glue8 gemma-3-4b-pt google/gemma-3-4b-pt \
      cc012e0a6d0787b4adcc0fa2c4da74402494554d configs/glue8_paper.json
    if [[ "${COVERAGE_PROFILE}" == baseline-reproduction ]]; then
      run_one_real_smoke \
        math10k gemma-3-12b-pt google/gemma-3-12b-pt \
        "${MODEL_12B_REVISION}" configs/math10k_paper.json
    fi
  fi
}

CURRENT_ARM="real-model-smoke"
write_lane_status running 0
if [[ "${RUN_REAL_SMOKE}" == true ]]; then
  run_real_smoke
elif [[ "${RUN_REAL_SMOKE}" != false ]]; then
  echo "error: RUN_REAL_SMOKE must be true or false" >&2
  exit 2
fi

actual="$(awk 'NF {n++} END {print n+0}' "${PLAN}")"
if (( actual < 1 )); then
  echo "error: lane ${LANE} has no planned arms" >&2
  exit 2
fi
while IFS= read -r spec; do
  [[ -n "${spec}" ]] || continue
  run_training "${spec}"
done <"${PLAN}"

CURRENT_ARM_STATUS=""
CURRENT_ARM="complete"
write_lane_status completed 0
