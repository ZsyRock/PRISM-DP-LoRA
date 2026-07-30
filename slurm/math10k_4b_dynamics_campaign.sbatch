#!/usr/bin/env bash

# Worker for one two-H200, two-lane Math-10K development campaign.
# Scheduler options and every portable path are supplied by the tracked
# submission wrapper.  Keeping this file free of #SBATCH directives prevents
# variables or cluster-specific usernames from being embedded in the protocol.
set -Eeuo pipefail
umask 077

USER_NAME="${1:?missing user name}"
REPO_ROOT="${2:?missing immutable repository path}"
ENV_PREFIX="${3:?missing environment prefix}"
RUN_ROOT="${4:?missing run root}"
SHARED_HF_HOME="${5:?missing Hugging Face cache root}"
EXPECTED_REPO_SHA="${6:?missing locked repository SHA}"
CAMPAIGN_ID="${7:?missing campaign ID}"
SCRATCH_ROOT="${8:?missing scratch root}"
MODEL_ID="${9:?missing model ID}"
MODEL_REVISION="${10:?missing model revision}"
MODEL_SLUG="${11:?missing model slug}"

PYTHON_BIN="${ENV_PREFIX}/bin/python"
CAMPAIGN_ROOT="${RUN_ROOT}/campaigns/${CAMPAIGN_ID}"
JOB_ID="${SLURM_JOB_ID:-manual}"
JOB_TMP="${SLURM_TMPDIR:-${SCRATCH_ROOT}/tmp/prism-4b-dynamics-${JOB_ID}}"
STATUS_ROOT="${CAMPAIGN_ROOT}/status"
PLAN_ROOT="${CAMPAIGN_ROOT}/plans"
REGISTRY_ROOT="${CAMPAIGN_ROOT}/screen"
REGISTRY_FILE="${REGISTRY_ROOT}/candidate_registry.json"
SELECTION_ROOT="${CAMPAIGN_ROOT}/selection"
STAGE1_SELECTION="${SELECTION_ROOT}/stage1-selection.json"
LOCKED_SELECTION="${SELECTION_ROOT}/selection.json"
SCHEDULE_ROOT="${CAMPAIGN_ROOT}/schedule"
REPLAY_SCHEDULE="${SCHEDULE_ROOT}/replay_schedule.json"
CONTROL_MANIFEST="${SCHEDULE_ROOT}/control_manifest.json"
SUBMISSION_RECEIPT="${CAMPAIGN_ROOT}/submission_receipt.json"
ENVIRONMENT_FREEZE="${CAMPAIGN_ROOT}/artifacts/environment_freeze.txt"
ALLOCATION_SNAPSHOT="${CAMPAIGN_ROOT}/artifacts/allocation-${JOB_ID}.txt"
JOB_STATUS="${STATUS_ROOT}/job-${JOB_ID}.txt"
CURRENT_PHASE="bootstrap"
WAVE_COUNTER=0
LANE_SRUN_PID=""

atomic_sha256_sidecar() {
  local source="$1"
  "${PYTHON_BIN}" - "${source}" <<'PY'
import hashlib
import os
import sys
import tempfile
from pathlib import Path

source = Path(sys.argv[1]).resolve()
digest = hashlib.sha256(source.read_bytes()).hexdigest()
target = source.with_name(source.name + ".sha256")
content = f"{digest}  {source.name}\n"
if target.exists():
    if target.read_text(encoding="utf-8") != content:
        raise SystemExit(f"refusing inconsistent SHA sidecar: {target}")
else:
    fd, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
os.chmod(target, 0o600)
print(f"sha256={digest} path={source}")
PY
}

install_immutable_file() {
  local temporary="$1"
  local target="$2"
  mkdir -p "$(dirname -- "${target}")"
  if [[ -e "${target}" ]]; then
    if ! cmp -s -- "${temporary}" "${target}"; then
      rm -f -- "${temporary}"
      echo "error: refusing to replace inconsistent immutable file: ${target}" >&2
      return 2
    fi
  elif ! ln -- "${temporary}" "${target}" 2>/dev/null; then
    if [[ ! -e "${target}" ]] || ! cmp -s -- "${temporary}" "${target}"; then
      rm -f -- "${temporary}"
      echo "error: could not atomically create immutable file: ${target}" >&2
      return 2
    fi
  fi
  chmod 600 "${target}"
  rm -f -- "${temporary}"
}

create_environment_and_allocation_snapshots() {
  local environment_temporary allocation_temporary
  environment_temporary="$(mktemp "${CAMPAIGN_ROOT}/artifacts/.environment-freeze.XXXXXX")"
  {
    echo "schema_version=1"
    echo "code_sha=${EXPECTED_REPO_SHA}"
    echo "python_executable=${PYTHON_BIN}"
    echo "python_version=$("${PYTHON_BIN}" -c 'import platform; print(platform.python_version())')"
    echo "environment_prefix=${ENV_PREFIX}"
    echo "package_versions_begin"
    "${PYTHON_BIN}" -m pip freeze --all | LC_ALL=C sort
    echo "package_versions_end"
    echo "runtime_versions_begin"
    "${PYTHON_BIN}" - <<'PY'
import importlib.metadata
import json
import torch

names = ("accelerate", "datasets", "opacus", "peft", "transformers")
print(json.dumps({
    "cuda_runtime": torch.version.cuda,
    "cudnn": torch.backends.cudnn.version(),
    "packages": {name: importlib.metadata.version(name) for name in names},
    "torch": torch.__version__,
}, sort_keys=True))
PY
    echo "runtime_versions_end"
    echo "gpu_driver_begin"
    nvidia-smi --query-gpu=name,driver_version --format=csv,noheader \
      | LC_ALL=C sort -u
    echo "gpu_driver_end"
  } >"${environment_temporary}"
  install_immutable_file "${environment_temporary}" "${ENVIRONMENT_FREEZE}"
  atomic_sha256_sidecar "${ENVIRONMENT_FREEZE}"

  allocation_temporary="$(mktemp "${CAMPAIGN_ROOT}/artifacts/.allocation-${JOB_ID}.XXXXXX")"
  {
    echo "schema_version=1"
    echo "job_id=${JOB_ID}"
    echo "campaign_id=${CAMPAIGN_ID}"
    echo "code_sha=${EXPECTED_REPO_SHA}"
    echo "host=$(hostname)"
    echo "slurm_job_name=${SLURM_JOB_NAME:-}"
    echo "slurm_job_partition=${SLURM_JOB_PARTITION:-}"
    echo "slurm_job_nodelist=${SLURM_JOB_NODELIST:-}"
    echo "slurm_ntasks=${SLURM_NTASKS:-}"
    echo "slurm_cpus_per_task=${SLURM_CPUS_PER_TASK:-}"
    echo "slurm_mem_per_node=${SLURM_MEM_PER_NODE:-}"
    echo "slurm_job_gpus=${SLURM_JOB_GPUS:-}"
    echo "requested_lanes=2"
    echo "requested_gpu_per_lane=1xh200"
    echo "requested_cpu_per_lane=8"
    echo "requested_memory_per_lane=128G"
    if command -v scontrol >/dev/null 2>&1 && [[ "${JOB_ID}" =~ ^[0-9]+$ ]]; then
      echo "scontrol_job_begin"
      scontrol show job -o "${JOB_ID}"
      echo "scontrol_job_end"
    fi
  } >"${allocation_temporary}"
  install_immutable_file "${allocation_temporary}" "${ALLOCATION_SNAPSHOT}"
  atomic_sha256_sidecar "${ALLOCATION_SNAPSHOT}"
}

write_job_status() {
  local state="$1"
  local exit_code="$2"
  local temporary="${JOB_STATUS}.tmp.$$"
  mkdir -p "${STATUS_ROOT}"
  {
    echo "schema_version=1"
    echo "job_id=${JOB_ID}"
    echo "state=${state}"
    echo "phase=${CURRENT_PHASE}"
    echo "exit_code=${exit_code}"
    echo "host=$(hostname)"
    echo "updated_at=$(date --iso-8601=seconds)"
    echo "campaign_root=${CAMPAIGN_ROOT}"
    echo "code_sha=${EXPECTED_REPO_SHA}"
  } >"${temporary}"
  mv -f "${temporary}" "${JOB_STATUS}"
}

finish() {
  local exit_code=$?
  local state="completed"
  trap - EXIT
  if [[ "${exit_code}" != "0" ]]; then
    state="failed"
  fi
  write_job_status "${state}" "${exit_code}"
  if [[ "${exit_code}" != "0" ]]; then
    echo "campaign failed phase=${CURRENT_PHASE} exit_code=${exit_code}" >&2
  fi
  exit "${exit_code}"
}
trap finish EXIT

write_arm_status() {
  local arm_root="$1"
  local state="$2"
  local phase="$3"
  local candidate_id="$4"
  local seed="$5"
  local exit_code="$6"
  local status_file="${arm_root}/orchestration-status.txt"
  local temporary="${status_file}.tmp.$$"
  mkdir -p "${arm_root}"
  {
    echo "schema_version=1"
    echo "job_id=${JOB_ID}"
    echo "state=${state}"
    echo "phase=${phase}"
    echo "candidate_id=${candidate_id}"
    echo "seed=${seed}"
    echo "exit_code=${exit_code}"
    echo "updated_at=$(date --iso-8601=seconds)"
  } >"${temporary}"
  mv -f "${temporary}" "${status_file}"
}

write_wave_status() {
  local wave_id="$1"
  local state="$2"
  local first_spec="$3"
  local second_spec="$4"
  local first_rc="$5"
  local second_rc="$6"
  local status_file="${STATUS_ROOT}/wave-${JOB_ID}-${wave_id}.txt"
  local temporary="${status_file}.tmp.$$"
  {
    echo "schema_version=1"
    echo "job_id=${JOB_ID}"
    echo "wave_id=${wave_id}"
    echo "state=${state}"
    echo "phase=${CURRENT_PHASE}"
    echo "first_rc=${first_rc}"
    echo "second_rc=${second_rc}"
    echo "first_spec=${first_spec}"
    echo "second_spec=${second_spec}"
    echo "updated_at=$(date --iso-8601=seconds)"
  } >"${temporary}"
  mv -f "${temporary}" "${status_file}"
}

terminate_lane_process() {
  if [[ -n "${LANE_SRUN_PID:-}" ]] && kill -0 "${LANE_SRUN_PID}" 2>/dev/null; then
    kill -TERM "${LANE_SRUN_PID}" 2>/dev/null || true
    wait "${LANE_SRUN_PID}" 2>/dev/null || true
  fi
  LANE_SRUN_PID=""
}

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "error: Python environment unavailable: ${PYTHON_BIN}" >&2
  exit 2
fi
if [[ ! "${EXPECTED_REPO_SHA}" =~ ^[0-9a-f]{40}$ ]]; then
  echo "error: campaign requires a full lowercase Git SHA" >&2
  exit 2
fi
if [[ "${SLURM_NTASKS:-2}" != "2" || "${SLURM_CPUS_PER_TASK:-8}" != "8" ]]; then
  echo "error: worker requires two parent tasks and eight CPUs per task" >&2
  exit 2
fi
if [[ "$(git -C "${REPO_ROOT}" rev-parse HEAD)" != "${EXPECTED_REPO_SHA}" ]]; then
  echo "error: staged repository SHA changed after submission" >&2
  exit 2
fi
if [[ -n "$(git -C "${REPO_ROOT}" status --porcelain)" ]]; then
  echo "error: staged repository is dirty" >&2
  exit 2
fi
for required_file in \
  train_eval.py \
  configs/math10k_paper.json \
  scripts/preflight_hpc.py \
  scripts/smoke_dp_path.py \
  scripts/select_validation_candidates.py \
  scripts/summarize_telemetry.py; do
  if [[ ! -f "${REPO_ROOT}/${required_file}" ]]; then
    echo "error: immutable source is missing ${required_file}" >&2
    exit 2
  fi
done

export PRISM_REPO_ROOT="${REPO_ROOT}"
export PRISM_RUN_ROOT="${RUN_ROOT}"
export HF_HOME="${SHARED_HF_HOME}"
export HF_HUB_CACHE="${SHARED_HF_HOME}/hub"
export HUGGINGFACE_HUB_CACHE="${SHARED_HF_HOME}/hub"
export HF_DATASETS_CACHE="${RUN_ROOT}/cache/huggingface/datasets-preflight"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_XET=1
export TORCH_HOME="${RUN_ROOT}/cache/torch-preflight"
export TORCH_EXTENSIONS_DIR="${RUN_ROOT}/cache/torch-extensions-preflight"
export TRITON_CACHE_DIR="${RUN_ROOT}/cache/triton-preflight"
export PYTHONPYCACHEPREFIX="${RUN_ROOT}/cache/pycache-preflight"
export TMPDIR="${JOB_TMP}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export SLURM_EXPORT_ENV=ALL

mkdir -p \
  "${CAMPAIGN_ROOT}/slurm" \
  "${CAMPAIGN_ROOT}/smoke" \
  "${CAMPAIGN_ROOT}/screen/runs" \
  "${CAMPAIGN_ROOT}/controls/runs" \
  "${CAMPAIGN_ROOT}/artifacts" \
  "${STATUS_ROOT}" \
  "${PLAN_ROOT}" \
  "${REGISTRY_ROOT}" \
  "${SELECTION_ROOT}" \
  "${SCHEDULE_ROOT}" \
  "${HF_DATASETS_CACHE}" \
  "${TORCH_HOME}" \
  "${TORCH_EXTENSIONS_DIR}" \
  "${TRITON_CACHE_DIR}" \
  "${PYTHONPYCACHEPREFIX}" \
  "${TMPDIR}"

cache_key="models--${MODEL_ID//\//--}"
snapshot="${SHARED_HF_HOME}/hub/${cache_key}/snapshots/${MODEL_REVISION}"
marker="${SHARED_HF_HOME}/staged/${cache_key}/${MODEL_REVISION}.complete"
if [[ ! -d "${snapshot}" || ! -f "${marker}" ]]; then
  echo "error: pinned offline model is missing: ${MODEL_ID}@${MODEL_REVISION}" >&2
  exit 2
fi

cd "${REPO_ROOT}"
write_job_status running 0

# Fail before consuming hours if the queued source lacks the exact validation
# and replay interfaces on which this protocol depends.
TRAIN_HELP="$("${PYTHON_BIN}" train_eval.py --help)"
for required_option in \
  "--clip_schedule_path" \
  "--validation_eval_interval" \
  "--validation_generate_numeric" \
  "--validation_num_beams" \
  "--validation_max_new_tokens" \
  "--validation_max_input_length"; do
  if [[ "${TRAIN_HELP}" != *"${required_option}"* ]]; then
    echo "error: locked trainer lacks required option ${required_option}" >&2
    exit 2
  fi
done
if [[ "${TRAIN_HELP}" != *"replay"* ]]; then
  echo "error: locked trainer does not expose method=replay" >&2
  exit 2
fi

lane_srun() {
  local arm_root="$1"
  local log_stem="$2"
  shift 2
  local cache_root="${arm_root}/cache"
  local tmp_root="${arm_root}/tmp"
  mkdir -p \
    "${arm_root}/logs" \
    "${cache_root}/huggingface" \
    "${cache_root}/datasets" \
    "${cache_root}/torch" \
    "${cache_root}/torch-extensions" \
    "${cache_root}/triton" \
    "${cache_root}/xdg" \
    "${cache_root}/cuda" \
    "${cache_root}/pycache" \
    "${tmp_root}"
  srun \
    --export=ALL \
    --exclusive \
    --exact \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task=8 \
    --mem=128G \
    --gres=gpu:h200:1 \
    --kill-on-bad-exit=1 \
    --output="${arm_root}/logs/${log_stem}-${JOB_ID}.out" \
    --error="${arm_root}/logs/${log_stem}-${JOB_ID}.err" \
    env \
      PRISM_REPO_ROOT="${REPO_ROOT}" \
      PRISM_RUN_ROOT="${RUN_ROOT}" \
      HF_HOME="${cache_root}/huggingface" \
      HF_HUB_CACHE="${SHARED_HF_HOME}/hub" \
      HUGGINGFACE_HUB_CACHE="${SHARED_HF_HOME}/hub" \
      HF_DATASETS_CACHE="${cache_root}/datasets" \
      HF_HUB_OFFLINE=1 \
      TRANSFORMERS_OFFLINE=1 \
      HF_HUB_DISABLE_XET=1 \
      TORCH_HOME="${cache_root}/torch" \
      TORCH_EXTENSIONS_DIR="${cache_root}/torch-extensions" \
      TRITON_CACHE_DIR="${cache_root}/triton" \
      XDG_CACHE_HOME="${cache_root}/xdg" \
      CUDA_CACHE_PATH="${cache_root}/cuda" \
      PYTHONPYCACHEPREFIX="${cache_root}/pycache" \
      TMPDIR="${tmp_root}" \
      TOKENIZERS_PARALLELISM=false \
      PYTHONUNBUFFERED=1 \
      OMP_NUM_THREADS=8 \
      MKL_NUM_THREADS=8 \
      "$@" &
  LANE_SRUN_PID=$!
  local lane_rc=0
  wait "${LANE_SRUN_PID}" || lane_rc=$?
  LANE_SRUN_PID=""
  return "${lane_rc}"
}

validate_selection_run() {
  local arm_root="$1"
  local expected_method="$2"
  local expected_seed="$3"
  "${PYTHON_BIN}" - "${arm_root}" "${expected_method}" "${expected_seed}" <<'PY'
import json
import math
import sys
from pathlib import Path

root = Path(sys.argv[1])
expected_method = sys.argv[2]
expected_seed = int(sys.argv[3])
status_path = root / "adapter" / "run_status.json"
metrics_path = root / "results" / "validation" / "validation_metrics.json"
curve_path = root / "results" / "validation" / "validation_curve.jsonl"
split_path = root / "results" / "validation" / "split_manifest.json"
for path in (status_path, metrics_path, curve_path, split_path):
    if not path.is_file() or path.stat().st_size == 0:
        raise SystemExit(f"missing required selection artifact: {path}")
status = json.loads(status_path.read_text(encoding="utf-8"))
if status.get("state") != "completed" or int(status.get("update_steps", -1)) != 300:
    raise SystemExit(f"incomplete 300-step run: {status_path}")
config = status.get("config") or {}
if config.get("method") != expected_method or int(config.get("seed", -1)) != expected_seed:
    raise SystemExit(f"run identity mismatch: {status_path}")
if config.get("protocol_stage") != "selection" or config.get("validation_data_is_public") is not True:
    raise SystemExit(f"run is not a declared public selection run: {status_path}")
privacy = status.get("privacy_accounting") or {}
epsilon = status.get("epsilon_spent")
if epsilon is None:
    epsilon = privacy.get("epsilon_spent")
if epsilon is not None and (not math.isfinite(float(epsilon)) or abs(float(epsilon) - 6.0) > 0.02):
    raise SystemExit(f"unexpected epsilon={epsilon}: {status_path}")
metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
if metrics.get("selection_metric") != "public_math10k_numeric_exact_match_accuracy":
    raise SystemExit(f"wrong selection metric: {metrics_path}")
accuracy = metrics.get("numeric_exact_accuracy")
if not isinstance(accuracy, (int, float)) or isinstance(accuracy, bool) or not math.isfinite(float(accuracy)):
    raise SystemExit(f"missing finite numeric_exact_accuracy: {metrics_path}")
if not 0.0 <= float(accuracy) <= 1.0:
    raise SystemExit(f"numeric_exact_accuracy outside [0,1]: {metrics_path}")
for key in ("numeric_exact_correct", "numeric_parse_failures"):
    if not isinstance(metrics.get(key), int) or isinstance(metrics.get(key), bool) or metrics[key] < 0:
        raise SystemExit(f"invalid {key}: {metrics_path}")
loss = metrics.get("loss_mean")
if not isinstance(loss, (int, float)) or isinstance(loss, bool) or not math.isfinite(float(loss)):
    raise SystemExit(f"missing endpoint loss_mean: {metrics_path}")
records = []
with curve_path.open("r", encoding="utf-8") as handle:
    for line_number, line in enumerate(handle, 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"invalid curve JSON {curve_path}:{line_number}: {exc}") from exc
        records.append(record)
steps = [record.get("step") for record in records]
expected_steps = [0, 50, 100, 150, 200, 250, 300]
if steps != expected_steps:
    raise SystemExit(f"validation curve steps={steps}, expected={expected_steps}: {curve_path}")
for record in records:
    value = record.get("loss_mean")
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)):
        raise SystemExit(f"non-finite curve loss at step={record.get('step')}: {curve_path}")
split = json.loads(split_path.read_text(encoding="utf-8"))
if split.get("validation_data_is_public") is not True or int(split.get("validation_rows", -1)) != 500:
    raise SystemExit(f"unexpected public holdout manifest: {split_path}")
print(
    f"validated method={expected_method} seed={expected_seed} "
    f"accuracy={float(accuracy):.8f} loss={float(loss):.8f}"
)
PY
}

validate_real_smoke() {
  local arm_root="$1"
  local expected_method="$2"
  "${PYTHON_BIN}" - "${arm_root}" "${expected_method}" <<'PY'
import json
import math
import sys
from pathlib import Path

root = Path(sys.argv[1])
method = sys.argv[2]
status_path = root / "adapter" / "run_status.json"
metrics_path = root / "results" / "validation" / "validation_metrics.json"
curve_path = root / "results" / "validation" / "validation_curve.jsonl"
split_path = root / "results" / "validation" / "split_manifest.json"
for path in (status_path, metrics_path, curve_path, split_path):
    if not path.is_file() or path.stat().st_size == 0:
        raise SystemExit(f"real smoke missing artifact: {path}")
status = json.loads(status_path.read_text(encoding="utf-8"))
if (
    status.get("state") != "completed"
    or int(status.get("update_steps", -1)) != 2
    or status.get("method") != method
):
    raise SystemExit(f"real smoke status mismatch: {status_path}")
metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
if metrics.get("selection_metric") != "public_math10k_numeric_exact_match_accuracy":
    raise SystemExit(f"real smoke selection metric mismatch: {metrics_path}")
accuracy = metrics.get("numeric_exact_accuracy")
if not isinstance(accuracy, (int, float)) or isinstance(accuracy, bool) or not math.isfinite(float(accuracy)):
    raise SystemExit(f"real smoke numeric accuracy is missing: {metrics_path}")
curve = [json.loads(line) for line in curve_path.read_text(encoding="utf-8").splitlines() if line.strip()]
if [item.get("step") for item in curve] != [0, 1, 2]:
    raise SystemExit(f"real smoke curve must cover [0,1,2]: {curve_path}")
if any(not math.isfinite(float(item.get("loss_mean", float("nan")))) for item in curve):
    raise SystemExit(f"real smoke curve contains non-finite loss: {curve_path}")
split = json.loads(split_path.read_text(encoding="utf-8"))
if split.get("validation_data_is_public") is not True or int(split.get("validation_rows", -1)) != 8:
    raise SystemExit(f"real smoke holdout mismatch: {split_path}")
print(f"validated real smoke method={method} numeric_exact_accuracy={float(accuracy):.8f}")
PY
}

summarize_run_telemetry() {
  local arm_root="$1"
  local raw_dir="${arm_root}/results/research_raw"
  local raw_log="${raw_dir}/NON_PRIVATE_train_log.jsonl"
  local safe_log="${arm_root}/adapter/train_log.jsonl"
  if [[ ! -s "${raw_log}" || ! -s "${safe_log}" ]]; then
    echo "error: required telemetry logs are missing: ${arm_root}" >&2
    return 3
  fi
  "${PYTHON_BIN}" scripts/summarize_telemetry.py \
    "${raw_log}" \
    --safe-log "${safe_log}" \
    --format both \
    --csv-out "${raw_dir}/telemetry_steps.csv" \
    --json-out "${raw_dir}/telemetry_summary.json" \
    >>"${arm_root}/logs/summarize-${JOB_ID}.out" \
    2>"${arm_root}/logs/summarize-${JOB_ID}.err"
  if [[ ! -s "${raw_dir}/telemetry_summary.json" ]]; then
    echo "error: telemetry summary was not created: ${arm_root}" >&2
    return 4
  fi
}

run_training_spec() {
  local spec="$1"
  local kind phase candidate_id role seed method clip rho eta schedule_path arm_root
  IFS='|' read -r kind phase candidate_id role seed method clip rho eta schedule_path arm_root <<<"${spec}"
  if [[ "${kind}" != "train" ]]; then
    echo "error: invalid training spec: ${spec}" >&2
    return 2
  fi
  local adapter_dir="${arm_root}/adapter"
  local result_dir="${arm_root}/results"
  mkdir -p "${adapter_dir}" "${result_dir}" "${arm_root}/logs"
  write_arm_status "${arm_root}" running "${phase}" "${candidate_id}" "${seed}" 0
  {
    echo "phase=${phase}"
    echo "model=${MODEL_ID}@${MODEL_REVISION}"
    echo "candidate_id=${candidate_id}"
    echo "role=${role}"
    echo "seed=${seed}"
    echo "method=${method}"
    echo "initial_or_fixed_clip=${clip}"
    echo "slaclip_target_non_small_clip_fraction=${rho}"
    echo "slaclip_eta=${eta}"
    echo "clip_schedule_path=${schedule_path}"
    echo "code_sha=${EXPECTED_REPO_SHA}"
    echo "selection_metric=public_math10k_numeric_exact_match_accuracy"
    echo "validation_curve_steps=0,50,100,150,200,250,300"
    echo "raw_telemetry=NON_PRIVATE"
    echo "development_artifact_bundle_privacy=NON_PRIVATE_due_to_research_raw_exact_training_statistics"
  } >"${arm_root}/arm-manifest.txt"
  if [[ "${role}" == "matched-fixed-noise-energy" ]]; then
    {
      echo "schedule_privacy_class=DP_DERIVED_FROM_3_SLACLIP_RUNS"
      echo "schedule_source=three_DP_derived_selected_SlaClip_trajectories"
      echo "control_classification=oracle_mechanistic_control"
      echo "matched_fixed_definition=RMS_C_matches_mean_squared_absolute_DP_noise_scale"
      echo "matched_fixed_is_schedule_release=false"
      echo "matched_fixed_conditional_run_privacy=epsilon6_delta1e-5_conditioned_on_locked_schedule_postprocessing"
      echo "one_matched_fixed_plus_schedule_basic_composition=epsilon24_delta4e-5"
      echo "three_matched_fixed_outputs_plus_schedule_basic_composition=epsilon36_delta6e-5"
      echo "schedule_plus_three_replay_plus_three_matched_fixed_basic_composition=epsilon54_delta9e-5"
    } >>"${arm_root}/arm-manifest.txt"
  elif [[ "${role}" == "replay" ]]; then
    {
      echo "schedule_privacy_class=DP_DERIVED_FROM_3_SLACLIP_RUNS"
      echo "schedule_source=three_DP_derived_selected_SlaClip_trajectories"
      echo "control_classification=oracle_mechanistic_control"
      echo "replay_uses_shared_locked_schedule=true"
      echo "replay_conditional_run_privacy=epsilon6_delta1e-5_conditioned_on_locked_schedule"
      echo "one_replay_plus_schedule_basic_composition=epsilon24_delta4e-5"
      echo "three_replay_outputs_plus_schedule_basic_composition=epsilon36_delta6e-5"
      echo "schedule_plus_three_replay_plus_three_matched_fixed_basic_composition=epsilon54_delta9e-5"
    } >>"${arm_root}/arm-manifest.txt"
  fi

  local -a train_args=(
    "${PYTHON_BIN}" -u train_eval.py
    --config configs/math10k_paper.json
    --dataset math10k
    --method "${method}"
    --privacy dp
    --base_model "${MODEL_ID}"
    --model_revision "${MODEL_REVISION}"
    --seed "${seed}"
    --run_name "${phase}-${MODEL_SLUG}-${role}-${candidate_id}-seed${seed}"
    --steps 300
    --batch_size 64
    --micro_batch_size 4
    --lr 0.0003
    --cutoff_len 256
    --train_on_inputs true
    --epsilon 6.0
    --delta 1e-5
    --initial_clip_threshold "${clip}"
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
    --telemetry_mode research_raw
    --allow_non_private_telemetry
    --raw_hist_bins 128
    --raw_hist_max 30.0
    --require_cuda
    --resume
    --checkpoint_every 25
    --run_train true
    --run_eval false
    --output_dir "${adapter_dir}"
    --result_dir "${result_dir}"
  )
  case "${method}" in
    slaclip)
      train_args+=(
        --slaclip_target_non_small_clip_fraction "${rho}"
        --slaclip_eta "${eta}"
        --slaclip_num_slots 15
        --slaclip_c_min 0.1
        --slaclip_c_max 15.0
      )
      ;;
    replay)
      if [[ ! -s "${schedule_path}" ]]; then
        echo "error: replay schedule is missing: ${schedule_path}" >&2
        write_arm_status "${arm_root}" failed "${phase}" "${candidate_id}" "${seed}" 2
        return 2
      fi
      train_args+=(--clip_schedule_path "${schedule_path}")
      ;;
    baseline) ;;
    *)
      echo "error: unsupported method in plan: ${method}" >&2
      return 2
      ;;
  esac

  trap 'terminate_lane_process; write_arm_status "${arm_root}" failed "${phase}" "${candidate_id}" "${seed}" 143; exit 143' TERM INT
  local train_rc=0
  lane_srun "${arm_root}" train "${train_args[@]}" || train_rc=$?
  trap - TERM INT
  if [[ "${train_rc}" != "0" ]]; then
    write_arm_status "${arm_root}" failed "${phase}" "${candidate_id}" "${seed}" "${train_rc}"
    return "${train_rc}"
  fi
  local validate_rc=0
  validate_selection_run "${arm_root}" "${method}" "${seed}" || validate_rc=$?
  if [[ "${validate_rc}" != "0" ]]; then
    write_arm_status "${arm_root}" failed "${phase}" "${candidate_id}" "${seed}" "${validate_rc}"
    return "${validate_rc}"
  fi
  local summary_rc=0
  summarize_run_telemetry "${arm_root}" || summary_rc=$?
  if [[ "${summary_rc}" != "0" ]]; then
    write_arm_status "${arm_root}" failed "${phase}" "${candidate_id}" "${seed}" "${summary_rc}"
    return "${summary_rc}"
  fi
  write_arm_status "${arm_root}" completed "${phase}" "${candidate_id}" "${seed}" 0
}

run_synthetic_spec() {
  local spec="$1"
  local kind method
  IFS='|' read -r kind method <<<"${spec}"
  if [[ "${kind}" != "synthetic" || ( "${method}" != "baseline" && "${method}" != "slaclip" ) ]]; then
    echo "error: invalid synthetic smoke spec: ${spec}" >&2
    return 2
  fi
  local arm_root="${CAMPAIGN_ROOT}/smoke/synthetic/${method}"
  mkdir -p "${arm_root}"
  local -a args=(
    "${PYTHON_BIN}" -u scripts/smoke_dp_path.py
    --device cuda
    --steps 2
    --method "${method}"
    --clip-norm 1.0
  )
  if [[ "${method}" == "slaclip" ]]; then
    args+=(
      --slaclip-beta 0.99
      --slaclip-eta 0.15
      --slaclip-num-slots 15
      --slaclip-c-min 0.1
      --slaclip-c-max 15.0
    )
  fi
  trap 'terminate_lane_process; exit 143' TERM INT
  local rc=0
  lane_srun "${arm_root}" synthetic "${args[@]}" || rc=$?
  trap - TERM INT
  return "${rc}"
}

run_real_smoke_spec() {
  local spec="$1"
  local kind method
  IFS='|' read -r kind method <<<"${spec}"
  if [[ "${kind}" != "real-smoke" ]]; then
    echo "error: invalid real smoke spec: ${spec}" >&2
    return 2
  fi
  local arm_root="${CAMPAIGN_ROOT}/smoke/real/${method}"
  local adapter_dir="${arm_root}/adapter"
  local result_dir="${arm_root}/results"
  local schedule_path="${CAMPAIGN_ROOT}/smoke/replay_schedule_2step.json"
  mkdir -p "${adapter_dir}" "${result_dir}"
  if [[ "${method}" == "replay" && ! -e "${schedule_path}" ]]; then
    "${PYTHON_BIN}" - "${schedule_path}" <<'PY'
import json
import os
import tempfile
import sys
from pathlib import Path

path = Path(sys.argv[1])
path.parent.mkdir(parents=True, exist_ok=True)
payload = {"schema_version": 1, "num_steps": 2, "clip_thresholds": [1.0, 1.0], "purpose": "real-model smoke only"}
encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
temporary = Path(name)
try:
    with os.fdopen(fd, "wb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.link(temporary, path)
    except FileExistsError:
        if path.read_bytes() != encoded:
            raise
finally:
    temporary.unlink(missing_ok=True)
PY
  fi
  local -a args=(
    "${PYTHON_BIN}" -u train_eval.py
    --config configs/math10k_paper.json
    --dataset math10k
    --method "${method}"
    --privacy dp
    --base_model "${MODEL_ID}"
    --model_revision "${MODEL_REVISION}"
    --seed 42
    --run_name "smoke-${MODEL_SLUG}-${method}"
    --steps 2
    --batch_size 4
    --micro_batch_size 1
    --initial_clip_threshold 1.0
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
      --slaclip_target_non_small_clip_fraction 0.99
      --slaclip_eta 0.15
      --slaclip_num_slots 15
      --slaclip_c_min 0.1
      --slaclip_c_max 15.0
    )
  elif [[ "${method}" == "replay" ]]; then
    args+=(--clip_schedule_path "${schedule_path}")
  elif [[ "${method}" != "baseline" ]]; then
    echo "error: unsupported real smoke method=${method}" >&2
    return 2
  fi
  trap 'terminate_lane_process; exit 143' TERM INT
  local rc=0
  lane_srun "${arm_root}" real-smoke "${args[@]}" || rc=$?
  trap - TERM INT
  if [[ "${rc}" == "0" ]]; then
    validate_real_smoke "${arm_root}" "${method}" || rc=$?
  fi
  return "${rc}"
}

run_spec() {
  local spec="$1"
  case "${spec%%|*}" in
    train) run_training_spec "${spec}" ;;
    synthetic) run_synthetic_spec "${spec}" ;;
    real-smoke) run_real_smoke_spec "${spec}" ;;
    *)
      echo "error: unknown lane spec: ${spec}" >&2
      return 2
      ;;
  esac
}

run_parallel_pair() {
  local first_spec="$1"
  local second_spec="${2:-}"
  local first_pid second_pid finished_pid other_pid first_rc=0 other_rc=0
  local wave_id state
  WAVE_COUNTER=$((WAVE_COUNTER + 1))
  wave_id="$(printf '%03d' "${WAVE_COUNTER}")"
  write_wave_status "${wave_id}" running "${first_spec}" "${second_spec}" NA NA
  run_spec "${first_spec}" &
  first_pid=$!
  if [[ -z "${second_spec}" ]]; then
    wait "${first_pid}" || first_rc=$?
    state=completed
    if [[ "${first_rc}" != "0" ]]; then
      state=failed
    fi
    write_wave_status "${wave_id}" "${state}" "${first_spec}" "${second_spec}" "${first_rc}" NA
    return "${first_rc}"
  fi
  run_spec "${second_spec}" &
  second_pid=$!
  while kill -0 "${first_pid}" 2>/dev/null && kill -0 "${second_pid}" 2>/dev/null; do
    sleep 2
  done
  if ! kill -0 "${first_pid}" 2>/dev/null; then
    finished_pid="${first_pid}"
    other_pid="${second_pid}"
  else
    finished_pid="${second_pid}"
    other_pid="${first_pid}"
  fi
  wait "${finished_pid}" || first_rc=$?
  if [[ "${first_rc}" != "0" ]]; then
    kill -TERM "${other_pid}" 2>/dev/null || true
    wait "${other_pid}" 2>/dev/null || other_rc=$?
    if [[ "${finished_pid}" == "${first_pid}" ]]; then
      write_wave_status "${wave_id}" failed "${first_spec}" "${second_spec}" "${first_rc}" "${other_rc}"
    else
      write_wave_status "${wave_id}" failed "${first_spec}" "${second_spec}" "${other_rc}" "${first_rc}"
    fi
    return "${first_rc}"
  fi
  wait "${other_pid}" || other_rc=$?
  state=completed
  if [[ "${other_rc}" != "0" ]]; then
    state=failed
  fi
  if [[ "${finished_pid}" == "${first_pid}" ]]; then
    write_wave_status "${wave_id}" "${state}" "${first_spec}" "${second_spec}" "${first_rc}" "${other_rc}"
  else
    write_wave_status "${wave_id}" "${state}" "${first_spec}" "${second_spec}" "${other_rc}" "${first_rc}"
  fi
  return "${other_rc}"
}

run_plan() {
  local plan_file="$1"
  local first_spec second_spec
  exec 3<"${plan_file}"
  while IFS= read -r first_spec <&3; do
    [[ -n "${first_spec}" ]] || continue
    second_spec=""
    IFS= read -r second_spec <&3 || true
    echo "launching wave first=${first_spec%%|*} second=${second_spec%%|*}"
    run_parallel_pair "${first_spec}" "${second_spec}"
  done
  exec 3<&-
}

create_candidate_registry() {
  "${PYTHON_BIN}" - "${REGISTRY_FILE}" "${MODEL_ID}" "${MODEL_REVISION}" "${EXPECTED_REPO_SHA}" <<'PY'
import hashlib
import json
import math
import os
import sys
import tempfile
from pathlib import Path

target = Path(sys.argv[1])
fixed_values = (0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0)
c0_values = (0.5, 1.0)
rhos = (0.95, 0.99, 0.995)
etas = (0.05, 0.15, 0.30)

def slug(value):
    text = format(float(value), ".12g").replace(".", "p")
    return text.replace("-", "m")

candidates = []
for clip in fixed_values:
    candidates.append({
        "id": f"fixed-c{slug(clip)}",
        "family": "fixed",
        "method": "baseline",
        "params": {"dp_max_grad_norm": clip},
    })
for c0 in c0_values:
    for rho in rhos:
        for eta in etas:
            candidates.append({
                "id": f"sla-c{slug(c0)}-r{slug(rho)}-e{slug(eta)}",
                "family": "slaclip",
                "method": "slaclip",
                "params": {
                    "dp_max_grad_norm": c0,
                    "slaclip_target_non_small_clip_fraction": rho,
                    "slaclip_eta": eta,
                    "slaclip_num_slots": 15,
                    "slaclip_c_min": 0.1,
                    "slaclip_c_max": 15.0,
                },
            })
if len(candidates) != 26 or len({item["id"] for item in candidates}) != 26:
    raise SystemExit("candidate grid must contain exactly 8 fixed and 18 SlaClip arms")
for candidate in candidates:
    runs = {}
    for seed in (42, 43, 44):
        relative = Path("screen") / "runs" / candidate["id"] / f"seed-{seed}"
        runs[str(seed)] = {
            "run_status": str(relative / "adapter" / "run_status.json"),
            "validation_metrics": str(relative / "results" / "validation" / "validation_metrics.json"),
            "split_manifest": str(relative / "results" / "validation" / "split_manifest.json"),
        }
    candidate["runs"] = runs
payload = {
    "schema_version": 1,
    "selection_protocol": {
        "name": "math10k_4b_dynamics_public_numeric_em_v1",
        "protocol_stage": "selection",
        "validation_data_is_public": True,
        "selection_metric": "public_math10k_numeric_exact_match_accuracy",
        "loss_definition": "response_only_per_record_mean_of_nonignored_next_token_losses",
        "required_update_steps": 300,
        "target_epsilon": 6.0,
        "epsilon_tolerance": 0.02,
        "stage1_seed": 42,
        "stage2_seeds": [42, 43, 44],
        "common_config": {
        "dataset": "math10k",
        "privacy": "dp",
        "base_model": sys.argv[2],
        "model_revision": sys.argv[3],
        "lora_r": 16,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "total_update_steps": 300,
        "batch_size": 64,
        "micro_batch_size": 4,
        "learning_rate": 0.0003,
        "cutoff_len": 256,
        "train_on_inputs": True,
        "val_set_size": 500,
        "validation_seed": 1729,
        "validation_batch_size": 8,
        "validation_eval_interval": 50,
        "validation_generate_numeric": True,
        "validation_num_beams": 1,
        "validation_max_new_tokens": 128,
        "validation_max_input_length": 512,
        "protocol_stage": "selection",
        "validation_data_is_public": True,
        "dp_epsilon": 6.0,
        "dp_delta": 1e-5,
        "dp_grad_sample_mode": "functorch",
        "dp_accountant": "prv",
        "dp_secure_mode": False,
        "telemetry_mode": "research_raw",
        "allow_non_private_telemetry": True,
        "raw_hist_bins": 128,
        "raw_hist_max": 30.0,
        "spectral_svd_device": "cpu",
        "spectral_oversample": 8,
        "spectral_n_iter": 2,
        "prism_floor_factor": 0.5,
        "prism_floor_mode": "scalar",
        "prism_cond_max": 10000.0,
        "prism_cond_strategy": "raise_small",
        "prism_lift_fix": "both",
        "prism_debias_second_moment": False,
        "max_update_norm": 0.0,
        },
    },
    "candidates": candidates,
}
encoded = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
target.parent.mkdir(parents=True, exist_ok=True)
if target.exists():
    if target.read_bytes() != encoded:
        raise SystemExit(f"refusing inconsistent registry: {target}")
else:
    fd, name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
os.chmod(target, 0o600)
PY
  atomic_sha256_sidecar "${REGISTRY_FILE}"
}

create_campaign_manifest() {
  local manifest="${CAMPAIGN_ROOT}/campaign_manifest.json"
  "${PYTHON_BIN}" - \
    "${manifest}" "${REGISTRY_FILE}" "${EXPECTED_REPO_SHA}" "${JOB_ID}" \
    "${MODEL_ID}" "${MODEL_REVISION}" "${ENV_PREFIX}" \
    "${ENVIRONMENT_FREEZE}" <<'PY'
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

target = Path(sys.argv[1])
registry = Path(sys.argv[2])
payload = {
    "schema_version": 1,
    "protocol": "math10k_4b_dynamics_public_numeric_em_v1",
    "code_sha": sys.argv[3],
    "job_id": sys.argv[4],
    "model_id": sys.argv[5],
    "model_revision": sys.argv[6],
    "environment": sys.argv[7],
    "environment_freeze": {
        "path": str(Path(sys.argv[8]).relative_to(target.parent)),
        "sha256": hashlib.sha256(Path(sys.argv[8]).read_bytes()).hexdigest(),
    },
    "candidate_registry_sha256": hashlib.sha256(registry.read_bytes()).hexdigest(),
    "resources": {
        "allocation": "one Slurm job",
        "gpu": "2xh200",
        "lanes": 2,
        "cpu_per_lane": 8,
        "host_memory_per_lane": "128G",
    },
    "privacy": {
        "per_training_run_accountant_target": {
            "epsilon": 6.0,
            "delta": 1e-5,
        },
        "telemetry": "research_raw_NON_PRIVATE",
        "schedule_privacy_class": "DP_DERIVED_FROM_3_SLACLIP_RUNS",
        "replay_control_classification": (
            "oracle_mechanistic_control_from_three_DP_derived_SlaClip_trajectories"
        ),
        "control_basic_composition_upper_bounds": {
            "one_replay_conditional_on_schedule": {"epsilon": 6.0, "delta": 1e-5},
            "one_replay_plus_schedule": {"epsilon": 24.0, "delta": 4e-5},
            "three_replay_outputs_plus_schedule": {"epsilon": 36.0, "delta": 6e-5},
            "one_matched_fixed_conditional_on_schedule": {
                "epsilon": 6.0,
                "delta": 1e-5,
            },
            "one_matched_fixed_plus_schedule": {"epsilon": 24.0, "delta": 4e-5},
            "three_matched_fixed_outputs_plus_schedule": {
                "epsilon": 36.0,
                "delta": 6e-5,
            },
            "schedule_plus_three_replay_plus_three_matched_fixed": {
                "epsilon": 54.0,
                "delta": 9e-5,
                "release_count": 9,
            },
        },
        "development_artifact_bundle": (
            "NON_PRIVATE because research_raw telemetry contains exact "
            "training-example statistics"
        ),
    },
    "test_sets_accessed": False,
}
encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
if target.exists():
    existing = json.loads(target.read_text(encoding="utf-8"))
    # A resumed allocation has a different job ID but must retain the original
    # immutable campaign definition.
    candidate = dict(payload)
    candidate["job_id"] = existing.get("job_id")
    if existing != candidate:
        raise SystemExit(f"refusing inconsistent campaign manifest: {target}")
else:
    fd, name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
os.chmod(target, 0o600)
PY
  atomic_sha256_sidecar "${manifest}"
}

create_plan() {
  local mode="$1"
  local output="$2"
  local temporary create_rc=0
  temporary="$(mktemp "${output}.tmp.XXXXXX")"
  "${PYTHON_BIN}" - \
    "${mode}" "${REGISTRY_FILE}" "${CAMPAIGN_ROOT}" "${STAGE1_SELECTION}" \
    "${LOCKED_SELECTION}" "${CONTROL_MANIFEST}" "${REPLAY_SCHEDULE}" \
    >"${temporary}" <<'PY' || create_rc=$?
import json
import sys
from pathlib import Path

mode = sys.argv[1]
registry = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
campaign_root = Path(sys.argv[3])
stage1_selection = Path(sys.argv[4])
locked_selection = Path(sys.argv[5])
control_manifest = Path(sys.argv[6])
replay_schedule = Path(sys.argv[7])
candidates = {item["id"]: item for item in registry["candidates"]}

def emit(phase, candidate, role, seed, arm_root, schedule="NA"):
    params = candidate["params"]
    method = candidate["method"]
    clip = params["dp_max_grad_norm"]
    # New registries use the semantic target name. The fallback keeps this
    # planner able to consume already-locked pre-migration registries.
    rho = params.get(
        "slaclip_target_non_small_clip_fraction",
        params.get("slaclip_beta", "NA"),
    )
    eta = params.get("slaclip_eta", "NA")
    print("|".join(str(item) for item in (
        "train", phase, candidate["id"], role, seed, method, clip, rho, eta,
        schedule, arm_root,
    )))

if mode == "stage1":
    for candidate in registry["candidates"]:
        emit(
            "stage1",
            candidate,
            "grid",
            42,
            campaign_root / "screen" / "runs" / candidate["id"] / "seed-42",
        )
elif mode == "stage2":
    selection = json.loads(stage1_selection.read_text(encoding="utf-8"))
    selected_ids = [item["candidate_id"] for item in selection["top_slaclip"]]
    canonical_ids = [
        candidate_id
        for candidate_id, candidate in candidates.items()
        if candidate["family"] == "fixed"
        and float(candidate["params"]["dp_max_grad_norm"]) == 1.0
    ]
    if len(canonical_ids) != 1:
        raise SystemExit("registry must contain exactly one canonical fixed C=1 candidate")
    selected_ids.extend((
        selection["best_fixed"]["candidate_id"],
        canonical_ids[0],
    ))
    selected_ids = list(dict.fromkeys(selected_ids))
    if len(selection["top_slaclip"]) != 2 or len(selected_ids) not in {3, 4}:
        raise SystemExit("stage2 requires top2 SlaClip plus best-fixed/canonical-C1")
    for seed in (43, 44):
        for candidate_id in selected_ids:
            candidate = candidates[candidate_id]
            emit(
                "stage2",
                candidate,
                "confirmation",
                seed,
                campaign_root / "screen" / "runs" / candidate_id / f"seed-{seed}",
            )
elif mode == "controls":
    locked = json.loads(locked_selection.read_text(encoding="utf-8"))
    controls = json.loads(control_manifest.read_text(encoding="utf-8"))
    selected_id = locked["selected_slaclip"]["candidate_id"]
    if controls["selected_slaclip_candidate_id"] != selected_id:
        raise SystemExit("control manifest does not match locked SlaClip selection")
    replay = {
        "id": "replay-selected-shared-schedule",
        "family": "replay",
        "method": "replay",
        "params": {"dp_max_grad_norm": controls["schedule_initial_clip"]},
    }
    matched = {
        "id": "matched-fixed-noise-energy-rms",
        "family": "fixed-control",
        "method": "baseline",
        "params": {"dp_max_grad_norm": controls["matched_fixed_noise_rms_clip"]},
    }
    for seed in (42, 43, 44):
        emit(
            "controls",
            replay,
            "replay",
            seed,
            campaign_root / "controls" / "runs" / replay["id"] / f"seed-{seed}",
            replay_schedule,
        )
        emit(
            "controls",
            matched,
            "matched-fixed-noise-energy",
            seed,
            campaign_root / "controls" / "runs" / matched["id"] / f"seed-{seed}",
        )
else:
    raise SystemExit(f"unknown plan mode: {mode}")
PY
  if [[ "${create_rc}" != "0" ]]; then
    rm -f -- "${temporary}"
    return "${create_rc}"
  fi
  install_immutable_file "${temporary}" "${output}"
  atomic_sha256_sidecar "${output}"
}

run_formal_selector() {
  local stage="$1"
  local output="$2"
  local -a args=(
    "${PYTHON_BIN}" scripts/select_validation_candidates.py
    --campaign-root "${CAMPAIGN_ROOT}"
    --registry "screen/candidate_registry.json"
    --stage "${stage}"
    --output "${output}"
    --top-slaclip 2
  )
  if [[ "${stage}" == "stage2" ]]; then
    args+=(--stage1-selection "${STAGE1_SELECTION}")
  fi
  "${args[@]}" \
    >"${SELECTION_ROOT}/${stage}-selector-${JOB_ID}.out" \
    2>"${SELECTION_ROOT}/${stage}-selector-${JOB_ID}.err"
  atomic_sha256_sidecar "${output}"
  atomic_sha256_sidecar "${output%.json}.csv"
  if [[ "${stage}" == "stage2" ]]; then
    atomic_sha256_sidecar "${SELECTION_ROOT}/selected.env"
  fi
}

build_replay_schedule_and_controls() {
  "${PYTHON_BIN}" - \
    "${CAMPAIGN_ROOT}" "${LOCKED_SELECTION}" "${REPLAY_SCHEDULE}" "${CONTROL_MANIFEST}" <<'PY'
import hashlib
import json
import math
import os
import statistics
import sys
import tempfile
from pathlib import Path

campaign_root = Path(sys.argv[1])
selection_path = Path(sys.argv[2])
schedule_path = Path(sys.argv[3])
control_path = Path(sys.argv[4])
selection = json.loads(selection_path.read_text(encoding="utf-8"))
selected = selection["selected_slaclip"]
candidate_id = selected["candidate_id"]
seeds = (42, 43, 44)

def source_root(seed):
    return campaign_root / "screen" / "runs" / candidate_id / f"seed-{seed}"

trajectories = []
sources = []
for seed in seeds:
    root = source_root(seed)
    path = root / "adapter" / "train_log.jsonl"
    status_path = root / "adapter" / "run_status.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    if status.get("state") != "completed" or status.get("config", {}).get("method") != "slaclip":
        raise SystemExit(f"invalid selected-SlaClip source status: {status_path}")
    by_step = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            step = int(record["step"])
            value = record.get("dp_clip_threshold")
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise SystemExit(f"missing clip threshold {path}:{line_number}")
            value = float(value)
            if not math.isfinite(value) or value <= 0:
                raise SystemExit(f"invalid clip threshold {path}:{line_number}")
            by_step[step] = value
    if sorted(by_step) != list(range(1, 301)):
        raise SystemExit(f"DP-safe trajectory must cover exactly steps 1..300: {path}")
    trajectories.append([by_step[step] for step in range(1, 301)])
    sources.append({
        "seed": seed,
        "path": str(path.relative_to(campaign_root)),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "config_fingerprint": status["config_fingerprint"],
    })

clip_thresholds = [
    math.fsum(trajectory[index] for trajectory in trajectories) / len(trajectories)
    for index in range(300)
]
expected_initial = float(selected["params"]["dp_max_grad_norm"])
if not math.isclose(clip_thresholds[0], expected_initial, rel_tol=1e-12, abs_tol=1e-12):
    raise SystemExit(
        f"shared replay schedule begins at {clip_thresholds[0]}, selected C0={expected_initial}"
    )
statistics_rows = []
for index, mean_value in enumerate(clip_thresholds):
    values = [trajectory[index] for trajectory in trajectories]
    statistics_rows.append({
        "step": index + 1,
        "clip_threshold": mean_value,
        "source_seed_count": len(values),
        "min": min(values),
        "max": max(values),
        "sample_sd": statistics.stdev(values),
    })
schedule = {
    "schema_version": 1,
    "num_steps": 300,
    "construction": "per_step_arithmetic_mean_of_locked_top1_slaclip_dp_safe_trajectories",
    "schedule_privacy_class": "DP_DERIVED_FROM_3_SLACLIP_RUNS",
    "control_classification": (
        "oracle_mechanistic_control_derived_from_three_DP_SlaClip_trajectories; "
        "not an independently public threshold schedule"
    ),
    "selected_slaclip_candidate_id": candidate_id,
    "selection_sha256": hashlib.sha256(selection_path.read_bytes()).hexdigest(),
    "source_logs": sources,
    "clip_thresholds": clip_thresholds,
    "statistics": statistics_rows,
    "privacy_accounting": {
        "adjacency": "add_remove",
        "basic_composition_upper_bound": True,
        "source_schedule_three_runs": {
            "epsilon": 18.0,
            "delta": 3e-5,
            "derivation": "three DP-derived source trajectories, each epsilon=6/delta=1e-5",
        },
        "one_replay_conditional_on_locked_schedule": {
            "epsilon": 6.0,
            "delta": 1e-5,
        },
        "one_replay_output_plus_schedule": {
            "epsilon": 24.0,
            "delta": 4e-5,
        },
        "three_replay_outputs_plus_schedule": {
            "epsilon": 36.0,
            "delta": 6e-5,
        },
        "one_matched_fixed_conditional_on_locked_schedule": {
            "epsilon": 6.0,
            "delta": 1e-5,
        },
        "one_matched_fixed_output_plus_schedule": {
            "epsilon": 24.0,
            "delta": 4e-5,
        },
        "three_matched_fixed_outputs_plus_schedule": {
            "epsilon": 36.0,
            "delta": 6e-5,
        },
        "schedule_plus_three_replay_plus_three_matched_fixed": {
            "epsilon": 54.0,
            "delta": 9e-5,
            "release_count": 9,
            "derivation": (
                "3 schedule-source runs + 3 replay runs + 3 matched-fixed runs, "
                "each epsilon=6/delta=1e-5"
            ),
        },
        "development_artifact_bundle": (
            "NON_PRIVATE because research_raw telemetry exposes exact "
            "training-example statistics"
        ),
    },
}
canonical = json.dumps(schedule, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
schedule["schedule_content_sha256"] = hashlib.sha256(canonical).hexdigest()
arithmetic_mean = math.fsum(clip_thresholds) / len(clip_thresholds)
geometric_mean = math.exp(
    math.fsum(math.log(value) for value in clip_thresholds) / len(clip_thresholds)
)
noise_rms = math.sqrt(
    math.fsum(value * value for value in clip_thresholds) / len(clip_thresholds)
)
controls = {
    "schema_version": 1,
    "selected_slaclip_candidate_id": candidate_id,
    "selection_sha256": hashlib.sha256(selection_path.read_bytes()).hexdigest(),
    "schedule_content_sha256": schedule["schedule_content_sha256"],
    "schedule_initial_clip": clip_thresholds[0],
    "schedule_arithmetic_mean_clip": arithmetic_mean,
    "schedule_geometric_mean_clip": geometric_mean,
    "schedule_noise_rms_clip": noise_rms,
    "matched_fixed_definition": (
        "root_mean_square_of_shared_replay_schedule_over_300_steps; "
        "matches mean squared absolute Gaussian noise scale because std_t is proportional to C_t"
    ),
    "matched_fixed_noise_rms_clip": noise_rms,
    "schedule_privacy_class": "DP_DERIVED_FROM_3_SLACLIP_RUNS",
    "control_classification": "oracle_mechanistic_control_from_three_DP_derived_trajectories",
    "privacy_basic_composition": {
        "one_replay_conditional_epsilon": 6.0,
        "one_replay_conditional_delta": 1e-5,
        "one_replay_plus_schedule_epsilon": 24.0,
        "one_replay_plus_schedule_delta": 4e-5,
        "three_replay_outputs_plus_schedule_epsilon": 36.0,
        "three_replay_outputs_plus_schedule_delta": 6e-5,
        "one_matched_fixed_conditional_epsilon": 6.0,
        "one_matched_fixed_conditional_delta": 1e-5,
        "one_matched_fixed_plus_schedule_epsilon": 24.0,
        "one_matched_fixed_plus_schedule_delta": 4e-5,
        "three_matched_fixed_outputs_plus_schedule_epsilon": 36.0,
        "three_matched_fixed_outputs_plus_schedule_delta": 6e-5,
        "schedule_plus_three_replay_plus_three_matched_fixed_epsilon": 54.0,
        "schedule_plus_three_replay_plus_three_matched_fixed_delta": 9e-5,
        "schedule_plus_three_replay_plus_three_matched_fixed_release_count": 9,
        "development_bundle": "NON_PRIVATE_due_to_research_raw",
    },
    "control_seeds": list(seeds),
}

def write_locked(path, payload):
    encoded = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != encoded:
            raise SystemExit(f"refusing to replace immutable artifact: {path}")
        return
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    os.chmod(path, 0o600)

write_locked(schedule_path, schedule)
write_locked(control_path, controls)
print(
    f"replay_schedule={schedule_path} selected={candidate_id} "
    f"matched_fixed_noise_rms_C={noise_rms:.12g} "
    f"arithmetic_mean_C={arithmetic_mean:.12g} geometric_mean_C={geometric_mean:.12g}"
)
PY
  atomic_sha256_sidecar "${REPLAY_SCHEDULE}"
  atomic_sha256_sidecar "${CONTROL_MANIFEST}"
}

write_artifact_manifest() {
  local output="${CAMPAIGN_ROOT}/artifacts/manifest.json"
  "${PYTHON_BIN}" - \
    "${CAMPAIGN_ROOT}" "${EXPECTED_REPO_SHA}" "${REGISTRY_FILE}" "${STAGE1_SELECTION}" \
    "${LOCKED_SELECTION}" "${REPLAY_SCHEDULE}" "${CONTROL_MANIFEST}" "${output}" <<'PY'
import csv
import hashlib
import io
import json
import math
import os
import sys
import tempfile
from pathlib import Path

root = Path(sys.argv[1]).resolve()
code_sha = sys.argv[2]
locked_files = [Path(item).resolve() for item in sys.argv[3:8]]
output = Path(sys.argv[8]).resolve()
plan_files = [
    root / "plans" / "stage1.tsv",
    root / "plans" / "stage2.tsv",
    root / "plans" / "controls.tsv",
]

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def specs(path):
    result = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            fields = line.rstrip("\n").split("|")
            if len(fields) != 11 or fields[0] != "train":
                raise SystemExit(f"invalid plan record: {path}: {line!r}")
            result.append(fields)
    return result

all_specs = [item for path in plan_files for item in specs(path)]
if len(all_specs) not in {38, 40}:
    raise SystemExit(f"expected 38 or 40 unique experiment arms, found {len(all_specs)}")
arm_roots = [Path(fields[10]).resolve() for fields in all_specs]
if len(set(arm_roots)) != len(arm_roots):
    raise SystemExit("plans reuse an arm output directory")
runs = []
run_index_rows = []
for fields, arm_root in zip(all_specs, arm_roots):
    if root not in arm_root.parents:
        raise SystemExit(f"arm escapes campaign root: {arm_root}")
    required = {
        "orchestration_status": arm_root / "orchestration-status.txt",
        "arm_manifest": arm_root / "arm-manifest.txt",
        "adapter_status": arm_root / "adapter" / "run_status.json",
        "validation_metrics": arm_root / "results" / "validation" / "validation_metrics.json",
        "validation_curve": arm_root / "results" / "validation" / "validation_curve.jsonl",
        "split_manifest": arm_root / "results" / "validation" / "split_manifest.json",
        "public_numeric_predictions": arm_root / "results" / "validation" / "public_numeric_predictions.json",
        "telemetry_summary": arm_root / "results" / "research_raw" / "telemetry_summary.json",
        "telemetry_steps": arm_root / "results" / "research_raw" / "telemetry_steps.csv",
        "safe_train_log": arm_root / "adapter" / "train_log.jsonl",
        "raw_train_log": arm_root / "results" / "research_raw" / "NON_PRIVATE_train_log.jsonl",
        "adapter_config": arm_root / "adapter" / "adapter_config.json",
    }
    adapter_weights = [
        path
        for path in (
            arm_root / "adapter" / "adapter_model.safetensors",
            arm_root / "adapter" / "adapter_model.bin",
        )
        if path.is_file()
    ]
    if len(adapter_weights) != 1:
        raise SystemExit(
            f"expected exactly one adapter weight artifact, found {adapter_weights}: {arm_root}"
        )
    required[f"adapter_weights_{adapter_weights[0].suffix.lstrip('.')}"] = adapter_weights[0]
    for label, path in required.items():
        if not path.is_file() or path.stat().st_size == 0:
            raise SystemExit(f"missing {label}: {path}")
    status = json.loads(required["adapter_status"].read_text(encoding="utf-8"))
    if status.get("state") != "completed" or int(status.get("update_steps", -1)) != 300:
        raise SystemExit(f"incomplete adapter status: {required['adapter_status']}")
    validation = json.loads(required["validation_metrics"].read_text(encoding="utf-8"))
    telemetry = json.loads(required["telemetry_summary"].read_text(encoding="utf-8"))
    curve = [
        json.loads(line)
        for line in required["validation_curve"].read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if [int(item.get("step", -1)) for item in curve] != [0, 50, 100, 150, 200, 250, 300]:
        raise SystemExit(f"unexpected validation curve coverage: {required['validation_curve']}")

    def finite(value, label):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise SystemExit(f"{label} must be numeric")
        result = float(value)
        if not math.isfinite(result):
            raise SystemExit(f"{label} must be finite")
        return result

    def telemetry_mean(name):
        metric = (telemetry.get("metrics") or {}).get(name)
        if not isinstance(metric, dict):
            raise SystemExit(f"telemetry summary lacks metric={name}: {required['telemetry_summary']}")
        return finite(metric.get("mean"), f"telemetry {name}.mean")

    def optional_telemetry_mean(name):
        metric = (telemetry.get("metrics") or {}).get(name)
        if not isinstance(metric, dict) or metric.get("mean") is None:
            return ""
        return finite(metric.get("mean"), f"telemetry {name}.mean")

    curve_steps = [finite(item["step"], "curve step") for item in curve]
    curve_losses = [finite(item.get("loss_mean"), "curve loss") for item in curve]
    curve_width = curve_steps[-1] - curve_steps[0]
    curve_auc = math.fsum(
        (curve_steps[index + 1] - curve_steps[index])
        * (curve_losses[index + 1] + curve_losses[index])
        / 2.0
        for index in range(len(curve_steps) - 1)
    ) / curve_width
    accounting = status.get("privacy_accounting") or {}
    epsilon = finite(accounting.get("epsilon_spent"), "epsilon_spent")
    run_index_rows.append({
        "NON_PRIVATE_TELEMETRY": True,
        "phase": fields[1],
        "candidate_id": fields[2],
        "role": fields[3],
        "seed": int(fields[4]),
        "method": fields[5],
        "initial_or_fixed_C": finite(float(fields[6]), "plan C"),
        "slaclip_target_non_small_clip_fraction": (
            "" if fields[7] == "NA" else finite(float(fields[7]), "plan target")
        ),
        # Preserve the old column for readers of pre-migration campaign
        # indexes. It is an alias of the canonical full-SlaClip target.
        "slaclip_beta": "" if fields[7] == "NA" else finite(float(fields[7]), "plan target"),
        "slaclip_eta": "" if fields[8] == "NA" else finite(float(fields[8]), "plan eta"),
        "numeric_exact_accuracy": finite(validation.get("numeric_exact_accuracy"), "numeric accuracy"),
        "numeric_exact_correct": int(validation.get("numeric_exact_correct")),
        "numeric_parse_failures": int(validation.get("numeric_parse_failures")),
        "endpoint_validation_loss": finite(validation.get("loss_mean"), "endpoint loss"),
        "validation_loss_step0": curve_losses[0],
        "validation_loss_step300": curve_losses[-1],
        "validation_loss_normalized_auc": curve_auc,
        "epsilon_spent": epsilon,
        "raw_clip_fraction_mean": telemetry_mean("raw_clip_fraction"),
        "dp_clip_threshold_mean": telemetry_mean("dp_clip_threshold"),
        "raw_signal_to_noise_ratio_mean": telemetry_mean("raw_signal_to_noise_ratio"),
        "raw_clipping_bias_norm_mean": telemetry_mean("raw_clipping_bias_norm"),
        "raw_realized_noise_norm_mean": telemetry_mean("raw_realized_noise_norm"),
        "raw_signal_retention_ratio_mean": telemetry_mean(
            "raw_signal_retention_ratio"
        ),
        "raw_clipping_bias_to_noise_ratio_mean": telemetry_mean(
            "raw_clipping_bias_to_noise_ratio"
        ),
        "raw_bias_noise_squared_error_proxy_mean": telemetry_mean(
            "raw_bias_noise_squared_error_proxy"
        ),
        "raw_unclipped_clipped_cosine_mean": telemetry_mean(
            "raw_unclipped_clipped_cosine"
        ),
        "raw_clipped_noisy_cosine_mean": telemetry_mean(
            "raw_clipped_noisy_cosine"
        ),
        "raw_slack_indicator_noise_residual_rmse_mean": (
            optional_telemetry_mean(
                "raw_slack_indicator_noise_residual_rmse"
            )
        ),
        "slaclip_controller_error_mean": optional_telemetry_mean(
            "slaclip_controller_error"
        ),
        "config_fingerprint": status["config_fingerprint"],
        "arm_root": str(arm_root.relative_to(root)),
    })
    runs.append({
        "phase": fields[1],
        "candidate_id": fields[2],
        "role": fields[3],
        "seed": int(fields[4]),
        "method": fields[5],
        "arm_root": str(arm_root.relative_to(root)),
        "config_fingerprint": status["config_fingerprint"],
        "artifacts": {
            label: {
                "path": str(path.relative_to(root)),
                "sha256": sha(path),
                "bytes": path.stat().st_size,
            }
            for label, path in required.items()
        },
    })

run_index_path = root / "artifacts" / "run_index.csv"
run_index_fields = [
    "NON_PRIVATE_TELEMETRY",
    "phase",
    "candidate_id",
    "role",
    "seed",
    "method",
    "initial_or_fixed_C",
    "slaclip_target_non_small_clip_fraction",
    "slaclip_beta",
    "slaclip_eta",
    "numeric_exact_accuracy",
    "numeric_exact_correct",
    "numeric_parse_failures",
    "endpoint_validation_loss",
    "validation_loss_step0",
    "validation_loss_step300",
    "validation_loss_normalized_auc",
    "epsilon_spent",
    "raw_clip_fraction_mean",
    "dp_clip_threshold_mean",
    "raw_signal_to_noise_ratio_mean",
    "raw_clipping_bias_norm_mean",
    "raw_realized_noise_norm_mean",
    "raw_signal_retention_ratio_mean",
    "raw_clipping_bias_to_noise_ratio_mean",
    "raw_bias_noise_squared_error_proxy_mean",
    "raw_unclipped_clipped_cosine_mean",
    "raw_clipped_noisy_cosine_mean",
    "raw_slack_indicator_noise_residual_rmse_mean",
    "slaclip_controller_error_mean",
    "config_fingerprint",
    "arm_root",
]
buffer = io.StringIO(newline="")
writer = csv.DictWriter(buffer, fieldnames=run_index_fields, lineterminator="\n")
writer.writeheader()
writer.writerows(run_index_rows)
run_index_encoded = buffer.getvalue().encode()
if run_index_path.exists():
    if run_index_path.read_bytes() != run_index_encoded:
        raise SystemExit(f"refusing inconsistent run index: {run_index_path}")
else:
    fd, name = tempfile.mkstemp(
        prefix=f".{run_index_path.name}.", suffix=".tmp", dir=run_index_path.parent
    )
    temporary = Path(name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(run_index_encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, run_index_path)
    finally:
        temporary.unlink(missing_ok=True)
os.chmod(run_index_path, 0o600)

campaign_snapshot_files = [
    root / "campaign_manifest.json",
    root / "submission_receipt.json",
    root / "artifacts" / "environment_freeze.txt",
    *sorted((root / "artifacts").glob("allocation-*.txt")),
    *sorted(root.glob("preflight/**/logs/preflight-*.out")),
    *sorted(root.glob("preflight/**/logs/preflight-*.err")),
]
for path in campaign_snapshot_files:
    if not path.is_file():
        raise SystemExit(f"missing campaign/environment/preflight snapshot: {path}")
payload = {
    "schema_version": 1,
    "analysis_status": "DEVELOPMENT_CAMPAIGN_COMPLETE",
    "code_sha": code_sha,
    "raw_exact_telemetry_is_non_private": True,
    "privacy_accounting": {
        "schedule_privacy_class": "DP_DERIVED_FROM_3_SLACLIP_RUNS",
        "replay_control_classification": (
            "oracle_mechanistic_control_from_three_DP_derived_SlaClip_trajectories"
        ),
        "one_replay_conditional": {"epsilon": 6.0, "delta": 1e-5},
        "one_replay_plus_schedule_basic_composition": {"epsilon": 24.0, "delta": 4e-5},
        "three_replay_outputs_plus_schedule_basic_composition": {"epsilon": 36.0, "delta": 6e-5},
        "one_matched_fixed_conditional": {"epsilon": 6.0, "delta": 1e-5},
        "one_matched_fixed_plus_schedule_basic_composition": {
            "epsilon": 24.0,
            "delta": 4e-5,
        },
        "three_matched_fixed_outputs_plus_schedule_basic_composition": {
            "epsilon": 36.0,
            "delta": 6e-5,
        },
        "schedule_plus_three_replay_plus_three_matched_fixed_basic_composition": {
            "epsilon": 54.0,
            "delta": 9e-5,
            "release_count": 9,
        },
        "development_artifact_bundle": "NON_PRIVATE_due_to_research_raw",
    },
    "test_sets_accessed": False,
    "arm_count": len(runs),
    "locked_artifacts": [
        {"path": str(path.relative_to(root)), "sha256": sha(path), "bytes": path.stat().st_size}
        for path in [*locked_files, *plan_files, run_index_path]
    ],
    "run_index": {
        "path": str(run_index_path.relative_to(root)),
        "sha256": sha(run_index_path),
        "bytes": run_index_path.stat().st_size,
        "rows": len(run_index_rows),
        "privacy": "NON_PRIVATE_TELEMETRY",
    },
    "campaign_environment_preflight_snapshots": [
        {
            "path": str(path.relative_to(root)),
            "sha256": sha(path),
            "bytes": path.stat().st_size,
        }
        for path in campaign_snapshot_files
    ],
    "runs": runs,
}
encoded = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
output.parent.mkdir(parents=True, exist_ok=True)
if output.exists():
    if output.read_bytes() != encoded:
        raise SystemExit(f"refusing inconsistent final artifact manifest: {output}")
else:
    fd, name = tempfile.mkstemp(prefix=f".{output.name}.", suffix=".tmp", dir=output.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
os.chmod(output, 0o600)
print(f"artifact_manifest={output} arms={len(runs)}")
PY
  atomic_sha256_sidecar "${output}"
}

CURRENT_PHASE="preflight"
write_job_status running 0
echo "job_id=${JOB_ID} campaign=${CAMPAIGN_ID} host=$(hostname)"
echo "code_sha=${EXPECTED_REPO_SHA} environment=${ENV_PREFIX}"
nvidia-smi
gpu_count="$("${PYTHON_BIN}" -c 'import torch; print(torch.cuda.device_count())')"
if [[ "${gpu_count}" != "2" ]]; then
  echo "error: expected exactly two visible GPUs, found ${gpu_count}" >&2
  exit 2
fi
"${PYTHON_BIN}" -m pip check
"${PYTHON_BIN}" -m compileall -q train_eval.py src scripts
create_environment_and_allocation_snapshots
lane_srun \
  "${CAMPAIGN_ROOT}/preflight/${MODEL_SLUG}" \
  preflight \
  "${PYTHON_BIN}" scripts/preflight_hpc.py \
    --dataset math10k \
    --output-root "${CAMPAIGN_ROOT}/preflight/${MODEL_SLUG}" \
    --require-cuda \
    --model-id "${MODEL_ID}" \
    --model-revision "${MODEL_REVISION}"

CURRENT_PHASE="smoke"
write_job_status running 0
run_parallel_pair "synthetic|baseline" "synthetic|slaclip"
run_parallel_pair "real-smoke|baseline" "real-smoke|slaclip"
run_parallel_pair "real-smoke|replay"

CURRENT_PHASE="registry"
write_job_status running 0
create_candidate_registry
create_campaign_manifest

CURRENT_PHASE="stage1"
write_job_status running 0
STAGE1_PLAN="${PLAN_ROOT}/stage1.tsv"
create_plan stage1 "${STAGE1_PLAN}"
if [[ "$(wc -l <"${STAGE1_PLAN}")" != "26" ]]; then
  echo "error: stage1 plan must contain exactly 26 arms" >&2
  exit 2
fi
run_plan "${STAGE1_PLAN}"

CURRENT_PHASE="stage1-selection"
write_job_status running 0
run_formal_selector stage1 "${STAGE1_SELECTION}"

CURRENT_PHASE="stage2-confirmation"
write_job_status running 0
STAGE2_PLAN="${PLAN_ROOT}/stage2.tsv"
create_plan stage2 "${STAGE2_PLAN}"
stage2_count="$(wc -l <"${STAGE2_PLAN}")"
if [[ "${stage2_count}" != "6" && "${stage2_count}" != "8" ]]; then
  echo "error: stage2 plan must contain six or eight arms, found ${stage2_count}" >&2
  exit 2
fi
run_plan "${STAGE2_PLAN}"

CURRENT_PHASE="locked-selection"
write_job_status running 0
run_formal_selector stage2 "${LOCKED_SELECTION}"

CURRENT_PHASE="replay-lock"
write_job_status running 0
build_replay_schedule_and_controls

CURRENT_PHASE="controls"
write_job_status running 0
CONTROL_PLAN="${PLAN_ROOT}/controls.tsv"
create_plan controls "${CONTROL_PLAN}"
if [[ "$(wc -l <"${CONTROL_PLAN}")" != "6" ]]; then
  echo "error: replay/matched-fixed plan must contain exactly six arms" >&2
  exit 2
fi
run_plan "${CONTROL_PLAN}"

CURRENT_PHASE="artifact-manifest"
write_job_status running 0
write_artifact_manifest

CURRENT_PHASE="complete"
write_job_status completed 0
echo "Math-10K 4B dynamics campaign completed at $(date --iso-8601=seconds)."
