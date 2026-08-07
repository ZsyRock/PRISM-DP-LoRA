#!/usr/bin/env bash

# Worker for one two-A100, two-lane Math-10K local refinement campaign.
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
JOB_TMP="${SLURM_TMPDIR:-${SCRATCH_ROOT}/tmp/prism-4b-refinement-${JOB_ID}}"
STATUS_ROOT="${CAMPAIGN_ROOT}/status"
PLAN_ROOT="${CAMPAIGN_ROOT}/plans"
REGISTRY_ROOT="${CAMPAIGN_ROOT}/screen"
FIXED_MANIFEST="${REGISTRY_ROOT}/fixed_scan_manifest.json"
REGISTRY_FILE="${REGISTRY_ROOT}/candidate_registry.json"
SELECTION_ROOT="${CAMPAIGN_ROOT}/selection"
TARGET_GRID_FILE="${SELECTION_ROOT}/slaclip_target_grid.json"
STAGE1_SELECTION="${SELECTION_ROOT}/stage1-selection.json"
LOCKED_SELECTION="${SELECTION_ROOT}/selection.json"
SCHEDULE_ROOT="${CAMPAIGN_ROOT}/schedule"
REPLAY_SCHEDULE="${SCHEDULE_ROOT}/replay_schedule.json"
CONTROL_MANIFEST="${SCHEDULE_ROOT}/control_manifest.json"
FINAL_EVAL_ASSET_MANIFEST="${SELECTION_ROOT}/final-evaluation-assets.json"
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
    echo "requested_gpu_per_lane=1xa100"
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
  scripts/analyze_math10k_4b_refinement_campaign.py \
  scripts/analyze_mechanism_campaign.py \
  scripts/build_math10k_4b_refinement_registry.py \
  scripts/preflight_hpc.py \
  scripts/cuda_step_guard.py \
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
  "${CAMPAIGN_ROOT}/final/${MODEL_SLUG}" \
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
# and configurable full-SlaClip interfaces on which this protocol depends.
TRAIN_HELP="$("${PYTHON_BIN}" train_eval.py --help)"
for required_option in \
  "--slaclip_target_non_small_clip_fraction" \
  "--slaclip_eta" \
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
  local guard_root="${arm_root}/cuda-guards"
  local max_attempts=3
  local attempt lane_rc marker_path
  mkdir -p "${guard_root}"
  for ((attempt = 1; attempt <= max_attempts; attempt++)); do
    marker_path="${guard_root}/${log_stem}-${JOB_ID}-attempt-${attempt}.json"
    lane_rc=0
    srun \
      --export=ALL \
      --exclusive \
      --exact \
      --nodes=1 \
      --ntasks=1 \
      --cpus-per-task=8 \
      --mem=128G \
      --gres=gpu:a100:1 \
      --kill-on-bad-exit=1 \
      --output="${arm_root}/logs/${log_stem}-${JOB_ID}-attempt-${attempt}.out" \
      --error="${arm_root}/logs/${log_stem}-${JOB_ID}-attempt-${attempt}.err" \
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
        "${PYTHON_BIN}" -u scripts/cuda_step_guard.py \
          --marker "${marker_path}" -- "$@" &
    LANE_SRUN_PID=$!
    wait "${LANE_SRUN_PID}" || lane_rc=$?
    LANE_SRUN_PID=""
    if [[ "${lane_rc}" == "0" ]]; then
      return 0
    fi
    # A marker is created immediately before exec.  Once it exists, any
    # nonzero result belongs to the workload and must never be auto-repeated.
    if [[ -e "${marker_path}" ]]; then
      return "${lane_rc}"
    fi
    echo "warning: retry-safe CUDA/Slurm step startup failure " \
      "attempt=${attempt}/${max_attempts} arm=${arm_root} rc=${lane_rc}" >&2
    if (( attempt < max_attempts )); then
      sleep $((5 * attempt))
    fi
  done
  return "${lane_rc}"
}

run_dual_cuda_guard_pair() {
  local first_pid second_pid first_rc=0 second_rc=0
  lane_srun \
    "${CAMPAIGN_ROOT}/preflight/cuda-lane-0" \
    cuda-lane-0 \
    "${PYTHON_BIN}" -c \
      'import torch; print(f"guarded_lane_device={torch.cuda.get_device_name(0)}", flush=True)' &
  first_pid=$!
  lane_srun \
    "${CAMPAIGN_ROOT}/preflight/cuda-lane-1" \
    cuda-lane-1 \
    "${PYTHON_BIN}" -c \
      'import torch; print(f"guarded_lane_device={torch.cuda.get_device_name(0)}", flush=True)' &
  second_pid=$!
  wait "${first_pid}" || first_rc=$?
  wait "${second_pid}" || second_rc=$?
  if [[ "${first_rc}" != "0" || "${second_rc}" != "0" ]]; then
    echo "error: concurrent CUDA lane guard failed " \
      "lane0_rc=${first_rc} lane1_rc=${second_rc}" >&2
    return 86
  fi
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

validate_schedule_source_run() {
  local arm_root="$1"
  local expected_seed="$2"
  "${PYTHON_BIN}" - "${arm_root}" "${expected_seed}" <<'PY'
import json
import math
import sys
from pathlib import Path

root = Path(sys.argv[1])
expected_seed = int(sys.argv[2])
status_path = root / "adapter" / "run_status.json"
log_path = root / "adapter" / "train_log.jsonl"
split_path = root / "results" / "validation" / "split_manifest.json"
for path in (status_path, log_path, split_path):
    if not path.is_file() or path.stat().st_size == 0:
        raise SystemExit(f"missing full-data schedule-source artifact: {path}")
status = json.loads(status_path.read_text(encoding="utf-8"))
config = status.get("config") or {}
expected = {
    "method": "slaclip",
    "seed": expected_seed,
    "protocol_stage": "final",
    "val_set_size": 0,
    "run_eval": False,
}
if status.get("state") != "completed" or int(status.get("update_steps", -1)) != 300:
    raise SystemExit(f"incomplete full-data schedule-source run: {status_path}")
for key, value in expected.items():
    if config.get(key) != value:
        raise SystemExit(
            f"schedule-source identity mismatch {key}={config.get(key)!r}, "
            f"expected={value!r}: {status_path}"
        )
epsilon = (status.get("privacy_accounting") or {}).get("epsilon_spent")
if epsilon is None or not math.isfinite(float(epsilon)) or abs(float(epsilon) - 6.0) > 0.02:
    raise SystemExit(f"unexpected schedule-source epsilon={epsilon}: {status_path}")
for forbidden in (
    root / "results" / "summary.csv",
    root / "results" / "details.csv",
    root / "results" / "evaluation_config.json",
    root / "results" / "gsm8k.json",
    root / "results" / "AQuA.json",
    root / "results" / "mawps.json",
    root / "results" / "SVAMP.json",
    root / "results" / "validation" / "validation_metrics.json",
    root / "results" / "validation" / "public_numeric_predictions.json",
):
    if forbidden.exists():
        raise SystemExit(f"schedule-source must not access evaluation/validation data: {forbidden}")
split = json.loads(split_path.read_text(encoding="utf-8"))
if (
    split.get("protocol_stage") != "final"
    or int(split.get("source_rows", -1)) != 9919
    or int(split.get("train_rows", -1)) != 9919
    or int(split.get("validation_rows", -1)) != 0
    or split.get("validation_data_is_public") is not False
):
    raise SystemExit(f"invalid full-data schedule-source split: {split_path}")
if status.get("data_split") != split:
    raise SystemExit(f"schedule-source status/split manifest mismatch: {split_path}")
if not isinstance(split.get("manifest_sha256"), str) or len(split["manifest_sha256"]) != 64:
    raise SystemExit(f"schedule-source split lacks manifest SHA: {split_path}")
by_step = {}
with log_path.open("r", encoding="utf-8") as handle:
    for line_number, line in enumerate(handle, 1):
        if not line.strip():
            continue
        record = json.loads(line)
        step = int(record.get("step", -1))
        threshold = record.get("dp_clip_threshold")
        if (
            isinstance(threshold, bool)
            or not isinstance(threshold, (int, float))
            or not math.isfinite(float(threshold))
            or float(threshold) <= 0.0
        ):
            raise SystemExit(f"invalid schedule threshold {log_path}:{line_number}")
        by_step[step] = float(threshold)
if sorted(by_step) != list(range(1, 301)):
    raise SystemExit(f"schedule-source thresholds must cover steps 1..300: {log_path}")
print(f"validated full-data schedule-source seed={expected_seed}")
PY
}

validate_final_run() {
  local arm_root="$1"
  local expected_method="$2"
  local expected_seed="$3"
  "${PYTHON_BIN}" - \
    "${arm_root}" "${expected_method}" "${expected_seed}" \
    "${FINAL_EVAL_ASSET_MANIFEST}" <<'PY'
import csv
import hashlib
import json
import math
import os
import sys
import tempfile
import unicodedata
from pathlib import Path

root = Path(sys.argv[1])
expected_method = sys.argv[2]
expected_seed = int(sys.argv[3])
asset_manifest_path = Path(sys.argv[4]).resolve(strict=True)
status_path = root / "adapter" / "run_status.json"
summary_path = root / "results" / "summary.csv"
details_path = root / "results" / "details.csv"
evaluation_config_path = root / "results" / "evaluation_config.json"
decontaminated_path = root / "results" / "decontaminated_metrics.json"
task_paths = {
    task: root / "results" / f"{task}.json"
    for task in ("gsm8k", "AQuA", "mawps", "SVAMP")
}
for path in (
    status_path,
    summary_path,
    details_path,
    evaluation_config_path,
    *task_paths.values(),
):
    if not path.is_file() or path.stat().st_size == 0:
        raise SystemExit(f"missing required locked-final artifact: {path}")
status = json.loads(status_path.read_text(encoding="utf-8"))
if status.get("state") != "completed" or int(status.get("update_steps", -1)) != 300:
    raise SystemExit(f"incomplete 300-step final run: {status_path}")
config = status.get("config") or {}
expected = {
    "method": expected_method,
    "seed": expected_seed,
    "protocol_stage": "final",
    "val_set_size": 0,
    "run_eval": True,
}
for key, value in expected.items():
    if config.get(key) != value:
        raise SystemExit(
            f"final run identity mismatch {key}={config.get(key)!r}, expected={value!r}: {status_path}"
        )
privacy = status.get("privacy_accounting") or {}
epsilon = privacy.get("epsilon_spent")
if epsilon is None or not math.isfinite(float(epsilon)) or abs(float(epsilon) - 6.0) > 0.02:
    raise SystemExit(f"unexpected final epsilon={epsilon}: {status_path}")
evaluation_config = json.loads(evaluation_config_path.read_text(encoding="utf-8"))
expected_evaluation = {
    "dataset": "math10k",
    "base_model": "google/gemma-3-4b-pt",
    "batch_size": 8,
    "num_beams": 4,
    "max_new_tokens": 256,
    "max_input_length": 1024,
    "tasks": ["gsm8k", "AQuA", "mawps", "SVAMP"],
}
for key, value in expected_evaluation.items():
    if evaluation_config.get(key) != value:
        raise SystemExit(
            f"final evaluation config mismatch {key}={evaluation_config.get(key)!r}, "
            f"expected={value!r}: {evaluation_config_path}"
        )
if evaluation_config.get("config_fingerprint") != status.get("config_fingerprint"):
    raise SystemExit(f"final evaluation/config fingerprint mismatch: {evaluation_config_path}")
if evaluation_config.get("requested_model_revision") != status.get("model_revision"):
    raise SystemExit(f"final evaluation requested revision mismatch: {evaluation_config_path}")
if evaluation_config.get("resolved_model_revision") != status.get("resolved_model_revision"):
    raise SystemExit(f"final evaluation resolved revision mismatch: {evaluation_config_path}")
with summary_path.open("r", encoding="utf-8", newline="") as handle:
    rows = list(csv.DictReader(handle))
expected_columns = ["gsm8k", "AQuA", "mawps", "SVAMP", "Average"]
if len(rows) != 1 or list(rows[0]) != expected_columns:
    raise SystemExit(f"final summary must contain exactly {expected_columns}: {summary_path}")
values = []
for key in expected_columns:
    value = float(rows[0][key])
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise SystemExit(f"invalid final accuracy {key}={value}: {summary_path}")
    values.append(value)
if not math.isclose(values[-1], math.fsum(values[:-1]) / 4.0, rel_tol=0.0, abs_tol=1e-12):
    raise SystemExit(f"final macro accuracy is inconsistent: {summary_path}")

def canonical_sha256(value):
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()

def normalize_prompt(record):
    def normalize_field(value):
        return " ".join(
            unicodedata.normalize("NFKC", str(value)).casefold().split()
        )
    return json.dumps(
        {
            "instruction": normalize_field(record.get("instruction", "")),
            "input": normalize_field(record.get("input", "")),
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )

def record_identity(record):
    return {
        "instruction": record.get("instruction"),
        "input": record.get("input"),
        "answer": record.get("answer"),
    }

asset_manifest = json.loads(asset_manifest_path.read_text(encoding="utf-8"))
if asset_manifest.get("schema_version") != 3:
    raise SystemExit(f"unsupported final evaluation asset manifest: {asset_manifest_path}")
normalization = (
    (asset_manifest.get("decontamination") or {}).get("normalization") or {}
)
if normalization.get("id") != "instruction_input_nfkc_casefold_whitespace_collapse_v1":
    raise SystemExit(f"unexpected decontamination normalization: {asset_manifest_path}")
assets_by_task = {
    item.get("task"): item for item in asset_manifest.get("assets", [])
    if isinstance(item, dict)
}
if set(assets_by_task) != set(task_paths):
    raise SystemExit(f"final evaluation asset manifest task mismatch: {asset_manifest_path}")
summary_by_task = dict(zip(expected_columns[:-1], values[:-1]))
expected_task_rows = {"gsm8k": 1319, "AQuA": 254, "mawps": 238, "SVAMP": 1000}
expected_clean_rows = {"gsm8k": 1319, "AQuA": 254, "mawps": 185, "SVAMP": 1000}
expected_overlap_rows = {"gsm8k": 0, "AQuA": 0, "mawps": 53, "SVAMP": 0}
raw_accuracy = {}
clean_accuracy = {}
clean_correct = {}
clean_records = {}
for task, path in task_paths.items():
    records = json.loads(path.read_text(encoding="utf-8"))
    asset = assets_by_task[task]
    if int(asset.get("rows", -1)) != expected_task_rows[task]:
        raise SystemExit(f"locked task count changed for task={task}: {asset_manifest_path}")
    if not isinstance(records, list) or len(records) != expected_task_rows[task]:
        raise SystemExit(f"task output count does not match locked asset: {path}")
    normalized_prompts = [normalize_prompt(record) for record in records]
    if canonical_sha256(normalized_prompts) != asset.get(
        "normalized_prompt_sequence_sha256"
    ):
        raise SystemExit(f"task output order/content does not match locked asset: {path}")
    if canonical_sha256([record_identity(record) for record in records]) != asset.get(
        "record_identity_sequence_sha256"
    ):
        raise SystemExit(f"task output row identity does not match locked asset: {path}")
    flags = [record.get("flag") for record in records]
    if any(not isinstance(flag, bool) for flag in flags):
        raise SystemExit(f"task output contains a non-boolean correctness flag: {path}")
    for index, (record, flag) in enumerate(zip(records, flags)):
        if "pred" not in record or not isinstance(record.get("output_pred"), str):
            raise SystemExit(f"task output lacks stored prediction fields: {path}[{index}]")
        if task == "AQuA":
            label = str(record.get("answer")).strip().upper() if record.get("answer") is not None else ""
            recomputed_flag = label == str(record.get("pred", "")).strip().upper()
        else:
            try:
                label = float(record.get("answer"))
            except (TypeError, ValueError):
                label = float("inf")
            try:
                prediction = float(record.get("pred"))
            except (TypeError, ValueError):
                prediction = float("inf")
            recomputed_flag = abs(label - prediction) <= 0.001
        if flag is not recomputed_flag:
            raise SystemExit(f"task output flag disagrees with paper evaluator: {path}[{index}]")
    raw_accuracy[task] = math.fsum(int(flag) for flag in flags) / len(flags)
    if not math.isclose(
        raw_accuracy[task], summary_by_task[task], rel_tol=0.0, abs_tol=1e-12
    ):
        raise SystemExit(f"task output accuracy does not match summary: {path}")
    overlap = asset.get("prompt_overlap") or {}
    clean_indices = overlap.get("clean_test_indices")
    if not isinstance(clean_indices, list) or any(
        isinstance(index, bool) or not isinstance(index, int) for index in clean_indices
    ):
        raise SystemExit(f"invalid locked clean indices for task={task}")
    if clean_indices != sorted(set(clean_indices)) or any(
        not 0 <= index < len(records) for index in clean_indices
    ):
        raise SystemExit(f"unsafe locked clean indices for task={task}")
    if canonical_sha256(clean_indices) != overlap.get("clean_test_indices_sha256"):
        raise SystemExit(f"locked clean-index hash mismatch for task={task}")
    overlap_indices = overlap.get("overlap_test_indices")
    if (
        not isinstance(overlap_indices, list)
        or canonical_sha256(overlap_indices)
        != overlap.get("overlap_test_indices_sha256")
    ):
        raise SystemExit(f"locked overlap-index hash mismatch for task={task}")
    overlap_mapping = overlap.get("overlap_training_index_mapping")
    if not isinstance(overlap_mapping, list) or canonical_sha256(
        overlap_mapping
    ) != overlap.get("overlap_training_index_mapping_sha256"):
        raise SystemExit(f"locked overlap mapping/hash mismatch for task={task}")
    if [item.get("test_index") for item in overlap_mapping] != overlap.get(
        "overlap_test_indices"
    ):
        raise SystemExit(f"locked overlap mapping indices mismatch for task={task}")
    if len(clean_indices) != int(overlap.get("clean_count", -1)):
        raise SystemExit(f"locked clean-index count mismatch for task={task}")
    if (
        len(clean_indices) != expected_clean_rows[task]
        or int(overlap.get("overlap_count", -1)) != expected_overlap_rows[task]
    ):
        raise SystemExit(f"locked conservative decontamination count changed for task={task}")
    clean_records[task] = len(clean_indices)
    clean_correct[task] = sum(int(flags[index]) for index in clean_indices)
    clean_accuracy[task] = clean_correct[task] / clean_records[task]

with details_path.open("r", encoding="utf-8", newline="") as handle:
    detail_rows = list(csv.DictReader(handle))
if len(detail_rows) != 4 or {row.get("dataset") for row in detail_rows} != set(task_paths):
    raise SystemExit(f"final details must contain exactly one row per task: {details_path}")
for row in detail_rows:
    task = row["dataset"]
    try:
        detail_n = int(row["n"])
        detail_accuracy = float(row["accuracy"])
    except (KeyError, TypeError, ValueError) as exc:
        raise SystemExit(f"invalid final details row for task={task}: {details_path}") from exc
    if detail_n != expected_task_rows[task]:
        raise SystemExit(f"final details count mismatch for task={task}: {details_path}")
    if not math.isclose(
        detail_accuracy, raw_accuracy[task], rel_tol=0.0, abs_tol=1e-12
    ):
        raise SystemExit(f"final details accuracy mismatch for task={task}: {details_path}")
    if Path(row.get("json", "")).resolve() != task_paths[task].resolve():
        raise SystemExit(f"final details artifact path mismatch for task={task}: {details_path}")

evaluation_assets = evaluation_config.get("test_assets")
if not isinstance(evaluation_assets, dict) or set(evaluation_assets) != set(task_paths):
    raise SystemExit(f"final evaluation test-asset identity is missing: {evaluation_config_path}")
for task, asset in assets_by_task.items():
    cached_asset = evaluation_assets[task]
    if cached_asset != {"rows": expected_task_rows[task], "sha256": asset["sha256"]}:
        raise SystemExit(f"final evaluation test-asset mismatch for task={task}: {evaluation_config_path}")
paper_macro = values[-1]
decontaminated_four_task_macro = math.fsum((
    raw_accuracy["gsm8k"],
    raw_accuracy["AQuA"],
    clean_accuracy["mawps"],
    raw_accuracy["SVAMP"],
)) / 4.0
clean_three_task_macro = math.fsum((
    raw_accuracy["gsm8k"],
    raw_accuracy["AQuA"],
    raw_accuracy["SVAMP"],
)) / 3.0
payload = {
    "schema_version": 2,
    "primary_metric": "clean_three_task_macro_accuracy",
    "clean_three_task_macro_accuracy": clean_three_task_macro,
    "sensitivity_metric": "decontaminated_four_task_macro_accuracy",
    "decontaminated_four_task_macro_accuracy": decontaminated_four_task_macro,
    "clean_mawps_accuracy": clean_accuracy["mawps"],
    "clean_mawps_correct": clean_correct["mawps"],
    "clean_mawps_records": clean_records["mawps"],
    "paper_raw_four_task_macro_accuracy": paper_macro,
    "raw_task_accuracy": raw_accuracy,
    "decontaminated_task_accuracy": clean_accuracy,
    "decontaminated_task_records": clean_records,
    "asset_manifest_path": str(asset_manifest_path),
    "asset_manifest_sha256": hashlib.sha256(asset_manifest_path.read_bytes()).hexdigest(),
    "selection_sha256": asset_manifest["selection_sha256"],
    "normalization_id": normalization["id"],
    "contamination_disclosure": {
        "mawps_raw_records": len(json.loads(task_paths["mawps"].read_text(encoding="utf-8"))),
        "mawps_prompt_overlap_records": int(
            assets_by_task["mawps"]["prompt_overlap"]["overlap_count"]
        ),
        "mawps_clean_records": clean_records["mawps"],
        "mawps_overlap_answer_string_exact_records": int(
            assets_by_task["mawps"]["prompt_overlap"][
                "answer_string_exact_overlap_count"
            ]
        ),
        "mawps_overlap_answer_evaluator_equivalent_records": int(
            assets_by_task["mawps"]["prompt_overlap"][
                "answer_evaluator_equivalent_overlap_count"
            ]
        ),
    },
}
encoded = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
decontaminated_path.parent.mkdir(parents=True, exist_ok=True)
if decontaminated_path.exists():
    if decontaminated_path.read_bytes() != encoded:
        raise SystemExit(
            f"refusing inconsistent decontaminated metrics: {decontaminated_path}"
        )
else:
    fd, name = tempfile.mkstemp(
        prefix=f".{decontaminated_path.name}.",
        suffix=".tmp",
        dir=decontaminated_path.parent,
    )
    temporary = Path(name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, decontaminated_path)
    finally:
        temporary.unlink(missing_ok=True)
os.chmod(decontaminated_path, 0o600)
print(
    f"validated locked-final method={expected_method} seed={expected_seed} "
    f"primary_clean_three_task_macro={clean_three_task_macro:.8f} "
    f"decontaminated_four_task_macro={decontaminated_four_task_macro:.8f} "
    f"paper_macro={paper_macro:.8f} "
    f"clean_mawps={clean_accuracy['mawps']:.8f}"
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
    if [[ "${phase}" == "final" ]]; then
      local screen_run_count=$((27 + stage2_count))
      local total_dp_run_count=$((screen_run_count + final_count))
      echo "selection_sha256=$(sha256sum "${LOCKED_SELECTION}" | awk '{print $1}')"
      echo "primary_metric=clean_three_task_macro_accuracy"
      echo "sensitivity_metric=decontaminated_four_task_macro_accuracy"
      echo "paper_comparability_metric=raw_four_task_macro_accuracy"
      echo "campaign_worker_model_evaluation_begins_after_stage2_selection_lock=true"
      echo "test_assets_and_prior_test_results_were_accessed_before_campaign_design=true"
      echo "not_an_untouched_test_set=true"
      echo "stage2_run_count=${stage2_count}"
      echo "screen_run_count=${screen_run_count}"
      echo "final_run_count=${final_count}"
      echo "schedule_source_run_count=0"
      echo "selection_privacy_class=DP_DERIVED_VIA_${screen_run_count}_RUN_PUBLIC_VALIDATION_SELECTION"
      echo "single_run_privacy_conditioned_on_unreleased_locked_selection=epsilon6_delta1e-5"
      echo "conditional_privacy_warning=does_not_apply_if_selection_or_other_screen_outputs_are_released"
      echo "dp_only_full_campaign_basic_composition=epsilon$((6 * total_dp_run_count))_delta${total_dp_run_count}e-5_release_count${total_dp_run_count}"
    elif [[ "${phase}" == "schedule-source" ]]; then
      local screen_run_count=$((18 + stage2_count))
      echo "selection_sha256=$(sha256sum "${LOCKED_SELECTION}" | awk '{print $1}')"
      echo "data_scope=full_math10k_9919_rows"
      echo "model_evaluation=false"
      echo "test_sets_accessed=false"
      echo "schedule_source_role=full_data_matched_schedule_transfer"
      echo "screen_run_count=${screen_run_count}"
      echo "schedule_privacy_class=DP_DERIVED_VIA_${screen_run_count}_RUN_SELECTION_PLUS_3_FULL_DATA_SOURCES"
    else
      echo "selection_metric=public_math10k_numeric_exact_match_accuracy"
      echo "validation_curve_steps=0,50,100,150,200,250,300"
      echo "test_sets_accessed=false"
    fi
    echo "raw_telemetry=NON_PRIVATE"
    echo "development_artifact_bundle_privacy=NON_PRIVATE_due_to_research_raw_exact_training_statistics"
  } >"${arm_root}/arm-manifest.txt"
  if [[ "${phase}" == "final" && "${role}" == "matched-fixed-noise-energy" ]]; then
    {
      echo "schedule_construction_source=three_selected_full_data_DP_safe_SlaClip_trajectories"
      echo "control_classification=oracle_mechanistic_control"
      echo "matched_fixed_definition=RMS_C_matches_mean_squared_absolute_DP_noise_scale"
      echo "matched_fixed_is_schedule_release=false"
      echo "matched_fixed_conditional_run_privacy=epsilon6_delta1e-5_conditioned_on_unreleased_locked_selection_and_schedule"
    } >>"${arm_root}/arm-manifest.txt"
  elif [[ "${phase}" == "final" && "${role}" == "replay" ]]; then
    {
      echo "schedule_construction_source=three_selected_full_data_DP_safe_SlaClip_trajectories"
      echo "control_classification=oracle_mechanistic_control"
      echo "replay_uses_shared_locked_full_data_schedule=true"
      echo "replay_interpretation=schedule_transfer_across_fresh_seeds_not_exact_same_seed_replay"
      echo "replay_conditional_run_privacy=epsilon6_delta1e-5_conditioned_on_unreleased_locked_selection_and_schedule"
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
    --telemetry_mode research_raw
    --allow_non_private_telemetry
    --raw_hist_bins 128
    --raw_hist_max 30.0
    --slaclip_num_slots 15
    --require_cuda
    --resume
    --checkpoint_every 25
    --run_train true
    --output_dir "${adapter_dir}"
    --result_dir "${result_dir}"
  )
  if [[ "${phase}" == "final" ]]; then
    train_args+=(
      --val_set_size 0
      --validation_eval_interval 0
      --protocol_stage final
      --run_eval true
      --eval_batch_size 8
      --num_beams 4
      --max_new_tokens 256
      --max_input_length 1024
    )
  elif [[ "${phase}" == "schedule-source" ]]; then
    train_args+=(
      --val_set_size 0
      --validation_eval_interval 0
      --protocol_stage final
      --run_eval false
    )
  else
    train_args+=(
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
  fi
  case "${method}" in
    slaclip)
      train_args+=(
        --slaclip_target_non_small_clip_fraction "${rho}"
        --slaclip_eta "${eta}"
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
  if [[ "${phase}" == "final" ]]; then
    validate_final_run "${arm_root}" "${method}" "${seed}" || validate_rc=$?
  elif [[ "${phase}" == "schedule-source" ]]; then
    validate_schedule_source_run "${arm_root}" "${seed}" || validate_rc=$?
  else
    validate_selection_run "${arm_root}" "${method}" "${seed}" || validate_rc=$?
  fi
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
    --batch_size 64
    --micro_batch_size 4
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

create_fixed_scan_manifest() {
  "${PYTHON_BIN}" - "${FIXED_MANIFEST}" "${MODEL_ID}" "${MODEL_REVISION}" "${EXPECTED_REPO_SHA}" <<'PY'
import hashlib
import json
import math
import os
import sys
import tempfile
from pathlib import Path

target = Path(sys.argv[1])
fixed_values = (0.1, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 15.0)

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
if len(candidates) != 8 or len({item["id"] for item in candidates}) != 8:
    raise SystemExit("fixed scan must contain exactly eight unique candidates")
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
        "name": "math10k_4b_fixed_derived_targets_numeric_selection_fresh_final_v3",
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
    "fixed_candidates": candidates,
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
  atomic_sha256_sidecar "${FIXED_MANIFEST}"
}

derive_locked_slaclip_grid() {
  "${PYTHON_BIN}" scripts/derive_slaclip_target_grid.py \
    --campaign-root "${CAMPAIGN_ROOT}" \
    --fixed-manifest "${FIXED_MANIFEST}" \
    --output "${TARGET_GRID_FILE}" \
    --registry-out "${REGISTRY_FILE}" \
    >"${SELECTION_ROOT}/target-grid-${JOB_ID}.out" \
    2>"${SELECTION_ROOT}/target-grid-${JOB_ID}.err"
  atomic_sha256_sidecar "${TARGET_GRID_FILE}"
  atomic_sha256_sidecar "${REGISTRY_FILE}"
}

create_campaign_manifest() {
  local stage2_runs="$1"
  local final_runs="$2"
  local manifest="${CAMPAIGN_ROOT}/campaign_manifest.json"
  "${PYTHON_BIN}" - \
    "${manifest}" "${REGISTRY_FILE}" "${EXPECTED_REPO_SHA}" "${JOB_ID}" \
    "${MODEL_ID}" "${MODEL_REVISION}" "${ENV_PREFIX}" \
    "${ENVIRONMENT_FREEZE}" "${stage2_runs}" "${final_runs}" \
    "${FIXED_MANIFEST}" "${TARGET_GRID_FILE}" <<'PY'
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

target = Path(sys.argv[1])
registry = Path(sys.argv[2])
stage1_run_count = 18
stage2_run_count = int(sys.argv[9])
final_run_count = int(sys.argv[10])
if stage2_run_count not in (10, 12) or final_run_count not in (20, 25, 30, 35):
    raise SystemExit("invalid dynamic campaign run counts")
screen_run_count = stage1_run_count + stage2_run_count
schedule_source_run_count = 3
pre_final_dp_run_count = screen_run_count + schedule_source_run_count
total_dp_run_count = pre_final_dp_run_count + final_run_count

def basic_composition(release_count):
    return {
        "epsilon": 6.0 * release_count,
        "delta": 1e-5 * release_count,
        "release_count": release_count,
    }

payload = {
    "schema_version": 1,
    "protocol": "math10k_4b_fixed_derived_target_selection_fresh_final_v3",
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
    "fixed_scan_manifest_sha256": hashlib.sha256(Path(sys.argv[11]).read_bytes()).hexdigest(),
    "slaclip_target_grid_sha256": hashlib.sha256(Path(sys.argv[12]).read_bytes()).hexdigest(),
    "resources": {
        "allocation": "one Slurm job",
        "gpu": "2xa100",
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
        "target_calibration": (
            "NON_PRIVATE_data_dependent_fixed_scan_to_full_slaclip_grid;_"
            "not_an_end_to_end_DP_protocol"
        ),
        "stage1_run_count": stage1_run_count,
        "stage2_run_count": stage2_run_count,
        "screen_run_count": screen_run_count,
        "pre_final_dp_run_count": pre_final_dp_run_count,
        "final_run_count": final_run_count,
        "schedule_construction_source_run_count": schedule_source_run_count,
        "schedule_privacy_class": (
            f"DP_DERIVED_VIA_{screen_run_count}_RUN_SELECTION_PLUS_3_FULL_DATA_SOURCES"
        ),
        "replay_control_classification": (
            "full_data_matched_schedule_transfer_from_selected_DP_SlaClip_trajectories"
        ),
        "control_basic_composition_upper_bounds": {
            "one_run_conditioned_on_unreleased_locked_selection": {
                "epsilon": 6.0,
                "delta": 1e-5,
                "warning": (
                    "not a release bound if selection or other screen outputs are released"
                ),
            },
            "selection_and_full_data_schedule_sources": basic_composition(
                pre_final_dp_run_count
            ),
            "selection_schedule_plus_one_final_control": basic_composition(
                pre_final_dp_run_count + 1
            ),
            "selection_schedule_plus_five_replay": basic_composition(
                pre_final_dp_run_count + 5
            ),
            "selection_plus_five_replay_plus_five_matched_fixed": basic_composition(
                pre_final_dp_run_count + 10
            ),
            "all_screen_and_fresh_final_dp_outputs": basic_composition(
                total_dp_run_count
            ),
        },
        "development_artifact_bundle": (
            "NON_PRIVATE because research_raw telemetry contains exact "
            "training-example statistics"
        ),
    },
    "campaign_worker_model_evaluation_begins_after_stage2_selection_lock": True,
    "selection_algorithm_uses_test_outputs": False,
    "test_assets_and_prior_test_results_were_accessed_before_campaign_design": True,
    "not_an_untouched_test_set": True,
    "test_sets_accessed": True,
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
  local source_manifest="${REGISTRY_FILE}"
  if [[ "${mode}" == "fixed-scan" ]]; then
    source_manifest="${FIXED_MANIFEST}"
  fi
  local temporary create_rc=0
  temporary="$(mktemp "${output}.tmp.XXXXXX")"
  "${PYTHON_BIN}" - \
    "${mode}" "${source_manifest}" "${CAMPAIGN_ROOT}" "${STAGE1_SELECTION}" \
    "${LOCKED_SELECTION}" "${CONTROL_MANIFEST}" "${REPLAY_SCHEDULE}" "${MODEL_SLUG}" \
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
model_slug = sys.argv[8]
candidate_rows = (
    registry["fixed_candidates"] if mode == "fixed-scan" else registry["candidates"]
)
candidates = {item["id"]: item for item in candidate_rows}

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

if mode == "fixed-scan":
    for candidate in candidate_rows:
        emit(
            "fixed-scan",
            candidate,
            "fixed-grid-calibration",
            42,
            campaign_root / "screen" / "runs" / candidate["id"] / "seed-42",
        )
elif mode == "slaclip-screen":
    slaclip = [item for item in candidate_rows if item["family"] == "slaclip"]
    if len(slaclip) != 10:
        raise SystemExit("derived registry must contain exactly ten SlaClip candidates")
    for candidate in slaclip:
        emit(
            "slaclip-screen",
            candidate,
            "derived-target-grid",
            42,
            campaign_root / "screen" / "runs" / candidate["id"] / "seed-42",
        )
elif mode == "stage2":
    selection = json.loads(stage1_selection.read_text(encoding="utf-8"))
    selected_ids = [item["candidate_id"] for item in selection["top_slaclip"]]
    selected_ids.extend(item["candidate_id"] for item in selection["top_fixed"])
    canonical_ids = [
        candidate_id
        for candidate_id, candidate in candidates.items()
        if candidate["family"] == "fixed"
        and float(candidate["params"]["dp_max_grad_norm"]) == 1.0
    ]
    if len(canonical_ids) != 1:
        raise SystemExit("registry must contain exactly one canonical fixed C=1 candidate")
    selected_ids.append(canonical_ids[0])
    selected_ids = list(dict.fromkeys(selected_ids))
    if (
        len(selection["top_slaclip"]) != 3
        or len(selection["top_fixed"]) != 2
        or len(selected_ids) not in {5, 6}
    ):
        raise SystemExit("stage2 requires top3 SlaClip plus top2 fixed and canonical C=1")
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
elif mode == "schedule-source":
    locked = json.loads(locked_selection.read_text(encoding="utf-8"))
    selected_id = locked["selected_slaclip"]["candidate_id"]
    selected = candidates[selected_id]
    if selected["family"] != "slaclip":
        raise SystemExit("schedule-source plan requires a selected SlaClip candidate")
    for seed in (42, 43, 44):
        emit(
            "schedule-source",
            selected,
            "schedule-source-full-data",
            seed,
            campaign_root / "schedule-source" / "runs" / selected_id / f"seed-{seed}",
        )
elif mode == "final":
    locked = json.loads(locked_selection.read_text(encoding="utf-8"))
    controls = json.loads(control_manifest.read_text(encoding="utf-8"))
    selected_id = locked["selected_slaclip"]["candidate_id"]
    if controls["selected_slaclip_candidate_id"] != selected_id:
        raise SystemExit("control manifest does not match locked SlaClip selection")
    selected = candidates[selected_id]
    best_fixed = candidates[locked["best_fixed"]["candidate_id"]]
    canonical_ids = [
        candidate_id
        for candidate_id, candidate in candidates.items()
        if candidate["family"] == "fixed"
        and float(candidate["params"]["dp_max_grad_norm"]) == 1.0
    ]
    if len(canonical_ids) != 1:
        raise SystemExit("registry must contain exactly one canonical fixed C=1 candidate")
    canonical = candidates[canonical_ids[0]]
    canonical_full = {
        "id": "canonical-full-slaclip-paper-default",
        "family": "slaclip-control",
        "method": "slaclip",
        "params": {
            "dp_max_grad_norm": 1.0,
            "slaclip_target_non_small_clip_fraction": 0.5,
            "slaclip_eta": 0.2,
            "slaclip_num_slots": 15,
            "slaclip_c_min": 0.1,
            "slaclip_c_max": 15.0,
        },
    }
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
    final_candidates = [
        (selected, "selected-slaclip", "NA"),
        (best_fixed, "best-fixed", "NA"),
        (replay, "replay", replay_schedule),
        (matched, "matched-fixed-noise-energy", "NA"),
    ]
    if canonical["id"] != best_fixed["id"]:
        final_candidates.append((canonical, "canonical-fixed-c1", "NA"))
    selected_c0 = float(selected["params"]["dp_max_grad_norm"])
    best_fixed_c = float(best_fixed["params"]["dp_max_grad_norm"])
    if abs(selected_c0 - best_fixed_c) <= 1e-12:
        initial_reference_role = "best-fixed"
    elif abs(selected_c0 - 1.0) <= 1e-12:
        initial_reference_role = "canonical-fixed-c1"
    else:
        initial_reference_role = "initial-C-matched-fixed"
        initial_matched = {
            "id": "initial-c-matched-fixed",
            "family": "fixed-control",
            "method": "baseline",
            "params": {"dp_max_grad_norm": selected_c0},
        }
        final_candidates.append((initial_matched, initial_reference_role, "NA"))
    if controls.get("initial_c_matched_fixed_reference_role") != initial_reference_role:
        raise SystemExit("control manifest initial-C reference role is inconsistent")
    if abs(float(controls.get("initial_c_matched_fixed_C")) - selected_c0) > 1e-12:
        raise SystemExit("control manifest initial-C reference threshold is inconsistent")
    selected_is_canonical_full = (
        abs(selected_c0 - 1.0) <= 1e-12
        and abs(float(selected["params"]["slaclip_target_non_small_clip_fraction"]) - 0.5) <= 1e-12
        and abs(float(selected["params"]["slaclip_eta"]) - 0.2) <= 1e-12
    )
    if selected_is_canonical_full:
        canonical_full_reference_role = "selected-slaclip"
    else:
        canonical_full_reference_role = "canonical-full-slaclip"
        final_candidates.append(
            (canonical_full, canonical_full_reference_role, "NA")
        )
    if controls.get("canonical_full_slaclip_reference_role") != canonical_full_reference_role:
        raise SystemExit("control manifest canonical full-SlaClip reference is inconsistent")
    for seed in (17, 29, 71, 101, 137):
        for candidate, role, schedule in final_candidates:
            emit(
                "final",
                candidate,
                role,
                seed,
                campaign_root / "final" / model_slug / f"seed-{seed}" / role,
                schedule,
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
    --top-slaclip 3
    --top-fixed 2
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

lock_final_evaluation_assets() {
  local temporary
  temporary="$(mktemp "${FINAL_EVAL_ASSET_MANIFEST}.tmp.XXXXXX")"
  "${PYTHON_BIN}" - "${REPO_ROOT}" "${LOCKED_SELECTION}" >"${temporary}" <<'PY'
import hashlib
import json
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path

repo_root = Path(sys.argv[1]).resolve(strict=True)
selection_path = Path(sys.argv[2]).resolve(strict=True)
datasets = {
    "gsm8k": "LLM-Adapters/dataset/gsm8k/test.json",
    "AQuA": "LLM-Adapters/dataset/AQuA/test.json",
    "mawps": "LLM-Adapters/dataset/mawps/test.json",
    "SVAMP": "LLM-Adapters/dataset/SVAMP/test.json",
}
normalization = {
    "id": "instruction_input_nfkc_casefold_whitespace_collapse_v1",
    "fields": ["instruction", "input"],
    "unicode_normalization": "NFKC",
    "case_normalization": "casefold",
    "whitespace_rule": "split then join with one ASCII space",
    "field_composition": "separate normalized fields in sorted-key canonical JSON",
}

def normalize_prompt(record):
    def normalize_field(value):
        return " ".join(
            unicodedata.normalize("NFKC", str(value)).casefold().split()
        )
    return json.dumps(
        {
            "instruction": normalize_field(record.get("instruction", "")),
            "input": normalize_field(record.get("input", "")),
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )

def record_identity(record):
    return {
        "instruction": record.get("instruction"),
        "input": record.get("input"),
        "answer": record.get("answer"),
    }

def canonical_sha256(value):
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()

training_relative = "LLM-Adapters/ft-training_set/math_10k.json"
training_path = (repo_root / training_relative).resolve(strict=True)
training_encoded = training_path.read_bytes()
training_records = json.loads(training_encoded)
if not isinstance(training_records, list) or len(training_records) != 9919:
    raise SystemExit("decontamination audit requires the locked 9,919-row Math-10K training set")
training_by_prompt = defaultdict(list)
for index, record in enumerate(training_records):
    if not isinstance(record, dict):
        raise SystemExit(f"invalid Math-10K training record at index={index}")
    normalized = normalize_prompt(record)
    if not normalized:
        raise SystemExit(f"empty normalized Math-10K training prompt at index={index}")
    training_by_prompt[normalized].append((index, record))

expected_audit = {
    "gsm8k": {"rows": 1319, "overlap": 0, "clean": 1319, "string_exact": 0, "evaluator_equivalent": 0},
    "AQuA": {"rows": 254, "overlap": 0, "clean": 254, "string_exact": 0, "evaluator_equivalent": 0},
    "mawps": {"rows": 238, "overlap": 53, "clean": 185, "string_exact": 38, "evaluator_equivalent": 53},
    "SVAMP": {"rows": 1000, "overlap": 0, "clean": 1000, "string_exact": 0, "evaluator_equivalent": 0},
}
assets = []
for task, relative in datasets.items():
    path = (repo_root / relative).resolve(strict=True)
    try:
        path.relative_to(repo_root)
    except ValueError as exc:
        raise SystemExit(f"final evaluation asset escapes repository: {path}") from exc
    encoded = path.read_bytes()
    try:
        records = json.loads(encoded)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"final evaluation asset is invalid JSON: {path}: {exc}") from exc
    if not isinstance(records, list) or not records:
        raise SystemExit(f"final evaluation asset must be a non-empty list: {path}")
    overlap_indices = []
    normalized_test_prompts = []
    answer_string_exact_overlap_count = 0
    answer_evaluator_equivalent_overlap_count = 0
    overlap_mapping = []
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise SystemExit(f"invalid final evaluation record {task}[{index}]")
        normalized = normalize_prompt(record)
        normalized_test_prompts.append(normalized)
        matches = training_by_prompt.get(normalized, ())
        if matches:
            overlap_indices.append(index)
            overlap_mapping.append({
                "test_index": index,
                "training_indices": [training_index for training_index, _ in matches],
            })
            reference = str(record.get("answer", "")).strip()
            answer_string_exact_overlap_count += int(
                any(
                    str(training.get("answer", "")).strip() == reference
                    for _, training in matches
                )
            )
            if task == "AQuA":
                equivalent = any(
                    str(training.get("answer", "")).strip().upper()
                    == reference.upper()
                    for _, training in matches
                )
            else:
                try:
                    reference_number = float(reference)
                except (TypeError, ValueError):
                    equivalent = False
                else:
                    equivalent = False
                    for _, training in matches:
                        try:
                            training_number = float(training.get("answer"))
                        except (TypeError, ValueError):
                            continue
                        if abs(training_number - reference_number) <= 0.001:
                            equivalent = True
                            break
            answer_evaluator_equivalent_overlap_count += int(equivalent)
    overlap_set = set(overlap_indices)
    clean_indices = [index for index in range(len(records)) if index not in overlap_set]
    expected = expected_audit[task]
    observed = {
        "rows": len(records),
        "overlap": len(overlap_indices),
        "clean": len(clean_indices),
        "string_exact": answer_string_exact_overlap_count,
        "evaluator_equivalent": answer_evaluator_equivalent_overlap_count,
    }
    if observed != expected:
        raise SystemExit(
            f"locked decontamination audit changed for {task}: "
            f"observed={observed}, expected={expected}"
        )
    assets.append({
        "task": task,
        "path": relative,
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "bytes": len(encoded),
        "rows": len(records),
        "normalized_prompt_sequence_sha256": canonical_sha256(
            normalized_test_prompts
        ),
        "record_identity_sequence_sha256": canonical_sha256([
            record_identity(record) for record in records
        ]),
        "prompt_overlap": {
            "overlap_count": len(overlap_indices),
            "clean_count": len(clean_indices),
            "answer_string_exact_overlap_count": answer_string_exact_overlap_count,
            "answer_evaluator_equivalent_overlap_count": (
                answer_evaluator_equivalent_overlap_count
            ),
            "overlap_test_indices": overlap_indices,
            "overlap_test_indices_sha256": canonical_sha256(overlap_indices),
            "clean_test_indices": clean_indices,
            "clean_test_indices_sha256": canonical_sha256(clean_indices),
            "overlap_training_index_mapping": overlap_mapping,
            "overlap_training_index_mapping_sha256": canonical_sha256(
                overlap_mapping
            ),
        },
    })
payload = {
    "schema_version": 3,
    "selection_sha256": hashlib.sha256(selection_path.read_bytes()).hexdigest(),
    "access_policy": "campaign_worker_model_evaluation_begins_after_stage2_selection_lock",
    "campaign_worker_model_evaluation_begins_after_stage2_selection_lock": True,
    "selection_algorithm_uses_test_outputs": False,
    "test_assets_and_prior_test_results_were_accessed_before_campaign_design": True,
    "not_an_untouched_test_set": True,
    "decontamination": {
        "normalization": normalization,
        "training_asset": {
            "path": training_relative,
            "sha256": hashlib.sha256(training_encoded).hexdigest(),
            "bytes": len(training_encoded),
            "rows": len(training_records),
            "unique_normalized_prompts": len(training_by_prompt),
            "normalized_prompt_set_sha256": canonical_sha256(
                sorted(training_by_prompt)
            ),
        },
        "primary_metric": "clean_three_task_macro_accuracy",
        "sensitivity_metric": "decontaminated_four_task_macro_accuracy",
        "paper_metric_retained": "raw_four_task_macro_accuracy",
    },
    "assets": assets,
}
print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
PY
  install_immutable_file "${temporary}" "${FINAL_EVAL_ASSET_MANIFEST}"
  atomic_sha256_sidecar "${FINAL_EVAL_ASSET_MANIFEST}"
}

build_replay_schedule_and_controls() {
  "${PYTHON_BIN}" - \
    "${CAMPAIGN_ROOT}" "${LOCKED_SELECTION}" "${REPLAY_SCHEDULE}" "${CONTROL_MANIFEST}" \
    "${stage2_count}" <<'PY'
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
stage2_run_count = int(sys.argv[5])
if stage2_run_count not in (10, 12):
    raise SystemExit(f"unexpected stage2 run count: {stage2_run_count}")
stage1_run_count = 18
screen_run_count = stage1_run_count + stage2_run_count
selection = json.loads(selection_path.read_text(encoding="utf-8"))
selected = selection["selected_slaclip"]
candidate_id = selected["candidate_id"]
best_fixed_clip = float(selection["best_fixed"]["params"]["dp_max_grad_norm"])
selected_c0 = float(selected["params"]["dp_max_grad_norm"])
selected_rho = float(selected["params"]["slaclip_target_non_small_clip_fraction"])
selected_eta = float(selected["params"]["slaclip_eta"])
if math.isclose(selected_c0, best_fixed_clip, rel_tol=0.0, abs_tol=1e-12):
    initial_c_reference_role = "best-fixed"
elif math.isclose(selected_c0, 1.0, rel_tol=0.0, abs_tol=1e-12):
    initial_c_reference_role = "canonical-fixed-c1"
else:
    initial_c_reference_role = "initial-C-matched-fixed"
canonical_full_selected = (
    math.isclose(selected_c0, 1.0, rel_tol=0.0, abs_tol=1e-12)
    and math.isclose(selected_rho, 0.5, rel_tol=0.0, abs_tol=1e-12)
    and math.isclose(selected_eta, 0.2, rel_tol=0.0, abs_tol=1e-12)
)
canonical_full_reference_role = (
    "selected-slaclip" if canonical_full_selected else "canonical-full-slaclip"
)
final_role_count = 4
final_role_count += int(not math.isclose(best_fixed_clip, 1.0, rel_tol=0.0, abs_tol=1e-12))
final_role_count += int(initial_c_reference_role == "initial-C-matched-fixed")
final_role_count += int(not canonical_full_selected)
final_run_count = 5 * final_role_count
if final_run_count not in (20, 25, 30, 35):
    raise SystemExit(f"unexpected derived final run count: {final_run_count}")
pre_final_dp_run_count = screen_run_count + 3
total_dp_run_count = pre_final_dp_run_count + final_run_count
seeds = (42, 43, 44)

def basic_composition(release_count):
    return {
        "epsilon": 6.0 * release_count,
        "delta": 1e-5 * release_count,
        "release_count": release_count,
    }

def source_root(seed):
    return campaign_root / "schedule-source" / "runs" / candidate_id / f"seed-{seed}"

trajectories = []
sources = []
for seed in seeds:
    root = source_root(seed)
    path = root / "adapter" / "train_log.jsonl"
    status_path = root / "adapter" / "run_status.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    source_config = status.get("config") or {}
    if (
        status.get("state") != "completed"
        or source_config.get("method") != "slaclip"
        or source_config.get("protocol_stage") != "final"
        or int(source_config.get("val_set_size", -1)) != 0
        or source_config.get("run_eval") is not False
    ):
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
schedule_privacy_class = (
    f"DP_DERIVED_VIA_{screen_run_count}_RUN_SELECTION_PLUS_3_FULL_DATA_SOURCES"
)
privacy_accounting = {
    "adjacency": "add_remove",
    "basic_composition_upper_bound": True,
    "stage1_run_count": stage1_run_count,
    "stage2_run_count": stage2_run_count,
    "screen_run_count": screen_run_count,
    "pre_final_dp_run_count": pre_final_dp_run_count,
    "final_run_count": final_run_count,
    "schedule_construction_source_run_count": len(seeds),
    "selection_and_full_data_schedule_sources": basic_composition(pre_final_dp_run_count),
    "selection_schedule_plus_one_final_control": basic_composition(pre_final_dp_run_count + 1),
    "selection_schedule_plus_five_replay": basic_composition(pre_final_dp_run_count + 5),
    "selection_plus_five_replay_plus_five_matched_fixed": basic_composition(
        pre_final_dp_run_count + 10
    ),
    "all_screen_and_fresh_final_dp_outputs": basic_composition(total_dp_run_count),
    "single_run_conditioned_on_unreleased_locked_selection": {
        "epsilon": 6.0,
        "delta": 1e-5,
        "warning": (
            "not a release bound if selection or other screen outputs are released"
        ),
    },
    "development_artifact_bundle": (
        "NON_PRIVATE because research_raw telemetry exposes exact "
        "training-example statistics"
    ),
}
schedule = {
    "schema_version": 2,
    "num_steps": 300,
    "construction": "per_step_arithmetic_mean_of_three_locked_full_data_top1_slaclip_dp_safe_trajectories",
    "schedule_privacy_class": schedule_privacy_class,
    "control_classification": (
        "full-data matched schedule-transfer control constructed from three selected DP SlaClip trajectories; "
        "winner identity depends on all screen runs and fresh-seed replay is not exact same-seed replay"
    ),
    "selected_slaclip_candidate_id": candidate_id,
    "selection_sha256": hashlib.sha256(selection_path.read_bytes()).hexdigest(),
    "source_logs": sources,
    "clip_thresholds": clip_thresholds,
    "statistics": statistics_rows,
    "privacy_accounting": privacy_accounting,
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
    "schema_version": 2,
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
    "initial_c_matched_fixed_C": selected_c0,
    "initial_c_matched_fixed_reference_role": initial_c_reference_role,
    "canonical_full_slaclip_reference_role": canonical_full_reference_role,
    "canonical_full_slaclip_definition": {
        "dp_max_grad_norm": 1.0,
        "slaclip_target_non_small_clip_fraction": 0.5,
        "slaclip_eta": 0.2,
        "slaclip_num_slots": 15,
        "slaclip_c_min": 0.1,
        "slaclip_c_max": 15.0,
        "interpretation": "paper-default-controller anchor, not a verbatim task reproduction",
    },
    "schedule_privacy_class": schedule_privacy_class,
    "control_classification": (
        "full_data_matched_schedule_transfer_from_selected_DP_SlaClip_trajectories"
    ),
    "privacy_accounting": privacy_accounting,
    "schedule_source_seeds": list(seeds),
    "final_control_seeds": [17, 29, 71, 101, 137],
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
    "${CAMPAIGN_ROOT}" "${EXPECTED_REPO_SHA}" "${FIXED_MANIFEST}" "${TARGET_GRID_FILE}" \
    "${REGISTRY_FILE}" "${STAGE1_SELECTION}" \
    "${LOCKED_SELECTION}" "${REPLAY_SCHEDULE}" "${CONTROL_MANIFEST}" \
    "${FINAL_EVAL_ASSET_MANIFEST}" "${output}" <<'PY'
import csv
import hashlib
import io
import json
import math
import os
import statistics
import sys
import tempfile
from pathlib import Path

root = Path(sys.argv[1]).resolve()
code_sha = sys.argv[2]
locked_files = [Path(item).resolve() for item in sys.argv[3:11]]
output = Path(sys.argv[11]).resolve()
plan_files = [
    root / "plans" / "fixed-scan.tsv",
    root / "plans" / "slaclip-screen.tsv",
    root / "plans" / "stage2.tsv",
    root / "plans" / "schedule-source.tsv",
    root / "plans" / "final.tsv",
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
if len(all_specs) not in {56, 58, 61, 63, 66, 68}:
    raise SystemExit(
        "unexpected formal experiment-arm count after full-data schedule sources: "
        f"{len(all_specs)}"
    )
arm_roots = [Path(fields[10]).resolve() for fields in all_specs]
if len(set(arm_roots)) != len(arm_roots):
    raise SystemExit("plans reuse an arm output directory")
fixed_scan_run_count = sum(fields[1] == "fixed-scan" for fields in all_specs)
slaclip_screen_run_count = sum(
    fields[1] == "slaclip-screen" for fields in all_specs
)
stage1_run_count = fixed_scan_run_count + slaclip_screen_run_count
stage2_run_count = sum(fields[1] == "stage2" for fields in all_specs)
final_run_count = sum(fields[1] == "final" for fields in all_specs)
schedule_source_run_count = sum(fields[1] == "schedule-source" for fields in all_specs)
if (
    fixed_scan_run_count != 8
    or slaclip_screen_run_count != 10
    or stage2_run_count not in (10, 12)
    or schedule_source_run_count != 3
    or final_run_count not in (25, 30, 35)
):
    raise SystemExit(
        "invalid dynamic phase counts: "
        f"fixed_scan={fixed_scan_run_count} slaclip_screen={slaclip_screen_run_count} "
        f"stage1={stage1_run_count} stage2={stage2_run_count} "
        f"schedule_source={schedule_source_run_count} final={final_run_count}"
    )
screen_run_count = stage1_run_count + stage2_run_count
pre_final_dp_run_count = screen_run_count + schedule_source_run_count
total_dp_run_count = pre_final_dp_run_count + final_run_count

def basic_composition(release_count):
    return {
        "epsilon": 6.0 * release_count,
        "delta": 1e-5 * release_count,
        "release_count": release_count,
    }

runs = []
run_index_rows = []
for fields, arm_root in zip(all_specs, arm_roots):
    if root not in arm_root.parents:
        raise SystemExit(f"arm escapes campaign root: {arm_root}")
    required = {
        "orchestration_status": arm_root / "orchestration-status.txt",
        "arm_manifest": arm_root / "arm-manifest.txt",
        "adapter_status": arm_root / "adapter" / "run_status.json",
        "telemetry_summary": arm_root / "results" / "research_raw" / "telemetry_summary.json",
        "telemetry_steps": arm_root / "results" / "research_raw" / "telemetry_steps.csv",
        "safe_train_log": arm_root / "adapter" / "train_log.jsonl",
        "raw_train_log": arm_root / "results" / "research_raw" / "NON_PRIVATE_train_log.jsonl",
        "adapter_config": arm_root / "adapter" / "adapter_config.json",
    }
    is_final = fields[1] == "final"
    is_schedule_source = fields[1] == "schedule-source"
    if is_final:
        required.update({
            "decontaminated_metrics": arm_root / "results" / "decontaminated_metrics.json",
            "evaluation_config": arm_root / "results" / "evaluation_config.json",
            "task_summary": arm_root / "results" / "summary.csv",
            "task_details": arm_root / "results" / "details.csv",
            "task_gsm8k": arm_root / "results" / "gsm8k.json",
            "task_AQuA": arm_root / "results" / "AQuA.json",
            "task_mawps": arm_root / "results" / "mawps.json",
            "task_SVAMP": arm_root / "results" / "SVAMP.json",
        })
    elif is_schedule_source:
        required.update({
            "split_manifest": arm_root / "results" / "validation" / "split_manifest.json",
        })
    else:
        required.update({
            "validation_metrics": arm_root / "results" / "validation" / "validation_metrics.json",
            "validation_curve": arm_root / "results" / "validation" / "validation_curve.jsonl",
            "split_manifest": arm_root / "results" / "validation" / "split_manifest.json",
            "public_numeric_predictions": arm_root / "results" / "validation" / "public_numeric_predictions.json",
        })
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
    if is_schedule_source:
        split = json.loads(required["split_manifest"].read_text(encoding="utf-8"))
        if (
            split.get("protocol_stage") != "final"
            or int(split.get("train_rows", -1)) != 9919
            or int(split.get("validation_rows", -1)) != 0
            or split.get("validation_data_is_public") is not False
            or status.get("data_split") != split
        ):
            raise SystemExit(f"invalid full-data schedule-source split: {required['split_manifest']}")
    telemetry = json.loads(required["telemetry_summary"].read_text(encoding="utf-8"))

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

    def telemetry_stat(name, statistic):
        metric = (telemetry.get("metrics") or {}).get(name)
        if not isinstance(metric, dict):
            raise SystemExit(f"telemetry summary lacks metric={name}: {required['telemetry_summary']}")
        return finite(
            metric.get(statistic),
            f"telemetry {name}.{statistic}",
        )

    def optional_telemetry_mean(name):
        metric = (telemetry.get("metrics") or {}).get(name)
        if not isinstance(metric, dict) or metric.get("mean") is None:
            return ""
        return finite(metric.get("mean"), f"telemetry {name}.mean")

    validation_values = {
        "numeric_exact_accuracy": "",
        "numeric_exact_correct": "",
        "numeric_parse_failures": "",
        "endpoint_validation_loss": "",
        "validation_loss_step0": "",
        "validation_loss_step300": "",
        "validation_loss_normalized_auc": "",
    }
    task_values = {
        "gsm8k_accuracy": "",
        "AQuA_accuracy": "",
        "mawps_accuracy": "",
        "SVAMP_accuracy": "",
        "macro_accuracy": "",
    }
    decontaminated_values = {
        "clean_mawps_accuracy": "",
        "clean_mawps_correct": "",
        "clean_mawps_records": "",
        "decontaminated_four_task_macro_accuracy": "",
        "clean_three_task_macro_accuracy": "",
        "primary_accuracy_metric": "",
    }
    if is_final:
        with required["task_summary"].open("r", encoding="utf-8", newline="") as handle:
            summary_rows = list(csv.DictReader(handle))
        expected_columns = ["gsm8k", "AQuA", "mawps", "SVAMP", "Average"]
        if len(summary_rows) != 1 or list(summary_rows[0]) != expected_columns:
            raise SystemExit(f"unexpected final task summary: {required['task_summary']}")
        task_values = {
            "gsm8k_accuracy": finite(float(summary_rows[0]["gsm8k"]), "gsm8k accuracy"),
            "AQuA_accuracy": finite(float(summary_rows[0]["AQuA"]), "AQuA accuracy"),
            "mawps_accuracy": finite(float(summary_rows[0]["mawps"]), "mawps accuracy"),
            "SVAMP_accuracy": finite(float(summary_rows[0]["SVAMP"]), "SVAMP accuracy"),
            "macro_accuracy": finite(float(summary_rows[0]["Average"]), "macro accuracy"),
        }
        if any(not 0.0 <= value <= 1.0 for value in task_values.values()):
            raise SystemExit(f"final task accuracy outside [0,1]: {required['task_summary']}")
        if not math.isclose(
            task_values["macro_accuracy"],
            math.fsum(value for key, value in task_values.items() if key != "macro_accuracy") / 4.0,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise SystemExit(f"final macro accuracy mismatch: {required['task_summary']}")
        decontaminated = json.loads(
            required["decontaminated_metrics"].read_text(encoding="utf-8")
        )
        if decontaminated.get("primary_metric") != "clean_three_task_macro_accuracy":
            raise SystemExit(
                f"unexpected primary decontaminated metric: {required['decontaminated_metrics']}"
            )
        decontaminated_values = {
            "clean_mawps_accuracy": finite(
                decontaminated.get("clean_mawps_accuracy"),
                "clean MAWPS accuracy",
            ),
            "clean_mawps_correct": int(decontaminated.get("clean_mawps_correct")),
            "clean_mawps_records": int(decontaminated.get("clean_mawps_records")),
            "decontaminated_four_task_macro_accuracy": finite(
                decontaminated.get(
                    "decontaminated_four_task_macro_accuracy"
                ),
                "decontaminated four-task macro accuracy",
            ),
            "clean_three_task_macro_accuracy": finite(
                decontaminated.get("clean_three_task_macro_accuracy"),
                "clean three-task macro accuracy",
            ),
            "primary_accuracy_metric": decontaminated["primary_metric"],
        }
        if decontaminated_values["clean_mawps_records"] != 185:
            raise SystemExit(
                f"decontaminated MAWPS evaluation must contain 185 clean records: "
                f"{required['decontaminated_metrics']}"
            )
        if not math.isclose(
            finite(
                decontaminated.get("paper_raw_four_task_macro_accuracy"),
                "paper raw four-task macro accuracy",
            ),
            task_values["macro_accuracy"],
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise SystemExit(
                f"paper raw macro/decontaminated artifact mismatch: "
                f"{required['decontaminated_metrics']}"
            )
    elif not is_schedule_source:
        validation = json.loads(required["validation_metrics"].read_text(encoding="utf-8"))
        curve = [
            json.loads(line)
            for line in required["validation_curve"].read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if [int(item.get("step", -1)) for item in curve] != [0, 50, 100, 150, 200, 250, 300]:
            raise SystemExit(f"unexpected validation curve coverage: {required['validation_curve']}")
        curve_steps = [finite(item["step"], "curve step") for item in curve]
        curve_losses = [finite(item.get("loss_mean"), "curve loss") for item in curve]
        curve_width = curve_steps[-1] - curve_steps[0]
        curve_auc = math.fsum(
            (curve_steps[index + 1] - curve_steps[index])
            * (curve_losses[index + 1] + curve_losses[index])
            / 2.0
            for index in range(len(curve_steps) - 1)
        ) / curve_width
        validation_values = {
            "numeric_exact_accuracy": finite(validation.get("numeric_exact_accuracy"), "numeric accuracy"),
            "numeric_exact_correct": int(validation.get("numeric_exact_correct")),
            "numeric_parse_failures": int(validation.get("numeric_parse_failures")),
            "endpoint_validation_loss": finite(validation.get("loss_mean"), "endpoint loss"),
            "validation_loss_step0": curve_losses[0],
            "validation_loss_step300": curve_losses[-1],
            "validation_loss_normalized_auc": curve_auc,
        }
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
        **validation_values,
        **task_values,
        **decontaminated_values,
        "epsilon_spent": epsilon,
        "raw_training_loss_mean": telemetry_stat("loss_mean", "mean"),
        "raw_training_loss_first": telemetry_stat("loss_mean", "first"),
        "raw_training_loss_last": telemetry_stat("loss_mean", "last"),
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
    "gsm8k_accuracy",
    "AQuA_accuracy",
    "mawps_accuracy",
    "SVAMP_accuracy",
    "macro_accuracy",
    "clean_mawps_accuracy",
    "clean_mawps_correct",
    "clean_mawps_records",
    "decontaminated_four_task_macro_accuracy",
    "clean_three_task_macro_accuracy",
    "primary_accuracy_metric",
    "epsilon_spent",
    "raw_training_loss_mean",
    "raw_training_loss_first",
    "raw_training_loss_last",
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

# Predeclared paired inference uses only the five fresh post-lock seeds.  A
# seed, not a training step or prediction row, is the independent unit.
final_rows = [row for row in run_index_rows if row["phase"] == "final"]
by_role_seed = {
    (str(row["role"]), int(row["seed"])): row
    for row in final_rows
}
fresh_seeds = [17, 29, 71, 101, 137]
required_roles = {
    "selected-slaclip",
    "best-fixed",
    "replay",
    "matched-fixed-noise-energy",
}
observed_roles = {str(row["role"]) for row in final_rows}
control_manifest = json.loads(locked_files[6].read_text(encoding="utf-8"))
initial_c_reference_role = control_manifest["initial_c_matched_fixed_reference_role"]
canonical_full_reference_role = control_manifest["canonical_full_slaclip_reference_role"]
if not required_roles.issubset(observed_roles):
    raise SystemExit(
        f"final paired analysis lacks required roles: {sorted(required_roles - observed_roles)}"
    )
if any((role, seed) not in by_role_seed for role in required_roles for seed in fresh_seeds):
    raise SystemExit("final paired analysis lacks a required role/seed run")
for reference_role in (initial_c_reference_role, canonical_full_reference_role):
    if reference_role not in observed_roles or any(
        (reference_role, seed) not in by_role_seed for seed in fresh_seeds
    ):
        raise SystemExit(f"final paired analysis lacks locked reference role={reference_role}")
comparisons = [
    ("selected-slaclip", "best-fixed"),
    ("selected-slaclip", "replay"),
    ("selected-slaclip", "matched-fixed-noise-energy"),
    ("replay", "best-fixed"),
    ("matched-fixed-noise-energy", "best-fixed"),
]
if "canonical-fixed-c1" in observed_roles:
    if any(("canonical-fixed-c1", seed) not in by_role_seed for seed in fresh_seeds):
        raise SystemExit("canonical C=1 final runs do not cover all fresh seeds")
    comparisons.extend((
        ("selected-slaclip", "canonical-fixed-c1"),
        ("canonical-fixed-c1", "best-fixed"),
    ))
for comparison in (
    ("selected-slaclip", initial_c_reference_role),
    ("selected-slaclip", canonical_full_reference_role),
    (canonical_full_reference_role, "best-fixed"),
):
    if comparison[0] != comparison[1] and comparison not in comparisons:
        comparisons.append(comparison)
accuracy_metrics = [
    ("clean_three_task_macro_accuracy", "primary", True),
    ("decontaminated_four_task_macro_accuracy", "sensitivity", False),
    ("macro_accuracy", "paper_comparability", False),
    ("gsm8k_accuracy", "task", False),
    ("AQuA_accuracy", "task", False),
    ("clean_mawps_accuracy", "decontaminated_task", False),
    ("mawps_accuracy", "raw_task", False),
    ("SVAMP_accuracy", "task", False),
]
paired_rows = []
critical_t_95_df4 = 2.7764451051977987
for candidate_role, reference_role in comparisons:
    for metric, metric_role, is_primary in accuracy_metrics:
        is_campaign_primary = bool(
            is_primary
            and candidate_role == "selected-slaclip"
            and reference_role == "best-fixed"
        )
        is_initial_c_matched = (
            candidate_role == "selected-slaclip"
            and reference_role == initial_c_reference_role
        )
        is_canonical_full_anchor = (
            (candidate_role, reference_role)
            in {
                ("selected-slaclip", canonical_full_reference_role),
                (canonical_full_reference_role, "best-fixed"),
            }
            and candidate_role != reference_role
        )
        deltas = [
            float(by_role_seed[(candidate_role, seed)][metric])
            - float(by_role_seed[(reference_role, seed)][metric])
            for seed in fresh_seeds
        ]
        mean_delta = math.fsum(deltas) / len(deltas)
        sample_sd = statistics.stdev(deltas)
        half_width = critical_t_95_df4 * sample_sd / math.sqrt(len(deltas))
        tolerance = 1e-15
        paired_rows.append({
            "comparison": f"{candidate_role}_vs_{reference_role}",
            "candidate_role": candidate_role,
            "reference_role": reference_role,
            "metric": metric,
            "metric_role": metric_role,
            "is_primary": is_campaign_primary,
            "comparison_scope": (
                "campaign_primary" if is_campaign_primary else "exploratory"
            ),
            "is_initial_c_matched_comparison": is_initial_c_matched,
            "is_canonical_full_controller_comparison": is_canonical_full_anchor,
            "n_paired_seeds": len(deltas),
            "paired_mean_delta": mean_delta,
            "sample_sd": sample_sd,
            "t95_ci_low": mean_delta - half_width,
            "t95_ci_high": mean_delta + half_width,
            "wins": sum(value > tolerance for value in deltas),
            "ties": sum(abs(value) <= tolerance for value in deltas),
            "losses": sum(value < -tolerance for value in deltas),
            "seeds": ",".join(str(seed) for seed in fresh_seeds),
        })
if sum(row["is_primary"] is True for row in paired_rows) != 1:
    raise SystemExit("paired accuracy table must contain exactly one campaign-primary row")
paired_path = root / "artifacts" / "paired_final_accuracy.csv"
paired_fields = [
    "comparison",
    "candidate_role",
    "reference_role",
    "metric",
    "metric_role",
    "is_primary",
    "comparison_scope",
    "is_initial_c_matched_comparison",
    "is_canonical_full_controller_comparison",
    "n_paired_seeds",
    "paired_mean_delta",
    "sample_sd",
    "t95_ci_low",
    "t95_ci_high",
    "wins",
    "ties",
    "losses",
    "seeds",
]
paired_buffer = io.StringIO(newline="")
paired_writer = csv.DictWriter(
    paired_buffer,
    fieldnames=paired_fields,
    lineterminator="\n",
)
paired_writer.writeheader()
paired_writer.writerows(paired_rows)
paired_encoded = paired_buffer.getvalue().encode()
if paired_path.exists():
    if paired_path.read_bytes() != paired_encoded:
        raise SystemExit(f"refusing inconsistent paired final analysis: {paired_path}")
else:
    fd, name = tempfile.mkstemp(
        prefix=f".{paired_path.name}.", suffix=".tmp", dir=paired_path.parent
    )
    temporary = Path(name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(paired_encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, paired_path)
    finally:
        temporary.unlink(missing_ok=True)
os.chmod(paired_path, 0o600)

mechanism_comparisons = [
    ("selected-slaclip", "best-fixed"),
    ("selected-slaclip", "replay"),
    ("selected-slaclip", "matched-fixed-noise-energy"),
]
if "canonical-fixed-c1" in observed_roles:
    mechanism_comparisons.append(("selected-slaclip", "canonical-fixed-c1"))
for comparison in (
    ("selected-slaclip", initial_c_reference_role),
    ("selected-slaclip", canonical_full_reference_role),
    (canonical_full_reference_role, "best-fixed"),
):
    if comparison[0] != comparison[1] and comparison not in mechanism_comparisons:
        mechanism_comparisons.append(comparison)
mechanism_metrics = {
    "raw_training_loss_mean": "lower_is_better",
    "raw_clip_fraction_mean": "target_dependent_diagnostic",
    "dp_clip_threshold_mean": "target_dependent_diagnostic",
    "raw_signal_to_noise_ratio_mean": "higher_is_better",
    "raw_clipping_bias_norm_mean": "lower_is_better",
    "raw_realized_noise_norm_mean": "lower_is_better",
    "raw_signal_retention_ratio_mean": "higher_is_better",
    "raw_clipping_bias_to_noise_ratio_mean": "lower_is_better",
    "raw_bias_noise_squared_error_proxy_mean": "lower_is_better",
    "raw_unclipped_clipped_cosine_mean": "higher_is_better",
    "raw_clipped_noisy_cosine_mean": "higher_is_better",
}
mechanism_rows = []
for candidate_role, reference_role in mechanism_comparisons:
    for metric, direction in mechanism_metrics.items():
        deltas = [
            float(by_role_seed[(candidate_role, seed)][metric])
            - float(by_role_seed[(reference_role, seed)][metric])
            for seed in fresh_seeds
        ]
        mean_delta = math.fsum(deltas) / len(deltas)
        sample_sd = statistics.stdev(deltas)
        half_width = critical_t_95_df4 * sample_sd / math.sqrt(len(deltas))
        tolerance = 1e-15
        if direction == "higher_is_better":
            wins = sum(value > tolerance for value in deltas)
            losses = sum(value < -tolerance for value in deltas)
        elif direction == "lower_is_better":
            wins = sum(value < -tolerance for value in deltas)
            losses = sum(value > tolerance for value in deltas)
        else:
            wins = ""
            losses = ""
        mechanism_rows.append({
            "comparison": f"{candidate_role}_vs_{reference_role}",
            "candidate_role": candidate_role,
            "reference_role": reference_role,
            "metric": metric,
            "preferred_direction": direction,
            "n_paired_seeds": len(deltas),
            "paired_mean_delta": mean_delta,
            "sample_sd": sample_sd,
            "t95_ci_low": mean_delta - half_width,
            "t95_ci_high": mean_delta + half_width,
            "candidate_better_wins": wins,
            "ties": sum(abs(value) <= tolerance for value in deltas),
            "candidate_worse_losses": losses,
            "seeds": ",".join(str(seed) for seed in fresh_seeds),
        })
mechanism_path = root / "artifacts" / "paired_final_mechanism.csv"
mechanism_fields = [
    "comparison",
    "candidate_role",
    "reference_role",
    "metric",
    "preferred_direction",
    "n_paired_seeds",
    "paired_mean_delta",
    "sample_sd",
    "t95_ci_low",
    "t95_ci_high",
    "candidate_better_wins",
    "ties",
    "candidate_worse_losses",
    "seeds",
]
mechanism_buffer = io.StringIO(newline="")
mechanism_writer = csv.DictWriter(
    mechanism_buffer,
    fieldnames=mechanism_fields,
    lineterminator="\n",
)
mechanism_writer.writeheader()
mechanism_writer.writerows(mechanism_rows)
mechanism_encoded = mechanism_buffer.getvalue().encode()
if mechanism_path.exists():
    if mechanism_path.read_bytes() != mechanism_encoded:
        raise SystemExit(f"refusing inconsistent paired final mechanism analysis: {mechanism_path}")
else:
    fd, name = tempfile.mkstemp(
        prefix=f".{mechanism_path.name}.", suffix=".tmp", dir=mechanism_path.parent
    )
    temporary = Path(name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(mechanism_encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, mechanism_path)
    finally:
        temporary.unlink(missing_ok=True)
os.chmod(mechanism_path, 0o600)

campaign_snapshot_files = [
    root / "campaign_manifest.json",
    root / "submission_receipt.json",
    root / "artifacts" / "environment_freeze.txt",
    *sorted((root / "artifacts").glob("allocation-*.txt")),
    *sorted(root.glob("preflight/**/cuda-guards/*.json")),
    *sorted(root.glob("preflight/**/logs/preflight-*.out")),
    *sorted(root.glob("preflight/**/logs/preflight-*.err")),
]
for path in campaign_snapshot_files:
    if not path.is_file():
        raise SystemExit(f"missing campaign/environment/preflight snapshot: {path}")
evaluation_asset_manifest = json.loads(locked_files[7].read_text(encoding="utf-8"))
decontamination_protocol = evaluation_asset_manifest.get("decontamination") or {}
payload = {
    "schema_version": 1,
    "analysis_status": "LOCKED_FRESH_SEED_FINAL_COMPLETE",
    "inference_scope": (
        "prospectively_locked_within_this_campaign_but_not_untouched-test_"
        "confirmatory_due_to_prior_access"
    ),
    "recommended_use": (
        "journal_development_evidence_requiring_independent_dataset_or_model_validation"
    ),
    "code_sha": code_sha,
    "raw_exact_telemetry_is_non_private": True,
    "target_calibration": {
        "classification": "NON_PRIVATE_data_dependent_exploratory_calibration",
        "fixed_scan_manifest_path": str(locked_files[0].relative_to(root)),
        "fixed_scan_manifest_sha256": sha(locked_files[0]),
        "slaclip_target_grid_path": str(locked_files[1].relative_to(root)),
        "slaclip_target_grid_sha256": sha(locked_files[1]),
        "mapping": "p=rho*(1-z_reference)",
    },
    "privacy_accounting": {
        "fixed_scan_run_count": fixed_scan_run_count,
        "slaclip_screen_run_count": slaclip_screen_run_count,
        "stage1_run_count": stage1_run_count,
        "stage2_run_count": stage2_run_count,
        "screen_run_count": screen_run_count,
        "pre_final_dp_run_count": pre_final_dp_run_count,
        "final_run_count": final_run_count,
        "schedule_construction_source_run_count": schedule_source_run_count,
        "schedule_privacy_class": (
            f"DP_DERIVED_VIA_{screen_run_count}_RUN_SELECTION_PLUS_3_FULL_DATA_SOURCES"
        ),
        "replay_control_classification": (
            "full_data_matched_schedule_transfer_from_selected_DP_SlaClip_trajectories"
        ),
        "one_run_conditioned_on_unreleased_locked_selection": {
            "epsilon": 6.0,
            "delta": 1e-5,
            "warning": (
                "not a release bound if selection or other screen outputs are released"
            ),
        },
        "selection_and_full_data_schedule_sources_basic_composition": basic_composition(
            pre_final_dp_run_count
        ),
        "selection_schedule_plus_one_final_control_basic_composition": basic_composition(
            pre_final_dp_run_count + 1
        ),
        "selection_schedule_plus_five_replay_basic_composition": basic_composition(
            pre_final_dp_run_count + 5
        ),
        "selection_plus_five_replay_plus_five_matched_fixed_basic_composition": (
            basic_composition(pre_final_dp_run_count + 10)
        ),
        "all_screen_and_fresh_final_dp_outputs_basic_composition": basic_composition(
            total_dp_run_count
        ),
        "development_artifact_bundle": "NON_PRIVATE_due_to_research_raw",
    },
    "campaign_worker_model_evaluation_begins_after_stage2_selection_lock": True,
    "selection_algorithm_uses_test_outputs": False,
    "test_assets_and_prior_test_results_were_accessed_before_campaign_design": True,
    "not_an_untouched_test_set": True,
    "test_sets_accessed": True,
    "evaluation_protocol": {
        "campaign_primary_metric": "clean_three_task_macro_accuracy",
        "campaign_primary_comparison": "selected-slaclip_vs_best-fixed",
        "sensitivity_metric": "decontaminated_four_task_macro_accuracy",
        "paper_comparability_metric": "raw_four_task_macro_accuracy",
        "decontamination_normalization": decontamination_protocol["normalization"],
        "decontamination_asset_manifest_path": str(locked_files[7].relative_to(root)),
        "decontamination_asset_manifest_sha256": sha(locked_files[7]),
        "mawps_raw_records": 238,
        "mawps_conservative_clean_records": 185,
        "mawps_conservative_overlap_records": 53,
        "mawps_overlap_answer_string_exact_records": 38,
        "mawps_overlap_answer_evaluator_equivalent_records": 53,
    },
    "control_protocol": {
        "initial_c_matched_fixed_C": control_manifest["initial_c_matched_fixed_C"],
        "initial_c_matched_fixed_reference_role": initial_c_reference_role,
        "initial_c_matched_paired_comparison": (
            f"selected-slaclip_vs_{initial_c_reference_role}"
        ),
        "canonical_full_slaclip_reference_role": canonical_full_reference_role,
        "canonical_full_slaclip_definition": control_manifest[
            "canonical_full_slaclip_definition"
        ],
        "canonical_full_slaclip_paired_comparison": (
            "selected_controller_is_canonical_anchor_no_distinct_comparison"
            if canonical_full_reference_role == "selected-slaclip"
            else f"selected-slaclip_vs_{canonical_full_reference_role}"
        ),
        "comparison_scope": "exploratory_except_tuned_selected_vs_best_fixed_primary",
    },
    "fresh_final_seeds": [17, 29, 71, 101, 137],
    "arm_count": len(runs),
    "locked_artifacts": [
        {"path": str(path.relative_to(root)), "sha256": sha(path), "bytes": path.stat().st_size}
        for path in [
            *locked_files,
            *plan_files,
            run_index_path,
            paired_path,
            mechanism_path,
        ]
    ],
    "run_index": {
        "path": str(run_index_path.relative_to(root)),
        "sha256": sha(run_index_path),
        "bytes": run_index_path.stat().st_size,
        "rows": len(run_index_rows),
        "privacy": "NON_PRIVATE_TELEMETRY",
    },
    "paired_final_accuracy": {
        "path": str(paired_path.relative_to(root)),
        "sha256": sha(paired_path),
        "bytes": paired_path.stat().st_size,
        "rows": len(paired_rows),
        "independent_unit": "fresh_seed",
        "confidence_interval": "two_sided_paired_t_95_df4",
        "campaign_primary_rows": sum(row["is_primary"] is True for row in paired_rows),
        "campaign_primary_metric": "clean_three_task_macro_accuracy",
        "campaign_primary_comparison": "selected-slaclip_vs_best-fixed",
        "inference_scope": (
            "prospectively_locked_within_this_campaign_but_not_untouched-test_"
            "confirmatory_due_to_prior_access"
        ),
    },
    "paired_final_mechanism": {
        "path": str(mechanism_path.relative_to(root)),
        "sha256": sha(mechanism_path),
        "bytes": mechanism_path.stat().st_size,
        "rows": len(mechanism_rows),
        "independent_unit": "fresh_seed",
        "confidence_interval": "two_sided_paired_t_95_df4",
        "step_level_source": "each_run/results/research_raw/telemetry_steps.csv",
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

# The completed c877ba1 campaign remains immutable.  The focused refinement
# protocol below deliberately overrides only orchestration; mature execution,
# validation, telemetry, atomic-write, and failure-handling helpers above are
# reused unchanged.
create_refinement_registry() {
  "${PYTHON_BIN}" scripts/build_math10k_4b_refinement_registry.py \
    --campaign-root "${CAMPAIGN_ROOT}" \
    --output "${REGISTRY_FILE}" \
    --model-id "${MODEL_ID}" \
    --model-revision "${MODEL_REVISION}" \
    --code-sha "${EXPECTED_REPO_SHA}" \
    >"${SELECTION_ROOT}/registry-${JOB_ID}.out" \
    2>"${SELECTION_ROOT}/registry-${JOB_ID}.err"
  atomic_sha256_sidecar "${REGISTRY_FILE}"
}

create_refinement_plan() {
  local mode="$1"
  local output="$2"
  local temporary create_rc=0
  temporary="$(mktemp "${output}.tmp.XXXXXX")"
  "${PYTHON_BIN}" - \
    "${mode}" "${REGISTRY_FILE}" "${CAMPAIGN_ROOT}" \
    "${STAGE1_SELECTION}" "${LOCKED_SELECTION}" "${MODEL_SLUG}" \
    >"${temporary}" <<'PY' || create_rc=$?
import json
import sys
from pathlib import Path

mode = sys.argv[1]
registry = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
campaign_root = Path(sys.argv[3])
stage1_selection = Path(sys.argv[4])
locked_selection = Path(sys.argv[5])
model_slug = sys.argv[6]
candidates = {item["id"]: item for item in registry["candidates"]}

def emit(phase, candidate, role, seed, arm_root):
    params = candidate["params"]
    rho = params.get(
        "slaclip_target_non_small_clip_fraction",
        params.get("slaclip_beta", "NA"),
    )
    eta = params.get("slaclip_eta", "NA")
    fields = (
        "train",
        phase,
        candidate["id"],
        role,
        seed,
        candidate["method"],
        params["dp_max_grad_norm"],
        rho,
        eta,
        "NA",
        arm_root,
    )
    if any("|" in str(value) or "\n" in str(value) for value in fields):
        raise SystemExit("plan field contains a forbidden delimiter")
    print("|".join(str(value) for value in fields))

if mode == "stage1":
    if len(candidates) != 27:
        raise SystemExit(f"stage1 registry must contain 27 candidates, got {len(candidates)}")
    for candidate in registry["candidates"]:
        emit(
            "stage1",
            candidate,
            "local-grid-screen",
            42,
            campaign_root / "screen" / "runs" / candidate["id"] / "seed-42",
        )
elif mode == "stage2":
    selection = json.loads(stage1_selection.read_text(encoding="utf-8"))
    selected_ids = [item["candidate_id"] for item in selection["top_slaclip"]]
    if len(selected_ids) != 3 or len(set(selected_ids)) != 3:
        raise SystemExit("stage2 requires exactly three unique SlaClip finalists")
    fixed_ids = [
        candidate_id
        for candidate_id, candidate in candidates.items()
        if candidate["family"] == "fixed"
    ]
    if set(fixed_ids) != {"fixed-c1", "fixed-c2"}:
        raise SystemExit(f"stage2 requires fixed C=1 and C=2, got {fixed_ids}")
    selected_ids.extend(("fixed-c1", "fixed-c2"))
    seeds = registry["selection_protocol"]["stage2_seeds"]
    if seeds != [42, 43, 44, 45, 46]:
        raise SystemExit(f"unexpected selection seeds: {seeds}")
    for seed in seeds[1:]:
        for candidate_id in selected_ids:
            candidate = candidates[candidate_id]
            emit(
                "stage2",
                candidate,
                "five-seed-confirmation",
                seed,
                campaign_root / "screen" / "runs" / candidate_id / f"seed-{seed}",
            )
elif mode == "final":
    selection = json.loads(locked_selection.read_text(encoding="utf-8"))
    selected_id = selection["selected_slaclip"]["candidate_id"]
    selected = candidates[selected_id]
    fixed = candidates["fixed-c2"]
    if selected["family"] != "slaclip" or fixed["params"] != {"dp_max_grad_norm": 2.0}:
        raise SystemExit("locked final candidates are invalid")
    for seed in (191, 223, 257, 293, 331):
        # Adjacency is intentional: run_plan assigns this pair to the same node
        # and launches one arm per A100 in one wave.
        emit(
            "final",
            selected,
            "slaclip",
            seed,
            campaign_root / "final" / model_slug / f"seed-{seed}" / "slaclip",
        )
        emit(
            "final",
            fixed,
            "baseline",
            seed,
            campaign_root / "final" / model_slug / f"seed-{seed}" / "baseline",
        )
else:
    raise SystemExit(f"unknown refinement plan mode: {mode}")
PY
  if [[ "${create_rc}" != "0" ]]; then
    rm -f -- "${temporary}"
    return "${create_rc}"
  fi
  install_immutable_file "${temporary}" "${output}"
  atomic_sha256_sidecar "${output}"
}

create_refinement_campaign_manifest() {
  local manifest="${CAMPAIGN_ROOT}/campaign_manifest.json"
  local temporary
  temporary="$(mktemp "${manifest}.tmp.XXXXXX")"
  "${PYTHON_BIN}" - \
    "${manifest}" "${REGISTRY_FILE}" "${LOCKED_SELECTION}" \
    "${EXPECTED_REPO_SHA}" "${JOB_ID}" "${MODEL_ID}" "${MODEL_REVISION}" \
    "${ENV_PREFIX}" "${ENVIRONMENT_FREEZE}" >"${temporary}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

target = Path(sys.argv[1])
registry_path = Path(sys.argv[2])
selection_path = Path(sys.argv[3])
environment_path = Path(sys.argv[9])
registry = json.loads(registry_path.read_text(encoding="utf-8"))
selection = json.loads(selection_path.read_text(encoding="utf-8"))
if len(registry.get("candidates", [])) != 27:
    raise SystemExit("campaign manifest requires exactly 27 candidates")
if selection.get("required_seeds") != [42, 43, 44, 45, 46]:
    raise SystemExit("campaign manifest requires the locked five-seed selection")
release_count = 27 + 20 + 10
payload = {
    "schema_version": 1,
    "protocol": "math10k_4b_c1_c2_high_rho_low_eta_refinement_v1",
    "code_sha": sys.argv[4],
    "job_id": sys.argv[5],
    "model_id": sys.argv[6],
    "model_revision": sys.argv[7],
    "environment": sys.argv[8],
    "environment_freeze": {
        "path": str(environment_path.relative_to(target.parent)),
        "sha256": hashlib.sha256(environment_path.read_bytes()).hexdigest(),
    },
    "candidate_registry": {
        "path": str(registry_path.relative_to(target.parent)),
        "sha256": hashlib.sha256(registry_path.read_bytes()).hexdigest(),
        "candidate_count": 27,
    },
    "locked_selection": {
        "path": str(selection_path.relative_to(target.parent)),
        "sha256": hashlib.sha256(selection_path.read_bytes()).hexdigest(),
        "selection_seeds": [42, 43, 44, 45, 46],
    },
    "resources": {
        "allocation": "one Slurm job",
        "gpu": "2xa100",
        "lanes": 2,
        "cpu_per_lane": 8,
        "host_memory_per_lane": "128G",
        "walltime": "60h",
    },
    "design": {
        "stage1_runs": 27,
        "stage2_runs": 20,
        "final_runs": 10,
        "total_dp_training_runs": release_count,
        "final_reference": "fixed_C=2",
        "final_fresh_paired_seeds": [191, 223, 257, 293, 331],
        "current_result_basis": (
            "fixed_C2_outperformed_the_previous_selected_C0=2_rho=.97_eta=.15_"
            "controller_while_rho=.99_kept_mean_C_closer_to_2;_therefore_scan_"
            "C0_in_[1.25,2],_rho_in_[.985,.995],_and_eta_in_[.05,.1]"
        ),
        "selection_uses_only_public_math10k_holdout": True,
        "test_assets_accessed_only_after_selection_lock": True,
        "prior_test_results_informed_this_design": True,
        "not_an_untouched_confirmatory_test": True,
    },
    "privacy": {
        "per_run_target": {"epsilon": 6.0, "delta": 1e-5},
        "all_runs_basic_composition_upper_bound": {
            "epsilon": 6.0 * release_count,
            "delta": 1e-5 * release_count,
            "release_count": release_count,
        },
        "telemetry": "research_raw_NON_PRIVATE",
        "development_bundle_is_non_private": True,
    },
}
print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
PY
  install_immutable_file "${temporary}" "${manifest}"
  atomic_sha256_sidecar "${manifest}"
}

create_refinement_final_manifest() {
  local manifest="${CAMPAIGN_ROOT}/final/manifest.json"
  local temporary
  temporary="$(mktemp "${manifest}.tmp.XXXXXX")"
  "${PYTHON_BIN}" - >"${temporary}" <<'PY'
import json

print(json.dumps({"expected_update_steps": 300}, indent=2, sort_keys=True))
PY
  install_immutable_file "${temporary}" "${manifest}"
  atomic_sha256_sidecar "${manifest}"
}

CURRENT_PHASE="preflight"
write_job_status running 0
echo "job_id=${JOB_ID} campaign=${CAMPAIGN_ID} host=$(hostname)"
echo "code_sha=${EXPECTED_REPO_SHA} environment=${ENV_PREFIX}"
nvidia-smi
"${PYTHON_BIN}" - <<'PY'
import torch

torch.cuda.init()
count = torch.cuda.device_count()
if count != 2:
    raise SystemExit(f"expected exactly two visible allocation GPUs, found {count}")
for index in range(count):
    with torch.cuda.device(index):
        value = torch.ones((16, 16), device=f"cuda:{index}")
        result = value @ value
        if float(result.sum().item()) != 4096.0:
            raise SystemExit(f"unexpected CUDA checksum on allocation GPU {index}")
        torch.cuda.synchronize(index)
        print(f"allocation_gpu[{index}]={torch.cuda.get_device_name(index)}")
PY
"${PYTHON_BIN}" -m pip check
"${PYTHON_BIN}" -m compileall -q train_eval.py src scripts
create_environment_and_allocation_snapshots
run_dual_cuda_guard_pair
lane_srun \
  "${CAMPAIGN_ROOT}/preflight/${MODEL_SLUG}" \
  preflight \
  "${PYTHON_BIN}" scripts/preflight_hpc.py \
    --dataset math10k \
    --data-path "${REPO_ROOT}/LLM-Adapters/ft-training_set/math_10k.json" \
    --expected-data-sha256 0342d0d860ad8592b579329337c90e42eefd3d9f2898043140cbd120630418b8 \
    --output-root "${CAMPAIGN_ROOT}/preflight/${MODEL_SLUG}" \
    --require-cuda \
    --model-id "${MODEL_ID}" \
    --model-revision "${MODEL_REVISION}"

CURRENT_PHASE="smoke"
write_job_status running 0
run_parallel_pair "synthetic|baseline" "synthetic|slaclip"
run_parallel_pair "real-smoke|baseline" "real-smoke|slaclip"

CURRENT_PHASE="registry-lock"
write_job_status running 0
create_refinement_registry

CURRENT_PHASE="stage1-local-grid"
write_job_status running 0
STAGE1_PLAN="${PLAN_ROOT}/stage1.tsv"
create_refinement_plan stage1 "${STAGE1_PLAN}"
if [[ "$(wc -l <"${STAGE1_PLAN}")" != "27" ]]; then
  echo "error: local-grid stage1 plan must contain exactly 27 arms" >&2
  exit 2
fi
run_plan "${STAGE1_PLAN}"

CURRENT_PHASE="stage1-selection"
write_job_status running 0
run_formal_selector stage1 "${STAGE1_SELECTION}"

CURRENT_PHASE="stage2-five-seed-confirmation"
write_job_status running 0
STAGE2_PLAN="${PLAN_ROOT}/stage2.tsv"
create_refinement_plan stage2 "${STAGE2_PLAN}"
stage2_count="$(wc -l <"${STAGE2_PLAN}")"
if [[ "${stage2_count}" != "20" ]]; then
  echo "error: stage2 plan must contain exactly 20 arms, found ${stage2_count}" >&2
  exit 2
fi
run_plan "${STAGE2_PLAN}"

CURRENT_PHASE="locked-selection"
write_job_status running 0
run_formal_selector stage2 "${LOCKED_SELECTION}"

CURRENT_PHASE="final-assets-lock"
write_job_status running 0
lock_final_evaluation_assets

CURRENT_PHASE="final"
write_job_status running 0
FINAL_PLAN="${PLAN_ROOT}/final.tsv"
create_refinement_plan final "${FINAL_PLAN}"
final_count="$(wc -l <"${FINAL_PLAN}")"
if [[ "${final_count}" != "10" ]]; then
  echo "error: locked paired final plan must contain exactly ten arms, found ${final_count}" >&2
  exit 2
fi
create_refinement_campaign_manifest
run_plan "${FINAL_PLAN}"

CURRENT_PHASE="refinement-analysis"
write_job_status running 0
create_refinement_final_manifest
"${PYTHON_BIN}" scripts/analyze_mechanism_campaign.py \
  --campaign-root "${CAMPAIGN_ROOT}" \
  --output-dir artifacts/deep-analysis-v1 \
  --strict
"${PYTHON_BIN}" scripts/analyze_math10k_4b_refinement_campaign.py \
  --campaign-root "${CAMPAIGN_ROOT}" \
  --output-dir artifacts
atomic_sha256_sidecar "${CAMPAIGN_ROOT}/artifacts/manifest.json"

CURRENT_PHASE="complete"
write_job_status completed 0
echo "Math-10K 4B C=1--2 refinement campaign completed at $(date --iso-8601=seconds)."
