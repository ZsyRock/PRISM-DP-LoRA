#!/usr/bin/env bash

# Submit one allocation containing one or two GPU lanes for all paper-coverage
# subexperiments.  There are no arrays, dependencies, child sbatch calls, or
# requeue attempts.
set -Eeuo pipefail
umask 077

MODE="${1:-}"
case "${MODE}" in
  --test-only|--submit|--resume-submit) ;;
  *)
    echo "usage: $0 --test-only|--submit|--resume-submit" >&2
    exit 2
    ;;
esac

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
USER_NAME="${PRISM_USER_NAME:-${USER:-$(id -un)}}"
if ! getent passwd "${USER_NAME}" >/dev/null; then USER_NAME="$(id -un)"; fi
USER_HOME="${PRISM_USER_HOME:-$(getent passwd "${USER_NAME}" | cut -d: -f6)}"
SCRATCH_ROOT="${PRISM_SCRATCH_ROOT:-${SCRATCH:-/scratch/${USER_NAME}}}"
REPO_ROOT="${PRISM_REPO_ROOT:-${DEFAULT_REPO_ROOT}}"
REPO_ROOT="$(cd -- "${REPO_ROOT}" && pwd)"
[[ -d "${REPO_ROOT}/.git" ]] || { echo "error: repository unavailable" >&2; exit 2; }

LOCKED_REPO_SHA="$(git -C "${REPO_ROOT}" rev-parse "${PRISM_CODE_REVISION:-HEAD}^{commit}")"
if [[ "$(git -C "${REPO_ROOT}" rev-parse HEAD)" != "${LOCKED_REPO_SHA}" \
    || -n "$(git -C "${REPO_ROOT}" status --porcelain)" ]]; then
  echo "error: formal submission requires a clean checked-out HEAD" >&2
  git -C "${REPO_ROOT}" status --short >&2
  exit 2
fi

SHORT_SHA="${LOCKED_REPO_SHA:0:12}"
ENV_PREFIX="${PRISM_ENV_PREFIX:-${SCRATCH_ROOT}/envs/prism-dp-lora-${PRISM_ENV_TAG:-71bce55}}"
RUN_ROOT="${PRISM_RUN_ROOT:-${SCRATCH_ROOT}/runs/prism-dp-lora}"
SHARED_HF_HOME="${PRISM_HF_HOME:-${RUN_ROOT}/cache/huggingface}"
GLUE_EVAL_REVISION="bcdcba79d07bc864c1c254ccfcedcce55bcc9a8c"
GLUE_EVAL_ROOT="${PRISM_GLUE_EVAL_ROOT:-${RUN_ROOT}/cache/glue-eval/${GLUE_EVAL_REVISION}}"
COVERAGE_PROFILE="${PRISM_COVERAGE_PROFILE:-}"
if [[ -z "${COVERAGE_PROFILE}" ]]; then
  echo "error: PRISM_COVERAGE_PROFILE must be set explicitly" >&2
  exit 2
fi
CAMPAIGN_ID="${PRISM_CAMPAIGN_ID:-paper-coverage-${SHORT_SHA}-${COVERAGE_PROFILE}-v2}"
STAGED_REPO_ROOT="${PRISM_STAGED_REPO_ROOT:-${RUN_ROOT}/sources/PRISM-DP-LoRA-${LOCKED_REPO_SHA}}"
MODEL_4B_REVISION="cc012e0a6d0787b4adcc0fa2c4da74402494554d"
MODEL_9B_REVISION="33c193028431c2fde6c6e51f29e6f17b60cbfac6"
MODEL_12B_REVISION="295efb63d01a7017928f273a94ebb86105c9526f"

ACCOUNT="${PRISM_SLURM_ACCOUNT:-normal}"
QOS="${PRISM_SLURM_QOS:-normal}"
# Current queue policy: one sequential A100 allocation with enough host memory
# for the validated single-process trainer.  Profile-specific larger historical
# shapes remain available only through explicit PRISM_* overrides.
DEFAULT_PARTITION=a100
DEFAULT_WALLTIME=1-00:00:00
DEFAULT_GPU_TYPE=a100
DEFAULT_GPU_LANES=1
DEFAULT_CPUS_PER_TASK=8
DEFAULT_HOST_MEMORY=80G
DEFAULT_STEP_MEMORY=76G
PARTITION="${PRISM_SLURM_PARTITION:-${DEFAULT_PARTITION}}"
EXCLUDE_NODES="${PRISM_SLURM_EXCLUDE:-}"
WALLTIME="${PRISM_SLURM_WALLTIME:-${DEFAULT_WALLTIME}}"
GPU_TYPE="${PRISM_GPU_TYPE:-${DEFAULT_GPU_TYPE}}"
GPU_LANES="${PRISM_GPU_LANES:-${DEFAULT_GPU_LANES}}"
CPUS_PER_TASK="${PRISM_CPUS_PER_TASK:-${DEFAULT_CPUS_PER_TASK}}"
HOST_MEMORY="${PRISM_SLURM_MEMORY:-${DEFAULT_HOST_MEMORY}}"
STEP_MEMORY="${PRISM_STEP_MEMORY:-${DEFAULT_STEP_MEMORY}}"
GPU_GRES="${PRISM_SLURM_GRES:-gpu:${GPU_TYPE}:${GPU_LANES}}"
STEP_GRES="${PRISM_SLURM_STEP_GRES:-gpu:${GPU_TYPE}:1}"

if [[ "${COVERAGE_PROFILE}" != paper-breadth \
    && "${COVERAGE_PROFILE}" != regime-map \
    && "${COVERAGE_PROFILE}" != baseline-reproduction \
    && "${COVERAGE_PROFILE}" != baseline-reproduction-cached \
    && "${COVERAGE_PROFILE}" != baseline-gap-fill-cached \
    && "${COVERAGE_PROFILE}" != baseline-gap-fill-all-cached \
    && "${COVERAGE_PROFILE}" != baseline-gap-fill-math-only-cached \
    && "${COVERAGE_PROFILE}" != glue-slaclip-screen \
    && "${COVERAGE_PROFILE}" != glue-high-c-refinement \
    && "${COVERAGE_PROFILE}" != glue-r8-slack-screen \
    && "${COVERAGE_PROFILE}" != glue-target-baseline-screen \
    && "${COVERAGE_PROFILE}" != glue-weighted-target-screen \
    && "${COVERAGE_PROFILE}" != glue-quantile-target-screen \
    && "${COVERAGE_PROFILE}" != glue-range-target-screen ]]; then
  echo "error: unsupported PRISM_COVERAGE_PROFILE" >&2
  exit 2
fi
if [[ "${GPU_LANES}" != 1 && "${GPU_LANES}" != 2 ]]; then
  echo "error: PRISM_GPU_LANES must be 1 or 2" >&2
  exit 2
fi
if [[ ( "${COVERAGE_PROFILE}" == glue-high-c-refinement \
      || "${COVERAGE_PROFILE}" == glue-r8-slack-screen \
      || "${COVERAGE_PROFILE}" == glue-target-baseline-screen \
      || "${COVERAGE_PROFILE}" == glue-weighted-target-screen \
      || "${COVERAGE_PROFILE}" == glue-quantile-target-screen \
      || "${COVERAGE_PROFILE}" == glue-range-target-screen \
      || "${COVERAGE_PROFILE}" == baseline-gap-fill-cached \
      || "${COVERAGE_PROFILE}" == baseline-gap-fill-all-cached \
      || "${COVERAGE_PROFILE}" == baseline-gap-fill-math-only-cached ) \
    && "${GPU_LANES}" != 1 ]]; then
  echo "error: ${COVERAGE_PROFILE} requires PRISM_GPU_LANES=1" >&2
  exit 2
fi
for numeric in "${CPUS_PER_TASK}"; do
  [[ "${numeric}" =~ ^[1-9][0-9]*$ ]] || { echo "error: invalid CPU count" >&2; exit 2; }
done
if [[ ! "${CAMPAIGN_ID}" =~ ^paper-coverage-${SHORT_SHA}-[A-Za-z0-9._-]+$ ]]; then
  echo "error: campaign ID must include locked SHA ${SHORT_SHA}" >&2
  exit 2
fi
if [[ -n "${EXCLUDE_NODES}" && "${EXCLUDE_NODES}" == *[!A-Za-z0-9_.,\[\]-]* ]]; then
  echo "error: invalid node exclusion" >&2
  exit 2
fi
[[ -x "${ENV_PREFIX}/bin/python" ]] || { echo "error: environment unavailable: ${ENV_PREFIX}" >&2; exit 2; }
for required in \
  scripts/build_paper_coverage_campaign.py \
  scripts/weighted_target_screen.py \
  scripts/quantile_target_screen.py \
  scripts/range_target_screen.py \
  scripts/prepare_glue_eval_assets.py \
  scripts/preflight_hpc.py \
  scripts/smoke_dp_path.py \
  scripts/summarize_telemetry.py \
  slurm/paper_coverage_campaign.sbatch \
  slurm/paper_coverage_lane.sh; do
  [[ -f "${REPO_ROOT}/${required}" ]] || { echo "error: missing ${required}" >&2; exit 2; }
done

if [[ "${COVERAGE_PROFILE}" != baseline-gap-fill-math-only-cached ]]; then
  "${ENV_PREFIX}/bin/python" "${REPO_ROOT}/scripts/prepare_glue_eval_assets.py" \
    --output-root "${GLUE_EVAL_ROOT}" \
    --cache-dir "${SHARED_HF_HOME}/datasets"
fi

check_model() {
  local model_id="$1" revision="$2"
  local key="models--${model_id//\//--}"
  local snapshot="${SHARED_HF_HOME}/hub/${key}/snapshots/${revision}"
  local marker="${SHARED_HF_HOME}/staged/${key}/${revision}.complete"
  [[ -d "${snapshot}" && -f "${marker}" ]] || {
    echo "error: pinned offline model missing: ${model_id}@${revision}" >&2
    return 2
  }
}
check_model google/gemma-3-4b-pt "${MODEL_4B_REVISION}"
if [[ "${COVERAGE_PROFILE}" != glue-slaclip-screen \
    && "${COVERAGE_PROFILE}" != glue-high-c-refinement \
    && "${COVERAGE_PROFILE}" != glue-r8-slack-screen \
    && "${COVERAGE_PROFILE}" != glue-target-baseline-screen \
    && "${COVERAGE_PROFILE}" != glue-weighted-target-screen \
    && "${COVERAGE_PROFILE}" != glue-quantile-target-screen \
    && "${COVERAGE_PROFILE}" != glue-range-target-screen \
    && "${COVERAGE_PROFILE}" != baseline-gap-fill-cached \
    && "${COVERAGE_PROFILE}" != baseline-gap-fill-all-cached \
    && "${COVERAGE_PROFILE}" != baseline-gap-fill-math-only-cached ]]; then
  check_model google/gemma-2-9b "${MODEL_9B_REVISION}"
fi
if [[ "${COVERAGE_PROFILE}" == baseline-reproduction \
    || "${COVERAGE_PROFILE}" == baseline-gap-fill-all-cached \
    || "${COVERAGE_PROFILE}" == baseline-gap-fill-math-only-cached ]]; then
  if ! check_model google/gemma-3-12b-pt "${MODEL_12B_REVISION}"; then
    if [[ "${MODE}" == --test-only ]]; then
      echo "warning: scheduler test continues without the gated 12B snapshot" >&2
    else
      echo "error: ${COVERAGE_PROFILE} requires the pinned 12B snapshot" >&2
      exit 2
    fi
  fi
fi

mkdir -p "$(dirname -- "${STAGED_REPO_ROOT}")"
command -v flock >/dev/null || { echo "error: flock is required" >&2; exit 2; }
exec 9>"${STAGED_REPO_ROOT}.lock"
flock 9
if [[ ! -e "${STAGED_REPO_ROOT}" ]]; then
  STAGED_TMP="$(mktemp -d "$(dirname -- "${STAGED_REPO_ROOT}")/.PRISM-DP-LoRA-${LOCKED_REPO_SHA}.XXXXXX")"
  if ! git clone --quiet --no-hardlinks --no-checkout "${REPO_ROOT}" "${STAGED_TMP}" \
      || ! git -C "${STAGED_TMP}" checkout --quiet --detach "${LOCKED_REPO_SHA}"; then
    rm -rf -- "${STAGED_TMP}"
    echo "error: immutable source staging failed" >&2
    exit 2
  fi
  origin="$(git -C "${REPO_ROOT}" remote get-url origin 2>/dev/null || true)"
  [[ -z "${origin}" ]] || git -C "${STAGED_TMP}" remote set-url origin "${origin}"
  mv -- "${STAGED_TMP}" "${STAGED_REPO_ROOT}"
fi
if [[ "$(git -C "${STAGED_REPO_ROOT}" rev-parse HEAD)" != "${LOCKED_REPO_SHA}" \
    || -n "$(git -C "${STAGED_REPO_ROOT}" status --porcelain)" ]]; then
  echo "error: staged source is dirty or at the wrong SHA" >&2
  exit 2
fi
flock -u 9
exec 9>&-

CAMPAIGN_ROOT="${RUN_ROOT}/campaigns/${CAMPAIGN_ID}"
LOG_ROOT="${CAMPAIGN_ROOT}/slurm"
RECEIPT="${CAMPAIGN_ROOT}/submission_receipt.json"
mkdir -p "${LOG_ROOT}"
WORKER="${STAGED_REPO_ROOT}/slurm/paper_coverage_campaign.sbatch"

# Materialize and validate the immutable plan on the login node before asking
# for queued GPU time. The compute job repeats this idempotently. For the
# focused screen this also reads, hashes, and recomputes the locked calibration
# source, so a missing/stale source cannot fail only after the queue wait.
"${ENV_PREFIX}/bin/python" \
  "${STAGED_REPO_ROOT}/scripts/build_paper_coverage_campaign.py" prepare \
  --campaign-root "${CAMPAIGN_ROOT}" \
  --code-sha "${LOCKED_REPO_SHA}" \
  --model-4b-revision "${MODEL_4B_REVISION}" \
  --model-9b-revision "${MODEL_9B_REVISION}" \
  --model-12b-revision "${MODEL_12B_REVISION}" \
  --profile "${COVERAGE_PROFILE}"

SBATCH_ARGS=(
  --account="${ACCOUNT}"
  --qos="${QOS}"
  --partition="${PARTITION}"
  --nodes=1
  --ntasks="${GPU_LANES}"
  --cpus-per-task="${CPUS_PER_TASK}"
  --mem="${HOST_MEMORY}"
  --time="${WALLTIME}"
  --gres="${GPU_GRES}"
  --job-name=prism-paper-cover
  --output="${LOG_ROOT}/%x-%j.out"
  --error="${LOG_ROOT}/%x-%j.err"
  --chdir="${STAGED_REPO_ROOT}"
  --export=NONE
  --no-requeue
  # Without B:, Slurm signals every active job step. This lets the lane and
  # trainer terminate before the hard limit; B: would signal only the batch
  # shell while it is blocked waiting for srun.
  --signal=TERM@180
)
[[ -z "${EXCLUDE_NODES}" ]] || SBATCH_ARGS+=(--exclude="${EXCLUDE_NODES}")
WORKER_ARGS=(
  "${USER_NAME}" "${STAGED_REPO_ROOT}" "${ENV_PREFIX}" "${RUN_ROOT}"
  "${SHARED_HF_HOME}" "${LOCKED_REPO_SHA}" "${CAMPAIGN_ID}" "${SCRATCH_ROOT}"
  "${MODEL_4B_REVISION}" "${MODEL_9B_REVISION}"
  "${GLUE_EVAL_ROOT}"
  "${COVERAGE_PROFILE}"
  "${MODEL_12B_REVISION}"
  "${STEP_GRES}"
  "${GPU_LANES}"
  "${STEP_MEMORY}"
  "${CPUS_PER_TASK}"
)

if [[ "${MODE}" == --test-only ]]; then
  sbatch --test-only "${SBATCH_ARGS[@]}" "${WORKER}" "${WORKER_ARGS[@]}"
  echo "scheduler_test=accepted resources=${GPU_GRES},tasks=${GPU_LANES},cpus_per_task=${CPUS_PER_TASK},mem=${HOST_MEMORY},time=${WALLTIME} campaign=${CAMPAIGN_ID}"
  exit 0
fi

old_job=""
old_state=""
if [[ -f "${RECEIPT}" ]]; then
  old_job="$("${ENV_PREFIX}/bin/python" - "${RECEIPT}" <<'PY'
import json, sys
print(json.load(open(sys.argv[1], encoding="utf-8")).get("current_job_id", ""))
PY
)"
  if [[ -n "${old_job}" ]]; then
    old_state="$(squeue -h -j "${old_job}" -o '%T' 2>/dev/null | head -1 || true)"
    if [[ -z "${old_state}" ]]; then
      old_state="$(sacct -X -n -P -j "${old_job}" -o JobIDRaw,State 2>/dev/null | awk -F'|' -v j="${old_job}" '$1==j {print $2; exit}')"
    fi
    old_state="${old_state%%+*}"
    old_state="${old_state%% *}"
  fi
fi
if [[ "${MODE}" == --submit && -n "${old_job}" ]]; then
  echo "error: campaign already submitted as job ${old_job} (${old_state:-unknown})" >&2
  exit 2
fi
if [[ "${MODE}" == --resume-submit ]]; then
  case "${old_state}" in
    FAILED|TIMEOUT|CANCELLED|OUT_OF_MEMORY|NODE_FAIL|PREEMPTED|BOOT_FAIL|DEADLINE|REVOKED) ;;
    *) echo "error: prior campaign is not terminally unsuccessful" >&2; exit 2 ;;
  esac
fi

submitted="$(sbatch --parsable "${SBATCH_ARGS[@]}" "${WORKER}" "${WORKER_ARGS[@]}")"
job_id="${submitted%%;*}"
[[ "${job_id}" =~ ^[0-9]+$ ]] || { echo "error: invalid job ID: ${submitted}" >&2; exit 2; }
"${ENV_PREFIX}/bin/python" - "${RECEIPT}" "${job_id}" "${old_job}" "${old_state}" \
  "${CAMPAIGN_ID}" "${CAMPAIGN_ROOT}" "${LOCKED_REPO_SHA}" "${STAGED_REPO_ROOT}" \
  "${ENV_PREFIX}" "${ACCOUNT}" "${QOS}" "${PARTITION}" "${GLUE_EVAL_ROOT}" \
  "${COVERAGE_PROFILE}" "${MODEL_12B_REVISION}" "${GPU_GRES}" \
  "${GPU_LANES}" "${CPUS_PER_TASK}" "${HOST_MEMORY}" "${WALLTIME}" \
  "${STEP_GRES}" "${STEP_MEMORY}" <<'PY'
import json, os, sys, tempfile
from datetime import datetime, timezone
from pathlib import Path

(target, job, previous, previous_state, campaign, root, sha, source, env,
 account, qos, partition, glue_eval_root, coverage_profile, model_12b_revision,
 gpu_gres, gpu_lanes, cpus_per_task, host_memory, walltime, step_gres,
 step_memory) = sys.argv[1:]
payload = {
    "schema_version": 1,
    "campaign_id": campaign,
    "campaign_root": root,
    "code_sha": sha,
    "staged_repo": source,
    "environment": env,
    "current_job_id": job,
    "previous_job_id": previous or None,
    "previous_state": previous_state or None,
    "submitted_at": datetime.now(timezone.utc).isoformat(),
    "models": {
        "google/gemma-3-4b-pt": "cc012e0a6d0787b4adcc0fa2c4da74402494554d",
        "google/gemma-2-9b": "33c193028431c2fde6c6e51f29e6f17b60cbfac6",
        "google/gemma-3-12b-pt": (
            model_12b_revision
            if coverage_profile in {
                "baseline-reproduction", "baseline-gap-fill-all-cached",
                "baseline-gap-fill-math-only-cached"
            }
            else None
        ),
    },
    "glue_eval_assets": glue_eval_root,
    "coverage_profile": coverage_profile,
    "resources": {
        "account": account, "qos": qos, "partition": partition,
        "nodes": 1, "tasks": int(gpu_lanes),
        "cpus_per_task": int(cpus_per_task), "memory": host_memory,
        "gpus": gpu_gres, "walltime": walltime,
        "step_gres": step_gres, "step_memory": step_memory,
        "single_allocation": True, "array": False, "requeue": False,
    },
}
path = Path(target); path.parent.mkdir(parents=True, exist_ok=True)
fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
try:
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True); handle.write("\n")
        handle.flush(); os.fsync(handle.fileno())
    os.chmod(name, 0o600); os.replace(name, path)
finally:
    Path(name).unlink(missing_ok=True)
PY

echo "submitted_job_id=${job_id}"
echo "campaign_root=${CAMPAIGN_ROOT}"
echo "slurm_stdout=${LOG_ROOT}/prism-paper-cover-${job_id}.out"
echo "slurm_stderr=${LOG_ROOT}/prism-paper-cover-${job_id}.err"
echo "resources=${GPU_GRES},tasks=${GPU_LANES},cpus_per_task=${CPUS_PER_TASK},mem=${HOST_MEMORY},time=${WALLTIME},one_allocation"
