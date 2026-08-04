#!/usr/bin/env bash

# Submit one portable two-H200 allocation for the complete 4B PRISM/SlaClip
# screening, lock, and fresh-seed confirmation campaign.  This wrapper performs
# only submission-time validation and immutable source staging.  All GPU smoke
# tests and experiment arms execute inside the single allocation.
set -Eeuo pipefail
umask 077

usage() {
  cat <<'EOF'
Usage:
  scripts/submit_math10k_4b_dynamics_campaign.sh --test-only
  scripts/submit_math10k_4b_dynamics_campaign.sh --submit
  scripts/submit_math10k_4b_dynamics_campaign.sh --resume-submit

Run --test-only first. It asks Slurm to validate the exact allocation without
queuing work. --submit creates one queued campaign containing all smoke,
screening, selection-seed confirmation, replay/matched-fixed controls, and
fresh-seed full-data task evaluation arms.

--resume-submit is accepted only after the campaign's previously receipted
Slurm job has reached a failed/timeout terminal state. It resumes the same
fingerprinted output directories and is rejected while an older job is active
or after the campaign has completed.

Portable path overrides:
  PRISM_USER_NAME, PRISM_USER_HOME, PRISM_SCRATCH_ROOT, PRISM_REPO_ROOT,
  PRISM_STAGED_REPO_ROOT, PRISM_ENV_PREFIX, PRISM_ENV_TAG, PRISM_RUN_ROOT,
  PRISM_HF_HOME, PRISM_CODE_REVISION, PRISM_CAMPAIGN_ID.

Scheduler overrides:
  PRISM_SLURM_ACCOUNT, PRISM_SLURM_QOS, PRISM_SLURM_PARTITION,
  PRISM_GPU_GRES, PRISM_CPUS_PER_TASK, PRISM_HOST_MEMORY, PRISM_WALLTIME.

Model overrides are intentionally not accepted: this protocol is locked to the
pinned Gemma-3-4B checkpoint declared below.
EOF
}

MODE="${1:-}"
case "${MODE}" in
  --test-only|--submit|--resume-submit) ;;
  -h|--help)
    usage
    exit 0
    ;;
  *)
    echo "error: choose exactly one of --test-only, --submit, or --resume-submit" >&2
    usage >&2
    exit 2
    ;;
esac

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

USER_NAME="${PRISM_USER_NAME:-${USER:-}}"
if [[ -z "${USER_NAME}" ]] || ! getent passwd "${USER_NAME}" >/dev/null; then
  USER_NAME="$(id -un)"
fi
USER_HOME="${PRISM_USER_HOME:-$(getent passwd "${USER_NAME}" | cut -d: -f6)}"
if [[ -z "${USER_HOME}" || ! -d "${USER_HOME}" ]]; then
  echo "error: could not resolve a home directory for ${USER_NAME}" >&2
  exit 2
fi

SCRATCH_ROOT="${PRISM_SCRATCH_ROOT:-${SCRATCH:-/scratch/${USER_NAME}}}"
REPO_ROOT="${PRISM_REPO_ROOT:-${DEFAULT_REPO_ROOT}}"
if [[ ! -d "${REPO_ROOT}/.git" ]]; then
  echo "error: PRISM repository not found: ${REPO_ROOT}" >&2
  exit 2
fi
REPO_ROOT="$(cd -- "${REPO_ROOT}" && pwd)"

REQUESTED_REVISION="${PRISM_CODE_REVISION:-HEAD}"
LOCKED_REPO_SHA="$(git -C "${REPO_ROOT}" rev-parse "${REQUESTED_REVISION}^{commit}")"
if [[ ! "${LOCKED_REPO_SHA}" =~ ^[0-9a-f]{40}$ ]]; then
  echo "error: could not resolve a full Git SHA from ${REQUESTED_REVISION}" >&2
  exit 2
fi
if [[ "$(git -C "${REPO_ROOT}" rev-parse HEAD)" != "${LOCKED_REPO_SHA}" ]]; then
  echo "error: requested revision is not the checked-out HEAD" >&2
  echo "expected=${LOCKED_REPO_SHA} actual=$(git -C "${REPO_ROOT}" rev-parse HEAD)" >&2
  exit 2
fi
if [[ -n "$(git -C "${REPO_ROOT}" status --porcelain)" ]]; then
  echo "error: refusing to stage a formal campaign from a dirty worktree" >&2
  git -C "${REPO_ROOT}" status --short >&2
  exit 2
fi

SHORT_SHA="${LOCKED_REPO_SHA:0:12}"
ENV_TAG="${PRISM_ENV_TAG:-71bce55}"
ENV_PREFIX="${PRISM_ENV_PREFIX:-${SCRATCH_ROOT}/envs/prism-dp-lora-${ENV_TAG}}"
RUN_ROOT="${PRISM_RUN_ROOT:-${SCRATCH_ROOT}/runs/prism-dp-lora}"
SHARED_HF_HOME="${PRISM_HF_HOME:-${RUN_ROOT}/cache/huggingface}"
CAMPAIGN_ID="${PRISM_CAMPAIGN_ID:-math10k-4b-dynamics-${SHORT_SHA}-v1}"
STAGED_REPO_ROOT="${PRISM_STAGED_REPO_ROOT:-${RUN_ROOT}/sources/PRISM-DP-LoRA-${LOCKED_REPO_SHA}}"

ACCOUNT="${PRISM_SLURM_ACCOUNT:-normal}"
QOS="${PRISM_SLURM_QOS:-normal}"
PARTITION="${PRISM_SLURM_PARTITION:-quad_h200,dual_h200}"
GPU_GRES="${PRISM_GPU_GRES:-gpu:h200:2}"
CPUS_PER_TASK="${PRISM_CPUS_PER_TASK:-8}"
HOST_MEMORY="${PRISM_HOST_MEMORY:-256G}"
WALLTIME="${PRISM_WALLTIME:-2-12:00:00}"

MODEL_ID="google/gemma-3-4b-pt"
MODEL_REVISION="cc012e0a6d0787b4adcc0fa2c4da74402494554d"
MODEL_SLUG="gemma-3-4b-pt"

if [[ ! "${CAMPAIGN_ID}" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "error: PRISM_CAMPAIGN_ID contains unsupported characters" >&2
  exit 2
fi
if [[ ! "${CPUS_PER_TASK}" =~ ^[1-9][0-9]*$ ]]; then
  echo "error: PRISM_CPUS_PER_TASK must be a positive integer" >&2
  exit 2
fi
if (( CPUS_PER_TASK * 2 != 16 )); then
  echo "error: this reviewed protocol requires two lanes x 8 CPUs" >&2
  exit 2
fi
if [[ "${GPU_GRES}" != "gpu:h200:2" ]]; then
  echo "error: this reviewed protocol requires PRISM_GPU_GRES=gpu:h200:2" >&2
  exit 2
fi
if [[ "${HOST_MEMORY}" != "256G" ]]; then
  echo "error: this reviewed protocol requires total PRISM_HOST_MEMORY=256G" >&2
  echo "the worker reserves two concurrent lanes of 128G each" >&2
  exit 2
fi
if [[ ! -x "${ENV_PREFIX}/bin/python" ]]; then
  echo "error: Python environment is unavailable: ${ENV_PREFIX}" >&2
  exit 2
fi

for tracked_file in \
  train_eval.py \
  scripts/preflight_hpc.py \
  scripts/smoke_dp_path.py \
  scripts/select_validation_candidates.py \
  scripts/summarize_telemetry.py \
  slurm/math10k_4b_dynamics_campaign.sbatch; do
  if [[ ! -f "${REPO_ROOT}/${tracked_file}" ]]; then
    echo "error: required tracked file is missing: ${tracked_file}" >&2
    exit 2
  fi
done

cache_key="models--${MODEL_ID//\//--}"
snapshot="${SHARED_HF_HOME}/hub/${cache_key}/snapshots/${MODEL_REVISION}"
marker="${SHARED_HF_HOME}/staged/${cache_key}/${MODEL_REVISION}.complete"
if [[ ! -d "${snapshot}" || ! -f "${marker}" ]]; then
  if [[ "${MODE}" == "--test-only" ]]; then
    echo "warning: pinned model is not fully staged; scheduler validation can continue" >&2
    echo "missing snapshot=${snapshot} or marker=${marker}" >&2
  else
    echo "error: pinned offline model is not staged: ${MODEL_ID}@${MODEL_REVISION}" >&2
    exit 2
  fi
fi

# Build an independent SHA-named clone.  It is safe if the development checkout
# changes while this job waits in the queue.
mkdir -p "$(dirname -- "${STAGED_REPO_ROOT}")"
if ! command -v flock >/dev/null 2>&1; then
  echo "error: flock is required for atomic source staging" >&2
  exit 2
fi
exec 9>"${STAGED_REPO_ROOT}.lock"
flock 9
if [[ ! -e "${STAGED_REPO_ROOT}" ]]; then
  STAGED_TMP="$(mktemp -d "$(dirname -- "${STAGED_REPO_ROOT}")/.PRISM-DP-LoRA-${LOCKED_REPO_SHA}.XXXXXX")"
  if ! git clone --quiet --no-hardlinks --no-checkout "${REPO_ROOT}" "${STAGED_TMP}" \
      || ! git -C "${STAGED_TMP}" checkout --quiet --detach "${LOCKED_REPO_SHA}"; then
    rm -rf -- "${STAGED_TMP}"
    echo "error: could not create immutable source staging" >&2
    exit 2
  fi
  SOURCE_ORIGIN="$(git -C "${REPO_ROOT}" remote get-url origin 2>/dev/null || true)"
  if [[ -n "${SOURCE_ORIGIN}" ]]; then
    git -C "${STAGED_TMP}" remote set-url origin "${SOURCE_ORIGIN}"
  fi
  mv -- "${STAGED_TMP}" "${STAGED_REPO_ROOT}"
fi
if [[ ! -d "${STAGED_REPO_ROOT}/.git" ]]; then
  echo "error: staged source is not an independent Git checkout" >&2
  exit 2
fi
if [[ "$(git -C "${STAGED_REPO_ROOT}" rev-parse HEAD)" != "${LOCKED_REPO_SHA}" ]]; then
  echo "error: staged source revision does not match ${LOCKED_REPO_SHA}" >&2
  exit 2
fi
if [[ -n "$(git -C "${STAGED_REPO_ROOT}" status --porcelain)" ]]; then
  echo "error: staged source is dirty: ${STAGED_REPO_ROOT}" >&2
  exit 2
fi
WORKER="${STAGED_REPO_ROOT}/slurm/math10k_4b_dynamics_campaign.sbatch"
if [[ ! -f "${WORKER}" ]]; then
  echo "error: staged revision does not contain the campaign worker" >&2
  exit 2
fi
flock -u 9
exec 9>&-

CAMPAIGN_ROOT="${RUN_ROOT}/campaigns/${CAMPAIGN_ID}"
LOG_ROOT="${CAMPAIGN_ROOT}/slurm"
mkdir -p "${LOG_ROOT}"
SUBMISSION_RECEIPT="${CAMPAIGN_ROOT}/submission_receipt.json"
SUBMISSION_LOCK="${CAMPAIGN_ROOT}/.submission.lock"
RECEIPT_LOCKED=0
OLD_JOB_ID=""
OLD_JOB_STATE=""

receipt_transition() {
  local action="$1"
  local mode="$2"
  local job_id="${3:-}"
  local scheduler_result="${4:-}"
  local previous_job_id="${5:-}"
  local previous_job_state="${6:-}"
  "${ENV_PREFIX}/bin/python" - \
    "${SUBMISSION_RECEIPT}" "${action}" "${mode}" "${job_id}" \
    "${scheduler_result}" "${previous_job_id}" "${previous_job_state}" \
    "${CAMPAIGN_ID}" "${CAMPAIGN_ROOT}" "${LOCKED_REPO_SHA}" \
    "${STAGED_REPO_ROOT}" "${ENV_PREFIX}" "${MODEL_ID}" "${MODEL_REVISION}" \
    "${ACCOUNT}" "${QOS}" "${PARTITION}" "${GPU_GRES}" \
    "${CPUS_PER_TASK}" "${HOST_MEMORY}" "${WALLTIME}" <<'PY'
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

(
    target_text,
    action,
    mode,
    job_id,
    scheduler_result,
    previous_job_id,
    previous_job_state,
    campaign_id,
    campaign_root,
    code_sha,
    staged_repo,
    environment,
    model_id,
    model_revision,
    account,
    qos,
    partition,
    gpu_gres,
    cpus_per_task,
    host_memory,
    walltime,
) = sys.argv[1:]
target = Path(target_text)
now = datetime.now(timezone.utc).isoformat()
if target.exists():
    payload = json.loads(target.read_text(encoding="utf-8"))
    if payload.get("campaign_id") != campaign_id or payload.get("code_sha") != code_sha:
        raise SystemExit(f"submission receipt identity mismatch: {target}")
else:
    payload = {
        "schema_version": 1,
        "campaign_id": campaign_id,
        "campaign_root": campaign_root,
        "code_sha": code_sha,
        "staged_repo": staged_repo,
        "environment": environment,
        "model_id": model_id,
        "model_revision": model_revision,
        "resources": {
            "account": account,
            "qos": qos,
            "partition": partition,
            "gpu_gres": gpu_gres,
            "nodes": 1,
            "lanes": 2,
            "cpus_per_lane": int(cpus_per_task),
            "total_memory": host_memory,
            "memory_per_lane": "128G",
            "walltime": walltime,
        },
        "attempts": [],
    }
attempts = payload.get("attempts")
if not isinstance(attempts, list):
    raise SystemExit(f"submission receipt attempts are malformed: {target}")
if action == "begin":
    attempt = {
        "attempt": len(attempts) + 1,
        "mode": mode,
        "state": "submitting",
        "requested_at": now,
        "previous_job_id": previous_job_id or None,
        "previous_job_terminal_state": previous_job_state or None,
    }
    attempts.append(attempt)
    payload["state"] = "submitting"
    payload["current_job_id"] = None
elif action in {"submitted", "submission_failed", "submission_unknown"}:
    if not attempts or attempts[-1].get("state") != "submitting":
        raise SystemExit(f"submission receipt has no active attempt to finalize: {target}")
    attempt = attempts[-1]
    attempt["state"] = action
    attempt["finished_at"] = now
    attempt["job_id"] = job_id or None
    attempt["scheduler_result"] = scheduler_result
    payload["state"] = action
    payload["current_job_id"] = job_id or None
else:
    raise SystemExit(f"unknown receipt transition: {action}")
payload["updated_at"] = now
encoded = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
target.parent.mkdir(parents=True, exist_ok=True)
descriptor, name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
temporary = Path(name)
try:
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, target)
    os.chmod(target, 0o600)
finally:
    temporary.unlink(missing_ok=True)
PY
}

read_receipt_state() {
  "${ENV_PREFIX}/bin/python" - "${SUBMISSION_RECEIPT}" "${CAMPAIGN_ID}" "${LOCKED_REPO_SHA}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.is_file():
    raise SystemExit(f"submission receipt does not exist: {path}")
payload = json.loads(path.read_text(encoding="utf-8"))
if payload.get("campaign_id") != sys.argv[2] or payload.get("code_sha") != sys.argv[3]:
    raise SystemExit(f"submission receipt identity mismatch: {path}")
print(str(payload.get("state") or ""))
print(str(payload.get("current_job_id") or ""))
PY
}

if [[ "${MODE}" != "--test-only" ]]; then
  exec 8>"${SUBMISSION_LOCK}"
  flock 8
  RECEIPT_LOCKED=1
  completed_status=""
  existing_status=""
  for status_file in "${CAMPAIGN_ROOT}"/status/job-*.txt; do
    [[ -e "${status_file}" ]] || continue
    existing_status="${status_file}"
    if grep -qx 'state=completed' "${status_file}"; then
      completed_status="${status_file}"
      break
    fi
  done
  if [[ -n "${completed_status}" || -e "${CAMPAIGN_ROOT}/artifacts/manifest.json" ]]; then
    echo "error: refusing to submit an already completed campaign: ${CAMPAIGN_ROOT}" >&2
    exit 2
  fi
  if [[ "${MODE}" == "--submit" ]]; then
    if [[ -e "${SUBMISSION_RECEIPT}" || -n "${existing_status}" ]]; then
      echo "error: campaign already has submission state; refusing a duplicate job" >&2
      echo "use --resume-submit only after the receipted job has failed or timed out" >&2
      exit 2
    fi
    receipt_transition begin submit
  else
    if [[ ! -e "${SUBMISSION_RECEIPT}" ]]; then
      echo "error: --resume-submit requires an existing submission receipt" >&2
      exit 2
    fi
    mapfile -t RECEIPT_STATE < <(read_receipt_state)
    RECEIPT_PHASE="${RECEIPT_STATE[0]:-}"
    OLD_JOB_ID="${RECEIPT_STATE[1]:-}"
    if [[ -z "${OLD_JOB_ID}" ]]; then
      if [[ "${RECEIPT_PHASE}" != "submission_failed" ]]; then
        echo "error: prior submission has no verifiable failed job ID/state" >&2
        exit 2
      fi
      OLD_JOB_STATE="SUBMISSION_FAILED"
    else
      if [[ ! "${OLD_JOB_ID}" =~ ^[0-9]+$ ]]; then
        echo "error: invalid prior job ID in receipt: ${OLD_JOB_ID}" >&2
        exit 2
      fi
      ACTIVE_JOB="$(squeue -h -j "${OLD_JOB_ID}" -o '%i|%T' 2>/dev/null || true)"
      if [[ -n "${ACTIVE_JOB}" ]]; then
        echo "error: prior job is still active; refusing duplicate resume: ${ACTIVE_JOB}" >&2
        exit 2
      fi
      OLD_JOB_STATE="$(
        sacct -nX -j "${OLD_JOB_ID}" --format=JobIDRaw,State -P 2>/dev/null \
          | awk -F'|' -v wanted="${OLD_JOB_ID}" '$1 == wanted {print $2; exit}'
      )"
      OLD_JOB_STATE="${OLD_JOB_STATE%%+}"
      case "${OLD_JOB_STATE}" in
        FAILED|TIMEOUT|CANCELLED|NODE_FAIL|OUT_OF_MEMORY|PREEMPTED|BOOT_FAIL|DEADLINE|REVOKED) ;;
        COMPLETED)
          echo "error: prior job completed; a resume submission is forbidden" >&2
          exit 2
          ;;
        *)
          echo "error: prior job has no allowed failed terminal state: ${OLD_JOB_STATE:-unknown}" >&2
          exit 2
          ;;
      esac
    fi
    receipt_transition begin resume-submit "" "" "${OLD_JOB_ID}" "${OLD_JOB_STATE}"
  fi
fi

SBATCH_ARGS=(
  --parsable
  --job-name=prism-4b-dynamics
  --account="${ACCOUNT}"
  --qos="${QOS}"
  --partition="${PARTITION}"
  --nodes=1
  --ntasks=2
  --cpus-per-task="${CPUS_PER_TASK}"
  --mem="${HOST_MEMORY}"
  --gres="${GPU_GRES}"
  --time="${WALLTIME}"
  --export=NONE
  --output="${LOG_ROOT}/%x-%j.out"
  --error="${LOG_ROOT}/%x-%j.err"
)
if [[ "${MODE}" == "--test-only" ]]; then
  SBATCH_ARGS+=(--test-only)
fi

echo "mode=${MODE}"
echo "campaign_id=${CAMPAIGN_ID}"
echo "campaign_root=${CAMPAIGN_ROOT}"
echo "development_repo=${REPO_ROOT}"
echo "staged_repo=${STAGED_REPO_ROOT}"
echo "code_sha=${LOCKED_REPO_SHA}"
echo "environment=${ENV_PREFIX}"
echo "model=${MODEL_ID}@${MODEL_REVISION}"
echo "resources=account:${ACCOUNT},qos:${QOS},partition:${PARTITION},2xH200,16cpu,${HOST_MEMORY},${WALLTIME}"
echo "protocol=stage1_grid_then_multiseed_lock_then_fresh_seed_final_controls"
echo "privacy_warning=research_raw artifacts are NON_PRIVATE and remain under scratch with umask 077"

SUBMISSION_RC=0
SUBMISSION_OUTPUT="$(sbatch "${SBATCH_ARGS[@]}" "${WORKER}" \
  "${USER_NAME}" \
  "${STAGED_REPO_ROOT}" \
  "${ENV_PREFIX}" \
  "${RUN_ROOT}" \
  "${SHARED_HF_HOME}" \
  "${LOCKED_REPO_SHA}" \
  "${CAMPAIGN_ID}" \
  "${SCRATCH_ROOT}" \
  "${MODEL_ID}" \
  "${MODEL_REVISION}" \
  "${MODEL_SLUG}" 2>&1)" || SUBMISSION_RC=$?
echo "scheduler_result=${SUBMISSION_OUTPUT}"
if [[ "${SUBMISSION_RC}" != "0" ]]; then
  if [[ "${MODE}" != "--test-only" ]]; then
    receipt_transition submission_failed "${MODE#--}" "" "${SUBMISSION_OUTPUT}" \
      "${OLD_JOB_ID}" "${OLD_JOB_STATE}"
  fi
  echo "error: sbatch failed with exit code ${SUBMISSION_RC}" >&2
  exit "${SUBMISSION_RC}"
fi
if [[ "${MODE}" != "--test-only" ]]; then
  SUBMITTED_JOB_ID="${SUBMISSION_OUTPUT%%;*}"
  if [[ ! "${SUBMITTED_JOB_ID}" =~ ^[0-9]+$ ]]; then
    receipt_transition submission_unknown "${MODE#--}" "" "${SUBMISSION_OUTPUT}" \
      "${OLD_JOB_ID}" "${OLD_JOB_STATE}"
    echo "error: could not parse Slurm job ID from: ${SUBMISSION_OUTPUT}" >&2
    exit 3
  fi
  receipt_transition submitted "${MODE#--}" "${SUBMITTED_JOB_ID}" \
    "${SUBMISSION_OUTPUT}" "${OLD_JOB_ID}" "${OLD_JOB_STATE}"
  echo "submitted_job_id=${SUBMITTED_JOB_ID}"
  if (( RECEIPT_LOCKED == 1 )); then
    flock -u 8
    exec 8>&-
    RECEIPT_LOCKED=0
  fi
  echo "submission_receipt=${SUBMISSION_RECEIPT}"
else
  echo "scheduler_validation=test-only"
fi
