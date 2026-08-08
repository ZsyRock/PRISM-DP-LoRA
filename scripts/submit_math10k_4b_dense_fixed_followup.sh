#!/usr/bin/env bash

# Submit the predeclared, sequential one-A100 Math10K fixed-control follow-up.
# The queued worker first finishes any missing fresh-seed reference arms and
# then evaluates all fixed controls in the same allocation.  No child Slurm
# jobs are created by this wrapper or by the worker.
set -Eeuo pipefail
umask 077

usage() {
  cat <<'EOF'
Usage:
  scripts/submit_math10k_4b_dense_fixed_followup.sh --test-only
  scripts/submit_math10k_4b_dense_fixed_followup.sh --submit
  scripts/submit_math10k_4b_dense_fixed_followup.sh --resume-submit

--test-only validates the exact Slurm request without queuing it. --submit
queues one job, dependent on the reference refinement job, containing every
sequential experiment arm. --resume-submit is accepted only when the prior
follow-up job is terminally unsuccessful and the campaign is not complete.

Portable path/identity overrides:
  PRISM_USER_NAME, PRISM_USER_HOME, PRISM_SCRATCH_ROOT, PRISM_REPO_ROOT,
  PRISM_STAGED_REPO_ROOT, PRISM_EXPERIMENT_STAGED_REPO_ROOT,
  PRISM_ENV_PREFIX, PRISM_ENV_TAG, PRISM_RUN_ROOT, PRISM_HF_HOME,
  PRISM_CODE_REVISION, PRISM_CAMPAIGN_ID, PRISM_SOURCE_CAMPAIGN_ROOT
  (alias PRISM_REFERENCE_CAMPAIGN_ROOT), PRISM_SOURCE_JOB_ID
  (alias PRISM_REFERENCE_JOB_ID), PRISM_EXPERIMENT_CODE_SHA.

Scheduler identity overrides:
  PRISM_SLURM_ACCOUNT, PRISM_SLURM_QOS, PRISM_SLURM_PARTITION,
  PRISM_SLURM_EXCLUDE (optional comma-separated node expression).

The model revision and one-A100 resource envelope are protocol locked.
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
ORCHESTRATOR_SHA="$(git -C "${REPO_ROOT}" rev-parse "${REQUESTED_REVISION}^{commit}")"
if [[ ! "${ORCHESTRATOR_SHA}" =~ ^[0-9a-f]{40}$ ]]; then
  echo "error: could not resolve a full Git SHA from ${REQUESTED_REVISION}" >&2
  exit 2
fi
if [[ "$(git -C "${REPO_ROOT}" rev-parse HEAD)" != "${ORCHESTRATOR_SHA}" ]]; then
  echo "error: requested revision is not the checked-out HEAD" >&2
  exit 2
fi
if [[ -n "$(git -C "${REPO_ROOT}" status --porcelain)" ]]; then
  echo "error: refusing to stage a formal campaign from a dirty worktree" >&2
  git -C "${REPO_ROOT}" status --short >&2
  exit 2
fi

SHORT_SHA="${ORCHESTRATOR_SHA:0:12}"
EXPERIMENT_SHA="${PRISM_EXPERIMENT_CODE_SHA:-feba5968285dc1651bf7726327932b3f625ace22}"
if [[ ! "${EXPERIMENT_SHA}" =~ ^[0-9a-f]{40}$ ]]; then
  echo "error: PRISM_EXPERIMENT_CODE_SHA must be a full Git SHA" >&2
  exit 2
fi

ENV_TAG="${PRISM_ENV_TAG:-71bce55}"
ENV_PREFIX="${PRISM_ENV_PREFIX:-${SCRATCH_ROOT}/envs/prism-dp-lora-${ENV_TAG}}"
RUN_ROOT="${PRISM_RUN_ROOT:-${SCRATCH_ROOT}/runs/prism-dp-lora}"
SHARED_HF_HOME="${PRISM_HF_HOME:-${RUN_ROOT}/cache/huggingface}"
CAMPAIGN_ID="${PRISM_CAMPAIGN_ID:-math10k-4b-dense-fixed-followup-${SHORT_SHA}-v1}"
STAGED_REPO_ROOT="${PRISM_STAGED_REPO_ROOT:-${RUN_ROOT}/sources/PRISM-DP-LoRA-${ORCHESTRATOR_SHA}}"
EXPERIMENT_STAGED_REPO_ROOT="${PRISM_EXPERIMENT_STAGED_REPO_ROOT:-${RUN_ROOT}/sources/PRISM-DP-LoRA-${EXPERIMENT_SHA}}"
REFERENCE_CAMPAIGN_ROOT="${PRISM_SOURCE_CAMPAIGN_ROOT:-${PRISM_REFERENCE_CAMPAIGN_ROOT:-${RUN_ROOT}/campaigns/math10k-4b-c1-c2-refinement-feba5968285d-v1}}"
REFERENCE_JOB_ID="${PRISM_SOURCE_JOB_ID:-${PRISM_REFERENCE_JOB_ID:-1365564}}"
DEPENDENCY_SPEC="afterany:${REFERENCE_JOB_ID}"

ACCOUNT="${PRISM_SLURM_ACCOUNT:-normal}"
QOS="${PRISM_SLURM_QOS:-normal}"
PARTITION="${PRISM_SLURM_PARTITION:-a100}"
EXCLUDE_NODES="${PRISM_SLURM_EXCLUDE:-}"

MODEL_ID="google/gemma-3-4b-pt"
MODEL_REVISION="cc012e0a6d0787b4adcc0fa2c4da74402494554d"
MODEL_SLUG="gemma-3-4b-pt"

if [[ ! "${CAMPAIGN_ID}" =~ ^math10k-4b-dense-fixed-followup-${SHORT_SHA}-[A-Za-z0-9._-]+$ ]]; then
  echo "error: PRISM_CAMPAIGN_ID must include the locked orchestrator SHA ${SHORT_SHA}" >&2
  exit 2
fi
if [[ ! "${REFERENCE_JOB_ID}" =~ ^[0-9]+$ ]]; then
  echo "error: PRISM_REFERENCE_JOB_ID must be numeric" >&2
  exit 2
fi
if [[ -n "${EXCLUDE_NODES}" && "${EXCLUDE_NODES}" == *[!A-Za-z0-9_.,\[\]-]* ]]; then
  echo "error: PRISM_SLURM_EXCLUDE contains unsupported characters" >&2
  exit 2
fi
if [[ ! -x "${ENV_PREFIX}/bin/python" ]]; then
  echo "error: Python environment is unavailable: ${ENV_PREFIX}" >&2
  exit 2
fi

for tracked_file in \
  train_eval.py \
  scripts/analyze_math10k_4b_dense_fixed_followup.py \
  scripts/build_math10k_4b_dense_fixed_followup.py \
  scripts/cuda_step_guard.py \
  scripts/preflight_hpc.py \
  scripts/smoke_dp_path.py \
  scripts/summarize_telemetry.py \
  slurm/math10k_4b_dense_fixed_followup.sbatch; do
  if [[ ! -f "${REPO_ROOT}/${tracked_file}" ]]; then
    echo "error: required tracked file is missing: ${tracked_file}" >&2
    exit 2
  fi
done

if [[ ! -d "${EXPERIMENT_STAGED_REPO_ROOT}/.git" ]]; then
  echo "error: immutable experiment source is unavailable: ${EXPERIMENT_STAGED_REPO_ROOT}" >&2
  exit 2
fi
if [[ "$(git -C "${EXPERIMENT_STAGED_REPO_ROOT}" rev-parse HEAD)" != "${EXPERIMENT_SHA}" ]]; then
  echo "error: experiment source SHA mismatch: ${EXPERIMENT_STAGED_REPO_ROOT}" >&2
  exit 2
fi
if [[ -n "$(git -C "${EXPERIMENT_STAGED_REPO_ROOT}" status --porcelain)" ]]; then
  echo "error: immutable experiment source is dirty" >&2
  exit 2
fi

REFERENCE_RECEIPT="${REFERENCE_CAMPAIGN_ROOT}/submission_receipt.json"
if [[ ! -f "${REFERENCE_RECEIPT}" ]]; then
  echo "error: reference campaign receipt is unavailable: ${REFERENCE_RECEIPT}" >&2
  exit 2
fi
"${ENV_PREFIX}/bin/python" - \
  "${REFERENCE_RECEIPT}" "${EXPERIMENT_SHA}" "${REFERENCE_JOB_ID}" \
  "${MODEL_ID}" "${MODEL_REVISION}" "${REFERENCE_CAMPAIGN_ROOT}" \
  "${EXPERIMENT_STAGED_REPO_ROOT}" "${ENV_PREFIX}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
payload = json.loads(path.read_text(encoding="utf-8"))
expected = {
    "code_sha": sys.argv[2],
    "current_job_id": sys.argv[3],
    "model_id": sys.argv[4],
    "model_revision": sys.argv[5],
    "campaign_root": sys.argv[6],
    "staged_repo": sys.argv[7],
    "environment": sys.argv[8],
}
for key, value in expected.items():
    if str(payload.get(key) or "") != value:
        raise SystemExit(
            f"reference receipt {key} mismatch: expected={value!r} "
            f"actual={payload.get(key)!r} path={path}"
        )
PY

cache_key="models--${MODEL_ID//\//--}"
snapshot="${SHARED_HF_HOME}/hub/${cache_key}/snapshots/${MODEL_REVISION}"
marker="${SHARED_HF_HOME}/staged/${cache_key}/${MODEL_REVISION}.complete"
if [[ ! -d "${snapshot}" || ! -f "${marker}" ]]; then
  if [[ "${MODE}" == "--test-only" ]]; then
    echo "warning: pinned model is not fully staged; scheduler validation can continue" >&2
  else
    echo "error: pinned offline model is not staged: ${MODEL_ID}@${MODEL_REVISION}" >&2
    exit 2
  fi
fi

# Build a SHA-named checkout independent of the mutable development tree.
mkdir -p "$(dirname -- "${STAGED_REPO_ROOT}")"
if ! command -v flock >/dev/null 2>&1; then
  echo "error: flock is required for atomic source staging" >&2
  exit 2
fi
exec 9>"${STAGED_REPO_ROOT}.lock"
flock 9
if [[ ! -e "${STAGED_REPO_ROOT}" ]]; then
  STAGED_TMP="$(mktemp -d "$(dirname -- "${STAGED_REPO_ROOT}")/.PRISM-DP-LoRA-${ORCHESTRATOR_SHA}.XXXXXX")"
  if ! git clone --quiet --no-hardlinks --no-checkout "${REPO_ROOT}" "${STAGED_TMP}" \
      || ! git -C "${STAGED_TMP}" checkout --quiet --detach "${ORCHESTRATOR_SHA}"; then
    rm -rf -- "${STAGED_TMP}"
    echo "error: could not create immutable orchestrator source staging" >&2
    exit 2
  fi
  SOURCE_ORIGIN="$(git -C "${REPO_ROOT}" remote get-url origin 2>/dev/null || true)"
  if [[ -n "${SOURCE_ORIGIN}" ]]; then
    git -C "${STAGED_TMP}" remote set-url origin "${SOURCE_ORIGIN}"
  fi
  mv -- "${STAGED_TMP}" "${STAGED_REPO_ROOT}"
fi
if [[ ! -d "${STAGED_REPO_ROOT}/.git" ]]; then
  echo "error: staged orchestrator source is not an independent Git checkout" >&2
  exit 2
fi
if [[ "$(git -C "${STAGED_REPO_ROOT}" rev-parse HEAD)" != "${ORCHESTRATOR_SHA}" ]]; then
  echo "error: staged orchestrator source revision mismatch" >&2
  exit 2
fi
if [[ -n "$(git -C "${STAGED_REPO_ROOT}" status --porcelain)" ]]; then
  echo "error: staged orchestrator source is dirty: ${STAGED_REPO_ROOT}" >&2
  exit 2
fi
WORKER="${STAGED_REPO_ROOT}/slurm/math10k_4b_dense_fixed_followup.sbatch"
if [[ ! -f "${WORKER}" ]]; then
  echo "error: staged revision does not contain the follow-up worker" >&2
  exit 2
fi
flock -u 9
exec 9>&-

CAMPAIGN_ROOT="${RUN_ROOT}/campaigns/${CAMPAIGN_ID}"
LOG_ROOT="${CAMPAIGN_ROOT}/slurm"
mkdir -p "${LOG_ROOT}"
SUBMISSION_RECEIPT="${CAMPAIGN_ROOT}/submission_receipt.json"
SUBMISSION_LOCK="${CAMPAIGN_ROOT}/.submission.lock"
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
    "${CAMPAIGN_ID}" "${CAMPAIGN_ROOT}" "${ORCHESTRATOR_SHA}" \
    "${EXPERIMENT_SHA}" "${STAGED_REPO_ROOT}" \
    "${EXPERIMENT_STAGED_REPO_ROOT}" "${ENV_PREFIX}" "${MODEL_ID}" \
    "${MODEL_REVISION}" "${REFERENCE_CAMPAIGN_ROOT}" "${REFERENCE_JOB_ID}" \
    "${DEPENDENCY_SPEC}" "${ACCOUNT}" "${QOS}" "${PARTITION}" \
    "${EXCLUDE_NODES}" <<'PY'
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

(
    target_text, action, mode, job_id, scheduler_result,
    previous_job_id, previous_job_state, campaign_id, campaign_root,
    orchestrator_sha, experiment_sha, staged_repo, experiment_staged_repo,
    environment, model_id, model_revision, reference_campaign_root,
    reference_job_id, dependency, account, qos, partition, exclude_nodes,
) = sys.argv[1:]
target = Path(target_text)
now = datetime.now(timezone.utc).isoformat()
resources = {
    "account": account,
    "qos": qos,
    "partition": partition,
    "nodes": 1,
    "tasks": 1,
    "cpus_per_task": 8,
    "memory": "128G",
    "gpu_gres": "gpu:a100:1",
    "walltime": "2-12:00:00",
    "no_requeue": True,
    "export": "NONE",
    "exclude_nodes": exclude_nodes or None,
}
identity = {
    "campaign_id": campaign_id,
    "campaign_root": campaign_root,
    "orchestrator_code_sha": orchestrator_sha,
    "experiment_code_sha": experiment_sha,
    "staged_orchestrator_repo": staged_repo,
    "staged_experiment_repo": experiment_staged_repo,
    "environment": environment,
    "model_id": model_id,
    "model_revision": model_revision,
    "reference_campaign_root": reference_campaign_root,
    "reference_job_id": reference_job_id,
    "dependency": dependency,
}
if target.exists():
    payload = json.loads(target.read_text(encoding="utf-8"))
    for key in ("campaign_id", "orchestrator_code_sha", "experiment_code_sha"):
        if payload.get(key) != identity[key]:
            raise SystemExit(f"submission receipt {key} mismatch: {target}")
else:
    payload = {"schema_version": 1, **identity, "resources": resources, "attempts": []}
attempts = payload.get("attempts")
if not isinstance(attempts, list):
    raise SystemExit(f"submission receipt attempts are malformed: {target}")
if action == "begin":
    attempts.append({
        "attempt": len(attempts) + 1,
        "mode": mode,
        "state": "submitting",
        "requested_at": now,
        "previous_job_id": previous_job_id or None,
        "previous_job_terminal_state": previous_job_state or None,
        "dependency": dependency,
        "resources": resources,
    })
    payload["state"] = "submitting"
    payload["current_job_id"] = None
elif action in {"submitted", "submission_failed", "submission_unknown"}:
    if not attempts or attempts[-1].get("state") != "submitting":
        raise SystemExit(f"submission receipt has no active attempt: {target}")
    attempts[-1].update({
        "state": action,
        "finished_at": now,
        "job_id": job_id or None,
        "scheduler_result": scheduler_result,
    })
    payload["state"] = action
    payload["current_job_id"] = job_id or None
else:
    raise SystemExit(f"unknown receipt transition: {action}")
payload.update(identity)
payload["resources"] = resources
payload["updated_at"] = now
encoded = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
target.parent.mkdir(parents=True, exist_ok=True)
descriptor, temporary_name = tempfile.mkstemp(
    prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
)
temporary = Path(temporary_name)
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
  "${ENV_PREFIX}/bin/python" - \
    "${SUBMISSION_RECEIPT}" "${CAMPAIGN_ID}" "${ORCHESTRATOR_SHA}" \
    "${EXPERIMENT_SHA}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
payload = json.loads(path.read_text(encoding="utf-8"))
expected = {
    "campaign_id": sys.argv[2],
    "orchestrator_code_sha": sys.argv[3],
    "experiment_code_sha": sys.argv[4],
}
for key, value in expected.items():
    if payload.get(key) != value:
        raise SystemExit(f"submission receipt {key} mismatch: {path}")
print(str(payload.get("state") or ""))
print(str(payload.get("current_job_id") or ""))
PY
}

if [[ "${MODE}" != "--test-only" ]]; then
  exec 8>"${SUBMISSION_LOCK}"
  flock 8
  completed_status=""
  any_status=""
  for status_file in "${CAMPAIGN_ROOT}"/status/job-*.txt; do
    [[ -e "${status_file}" ]] || continue
    any_status="${status_file}"
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
    if [[ -e "${SUBMISSION_RECEIPT}" || -n "${any_status}" ]]; then
      echo "error: campaign already has submission state; refusing a duplicate job" >&2
      echo "use --resume-submit only after a terminally unsuccessful attempt" >&2
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
        echo "error: prior submission has no verifiable failed state" >&2
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
          echo "error: prior job has no allowed terminal state: ${OLD_JOB_STATE:-unknown}" >&2
          exit 2
          ;;
      esac
    fi
    receipt_transition begin resume-submit "" "" "${OLD_JOB_ID}" "${OLD_JOB_STATE}"
  fi
fi

SBATCH_ARGS=(
  --parsable
  --job-name=prism-4b-fixed-follow
  --account="${ACCOUNT}"
  --qos="${QOS}"
  --partition="${PARTITION}"
  --dependency="${DEPENDENCY_SPEC}"
  --nodes=1
  --ntasks=1
  --cpus-per-task=8
  --mem=128G
  --gres=gpu:a100:1
  --time=2-12:00:00
  --no-requeue
  --export=NONE
  --output="${LOG_ROOT}/%x-%j.out"
  --error="${LOG_ROOT}/%x-%j.err"
)
if [[ -n "${EXCLUDE_NODES}" ]]; then
  SBATCH_ARGS+=(--exclude="${EXCLUDE_NODES}")
fi
if [[ "${MODE}" == "--test-only" ]]; then
  SBATCH_ARGS+=(--test-only)
fi

echo "mode=${MODE}"
echo "campaign_id=${CAMPAIGN_ID}"
echo "campaign_root=${CAMPAIGN_ROOT}"
echo "orchestrator_sha=${ORCHESTRATOR_SHA}"
echo "experiment_sha=${EXPERIMENT_SHA}"
echo "staged_orchestrator_repo=${STAGED_REPO_ROOT}"
echo "staged_experiment_repo=${EXPERIMENT_STAGED_REPO_ROOT}"
echo "reference_campaign=${REFERENCE_CAMPAIGN_ROOT}"
echo "dependency=${DEPENDENCY_SPEC}"
echo "environment=${ENV_PREFIX}"
echo "model=${MODEL_ID}@${MODEL_REVISION}"
echo "resources=account:${ACCOUNT},qos:${QOS},partition:${PARTITION},1xA100,8cpu,128G,2-12:00:00"
echo "protocol=finish_missing_locked_reference_then_sequential_predeclared_fixed_controls"
echo "privacy_warning=research_raw artifacts are NON_PRIVATE and remain under scratch with umask 077"

SUBMISSION_RC=0
SUBMISSION_OUTPUT="$(sbatch "${SBATCH_ARGS[@]}" "${WORKER}" \
  "${USER_NAME}" \
  "${STAGED_REPO_ROOT}" \
  "${ENV_PREFIX}" \
  "${RUN_ROOT}" \
  "${SHARED_HF_HOME}" \
  "${ORCHESTRATOR_SHA}" \
  "${CAMPAIGN_ID}" \
  "${SCRATCH_ROOT}" \
  "${MODEL_ID}" \
  "${MODEL_REVISION}" \
  "${MODEL_SLUG}" \
  "${REFERENCE_CAMPAIGN_ROOT}" \
  "${REFERENCE_JOB_ID}" \
  "${EXPERIMENT_SHA}" 2>&1)" || SUBMISSION_RC=$?
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
  flock -u 8
  exec 8>&-
  echo "submission_receipt=${SUBMISSION_RECEIPT}"
else
  echo "scheduler_validation=test-only"
fi
