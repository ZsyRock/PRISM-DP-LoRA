#!/usr/bin/env bash

# Submit one sequential one-A100 allocation for the Math-10K low-target
# Full-SlaClip campaign.  No arrays or child Slurm jobs are created.
set -Eeuo pipefail
umask 077

usage() {
  cat <<'EOF'
Usage:
  scripts/submit_math10k_4b_low_target_campaign.sh --test-only
  scripts/submit_math10k_4b_low_target_campaign.sh --submit
  scripts/submit_math10k_4b_low_target_campaign.sh --resume-submit

--test-only asks Slurm to validate the exact one-A100 request without queuing
it. --submit queues one allocation containing preflight, the complete public
validation screen, the immutable selection lock, all fresh paired confirmation
arms, and analysis. --resume-submit reuses the same
fingerprinted campaign only after its prior allocation is terminally
unsuccessful.

Portable overrides:
  PRISM_USER_NAME, PRISM_USER_HOME, PRISM_SCRATCH_ROOT, PRISM_REPO_ROOT,
  PRISM_STAGED_REPO_ROOT, PRISM_ENV_PREFIX, PRISM_ENV_TAG, PRISM_RUN_ROOT,
  PRISM_HF_HOME, PRISM_CODE_REVISION, PRISM_CAMPAIGN_ID.

Scheduler overrides:
  PRISM_SLURM_ACCOUNT, PRISM_SLURM_QOS, PRISM_SLURM_PARTITION,
  PRISM_SLURM_EXCLUDE.

The reviewed protocol locks one A100, eight CPUs, 128G host memory, and a
60-hour walltime. It runs four 150-step public-validation SlaClip screens,
then fifteen 300-step same-A100 fresh-seed confirmation arms: selected rho=0.8,
selected rho=0.9, and established fixed C=2 for each of five seeds. The model
ID/revision and experiment grid are not overridable.
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
    echo "error: choose --test-only, --submit, or --resume-submit" >&2
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
  echo "error: could not resolve home directory for ${USER_NAME}" >&2
  exit 2
fi
SCRATCH_ROOT="${PRISM_SCRATCH_ROOT:-${SCRATCH:-/scratch/${USER_NAME}}}"
REPO_ROOT="${PRISM_REPO_ROOT:-${DEFAULT_REPO_ROOT}}"
if [[ ! -d "${REPO_ROOT}/.git" ]]; then
  echo "error: repository not found: ${REPO_ROOT}" >&2
  exit 2
fi
REPO_ROOT="$(cd -- "${REPO_ROOT}" && pwd)"

REQUESTED_REVISION="${PRISM_CODE_REVISION:-HEAD}"
LOCKED_REPO_SHA="$(git -C "${REPO_ROOT}" rev-parse "${REQUESTED_REVISION}^{commit}")"
if [[ ! "${LOCKED_REPO_SHA}" =~ ^[0-9a-f]{40}$ ]]; then
  echo "error: could not resolve a full repository SHA" >&2
  exit 2
fi
if [[ "$(git -C "${REPO_ROOT}" rev-parse HEAD)" != "${LOCKED_REPO_SHA}" ]]; then
  echo "error: requested revision must be the checked-out HEAD" >&2
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
CAMPAIGN_ID="${PRISM_CAMPAIGN_ID:-math10k-4b-low-target-${SHORT_SHA}-v1}"
STAGED_REPO_ROOT="${PRISM_STAGED_REPO_ROOT:-${RUN_ROOT}/sources/PRISM-DP-LoRA-${LOCKED_REPO_SHA}}"

ACCOUNT="${PRISM_SLURM_ACCOUNT:-normal}"
QOS="${PRISM_SLURM_QOS:-normal}"
PARTITION="${PRISM_SLURM_PARTITION:-a100}"
EXCLUDE_NODES="${PRISM_SLURM_EXCLUDE:-}"
GPU_GRES="gpu:a100:1"
CPUS_PER_TASK="8"
HOST_MEMORY="128G"
WALLTIME="2-12:00:00"
MODEL_ID="google/gemma-3-4b-pt"
MODEL_REVISION="cc012e0a6d0787b4adcc0fa2c4da74402494554d"
MODEL_SLUG="gemma-3-4b-pt"

if [[ ! "${CAMPAIGN_ID}" =~ ^math10k-4b-low-target-${SHORT_SHA}-[A-Za-z0-9._-]+$ ]]; then
  echo "error: campaign ID must contain locked SHA ${SHORT_SHA}" >&2
  exit 2
fi
if [[ -n "${EXCLUDE_NODES}" && "${EXCLUDE_NODES}" == *[!A-Za-z0-9_.,\[\]-]* ]]; then
  echo "error: PRISM_SLURM_EXCLUDE contains unsupported characters" >&2
  exit 2
fi
if [[ ! -x "${ENV_PREFIX}/bin/python" ]]; then
  echo "error: Python environment unavailable: ${ENV_PREFIX}" >&2
  exit 2
fi
for required in \
  train_eval.py \
  scripts/build_math10k_4b_low_target_campaign.py \
  scripts/preflight_hpc.py \
  scripts/smoke_dp_path.py \
  scripts/summarize_telemetry.py \
  slurm/math10k_4b_low_target_campaign.sbatch; do
  if [[ ! -f "${REPO_ROOT}/${required}" ]]; then
    echo "error: required tracked file is missing: ${required}" >&2
    exit 2
  fi
done

cache_key="models--${MODEL_ID//\//--}"
snapshot="${SHARED_HF_HOME}/hub/${cache_key}/snapshots/${MODEL_REVISION}"
marker="${SHARED_HF_HOME}/staged/${cache_key}/${MODEL_REVISION}.complete"
if [[ ! -d "${snapshot}" || ! -f "${marker}" ]]; then
  if [[ "${MODE}" == "--test-only" ]]; then
    echo "warning: pinned model snapshot/marker is missing; scheduler test continues" >&2
  else
    echo "error: pinned offline model is not staged: ${MODEL_ID}@${MODEL_REVISION}" >&2
    exit 2
  fi
fi

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
    echo "error: could not create immutable staged source" >&2
    exit 2
  fi
  SOURCE_ORIGIN="$(git -C "${REPO_ROOT}" remote get-url origin 2>/dev/null || true)"
  if [[ -n "${SOURCE_ORIGIN}" ]]; then
    git -C "${STAGED_TMP}" remote set-url origin "${SOURCE_ORIGIN}"
  fi
  mv -- "${STAGED_TMP}" "${STAGED_REPO_ROOT}"
fi
if [[ ! -d "${STAGED_REPO_ROOT}/.git" ]] \
    || [[ "$(git -C "${STAGED_REPO_ROOT}" rev-parse HEAD)" != "${LOCKED_REPO_SHA}" ]] \
    || [[ -n "$(git -C "${STAGED_REPO_ROOT}" status --porcelain)" ]]; then
  echo "error: staged source is absent, dirty, or at the wrong SHA" >&2
  exit 2
fi
WORKER="${STAGED_REPO_ROOT}/slurm/math10k_4b_low_target_campaign.sbatch"
if [[ ! -f "${WORKER}" ]]; then
  echo "error: staged source does not contain the worker" >&2
  exit 2
fi
flock -u 9
exec 9>&-

CAMPAIGN_ROOT="${RUN_ROOT}/campaigns/${CAMPAIGN_ID}"
LOG_ROOT="${CAMPAIGN_ROOT}/slurm"
SUBMISSION_RECEIPT="${CAMPAIGN_ROOT}/submission_receipt.json"
SUBMISSION_LOCK="${CAMPAIGN_ROOT}/.submission.lock"
mkdir -p "${LOG_ROOT}"

scheduler_state() {
  local job_id="$1" state=""
  state="$(squeue -h -j "${job_id}" -o '%T' 2>/dev/null | head -1 || true)"
  if [[ -z "${state}" ]]; then
    state="$(sacct -X -n -P -j "${job_id}" -o JobIDRaw,State 2>/dev/null \
      | awk -F'|' -v wanted="${job_id}" '$1 == wanted {print $2; exit}' || true)"
  fi
  state="${state%%+*}"
  # sacct may report terminal states as, for example, "CANCELLED by 12345".
  state="${state%% *}"
  printf '%s\n' "${state}"
}

receipt_job_id() {
  if [[ ! -f "${SUBMISSION_RECEIPT}" ]]; then
    return 0
  fi
  "${ENV_PREFIX}/bin/python" - "${SUBMISSION_RECEIPT}" <<'PY'
import json, sys
print(json.load(open(sys.argv[1], encoding="utf-8")).get("current_job_id") or "")
PY
}

write_receipt() {
  local job_id="$1" mode="$2" previous_job_id="$3" previous_state="$4"
  "${ENV_PREFIX}/bin/python" - \
    "${SUBMISSION_RECEIPT}" "${job_id}" "${mode}" \
    "${previous_job_id}" "${previous_state}" \
    "${CAMPAIGN_ID}" "${CAMPAIGN_ROOT}" "${LOCKED_REPO_SHA}" \
    "${STAGED_REPO_ROOT}" "${ENV_PREFIX}" "${MODEL_ID}" "${MODEL_REVISION}" \
    "${ACCOUNT}" "${QOS}" "${PARTITION}" "${EXCLUDE_NODES}" <<'PY'
import json, os, sys, tempfile
from datetime import datetime, timezone
from pathlib import Path

(
    target_text, job_id, mode, previous_job_id, previous_state,
    campaign_id, campaign_root, code_sha, staged_repo, environment,
    model_id, model_revision, account, qos, partition, exclude_nodes,
) = sys.argv[1:]
target = Path(target_text)
history = []
if target.exists():
    previous = json.loads(target.read_text(encoding="utf-8"))
    history = list(previous.get("history") or [])
    if previous.get("current_job_id"):
        history.append({
            "job_id": str(previous["current_job_id"]),
            "last_observed_state": previous_state or None,
            "superseded_at": datetime.now(timezone.utc).isoformat(),
        })
payload = {
    "schema_version": 1,
    "campaign_id": campaign_id,
    "campaign_root": campaign_root,
    "code_sha": code_sha,
    "staged_repo": staged_repo,
    "environment": environment,
    "model_id": model_id,
    "model_revision": model_revision,
    "current_job_id": job_id,
    "submission_mode": mode,
    "submitted_at": datetime.now(timezone.utc).isoformat(),
    "previous_job_id": previous_job_id or None,
    "history": history,
    "resources": {
        "account": account,
        "qos": qos,
        "partition": partition,
        "nodes": 1,
        "tasks": 1,
        "cpus_per_task": 8,
        "memory": "128G",
        "gpu_gres": "gpu:a100:1",
        "walltime": "2-12:00:00",
        "single_allocation": True,
        "sequential_arms": True,
        "array": False,
        "exclude_nodes": exclude_nodes or None,
    },
}
target.parent.mkdir(parents=True, exist_ok=True)
fd, name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
temporary = Path(name)
try:
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(temporary, 0o600)
    os.replace(temporary, target)
finally:
    temporary.unlink(missing_ok=True)
PY
}

SBATCH_ARGS=(
  --account="${ACCOUNT}"
  --qos="${QOS}"
  --partition="${PARTITION}"
  --nodes=1
  --ntasks=1
  --cpus-per-task="${CPUS_PER_TASK}"
  --mem="${HOST_MEMORY}"
  --time="${WALLTIME}"
  --gres="${GPU_GRES}"
  --job-name=prism-4b-lowtarget
  --output="${LOG_ROOT}/%x-%j.out"
  --error="${LOG_ROOT}/%x-%j.err"
  --chdir="${STAGED_REPO_ROOT}"
  --export=NONE
  --no-requeue
)
if [[ -n "${EXCLUDE_NODES}" ]]; then
  SBATCH_ARGS+=(--exclude="${EXCLUDE_NODES}")
fi
WORKER_ARGS=(
  "${USER_NAME}"
  "${STAGED_REPO_ROOT}"
  "${ENV_PREFIX}"
  "${RUN_ROOT}"
  "${SHARED_HF_HOME}"
  "${LOCKED_REPO_SHA}"
  "${CAMPAIGN_ID}"
  "${SCRATCH_ROOT}"
  "${MODEL_ID}"
  "${MODEL_REVISION}"
  "${MODEL_SLUG}"
)

if [[ "${MODE}" == "--test-only" ]]; then
  sbatch --test-only "${SBATCH_ARGS[@]}" "${WORKER}" "${WORKER_ARGS[@]}"
  echo "scheduler_test=accepted gpu=${GPU_GRES} time=${WALLTIME} campaign=${CAMPAIGN_ID}"
  exit 0
fi

exec 8>"${SUBMISSION_LOCK}"
flock 8
OLD_JOB_ID="$(receipt_job_id)"
OLD_STATE=""
if [[ -n "${OLD_JOB_ID}" ]]; then
  OLD_STATE="$(scheduler_state "${OLD_JOB_ID}")"
fi
if [[ "${MODE}" == "--submit" && -n "${OLD_JOB_ID}" ]]; then
  echo "error: campaign already has submission job=${OLD_JOB_ID} state=${OLD_STATE:-unknown}" >&2
  exit 2
fi
if [[ "${MODE}" == "--resume-submit" ]]; then
  if [[ -z "${OLD_JOB_ID}" ]]; then
    echo "error: no prior campaign submission exists to resume" >&2
    exit 2
  fi
  case "${OLD_STATE}" in
    FAILED|TIMEOUT|CANCELLED|OUT_OF_MEMORY|NODE_FAIL|PREEMPTED|BOOT_FAIL|DEADLINE|REVOKED) ;;
    *)
      echo "error: prior job is not terminally unsuccessful: job=${OLD_JOB_ID} state=${OLD_STATE:-unknown}" >&2
      exit 2
      ;;
  esac
  if find "${CAMPAIGN_ROOT}/status" -maxdepth 1 -type f -name 'job-*.txt' \
      -exec grep -l '^state=completed$' {} + 2>/dev/null | grep -q .; then
    echo "error: campaign already contains a completed job status" >&2
    exit 2
  fi
fi

SUBMIT_OUTPUT="$(sbatch --parsable "${SBATCH_ARGS[@]}" "${WORKER}" "${WORKER_ARGS[@]}")"
JOB_ID="${SUBMIT_OUTPUT%%;*}"
if [[ ! "${JOB_ID}" =~ ^[0-9]+$ ]]; then
  echo "error: sbatch returned an unexpected job identifier: ${SUBMIT_OUTPUT}" >&2
  exit 2
fi
write_receipt "${JOB_ID}" "${MODE#--}" "${OLD_JOB_ID}" "${OLD_STATE}"
flock -u 8
exec 8>&-
echo "submitted_job_id=${JOB_ID}"
echo "campaign_root=${CAMPAIGN_ROOT}"
echo "slurm_stdout=${LOG_ROOT}/prism-4b-lowtarget-${JOB_ID}.out"
echo "slurm_stderr=${LOG_ROOT}/prism-4b-lowtarget-${JOB_ID}.err"
echo "resources=1xa100,8cpu,128G,60h single_allocation sequential"
