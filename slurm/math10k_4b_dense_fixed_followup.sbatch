#!/usr/bin/env bash

# One-allocation, one-A100 follow-up to the locked Math-10K 4B refinement
# campaign.  Scheduler directives intentionally live in the submission
# wrapper: this worker contains no site-, account-, or username-specific path.
set -Eeuo pipefail
umask 077

USER_NAME="${1:?missing user name}"
ORCHESTRATOR_REPO="${2:?missing immutable orchestrator repository}"
ENV_PREFIX="${3:?missing environment prefix}"
RUN_ROOT="${4:?missing run root}"
SHARED_HF_HOME="${5:?missing shared Hugging Face cache root}"
EXPECTED_ORCHESTRATOR_SHA="${6:?missing orchestrator Git SHA}"
CAMPAIGN_ID="${7:?missing campaign ID}"
SCRATCH_ROOT="${8:?missing scratch root}"
MODEL_ID="${9:?missing model ID}"
MODEL_REVISION="${10:?missing model revision}"
MODEL_SLUG="${11:?missing model slug}"
SOURCE_CAMPAIGN_ROOT="${12:?missing source campaign root}"
EXPECTED_SOURCE_JOB_ID="${13:?missing source job ID}"
EXPERIMENT_SHA="${14:?missing locked experiment Git SHA}"

PYTHON_BIN="${ENV_PREFIX}/bin/python"
CAMPAIGN_ROOT="${RUN_ROOT}/campaigns/${CAMPAIGN_ID}"
SOURCE_RECEIPT="${SOURCE_CAMPAIGN_ROOT}/submission_receipt.json"
SOURCE_SELECTION="${SOURCE_CAMPAIGN_ROOT}/selection/selection.json"
SOURCE_FINAL_ASSETS="${SOURCE_CAMPAIGN_ROOT}/selection/final-evaluation-assets.json"
SOURCE_FINAL_PLAN="${SOURCE_CAMPAIGN_ROOT}/plans/final.tsv"
BUILDER="${ORCHESTRATOR_REPO}/scripts/build_math10k_4b_dense_fixed_followup.py"
ANALYZER="${ORCHESTRATOR_REPO}/scripts/analyze_math10k_4b_dense_fixed_followup.py"
DENSE_SELECTION_PLAN="${CAMPAIGN_ROOT}/plans/dense-selection.tsv"
DENSE_LOCK="${CAMPAIGN_ROOT}/selection/dense-best-fixed.json"
DENSE_FINAL_PLAN="${CAMPAIGN_ROOT}/plans/dense-final.tsv"
JOB_ID="${SLURM_JOB_ID:-manual}"
JOB_TMP="${SLURM_TMPDIR:-${SCRATCH_ROOT}/tmp/prism-dense-fixed-${JOB_ID}}"
STATUS_ROOT="${CAMPAIGN_ROOT}/status"
JOB_STATUS="${STATUS_ROOT}/job-${JOB_ID}.txt"
CURRENT_PHASE="bootstrap"

SOURCE_REPO=""
ACTIVE_ARM_STATUS=""
ACTIVE_PHASE=""
ACTIVE_CANDIDATE=""
ACTIVE_SEED=""

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
    echo "orchestrator_sha=${EXPECTED_ORCHESTRATOR_SHA}"
    echo "experiment_sha=${EXPERIMENT_SHA}"
    echo "source_job_id=${EXPECTED_SOURCE_JOB_ID}"
  } >"${temporary}"
  mv -f -- "${temporary}" "${JOB_STATUS}"
}

write_arm_status() {
  local target="$1"
  local state="$2"
  local phase="$3"
  local candidate="$4"
  local seed="$5"
  local exit_code="$6"
  local temporary="${target}.tmp.$$"
  mkdir -p "$(dirname -- "${target}")"
  {
    echo "schema_version=1"
    echo "job_id=${JOB_ID}"
    echo "state=${state}"
    echo "phase=${phase}"
    echo "candidate_id=${candidate}"
    echo "seed=${seed}"
    echo "exit_code=${exit_code}"
    echo "updated_at=$(date --iso-8601=seconds)"
    echo "experiment_sha=${EXPERIMENT_SHA}"
  } >"${temporary}"
  mv -f -- "${temporary}" "${target}"
}

finish() {
  local exit_code=$?
  local state="completed"
  trap - EXIT TERM INT
  if [[ "${exit_code}" != "0" ]]; then
    state="failed"
    if [[ -n "${ACTIVE_ARM_STATUS}" ]]; then
      write_arm_status \
        "${ACTIVE_ARM_STATUS}" failed "${ACTIVE_PHASE}" \
        "${ACTIVE_CANDIDATE}" "${ACTIVE_SEED}" "${exit_code}" || true
    fi
  fi
  write_job_status "${state}" "${exit_code}" || true
  exit "${exit_code}"
}
trap finish EXIT
trap 'exit 143' TERM INT

atomic_sha256_sidecar() {
  local source="$1"
  "${PYTHON_BIN}" - "${source}" <<'PY'
import hashlib
import os
import sys
import tempfile
from pathlib import Path

source = Path(sys.argv[1]).resolve(strict=True)
digest = hashlib.sha256(source.read_bytes()).hexdigest()
target = source.with_name(source.name + ".sha256")
encoded = f"{digest}  {source.name}\n".encode()
if target.exists():
    if target.read_bytes() != encoded:
        raise SystemExit(f"inconsistent SHA sidecar: {target}")
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
print(f"sha256={digest} path={source}")
PY
}

verify_sha_sidecar_if_present() {
  local source="$1"
  local sidecar="${source}.sha256"
  if [[ -e "${sidecar}" ]]; then
    (cd "$(dirname -- "${source}")" && sha256sum --check --status "$(basename -- "${sidecar}")")
  fi
}

require_clean_sha() {
  local repository="$1"
  local expected_sha="$2"
  local label="$3"
  if [[ ! "${expected_sha}" =~ ^[0-9a-f]{40}$ ]]; then
    echo "error: ${label} SHA must be a full lowercase Git SHA" >&2
    return 2
  fi
  if [[ "$(git -C "${repository}" rev-parse HEAD)" != "${expected_sha}" ]]; then
    echo "error: ${label} repository SHA differs from the locked SHA" >&2
    return 2
  fi
  if [[ -n "$(git -C "${repository}" status --porcelain)" ]]; then
    echo "error: ${label} staged repository is dirty: ${repository}" >&2
    return 2
  fi
}

validate_terminal_source_job() {
  if ! command -v sacct >/dev/null 2>&1; then
    echo "error: sacct is required to prove the dependency job is terminal" >&2
    return 2
  fi
  local state
  state="$(sacct -X -n -P -j "${EXPECTED_SOURCE_JOB_ID}" -o JobIDRaw,State \
    | awk -F'|' -v wanted="${EXPECTED_SOURCE_JOB_ID}" '$1 == wanted {print $2; exit}')"
  state="${state%%+*}"
  case "${state}" in
    COMPLETED|FAILED|TIMEOUT|CANCELLED|OUT_OF_MEMORY|NODE_FAIL|PREEMPTED|BOOT_FAIL|DEADLINE|REVOKED) ;;
    "")
      echo "error: sacct has no allocation record for source job ${EXPECTED_SOURCE_JOB_ID}" >&2
      return 2
      ;;
    *)
      echo "error: source job is not terminal: job=${EXPECTED_SOURCE_JOB_ID} state=${state}" >&2
      return 2
      ;;
  esac
  echo "source_job_id=${EXPECTED_SOURCE_JOB_ID} terminal_state=${state}"
}

validate_receipt_and_get_source_repo() {
  "${PYTHON_BIN}" - \
    "${SOURCE_RECEIPT}" "${SOURCE_CAMPAIGN_ROOT}" "${EXPECTED_SOURCE_JOB_ID}" \
    "${EXPERIMENT_SHA}" "${ENV_PREFIX}" "${MODEL_ID}" "${MODEL_REVISION}" <<'PY'
import json
import sys
from pathlib import Path

receipt_path = Path(sys.argv[1]).resolve(strict=True)
expected_root = Path(sys.argv[2]).resolve(strict=True)
expected_job = sys.argv[3]
expected_sha = sys.argv[4]
expected_env = Path(sys.argv[5]).resolve(strict=True)
expected_model = sys.argv[6]
expected_revision = sys.argv[7]
receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
checks = {
    "campaign_root": (Path(receipt["campaign_root"]).resolve(strict=True), expected_root),
    "current_job_id": (str(receipt.get("current_job_id")), expected_job),
    "code_sha": (receipt.get("code_sha"), expected_sha),
    "environment": (Path(receipt["environment"]).resolve(strict=True), expected_env),
    "model_id": (receipt.get("model_id"), expected_model),
    "model_revision": (receipt.get("model_revision"), expected_revision),
}
for label, (observed, expected) in checks.items():
    if observed != expected:
        raise SystemExit(f"source receipt mismatch {label}: observed={observed!r} expected={expected!r}")
source_repo = Path(receipt["staged_repo"]).resolve(strict=True)
print(source_repo)
PY
}

validate_source_plan() {
  verify_sha_sidecar_if_present "${SOURCE_SELECTION}"
  verify_sha_sidecar_if_present "${SOURCE_FINAL_ASSETS}"
  verify_sha_sidecar_if_present "${SOURCE_FINAL_PLAN}"
  "${PYTHON_BIN}" - \
    "${SOURCE_FINAL_PLAN}" "${SOURCE_SELECTION}" "${SOURCE_CAMPAIGN_ROOT}" \
    "${MODEL_SLUG}" <<'PY'
import json
import math
import sys
from pathlib import Path

plan_path = Path(sys.argv[1]).resolve(strict=True)
selection_path = Path(sys.argv[2]).resolve(strict=True)
source_root = Path(sys.argv[3]).resolve(strict=True)
model_slug = sys.argv[4]
selection = json.loads(selection_path.read_text(encoding="utf-8"))
selected = selection.get("selected_slaclip") or {}
selected_id = selected.get("candidate_id")
params = selected.get("params") or {}
expected_slaclip = (
    float(params["dp_max_grad_norm"]),
    float(params["slaclip_target_non_small_clip_fraction"]),
    float(params["slaclip_eta"]),
)
seeds = (191, 223, 257, 293, 331)
lines = [line for line in plan_path.read_text(encoding="utf-8").splitlines() if line]
if len(lines) != 10:
    raise SystemExit(f"source final plan must contain ten rows, got {len(lines)}")
seen = set()
for index, line in enumerate(lines):
    fields = line.split("|")
    if len(fields) != 11:
        raise SystemExit(f"invalid source plan row {index + 1}: {line!r}")
    kind, phase, candidate, role, seed_text, method, clip, rho, eta, schedule, root_text = fields
    seed = int(seed_text)
    expected_seed = seeds[index // 2]
    expected_role = "slaclip" if index % 2 == 0 else "baseline"
    expected_method = "slaclip" if expected_role == "slaclip" else "baseline"
    expected_root = source_root / "final" / model_slug / f"seed-{seed}" / role
    if (kind, phase, role, method, seed) != (
        "train", "final", expected_role, expected_method, expected_seed
    ):
        raise SystemExit(f"source plan order/identity mismatch at row {index + 1}: {fields}")
    if Path(root_text).resolve() != expected_root.resolve():
        raise SystemExit(f"source arm root mismatch at row {index + 1}: {root_text}")
    if schedule != "NA" or (role, seed) in seen:
        raise SystemExit(f"source plan schedule/duplicate mismatch at row {index + 1}")
    seen.add((role, seed))
    if role == "baseline":
        if candidate != "fixed-c2" or not math.isclose(float(clip), 2.0, abs_tol=1e-12):
            raise SystemExit(f"source reference is not fixed C=2 at row {index + 1}")
        if rho != "NA" or eta != "NA":
            raise SystemExit(f"fixed source row carries SlaClip parameters at row {index + 1}")
    else:
        observed = (float(clip), float(rho), float(eta))
        if candidate != selected_id or any(
            not math.isclose(a, b, rel_tol=0.0, abs_tol=1e-12)
            for a, b in zip(observed, expected_slaclip)
        ):
            raise SystemExit(f"source SlaClip row differs from locked selection at row {index + 1}")
print(f"validated source final plan rows={len(lines)} selected={selected_id}")
PY
}

validate_dense_plan() {
  local mode="$1"
  local plan="$2"
  "${PYTHON_BIN}" - "${mode}" "${plan}" "${CAMPAIGN_ROOT}" "${DENSE_LOCK}" <<'PY'
import hashlib
import json
import math
import sys
from pathlib import Path

mode = sys.argv[1]
plan = Path(sys.argv[2]).resolve(strict=True)
root = Path(sys.argv[3]).resolve(strict=True)
lock_path = Path(sys.argv[4])
lines = [line for line in plan.read_text(encoding="utf-8").splitlines() if line]
rows = []
for number, line in enumerate(lines, 1):
    fields = line.split("|")
    if len(fields) != 11:
        raise SystemExit(f"invalid dense plan row {number}: {line!r}")
    kind, phase, candidate, role, seed, method, clip, rho, eta, schedule, arm = fields
    resolved_arm = Path(arm).resolve()
    if root != resolved_arm and root not in resolved_arm.parents:
        raise SystemExit(f"dense arm escapes campaign root: {arm}")
    if any(value != "NA" for value in (rho, eta, schedule)):
        raise SystemExit(f"fixed dense row carries dynamic parameters: row={number}")
    rows.append((kind, phase, candidate, role, int(seed), method, float(clip), resolved_arm))
if len({row[7] for row in rows}) != len(rows):
    raise SystemExit("dense plan reuses an arm output directory")
if mode == "selection":
    expected = [
        (clip, seed)
        for clip in (1.5, 1.75, 1.25)
        for seed in (42, 43, 44, 45, 46)
    ]
    if len(rows) != 15:
        raise SystemExit(f"dense selection plan must contain 15 rows, got {len(rows)}")
    for row, (clip, seed) in zip(rows, expected):
        if row[0:2] != ("train", "dense-selection") or row[3] != "dense-fixed-selection":
            raise SystemExit(f"invalid dense selection identity: {row}")
        if row[4] != seed or row[5] != "baseline" or not math.isclose(row[6], clip, abs_tol=1e-12):
            raise SystemExit(f"dense selection order/config mismatch: {row}, expected C={clip} seed={seed}")
elif mode == "final":
    if len(rows) not in (0, 5):
        raise SystemExit(f"dense final plan must contain zero or five rows, got {len(rows)}")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    gate = lock.get("gate") or {}
    passed = gate.get("passed")
    best = lock.get("dense_best_fixed") or {}
    best_clip = float(best["clip_threshold"])
    final = lock.get("fresh_final_plan") or {}
    if not isinstance(passed, bool):
        raise SystemExit("dense lock has no boolean gate.passed")
    if int(final.get("record_count", -1)) != len(rows):
        raise SystemExit("dense lock record count differs from final plan")
    if final.get("file_sha256") != hashlib.sha256(plan.read_bytes()).hexdigest():
        raise SystemExit("dense lock hash differs from final plan")
    status = final.get("status")
    if not passed:
        if rows or status != "empty-gate-failed":
            raise SystemExit("failed gate must have an empty, gate-failed final plan")
    elif math.isclose(best_clip, 2.0, abs_tol=1e-12):
        if rows or status != "empty-reuse-source-fixed-c2":
            raise SystemExit("best C=2 must reuse the five source fixed-C2 final arms")
    elif len(rows) != 5 or status != "enabled":
        raise SystemExit("passed non-C2 gate must enable five fresh dense-final arms")
    if rows:
        candidate_ids = {row[2] for row in rows}
        clips = {row[6] for row in rows}
        seeds = [row[4] for row in rows]
        if len(candidate_ids) != 1 or len(clips) != 1 or seeds != [191, 223, 257, 293, 331]:
            raise SystemExit("dense final rows must be one locked candidate across five fresh seeds")
        if any(
            row[0:2] != ("train", "dense-final")
            or row[3] != "dense-best-fixed"
            or row[5] != "baseline"
            for row in rows
        ):
            raise SystemExit("invalid dense final identity")
else:
    raise SystemExit(f"unknown plan mode: {mode}")
print(f"validated dense {mode} plan rows={len(rows)}")
PY
}

validate_selection_run() {
  local arm_root="$1"
  local expected_seed="$2"
  local expected_clip="$3"
  "${PYTHON_BIN}" - \
    "${arm_root}" "${expected_seed}" "${expected_clip}" "${EXPERIMENT_SHA}" \
    "${MODEL_ID}" "${MODEL_REVISION}" <<'PY'
import json
import math
import sys
from pathlib import Path

root = Path(sys.argv[1])
seed = int(sys.argv[2])
clip = float(sys.argv[3])
code_sha, model_id, revision = sys.argv[4:7]
status_path = root / "adapter" / "run_status.json"
metrics_path = root / "results" / "validation" / "validation_metrics.json"
curve_path = root / "results" / "validation" / "validation_curve.jsonl"
split_path = root / "results" / "validation" / "split_manifest.json"
for path in (status_path, metrics_path, curve_path, split_path):
    if not path.is_file() or path.stat().st_size == 0:
        raise SystemExit(f"missing dense-selection artifact: {path}")
status = json.loads(status_path.read_text(encoding="utf-8"))
config = status.get("config") or {}
expected = {
    "dataset": "math10k", "method": "baseline", "privacy": "dp",
    "seed": seed, "protocol_stage": "selection", "val_set_size": 500,
    "validation_seed": 1729, "validation_batch_size": 8,
    "validation_eval_interval": 50, "validation_generate_numeric": True,
    "validation_num_beams": 1, "validation_max_new_tokens": 128,
    "validation_max_input_length": 512, "validation_data_is_public": True,
    "run_eval": False, "total_update_steps": 300, "batch_size": 64,
    "micro_batch_size": 4, "learning_rate": 0.0003, "cutoff_len": 256,
    "train_on_inputs": True, "dp_epsilon": 6.0, "dp_delta": 1e-5,
    "telemetry_mode": "research_raw", "allow_non_private_telemetry": True,
    "raw_hist_bins": 128, "raw_hist_max": 30.0, "slaclip_num_slots": 15,
    "implementation_git_sha": code_sha, "implementation_git_dirty": False,
    "base_model": model_id, "model_revision": revision,
}
if status.get("state") != "completed" or int(status.get("update_steps", -1)) != 300:
    raise SystemExit(f"selection training is incomplete: {status_path}")
for key, value in expected.items():
    if config.get(key) != value:
        raise SystemExit(f"selection config mismatch {key}={config.get(key)!r}, expected={value!r}")
if not math.isclose(float(config.get("dp_max_grad_norm", -1)), clip, abs_tol=1e-12):
    raise SystemExit(f"selection clip mismatch: {status_path}")
if status.get("resolved_model_revision") != revision or config.get("resolved_model_revision") != revision:
    raise SystemExit(f"selection model revision was not resolved to the lock: {status_path}")
epsilon = (status.get("privacy_accounting") or {}).get("epsilon_spent")
if epsilon is None or not math.isfinite(float(epsilon)) or abs(float(epsilon) - 6.0) > 0.02:
    raise SystemExit(f"unexpected selection epsilon={epsilon}: {status_path}")
metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
if metrics.get("selection_metric") != "public_math10k_numeric_exact_match_accuracy":
    raise SystemExit(f"wrong selection metric: {metrics_path}")
accuracy = metrics.get("numeric_exact_accuracy")
loss = metrics.get("loss_mean")
if any(
    isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value))
    for value in (accuracy, loss)
):
    raise SystemExit(f"selection endpoint contains non-finite values: {metrics_path}")
if not 0.0 <= float(accuracy) <= 1.0:
    raise SystemExit(f"selection accuracy is outside [0,1]: {metrics_path}")
curve = [json.loads(line) for line in curve_path.read_text(encoding="utf-8").splitlines() if line]
if [row.get("step") for row in curve] != [0, 50, 100, 150, 200, 250, 300]:
    raise SystemExit(f"selection curve lacks the locked checkpoints: {curve_path}")
if any(not math.isfinite(float(row.get("loss_mean", float("nan")))) for row in curve):
    raise SystemExit(f"selection curve has non-finite loss: {curve_path}")
split = json.loads(split_path.read_text(encoding="utf-8"))
if (
    split.get("protocol_stage") != "selection"
    or split.get("validation_data_is_public") is not True
    or int(split.get("source_rows", -1)) != 9919
    or int(split.get("train_rows", -1)) != 9419
    or int(split.get("validation_rows", -1)) != 500
):
    raise SystemExit(f"selection split differs from the locked public holdout: {split_path}")
print(f"validated dense selection seed={seed} C={clip:.12g} accuracy={float(accuracy):.8f}")
PY
}

validate_final_run() {
  local arm_root="$1"
  local expected_method="$2"
  local expected_seed="$3"
  local expected_clip="$4"
  local expected_rho="$5"
  local expected_eta="$6"
  "${PYTHON_BIN}" - \
    "${arm_root}" "${expected_method}" "${expected_seed}" "${expected_clip}" \
    "${expected_rho}" "${expected_eta}" "${SOURCE_FINAL_ASSETS}" \
    "${EXPERIMENT_SHA}" "${MODEL_ID}" "${MODEL_REVISION}" <<'PY'
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
method = sys.argv[2]
seed = int(sys.argv[3])
clip = float(sys.argv[4])
rho_text, eta_text = sys.argv[5:7]
asset_path = Path(sys.argv[7]).resolve(strict=True)
code_sha, model_id, revision = sys.argv[8:11]
status_path = root / "adapter" / "run_status.json"
summary_path = root / "results" / "summary.csv"
details_path = root / "results" / "details.csv"
eval_path = root / "results" / "evaluation_config.json"
decontaminated_path = root / "results" / "decontaminated_metrics.json"
task_paths = {task: root / "results" / f"{task}.json" for task in ("gsm8k", "AQuA", "mawps", "SVAMP")}
for path in (status_path, summary_path, details_path, eval_path, *task_paths.values()):
    if not path.is_file() or path.stat().st_size == 0:
        raise SystemExit(f"missing final artifact: {path}")
weights = [path for path in (root / "adapter" / "adapter_model.safetensors", root / "adapter" / "adapter_model.bin") if path.is_file()]
if len(weights) != 1 or weights[0].stat().st_size == 0:
    raise SystemExit(f"final run must contain exactly one adapter weight file: {root}")
status = json.loads(status_path.read_text(encoding="utf-8"))
config = status.get("config") or {}
expected = {
    "dataset": "math10k", "method": method, "privacy": "dp", "seed": seed,
    "protocol_stage": "final", "val_set_size": 0, "validation_eval_interval": 0,
    "run_eval": True, "total_update_steps": 300, "batch_size": 64,
    "micro_batch_size": 4, "learning_rate": 0.0003, "cutoff_len": 256,
    "train_on_inputs": True, "dp_epsilon": 6.0, "dp_delta": 1e-5,
    "telemetry_mode": "research_raw", "allow_non_private_telemetry": True,
    "raw_hist_bins": 128, "raw_hist_max": 30.0, "slaclip_num_slots": 15,
    "implementation_git_sha": code_sha, "implementation_git_dirty": False,
    "base_model": model_id, "model_revision": revision,
}
if status.get("state") != "completed" or int(status.get("update_steps", -1)) != 300:
    raise SystemExit(f"final training is incomplete: {status_path}")
for key, value in expected.items():
    if config.get(key) != value:
        raise SystemExit(f"final config mismatch {key}={config.get(key)!r}, expected={value!r}")
if not math.isclose(float(config.get("dp_max_grad_norm", -1)), clip, abs_tol=1e-12):
    raise SystemExit(f"final clip mismatch: {status_path}")
if method == "slaclip":
    if rho_text == "NA" or eta_text == "NA":
        raise SystemExit("SlaClip final row lacks controller parameters")
    if not math.isclose(float(config.get("slaclip_target_non_small_clip_fraction")), float(rho_text), abs_tol=1e-12):
        raise SystemExit(f"SlaClip rho mismatch: {status_path}")
    if not math.isclose(float(config.get("slaclip_eta")), float(eta_text), abs_tol=1e-12):
        raise SystemExit(f"SlaClip eta mismatch: {status_path}")
elif rho_text != "NA" or eta_text != "NA":
    raise SystemExit("fixed final row unexpectedly has SlaClip parameters")
if status.get("resolved_model_revision") != revision or config.get("resolved_model_revision") != revision:
    raise SystemExit(f"final model revision was not resolved to the lock: {status_path}")
epsilon = (status.get("privacy_accounting") or {}).get("epsilon_spent")
if epsilon is None or not math.isfinite(float(epsilon)) or abs(float(epsilon) - 6.0) > 0.02:
    raise SystemExit(f"unexpected final epsilon={epsilon}: {status_path}")
evaluation = json.loads(eval_path.read_text(encoding="utf-8"))
eval_expected = {
    "dataset": "math10k", "base_model": model_id, "batch_size": 8,
    "num_beams": 4, "max_new_tokens": 256, "max_input_length": 1024,
    "tasks": ["gsm8k", "AQuA", "mawps", "SVAMP"],
}
for key, value in eval_expected.items():
    if evaluation.get(key) != value:
        raise SystemExit(f"final evaluation mismatch {key}: {eval_path}")
if evaluation.get("config_fingerprint") != status.get("config_fingerprint"):
    raise SystemExit(f"evaluation fingerprint mismatch: {eval_path}")
if evaluation.get("resolved_model_revision") != revision:
    raise SystemExit(f"evaluation revision mismatch: {eval_path}")
with summary_path.open("r", encoding="utf-8", newline="") as handle:
    summary_rows = list(csv.DictReader(handle))
columns = ["gsm8k", "AQuA", "mawps", "SVAMP", "Average"]
if len(summary_rows) != 1 or list(summary_rows[0]) != columns:
    raise SystemExit(f"invalid final summary schema: {summary_path}")
summary = {key: float(summary_rows[0][key]) for key in columns}
if any(not math.isfinite(value) or not 0 <= value <= 1 for value in summary.values()):
    raise SystemExit(f"non-finite final summary: {summary_path}")
if not math.isclose(summary["Average"], math.fsum(summary[key] for key in columns[:-1]) / 4, abs_tol=1e-12):
    raise SystemExit(f"inconsistent final macro: {summary_path}")

def canonical(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()

def normalize(record):
    def field(value):
        return " ".join(unicodedata.normalize("NFKC", str(value)).casefold().split())
    return json.dumps({"instruction": field(record.get("instruction", "")), "input": field(record.get("input", ""))}, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

def identity(record):
    return {"instruction": record.get("instruction"), "input": record.get("input"), "answer": record.get("answer")}

assets = json.loads(asset_path.read_text(encoding="utf-8"))
if assets.get("schema_version") != 3:
    raise SystemExit(f"unsupported final asset manifest: {asset_path}")
normalization = (assets.get("decontamination") or {}).get("normalization") or {}
if normalization.get("id") != "instruction_input_nfkc_casefold_whitespace_collapse_v1":
    raise SystemExit(f"unexpected decontamination normalization: {asset_path}")
by_task = {item["task"]: item for item in assets.get("assets", [])}
expected_rows = {"gsm8k": 1319, "AQuA": 254, "mawps": 238, "SVAMP": 1000}
raw_accuracy, clean_accuracy, clean_correct, clean_records = {}, {}, {}, {}
for task, path in task_paths.items():
    records = json.loads(path.read_text(encoding="utf-8"))
    asset = by_task.get(task) or {}
    if not isinstance(records, list) or len(records) != expected_rows[task] or int(asset.get("rows", -1)) != len(records):
        raise SystemExit(f"final task row count differs from locked asset: {path}")
    if canonical([normalize(row) for row in records]) != asset.get("normalized_prompt_sequence_sha256"):
        raise SystemExit(f"final prompt sequence differs from locked asset: {path}")
    if canonical([identity(row) for row in records]) != asset.get("record_identity_sequence_sha256"):
        raise SystemExit(f"final task identities differ from locked asset: {path}")
    flags = []
    for index, record in enumerate(records):
        flag = record.get("flag")
        if not isinstance(flag, bool) or "pred" not in record or not isinstance(record.get("output_pred"), str):
            raise SystemExit(f"invalid stored prediction: {path}[{index}]")
        if task == "AQuA":
            recomputed = str(record.get("answer", "")).strip().upper() == str(record.get("pred", "")).strip().upper()
        else:
            try:
                answer = float(record.get("answer"))
            except (TypeError, ValueError):
                answer = float("inf")
            try:
                prediction = float(record.get("pred"))
            except (TypeError, ValueError):
                prediction = float("inf")
            recomputed = abs(answer - prediction) <= 0.001
        if flag is not recomputed:
            raise SystemExit(f"stored correctness flag disagrees with evaluator: {path}[{index}]")
        flags.append(flag)
    raw_accuracy[task] = math.fsum(int(flag) for flag in flags) / len(flags)
    if not math.isclose(raw_accuracy[task], summary[task], abs_tol=1e-12):
        raise SystemExit(f"task output disagrees with summary: {path}")
    overlap = asset.get("prompt_overlap") or {}
    clean_indices = overlap.get("clean_test_indices")
    if not isinstance(clean_indices, list) or clean_indices != sorted(set(clean_indices)) or canonical(clean_indices) != overlap.get("clean_test_indices_sha256"):
        raise SystemExit(f"invalid locked clean indices for task={task}")
    clean_records[task] = len(clean_indices)
    clean_correct[task] = sum(int(flags[index]) for index in clean_indices)
    clean_accuracy[task] = clean_correct[task] / clean_records[task]
with details_path.open("r", encoding="utf-8", newline="") as handle:
    details = list(csv.DictReader(handle))
if len(details) != 4 or {row.get("dataset") for row in details} != set(task_paths):
    raise SystemExit(f"invalid final details: {details_path}")
for row in details:
    task = row["dataset"]
    if int(row["n"]) != expected_rows[task] or not math.isclose(float(row["accuracy"]), raw_accuracy[task], abs_tol=1e-12):
        raise SystemExit(f"final detail mismatch for task={task}")
clean_three = math.fsum(raw_accuracy[task] for task in ("gsm8k", "AQuA", "SVAMP")) / 3
clean_four = math.fsum((raw_accuracy["gsm8k"], raw_accuracy["AQuA"], clean_accuracy["mawps"], raw_accuracy["SVAMP"])) / 4
payload = {
    "schema_version": 2,
    "primary_metric": "clean_three_task_macro_accuracy",
    "clean_three_task_macro_accuracy": clean_three,
    "sensitivity_metric": "decontaminated_four_task_macro_accuracy",
    "decontaminated_four_task_macro_accuracy": clean_four,
    "clean_mawps_accuracy": clean_accuracy["mawps"],
    "clean_mawps_correct": clean_correct["mawps"],
    "clean_mawps_records": clean_records["mawps"],
    "paper_raw_four_task_macro_accuracy": summary["Average"],
    "raw_task_accuracy": raw_accuracy,
    "decontaminated_task_accuracy": clean_accuracy,
    "decontaminated_task_records": clean_records,
    "asset_manifest_path": str(asset_path),
    "asset_manifest_sha256": hashlib.sha256(asset_path.read_bytes()).hexdigest(),
    "selection_sha256": assets["selection_sha256"],
    "normalization_id": normalization["id"],
    "contamination_disclosure": {
        "mawps_raw_records": expected_rows["mawps"],
        "mawps_prompt_overlap_records": int(by_task["mawps"]["prompt_overlap"]["overlap_count"]),
        "mawps_clean_records": clean_records["mawps"],
        "mawps_overlap_answer_string_exact_records": int(by_task["mawps"]["prompt_overlap"]["answer_string_exact_overlap_count"]),
        "mawps_overlap_answer_evaluator_equivalent_records": int(by_task["mawps"]["prompt_overlap"]["answer_evaluator_equivalent_overlap_count"]),
    },
}
encoded = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
decontaminated_path.parent.mkdir(parents=True, exist_ok=True)
if decontaminated_path.exists():
    if decontaminated_path.read_bytes() != encoded:
        raise SystemExit(f"inconsistent decontaminated metrics: {decontaminated_path}")
else:
    fd, name = tempfile.mkstemp(prefix=f".{decontaminated_path.name}.", suffix=".tmp", dir=decontaminated_path.parent)
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
print(f"validated final method={method} seed={seed} C={clip:.12g} clean_three={clean_three:.8f}")
PY
}

summarize_telemetry() {
  local arm_root="$1"
  local raw_dir="${arm_root}/results/research_raw"
  local raw_log="${raw_dir}/NON_PRIVATE_train_log.jsonl"
  local safe_log="${arm_root}/adapter/train_log.jsonl"
  mkdir -p "${arm_root}/logs"
  if [[ ! -s "${raw_log}" || ! -s "${safe_log}" ]]; then
    echo "error: required raw/DP-safe telemetry is missing: ${arm_root}" >&2
    return 3
  fi
  "${PYTHON_BIN}" "${EXPERIMENT_REPO}/scripts/summarize_telemetry.py" \
    "${raw_log}" \
    --safe-log "${safe_log}" \
    --format both \
    --csv-out "${raw_dir}/telemetry_steps.csv" \
    --json-out "${raw_dir}/telemetry_summary.json" \
    >>"${arm_root}/logs/followup-summarize-${JOB_ID}.out" \
    2>"${arm_root}/logs/followup-summarize-${JOB_ID}.err"
  [[ -s "${raw_dir}/telemetry_steps.csv" && -s "${raw_dir}/telemetry_summary.json" ]]
}

serial_srun() {
  local arm_root="$1"
  local log_stem="$2"
  shift 2
  local tmp_root="${JOB_TMP}/arms/${log_stem}"
  mkdir -p "${arm_root}/logs" "${tmp_root}"
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
    --output="${arm_root}/logs/${log_stem}-${JOB_ID}.out" \
    --error="${arm_root}/logs/${log_stem}-${JOB_ID}.err" \
    env \
      PRISM_REPO_ROOT="${EXPERIMENT_REPO}" \
      PRISM_RUN_ROOT="${RUN_ROOT}" \
      HF_HOME="${SHARED_HF_HOME}" \
      HF_HUB_CACHE="${SHARED_HF_HOME}/hub" \
      HUGGINGFACE_HUB_CACHE="${SHARED_HF_HOME}/hub" \
      HF_DATASETS_CACHE="${CAMPAIGN_ROOT}/cache/datasets" \
      HF_HUB_OFFLINE=1 \
      TRANSFORMERS_OFFLINE=1 \
      HF_HUB_DISABLE_XET=1 \
      TORCH_HOME="${CAMPAIGN_ROOT}/cache/torch" \
      TORCH_EXTENSIONS_DIR="${CAMPAIGN_ROOT}/cache/torch-extensions" \
      TRITON_CACHE_DIR="${CAMPAIGN_ROOT}/cache/triton" \
      XDG_CACHE_HOME="${CAMPAIGN_ROOT}/cache/xdg" \
      CUDA_CACHE_PATH="${CAMPAIGN_ROOT}/cache/cuda" \
      TMPDIR="${tmp_root}" \
      PYTHONDONTWRITEBYTECODE=1 \
      TOKENIZERS_PARALLELISM=false \
      PYTHONUNBUFFERED=1 \
      OMP_NUM_THREADS=8 \
      MKL_NUM_THREADS=8 \
      "${PYTHON_BIN}" -u "${EXPERIMENT_REPO}/scripts/cuda_step_guard.py" \
        --marker "${arm_root}/logs/${log_stem}-${JOB_ID}-cuda-guard.json" -- "$@"
}

run_training_row() {
  local spec="$1"
  local status_namespace="$2"
  local kind phase candidate role seed method clip rho eta schedule arm_root extra_status
  IFS='|' read -r kind phase candidate role seed method clip rho eta schedule arm_root extra_status <<<"${spec}"
  if [[ "${kind}" != "train" || -n "${extra_status}" || "${schedule}" != "NA" ]]; then
    echo "error: malformed fixed-follow-up training spec: ${spec}" >&2
    return 2
  fi
  local adapter_dir="${arm_root}/adapter"
  local result_dir="${arm_root}/results"
  local status_file
  if [[ "${status_namespace}" == "source" ]]; then
    status_file="${STATUS_ROOT}/source-final-${role}-seed-${seed}.txt"
  else
    status_file="${arm_root}/followup-orchestration-status.txt"
  fi
  mkdir -p "${adapter_dir}" "${result_dir}" "${arm_root}/logs"
  ACTIVE_ARM_STATUS="${status_file}"
  ACTIVE_PHASE="${phase}"
  ACTIVE_CANDIDATE="${candidate}"
  ACTIVE_SEED="${seed}"
  write_arm_status "${status_file}" checking "${phase}" "${candidate}" "${seed}" 0

  local validation_rc=0
  if [[ "${phase}" == "dense-selection" ]]; then
    validate_selection_run "${arm_root}" "${seed}" "${clip}" \
      >"${arm_root}/logs/followup-precheck-${JOB_ID}.out" \
      2>"${arm_root}/logs/followup-precheck-${JOB_ID}.err" || validation_rc=$?
  else
    validate_final_run "${arm_root}" "${method}" "${seed}" "${clip}" "${rho}" "${eta}" \
      >"${arm_root}/logs/followup-precheck-${JOB_ID}.out" \
      2>"${arm_root}/logs/followup-precheck-${JOB_ID}.err" || validation_rc=$?
  fi
  if [[ "${validation_rc}" == "0" ]]; then
    summarize_telemetry "${arm_root}"
    write_arm_status "${status_file}" completed "${phase}" "${candidate}" "${seed}" 0
    ACTIVE_ARM_STATUS=""
    echo "skipping already-complete arm phase=${phase} candidate=${candidate} seed=${seed}"
    return 0
  fi

  write_arm_status "${status_file}" running "${phase}" "${candidate}" "${seed}" 0
  local -a args=(
    "${PYTHON_BIN}" -u "${EXPERIMENT_REPO}/train_eval.py"
    --config "${EXPERIMENT_REPO}/configs/math10k_paper.json"
    --dataset math10k
    --method "${method}"
    --privacy dp
    --base_model "${MODEL_ID}"
    --model_revision "${MODEL_REVISION}"
    --seed "${seed}"
    --run_name "${phase}-${MODEL_SLUG}-${role}-${candidate}-seed${seed}"
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
  if [[ "${phase}" == "dense-selection" ]]; then
    if [[ "${method}" != "baseline" || "${rho}" != "NA" || "${eta}" != "NA" ]]; then
      echo "error: dense-selection plan is not fixed baseline" >&2
      return 2
    fi
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
  else
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
    if [[ "${method}" == "slaclip" ]]; then
      args+=(
        --slaclip_target_non_small_clip_fraction "${rho}"
        --slaclip_eta "${eta}"
        --slaclip_c_min 0.1
        --slaclip_c_max 15.0
      )
    elif [[ "${method}" != "baseline" || "${rho}" != "NA" || "${eta}" != "NA" ]]; then
      echo "error: unsupported final method/parameters: ${spec}" >&2
      return 2
    fi
  fi

  local train_rc=0
  serial_srun "${arm_root}" train "${args[@]}" || train_rc=$?
  if [[ "${train_rc}" != "0" ]]; then
    write_arm_status "${status_file}" failed "${phase}" "${candidate}" "${seed}" "${train_rc}"
    return "${train_rc}"
  fi
  if [[ "${phase}" == "dense-selection" ]]; then
    validate_selection_run "${arm_root}" "${seed}" "${clip}"
  else
    validate_final_run "${arm_root}" "${method}" "${seed}" "${clip}" "${rho}" "${eta}"
  fi
  summarize_telemetry "${arm_root}"
  write_arm_status "${status_file}" completed "${phase}" "${candidate}" "${seed}" 0
  ACTIVE_ARM_STATUS=""
}

run_plan_serially() {
  local plan="$1"
  local namespace="$2"
  local spec
  while IFS= read -r spec || [[ -n "${spec}" ]]; do
    [[ -n "${spec}" ]] || continue
    run_training_row "${spec}" "${namespace}"
    write_job_status running 0
  done <"${plan}"
}

if [[ "$(id -un)" != "${USER_NAME}" ]]; then
  echo "error: worker user differs from submitted user" >&2
  exit 2
fi
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "error: locked Python environment is unavailable: ${PYTHON_BIN}" >&2
  exit 2
fi
for value in "${EXPECTED_ORCHESTRATOR_SHA}" "${EXPERIMENT_SHA}"; do
  if [[ ! "${value}" =~ ^[0-9a-f]{40}$ ]]; then
    echo "error: all Git revisions must be full lowercase SHAs" >&2
    exit 2
  fi
done
if [[ ! "${EXPECTED_SOURCE_JOB_ID}" =~ ^[0-9]+$ || ! "${JOB_ID}" =~ ^[0-9]+$ ]]; then
  echo "error: source and current Slurm job IDs must be numeric" >&2
  exit 2
fi
if [[ "${SLURM_JOB_NUM_NODES:-1}" != "1" || "${SLURM_NTASKS:-1}" != "1" || "${SLURM_CPUS_PER_TASK:-8}" != "8" ]]; then
  echo "error: worker requires exactly one node, one parent task, and eight CPUs" >&2
  exit 2
fi
if [[ -n "${SLURM_GPUS_ON_NODE:-}" && "${SLURM_GPUS_ON_NODE}" != "1" ]]; then
  echo "error: worker requires exactly one allocation GPU, found ${SLURM_GPUS_ON_NODE}" >&2
  exit 2
fi
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" && "${CUDA_VISIBLE_DEVICES}" == *,* ]]; then
  echo "error: worker received more than one visible allocation GPU: ${CUDA_VISIBLE_DEVICES}" >&2
  exit 2
fi
if [[ -n "${SLURM_MEM_PER_NODE:-}" && "${SLURM_MEM_PER_NODE}" =~ ^[0-9]+$ ]] && (( SLURM_MEM_PER_NODE < 128000 )); then
  echo "error: worker requires at least 128G host memory" >&2
  exit 2
fi
if ! command -v scontrol >/dev/null 2>&1; then
  echo "error: scontrol is required to audit the parent allocation" >&2
  exit 2
fi
ALLOCATION_RECORD="$(scontrol show job -o "${JOB_ID}")"
"${PYTHON_BIN}" - "${JOB_ID}" "${ALLOCATION_RECORD}" <<'PY'
import sys

job_id, record = sys.argv[1:]
fields = {}
for token in record.split():
    if "=" in token:
        key, value = token.split("=", 1)
        fields[key] = value
expected = {
    "JobId": job_id,
    "NumNodes": "1",
    "NumTasks": "1",
    "CPUs/Task": "8",
    "TresPerNode": "gres/gpu:a100:1",
}
for key, value in expected.items():
    if fields.get(key) != value:
        raise SystemExit(
            f"parent allocation mismatch {key}={fields.get(key)!r}, expected={value!r}"
        )
memory = fields.get("MinMemoryNode", "")
if memory not in {"128G", "131072M", "134217728K"}:
    raise SystemExit(f"parent allocation must request exactly 128G, found {memory!r}")
print("validated parent allocation: one node/task, 8 CPUs, 128G, 1xA100")
PY

require_clean_sha "${ORCHESTRATOR_REPO}" "${EXPECTED_ORCHESTRATOR_SHA}" orchestrator
for required in "${BUILDER}" "${ANALYZER}"; do
  if [[ ! -f "${required}" ]]; then
    echo "error: staged orchestrator lacks ${required}" >&2
    exit 2
  fi
done
SOURCE_REPO="$(validate_receipt_and_get_source_repo)"
EXPERIMENT_REPO="${SOURCE_REPO}"
require_clean_sha "${EXPERIMENT_REPO}" "${EXPERIMENT_SHA}" experiment
for required in \
  train_eval.py \
  configs/math10k_paper.json \
  scripts/cuda_step_guard.py \
  scripts/preflight_hpc.py \
  scripts/summarize_telemetry.py; do
  if [[ ! -f "${EXPERIMENT_REPO}/${required}" ]]; then
    echo "error: locked experiment repository lacks ${required}" >&2
    exit 2
  fi
done
for required in "${SOURCE_SELECTION}" "${SOURCE_FINAL_ASSETS}" "${SOURCE_FINAL_PLAN}"; do
  if [[ ! -s "${required}" ]]; then
    echo "error: required locked source artifact is missing: ${required}" >&2
    exit 2
  fi
done
validate_terminal_source_job
validate_source_plan

cache_key="models--${MODEL_ID//\//--}"
model_snapshot="${SHARED_HF_HOME}/hub/${cache_key}/snapshots/${MODEL_REVISION}"
model_marker="${SHARED_HF_HOME}/staged/${cache_key}/${MODEL_REVISION}.complete"
if [[ ! -d "${model_snapshot}" || ! -f "${model_marker}" ]]; then
  echo "error: pinned offline model is unavailable: ${MODEL_ID}@${MODEL_REVISION}" >&2
  exit 2
fi

export PRISM_REPO_ROOT="${EXPERIMENT_REPO}"
export PRISM_RUN_ROOT="${RUN_ROOT}"
export HF_HOME="${SHARED_HF_HOME}"
export HF_HUB_CACHE="${SHARED_HF_HOME}/hub"
export HUGGINGFACE_HUB_CACHE="${SHARED_HF_HOME}/hub"
export HF_DATASETS_CACHE="${CAMPAIGN_ROOT}/cache/datasets"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_XET=1
export TORCH_HOME="${CAMPAIGN_ROOT}/cache/torch"
export TORCH_EXTENSIONS_DIR="${CAMPAIGN_ROOT}/cache/torch-extensions"
export TRITON_CACHE_DIR="${CAMPAIGN_ROOT}/cache/triton"
export XDG_CACHE_HOME="${CAMPAIGN_ROOT}/cache/xdg"
export CUDA_CACHE_PATH="${CAMPAIGN_ROOT}/cache/cuda"
export TMPDIR="${JOB_TMP}"
export PYTHONDONTWRITEBYTECODE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export SLURM_EXPORT_ENV=ALL

mkdir -p \
  "${CAMPAIGN_ROOT}/artifacts" \
  "${CAMPAIGN_ROOT}/cache/datasets" \
  "${CAMPAIGN_ROOT}/cache/torch" \
  "${CAMPAIGN_ROOT}/cache/torch-extensions" \
  "${CAMPAIGN_ROOT}/cache/triton" \
  "${CAMPAIGN_ROOT}/cache/xdg" \
  "${CAMPAIGN_ROOT}/cache/cuda" \
  "${CAMPAIGN_ROOT}/logs" \
  "${CAMPAIGN_ROOT}/plans" \
  "${CAMPAIGN_ROOT}/preflight" \
  "${CAMPAIGN_ROOT}/selection" \
  "${STATUS_ROOT}" \
  "${JOB_TMP}"

CURRENT_PHASE="allocation-preflight"
write_job_status running 0
echo "job_id=${JOB_ID} host=$(hostname) campaign=${CAMPAIGN_ID}"
echo "orchestrator_sha=${EXPECTED_ORCHESTRATOR_SHA} experiment_sha=${EXPERIMENT_SHA}"
echo "source_job=${EXPECTED_SOURCE_JOB_ID} source_campaign=${SOURCE_CAMPAIGN_ROOT}"
nvidia-smi
"${PYTHON_BIN}" -m pip check
TRAIN_HELP="$("${PYTHON_BIN}" "${EXPERIMENT_REPO}/train_eval.py" --help)"
for option in \
  --validation_eval_interval \
  --validation_generate_numeric \
  --slaclip_target_non_small_clip_fraction \
  --allow_non_private_telemetry \
  --resume \
  --require_cuda; do
  if [[ "${TRAIN_HELP}" != *"${option}"* ]]; then
    echo "error: locked trainer lacks required option ${option}" >&2
    exit 2
  fi
done

serial_srun \
  "${CAMPAIGN_ROOT}/preflight" allocation-gpu \
  "${PYTHON_BIN}" -c \
  'import torch; torch.cuda.init(); assert torch.cuda.device_count() == 1, torch.cuda.device_count(); name=torch.cuda.get_device_name(0); assert "A100" in name, name; x=torch.ones((16,16),device="cuda"); assert float((x@x).sum()) == 4096.0; torch.cuda.synchronize(); print(f"validated_allocation_gpu={name}", flush=True)'
serial_srun \
  "${CAMPAIGN_ROOT}/preflight" model-data \
  "${PYTHON_BIN}" "${EXPERIMENT_REPO}/scripts/preflight_hpc.py" \
    --dataset math10k \
    --data-path "${EXPERIMENT_REPO}/LLM-Adapters/ft-training_set/math_10k.json" \
    --expected-data-sha256 0342d0d860ad8592b579329337c90e42eefd3d9f2898043140cbd120630418b8 \
    --output-root "${CAMPAIGN_ROOT}/preflight/model-data" \
    --require-cuda \
    --model-id "${MODEL_ID}" \
    --model-revision "${MODEL_REVISION}"

"${PYTHON_BIN}" - \
  "${CAMPAIGN_ROOT}/artifacts/environment-${JOB_ID}.txt" \
  "${EXPECTED_ORCHESTRATOR_SHA}" "${EXPERIMENT_SHA}" "${ENV_PREFIX}" <<'PY'
import importlib.metadata
import json
import os
import platform
import sys
import tempfile
from pathlib import Path

target = Path(sys.argv[1])
payload = {
    "schema_version": 1,
    "orchestrator_sha": sys.argv[2],
    "experiment_sha": sys.argv[3],
    "environment_prefix": sys.argv[4],
    "python": platform.python_version(),
    "packages": {name: importlib.metadata.version(name) for name in ("accelerate", "datasets", "opacus", "peft", "torch", "transformers")},
}
encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
target.parent.mkdir(parents=True, exist_ok=True)
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
atomic_sha256_sidecar "${CAMPAIGN_ROOT}/artifacts/environment-${JOB_ID}.txt"

cd "${EXPERIMENT_REPO}"
CURRENT_PHASE="source-final-resume"
write_job_status running 0
run_plan_serially "${SOURCE_FINAL_PLAN}" source

CURRENT_PHASE="dense-selection-prepare"
write_job_status running 0
"${PYTHON_BIN}" "${BUILDER}" prepare \
  --source-root "${SOURCE_CAMPAIGN_ROOT}" \
  --campaign-root "${CAMPAIGN_ROOT}" \
  --experiment-code-sha "${EXPERIMENT_SHA}" \
  >"${CAMPAIGN_ROOT}/logs/builder-prepare-${JOB_ID}.out" \
  2>"${CAMPAIGN_ROOT}/logs/builder-prepare-${JOB_ID}.err"
validate_dense_plan selection "${DENSE_SELECTION_PLAN}"
atomic_sha256_sidecar "${DENSE_SELECTION_PLAN}"

CURRENT_PHASE="dense-selection"
write_job_status running 0
run_plan_serially "${DENSE_SELECTION_PLAN}" dense

CURRENT_PHASE="dense-selection-lock"
write_job_status running 0
"${PYTHON_BIN}" "${BUILDER}" lock \
  --source-root "${SOURCE_CAMPAIGN_ROOT}" \
  --campaign-root "${CAMPAIGN_ROOT}" \
  --experiment-code-sha "${EXPERIMENT_SHA}" \
  >"${CAMPAIGN_ROOT}/logs/builder-lock-${JOB_ID}.out" \
  2>"${CAMPAIGN_ROOT}/logs/builder-lock-${JOB_ID}.err"
if [[ ! -s "${DENSE_LOCK}" || ! -e "${DENSE_FINAL_PLAN}" ]]; then
  echo "error: builder did not create the locked dense decision and final plan" >&2
  exit 2
fi
atomic_sha256_sidecar "${DENSE_LOCK}"
atomic_sha256_sidecar "${DENSE_FINAL_PLAN}"
validate_dense_plan final "${DENSE_FINAL_PLAN}"

CURRENT_PHASE="dense-final"
write_job_status running 0
if [[ -s "${DENSE_FINAL_PLAN}" ]]; then
  run_plan_serially "${DENSE_FINAL_PLAN}" dense
else
  "${PYTHON_BIN}" - "${DENSE_LOCK}" <<'PY'
import json
import math
import sys
from pathlib import Path

lock = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
passed = (lock.get("gate") or {}).get("passed")
best = float((lock.get("dense_best_fixed") or {})["clip_threshold"])
if passed is False:
    print("dense fixed-C gate did not pass; predeclared fresh final evaluation is skipped")
elif passed is True and math.isclose(best, 2.0, rel_tol=0.0, abs_tol=1e-12):
    print("dense fixed-C gate passed with best C=2; reusing the five locked source fixed-C2 final arms")
else:
    raise SystemExit("empty dense-final plan is inconsistent with its locked decision")
PY
fi

CURRENT_PHASE="strict-analysis"
write_job_status running 0
"${PYTHON_BIN}" "${ANALYZER}" \
  --source-root "${SOURCE_CAMPAIGN_ROOT}" \
  --campaign-root "${CAMPAIGN_ROOT}" \
  --experiment-code-sha "${EXPERIMENT_SHA}" \
  --strict \
  >"${CAMPAIGN_ROOT}/logs/analyzer-${JOB_ID}.out" \
  2>"${CAMPAIGN_ROOT}/logs/analyzer-${JOB_ID}.err"
ANALYSIS_MANIFEST="${CAMPAIGN_ROOT}/artifacts/dense_fixed_followup_manifest.json"
CANONICAL_MANIFEST="${CAMPAIGN_ROOT}/artifacts/manifest.json"
if [[ ! -s "${ANALYSIS_MANIFEST}" ]]; then
  echo "error: strict analyzer did not create its manifest" >&2
  exit 2
fi
"${PYTHON_BIN}" - "${ANALYSIS_MANIFEST}" "${CANONICAL_MANIFEST}" <<'PY'
import os
import sys
import tempfile
from pathlib import Path

source = Path(sys.argv[1]).resolve(strict=True)
target = Path(sys.argv[2])
encoded = source.read_bytes()
if target.exists():
    if target.read_bytes() != encoded:
        raise SystemExit(f"refusing inconsistent canonical campaign manifest: {target}")
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
atomic_sha256_sidecar "${CANONICAL_MANIFEST}"

CURRENT_PHASE="complete"
write_job_status completed 0
echo "Math-10K 4B dense fixed-C follow-up completed at $(date --iso-8601=seconds)."
