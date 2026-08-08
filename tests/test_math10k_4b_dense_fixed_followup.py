from __future__ import annotations

import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WRAPPER = ROOT / "scripts" / "submit_math10k_4b_dense_fixed_followup.sh"
WORKER = ROOT / "slurm" / "math10k_4b_dense_fixed_followup.sbatch"


def _embedded_python(path: Path) -> list[tuple[int, str]]:
    current: list[str] | None = None
    start = 0
    blocks: list[tuple[int, str]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if current is None and "<<'PY'" in line:
            current = []
            start = line_number + 1
        elif current is not None and line == "PY":
            blocks.append((start, "\n".join(current) + "\n"))
            current = None
        elif current is not None:
            current.append(line)
    assert current is None
    return blocks


def test_dense_followup_shell_syntax_and_portability() -> None:
    subprocess.run(["bash", "-n", str(WRAPPER)], check=True)
    wrapper = WRAPPER.read_text(encoding="utf-8")

    assert "sz1c24" not in wrapper
    assert "/iridisfs/home/" not in wrapper
    assert "/iridisfs/scratch/" not in wrapper
    assert 'SCRATCH_ROOT="${PRISM_SCRATCH_ROOT:-${SCRATCH:-/scratch/${USER_NAME}}}"' in wrapper
    assert 'USER_HOME="${PRISM_USER_HOME:' in wrapper
    assert 'REPO_ROOT="${PRISM_REPO_ROOT:' in wrapper
    assert 'REFERENCE_CAMPAIGN_ROOT="${PRISM_SOURCE_CAMPAIGN_ROOT:' in wrapper
    assert 'STAGED_REPO_ROOT="${PRISM_STAGED_REPO_ROOT:' in wrapper
    assert 'LOG_ROOT="${CAMPAIGN_ROOT}/slurm"' in wrapper
    assert '--output="${LOG_ROOT}/%x-%j.out"' in wrapper
    assert '--error="${LOG_ROOT}/%x-%j.err"' in wrapper


def test_dense_followup_is_exactly_one_single_a100_allocation() -> None:
    wrapper = WRAPPER.read_text(encoding="utf-8")

    assert wrapper.count('SUBMISSION_OUTPUT="$(sbatch ') == 1
    assert "--nodes=1" in wrapper
    assert "--ntasks=1" in wrapper
    assert "--cpus-per-task=8" in wrapper
    assert "--mem=128G" in wrapper
    assert "--gres=gpu:a100:1" in wrapper
    assert "--time=2-12:00:00" in wrapper
    assert "--no-requeue" in wrapper
    assert "--export=NONE" in wrapper
    assert '--dependency="${DEPENDENCY_SPEC}"' in wrapper
    assert 'DEPENDENCY_SPEC="afterany:${REFERENCE_JOB_ID}"' in wrapper
    assert (
        'REFERENCE_JOB_ID="${PRISM_SOURCE_JOB_ID:-'
        '${PRISM_REFERENCE_JOB_ID:-1365564}}"' in wrapper
    )
    assert "--array" not in wrapper
    assert "srun " not in wrapper


def test_dense_followup_uses_worker_fourteen_argument_contract() -> None:
    wrapper = WRAPPER.read_text(encoding="utf-8")
    invocation = re.search(
        r'SUBMISSION_OUTPUT="\$\(sbatch .*?"\$\{WORKER\}" \\\n(.*?) 2>&1\)"',
        wrapper,
        flags=re.DOTALL,
    )
    assert invocation is not None
    arguments = re.findall(
        r'^\s+"\$\{([A-Z_]+)\}"(?: \\)?$',
        invocation.group(1),
        re.MULTILINE,
    )
    assert arguments == [
        "USER_NAME",
        "STAGED_REPO_ROOT",
        "ENV_PREFIX",
        "RUN_ROOT",
        "SHARED_HF_HOME",
        "ORCHESTRATOR_SHA",
        "CAMPAIGN_ID",
        "SCRATCH_ROOT",
        "MODEL_ID",
        "MODEL_REVISION",
        "MODEL_SLUG",
        "REFERENCE_CAMPAIGN_ROOT",
        "REFERENCE_JOB_ID",
        "EXPERIMENT_SHA",
    ]


def test_dense_followup_locks_source_model_environment_and_clean_git() -> None:
    wrapper = WRAPPER.read_text(encoding="utf-8")

    assert (
        'EXPERIMENT_SHA="${PRISM_EXPERIMENT_CODE_SHA:-'
        'feba5968285dc1651bf7726327932b3f625ace22}"' in wrapper
    )
    assert 'MODEL_ID="google/gemma-3-4b-pt"' in wrapper
    assert (
        'MODEL_REVISION="cc012e0a6d0787b4adcc0fa2c4da74402494554d"'
        in wrapper
    )
    assert 'ENV_TAG="${PRISM_ENV_TAG:-71bce55}"' in wrapper
    assert 'status --porcelain' in wrapper
    assert "refusing to stage a formal campaign from a dirty worktree" in wrapper
    assert "git clone --quiet --no-hardlinks --no-checkout" in wrapper
    assert 'math10k-4b-dense-fixed-followup-${SHORT_SHA}-v1' in wrapper
    assert "must include the locked orchestrator SHA" in wrapper
    assert "scripts/build_math10k_4b_dense_fixed_followup.py" in wrapper
    assert "scripts/analyze_math10k_4b_dense_fixed_followup.py" in wrapper
    assert "slurm/math10k_4b_dense_fixed_followup.sbatch" in wrapper


def test_dense_followup_receipt_records_dual_sha_resources_and_dependency() -> None:
    wrapper = WRAPPER.read_text(encoding="utf-8")

    for field in (
        '"orchestrator_code_sha"',
        '"experiment_code_sha"',
        '"staged_orchestrator_repo"',
        '"staged_experiment_repo"',
        '"reference_campaign_root"',
        '"reference_job_id"',
        '"dependency"',
        '"gpu_gres": "gpu:a100:1"',
        '"tasks": 1',
        '"cpus_per_task": 8',
        '"memory": "128G"',
        '"walltime": "2-12:00:00"',
    ):
        assert field in wrapper
    assert 'SUBMISSION_RECEIPT="${CAMPAIGN_ROOT}/submission_receipt.json"' in wrapper
    assert "os.replace(temporary, target)" in wrapper


def test_dense_followup_rejects_duplicates_and_unsafe_resume() -> None:
    wrapper = WRAPPER.read_text(encoding="utf-8")

    assert "campaign already has submission state; refusing a duplicate job" in wrapper
    assert "prior job is still active; refusing duplicate resume" in wrapper
    assert "prior job completed; a resume submission is forbidden" in wrapper
    assert "FAILED|TIMEOUT|CANCELLED|NODE_FAIL|OUT_OF_MEMORY|PREEMPTED" in wrapper
    assert 'grep -qx \'state=completed\'' in wrapper
    assert '"${CAMPAIGN_ROOT}/artifacts/manifest.json"' in wrapper


def test_dense_followup_embedded_python_compiles() -> None:
    blocks = _embedded_python(WRAPPER)
    assert len(blocks) == 3
    for line_number, source in blocks:
        compile(source, f"{WRAPPER}:{line_number}", "exec")


def test_dense_followup_worker_exists_and_is_shell_valid() -> None:
    # Kept separate so a syntax regression in the worker is attributed to the
    # queued executable rather than to the submission wrapper.
    assert WORKER.is_file()
    subprocess.run(["bash", "-n", str(WORKER)], check=True)
    worker = WORKER.read_text(encoding="utf-8")
    assert 'status != "empty-reuse-source-fixed-c2"' in worker
