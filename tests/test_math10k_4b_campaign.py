from __future__ import annotations

import importlib.util
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKER = ROOT / "slurm" / "math10k_4b_dynamics_campaign.sbatch"
WRAPPER = ROOT / "scripts" / "submit_math10k_4b_dynamics_campaign.sh"
SELECTOR_PATH = ROOT / "scripts" / "select_validation_candidates.py"


def _embedded_python(function_name: str) -> str:
    text = WORKER.read_text(encoding="utf-8")
    start = text.index(f"{function_name}() {{")
    match = re.search(r"<<'PY'\n(.*?)\nPY\n", text[start:], flags=re.DOTALL)
    assert match is not None
    return match.group(1)


def test_campaign_shell_syntax_and_portability() -> None:
    subprocess.run(["bash", "-n", str(WORKER), str(WRAPPER)], check=True)
    worker = WORKER.read_text(encoding="utf-8")
    wrapper = WRAPPER.read_text(encoding="utf-8")
    combined = worker + wrapper
    assert "sz1c24" not in combined
    assert "/iridisfs/home/" not in combined
    assert "/iridisfs/scratch/" not in combined
    assert "--gres=gpu:h200:1" in worker
    assert "--gres=\"${GPU_GRES}\"" in wrapper
    assert 'WALLTIME="${PRISM_WALLTIME:-1-16:00:00}"' in wrapper
    assert 'HOST_MEMORY="${PRISM_HOST_MEMORY:-256G}"' in wrapper
    assert 'echo "the worker reserves two concurrent lanes of 128G each"' in wrapper
    assert "--resume-submit" in wrapper
    assert 'SUBMISSION_RECEIPT="${CAMPAIGN_ROOT}/submission_receipt.json"' in combined
    assert 'echo "submitted_job_id=${SUBMITTED_JOB_ID}"' in wrapper


def test_worker_uses_formal_selector_and_validation_smoke_contract() -> None:
    text = WORKER.read_text(encoding="utf-8")
    assert "select_candidates() {" not in text
    assert "run_formal_selector stage1" in text
    assert "run_formal_selector stage2" in text
    assert "scripts/select_validation_candidates.py" in text
    assert '--registry "screen/candidate_registry.json"' in text
    assert "--validation_eval_interval 1" in text
    assert "--validation_generate_numeric" in text
    assert '!= [0, 1, 2]' in text
    assert 'int(split.get("validation_rows", -1)) != 8' in text
    assert 'for key in ("numeric_exact_correct", "numeric_parse_failures")' in text
    assert 'locked_files = [Path(item).resolve() for item in sys.argv[3:8]]' in text
    assert "output = Path(sys.argv[8]).resolve()" in text
    assert 'temporary="$(mktemp "${output}.tmp.XXXXXX")"' in text
    assert 'install_immutable_file "${temporary}" "${output}"' in text
    assert "environment_freeze.txt" in text
    assert "public_numeric_predictions.json" in text
    assert "telemetry_steps.csv" in text
    assert "adapter_config.json" in text
    assert "adapter_model.safetensors" in text
    assert "run_index.csv" in text
    assert '"schedule_privacy_class": "DP_DERIVED_FROM_3_SLACLIP_RUNS"' in text
    assert '"epsilon": 54.0' in text
    assert '"delta": 9e-5' in text
    assert '"release_count": 9' in text


def test_all_embedded_python_compiles() -> None:
    for path in (WORKER, WRAPPER):
        lines = path.read_text(encoding="utf-8").splitlines()
        blocks: list[tuple[int, str]] = []
        current: list[str] | None = None
        start = 0
        for line_number, line in enumerate(lines, 1):
            if current is None and "<<'PY'" in line:
                current = []
                start = line_number + 1
            elif current is not None and line == "PY":
                blocks.append((start, "\n".join(current) + "\n"))
                current = None
            elif current is not None:
                current.append(line)
        assert current is None
        assert blocks
        for line_number, source in blocks:
            compile(source, f"{path}:{line_number}", "exec")


def test_generated_registry_matches_formal_selector_schema(tmp_path: Path) -> None:
    registry_path = tmp_path / "screen" / "candidate_registry.json"
    subprocess.run(
        [
            sys.executable,
            "-",
            str(registry_path),
            "google/gemma-3-4b-pt",
            "c" * 40,
            "d" * 40,
        ],
        input=_embedded_python("create_candidate_registry"),
        text=True,
        check=True,
    )
    payload = json.loads(registry_path.read_text(encoding="utf-8"))
    assert len(payload["candidates"]) == 26
    assert sum(item["family"] == "fixed" for item in payload["candidates"]) == 8
    assert sum(item["family"] == "slaclip" for item in payload["candidates"]) == 18
    assert all(set(item["runs"]) == {"42", "43", "44"} for item in payload["candidates"])
    assert all(
        run["run_status"].startswith("screen/runs/")
        for item in payload["candidates"]
        for run in item["runs"].values()
    )
    spec = importlib.util.spec_from_file_location("campaign_selector", SELECTOR_PATH)
    assert spec is not None and spec.loader is not None
    selector = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(selector)
    protocol = selector._validate_protocol(payload)
    candidates = selector._candidate_map(payload)
    assert protocol["selection_metric"] == selector.NUMERIC_EXACT_METRIC
    assert len(candidates) == 26


def test_replay_schedule_averages_three_seed_trajectories(tmp_path: Path) -> None:
    campaign = tmp_path / "campaign"
    selection = campaign / "selection" / "selection.json"
    schedule = campaign / "schedule" / "replay_schedule.json"
    controls = campaign / "schedule" / "control_manifest.json"
    selection.parent.mkdir(parents=True)
    candidate_id = "sla-selected"
    selection.write_text(
        json.dumps(
            {
                "selected_slaclip": {
                    "candidate_id": candidate_id,
                    "params": {"dp_max_grad_norm": 1.0},
                }
            }
        ),
        encoding="utf-8",
    )
    for seed, offset in ((42, 0.0), (43, 0.3), (44, 0.6)):
        root = campaign / "screen" / "runs" / candidate_id / f"seed-{seed}" / "adapter"
        root.mkdir(parents=True)
        root.joinpath("run_status.json").write_text(
            json.dumps(
                {
                    "state": "completed",
                    "config": {"method": "slaclip"},
                    "config_fingerprint": f"fp-{seed}",
                }
            ),
            encoding="utf-8",
        )
        values = [1.0 if step == 1 else 1.0 + offset + step / 1000 for step in range(1, 301)]
        root.joinpath("train_log.jsonl").write_text(
            "".join(
                json.dumps({"step": step, "dp_clip_threshold": value}) + "\n"
                for step, value in enumerate(values, 1)
            ),
            encoding="utf-8",
        )
    subprocess.run(
        [sys.executable, "-", str(campaign), str(selection), str(schedule), str(controls)],
        input=_embedded_python("build_replay_schedule_and_controls"),
        text=True,
        check=True,
    )
    replay = json.loads(schedule.read_text(encoding="utf-8"))
    control = json.loads(controls.read_text(encoding="utf-8"))
    assert len(replay["source_logs"]) == 3
    assert len(replay["clip_thresholds"]) == 300
    assert replay["clip_thresholds"][0] == 1.0
    assert math.isclose(replay["clip_thresholds"][1], 1.302, abs_tol=1e-12)
    expected_rms = math.sqrt(
        sum(value * value for value in replay["clip_thresholds"]) / 300
    )
    expected_geometric = math.exp(
        sum(math.log(value) for value in replay["clip_thresholds"]) / 300
    )
    assert math.isclose(control["matched_fixed_noise_rms_clip"], expected_rms, abs_tol=1e-12)
    assert math.isclose(control["schedule_noise_rms_clip"], expected_rms, abs_tol=1e-12)
    assert math.isclose(
        control["schedule_geometric_mean_clip"], expected_geometric, abs_tol=1e-12
    )
    assert control["schedule_privacy_class"] == "DP_DERIVED_FROM_3_SLACLIP_RUNS"
    assert replay["schedule_privacy_class"] == "DP_DERIVED_FROM_3_SLACLIP_RUNS"
    composition = replay["privacy_accounting"][
        "schedule_plus_three_replay_plus_three_matched_fixed"
    ]
    assert composition == {"epsilon": 54.0, "delta": 9e-5, "release_count": 9,
                           "derivation": composition["derivation"]}
    assert composition["derivation"].startswith("3 schedule-source runs")


def test_wrapper_receipt_blocks_duplicates_and_requires_explicit_resume(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    required_files = (
        "train_eval.py",
        "scripts/preflight_hpc.py",
        "scripts/smoke_dp_path.py",
        "scripts/select_validation_candidates.py",
        "scripts/summarize_telemetry.py",
    )
    for relative in required_files:
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# fixture\n", encoding="utf-8")
    worker_copy = repo / "slurm" / WORKER.name
    worker_copy.parent.mkdir(parents=True, exist_ok=True)
    worker_copy.write_bytes(WORKER.read_bytes())
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.email", "fixture@example.invalid"], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.name", "Fixture"], check=True)
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-qm", "fixture"], check=True)

    fake_bin = tmp_path / "fake-bin"
    fake_bin.mkdir()
    state = tmp_path / "fake-slurm-state"
    commands = {
        "sbatch": """#!/usr/bin/env bash
set -euo pipefail
for argument in "$@"; do
  if [[ "${argument}" == "--test-only" ]]; then
    echo "test-only accepted"
    exit 0
  fi
done
state="${FAKE_SLURM_STATE:?}"
if [[ -s "${state}" ]]; then
  job_id="$(( $(<"${state}") + 1 ))"
else
  job_id=810001
fi
printf '%s\n' "${job_id}" >"${state}"
printf '%s\n' "${job_id}"
""",
        "squeue": """#!/usr/bin/env bash
exit 0
""",
        "sacct": """#!/usr/bin/env bash
set -euo pipefail
wanted=""
while (($#)); do
  if [[ "$1" == "-j" ]]; then
    wanted="$2"
    shift 2
  else
    shift
  fi
done
printf '%s|TIMEOUT\n' "${wanted}"
""",
    }
    for name, source in commands.items():
        path = fake_bin / name
        path.write_text(source, encoding="utf-8")
        path.chmod(0o755)

    environment = tmp_path / "environment"
    (environment / "bin").mkdir(parents=True)
    (environment / "bin" / "python").symlink_to(Path(sys.executable).resolve())
    user_home = tmp_path / "home"
    user_home.mkdir()
    scratch = tmp_path / "scratch"
    run_root = scratch / "runs"
    hf_home = scratch / "hf"
    revision = "cc012e0a6d0787b4adcc0fa2c4da74402494554d"
    cache_key = "models--google--gemma-3-4b-pt"
    (hf_home / "hub" / cache_key / "snapshots" / revision).mkdir(parents=True)
    marker = hf_home / "staged" / cache_key / f"{revision}.complete"
    marker.parent.mkdir(parents=True)
    marker.write_text("complete\n", encoding="utf-8")
    campaign_id = "receipt-fixture"
    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{fake_bin}:{env['PATH']}",
            "FAKE_SLURM_STATE": str(state),
            "PRISM_USER_NAME": subprocess.check_output(
                ["id", "-un"], text=True
            ).strip(),
            "PRISM_USER_HOME": str(user_home),
            "PRISM_SCRATCH_ROOT": str(scratch),
            "PRISM_REPO_ROOT": str(repo),
            "PRISM_ENV_PREFIX": str(environment),
            "PRISM_RUN_ROOT": str(run_root),
            "PRISM_HF_HOME": str(hf_home),
            "PRISM_CAMPAIGN_ID": campaign_id,
        }
    )

    def invoke(mode: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["bash", str(WRAPPER), mode],
            env=env,
            text=True,
            capture_output=True,
            timeout=30,
        )

    campaign_root = run_root / "campaigns" / campaign_id
    receipt = campaign_root / "submission_receipt.json"
    test_only = invoke("--test-only")
    assert test_only.returncode == 0, test_only.stderr
    assert not receipt.exists()

    submitted = invoke("--submit")
    assert submitted.returncode == 0, submitted.stderr
    first = json.loads(receipt.read_text(encoding="utf-8"))
    assert first["state"] == "submitted"
    assert first["current_job_id"] == "810001"
    assert len(first["attempts"]) == 1
    assert first["resources"]["total_memory"] == "256G"
    assert first["resources"]["memory_per_lane"] == "128G"
    assert first["resources"]["walltime"] == "1-16:00:00"

    duplicate = invoke("--submit")
    assert duplicate.returncode != 0
    assert "refusing a duplicate job" in duplicate.stderr
    assert len(json.loads(receipt.read_text(encoding="utf-8"))["attempts"]) == 1

    resumed = invoke("--resume-submit")
    assert resumed.returncode == 0, resumed.stderr
    second = json.loads(receipt.read_text(encoding="utf-8"))
    assert second["state"] == "submitted"
    assert second["current_job_id"] == "810002"
    assert len(second["attempts"]) == 2
    assert second["attempts"][1]["previous_job_id"] == "810001"
    assert second["attempts"][1]["previous_job_terminal_state"] == "TIMEOUT"
