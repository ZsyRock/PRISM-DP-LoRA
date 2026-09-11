from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKER = ROOT / "slurm" / "math10k_4b_refinement_campaign.sbatch"
WRAPPER = ROOT / "scripts" / "submit_math10k_4b_refinement_campaign.sh"
REGISTRY_BUILDER = ROOT / "scripts" / "build_math10k_4b_refinement_registry.py"


def _embedded_python(path: Path, function_name: str) -> str:
    text = path.read_text(encoding="utf-8")
    start = text.index(f"{function_name}() {{")
    match = re.search(r"<<'PY'[^\n]*\n(.*?)\nPY\n", text[start:], flags=re.DOTALL)
    assert match is not None
    return match.group(1)


def _run_plan_builder(
    *,
    mode: str,
    campaign: Path,
    stage1_selection: Path,
    locked_selection: Path,
) -> list[list[str]]:
    result = subprocess.run(
        [
            sys.executable,
            "-",
            mode,
            str(campaign / "screen" / "candidate_registry.json"),
            str(campaign),
            str(stage1_selection),
            str(locked_selection),
            "gemma-3-4b-pt",
        ],
        input=_embedded_python(WORKER, "create_refinement_plan"),
        text=True,
        capture_output=True,
        check=True,
    )
    return [line.split("|") for line in result.stdout.splitlines() if line]


def test_refinement_shell_syntax_resources_and_portability() -> None:
    subprocess.run(["bash", "-n", str(WORKER), str(WRAPPER)], check=True)
    worker = WORKER.read_text(encoding="utf-8")
    wrapper = WRAPPER.read_text(encoding="utf-8")
    combined = worker + wrapper
    assert "sz1c24" not in combined
    assert "/iridisfs/home/" not in combined
    assert "/iridisfs/scratch/" not in combined
    assert "--gres=gpu:a100:1" in worker
    assert 'PARTITION="${PRISM_SLURM_PARTITION:-a100}"' in wrapper
    assert 'GPU_GRES="${PRISM_GPU_GRES:-gpu:a100:2}"' in wrapper
    assert 'WALLTIME="${PRISM_WALLTIME:-2-12:00:00}"' in wrapper
    assert 'HOST_MEMORY="${PRISM_HOST_MEMORY:-256G}"' in wrapper
    assert "--no-requeue" in wrapper
    assert "--nodes=1" in wrapper
    assert "--ntasks=2" in wrapper
    assert 'SUBMISSION_RECEIPT="${CAMPAIGN_ROOT}/submission_receipt.json"' in wrapper
    assert "--resume-submit" in wrapper
    assert 'echo "submitted_job_id=${SUBMITTED_JOB_ID}"' in wrapper
    assert "scripts/build_math10k_4b_refinement_registry.py" in combined
    assert "scripts/analyze_math10k_4b_refinement_campaign.py" in combined
    assert "--batch_size 64" in worker
    assert "--micro_batch_size 4" in worker


def test_refinement_main_is_one_allocation_without_subjob_resubmission() -> None:
    worker = WORKER.read_text(encoding="utf-8")
    wrapper = WRAPPER.read_text(encoding="utf-8")
    main = worker[worker.rindex('CURRENT_PHASE="preflight"') :]
    assert wrapper.count('SUBMISSION_OUTPUT="$(sbatch ') == 1
    assert "create_refinement_plan stage1" in main
    assert "create_refinement_plan stage2" in main
    assert "create_refinement_plan final" in main
    assert main.index("run_formal_selector stage2") < main.index(
        "lock_final_evaluation_assets"
    ) < main.index("create_refinement_plan final")
    assert "schedule-source" not in main
    assert "replay-lock" not in main
    assert main.index("scripts/analyze_mechanism_campaign.py") < main.index(
        "scripts/analyze_math10k_4b_refinement_campaign.py"
    )
    assert '!= "27"' in main
    assert '!= "20"' in main
    assert '!= "10"' in main


def test_refinement_plan_has_27_then_20_then_five_exact_pairs(
    tmp_path: Path,
) -> None:
    campaign = tmp_path / "campaign"
    (campaign / "screen").mkdir(parents=True)
    (campaign / "selection").mkdir()
    registry_path = campaign / "screen" / "candidate_registry.json"
    subprocess.run(
        [
            sys.executable,
            str(REGISTRY_BUILDER),
            "--campaign-root",
            str(campaign),
            "--output",
            str(registry_path),
            "--model-id",
            "google/gemma-3-4b-pt",
            "--model-revision",
            "a" * 40,
            "--code-sha",
            "b" * 40,
        ],
        check=True,
    )
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    slaclip = [
        candidate
        for candidate in registry["candidates"]
        if candidate["family"] == "slaclip"
    ]
    stage1_selection = campaign / "selection" / "stage1-selection.json"
    stage1_selection.write_text(
        json.dumps(
            {
                "top_slaclip": [
                    {"candidate_id": candidate["id"]} for candidate in slaclip[:3]
                ]
            }
        ),
        encoding="utf-8",
    )
    locked_selection = campaign / "selection" / "selection.json"
    locked_selection.write_text(
        json.dumps({"selected_slaclip": {"candidate_id": slaclip[0]["id"]}}),
        encoding="utf-8",
    )

    stage1 = _run_plan_builder(
        mode="stage1",
        campaign=campaign,
        stage1_selection=stage1_selection,
        locked_selection=locked_selection,
    )
    assert len(stage1) == 27
    assert {row[4] for row in stage1} == {"42"}
    assert len({row[2] for row in stage1}) == 27

    stage2 = _run_plan_builder(
        mode="stage2",
        campaign=campaign,
        stage1_selection=stage1_selection,
        locked_selection=locked_selection,
    )
    assert len(stage2) == 20
    assert {row[4] for row in stage2} == {"43", "44", "45", "46"}
    for seed in ("43", "44", "45", "46"):
        rows = [row for row in stage2 if row[4] == seed]
        assert len(rows) == 5
        assert {row[2] for row in rows} == {
            slaclip[0]["id"],
            slaclip[1]["id"],
            slaclip[2]["id"],
            "fixed-c1",
            "fixed-c2",
        }

    final = _run_plan_builder(
        mode="final",
        campaign=campaign,
        stage1_selection=stage1_selection,
        locked_selection=locked_selection,
    )
    assert len(final) == 10
    assert [int(final[index][4]) for index in range(0, 10, 2)] == [
        191,
        223,
        257,
        293,
        331,
    ]
    for index in range(0, 10, 2):
        candidate, reference = final[index : index + 2]
        assert candidate[4] == reference[4]
        assert candidate[3:10] == [
            "slaclip",
            candidate[4],
            "slaclip",
            "1.25",
            "0.985",
            "0.05",
            "NA",
        ]
        assert reference[2:10] == [
            "fixed-c2",
            "baseline",
            reference[4],
            "baseline",
            "2.0",
            "NA",
            "NA",
            "NA",
        ]
        assert candidate[-1].endswith(f"seed-{candidate[4]}/slaclip")
        assert reference[-1].endswith(f"seed-{reference[4]}/baseline")


def test_all_embedded_python_compiles() -> None:
    for path in (WORKER, WRAPPER):
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
        assert blocks
        for line_number, source in blocks:
            compile(source, f"{path}:{line_number}", "exec")
