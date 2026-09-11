from __future__ import annotations

import importlib.util
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
WORKER = ROOT / "slurm" / "math10k_4b_dynamics_campaign.sbatch"
WRAPPER = ROOT / "scripts" / "submit_math10k_4b_dynamics_campaign.sh"
SELECTOR_PATH = ROOT / "scripts" / "select_validation_candidates.py"


def _embedded_python(function_name: str) -> str:
    text = WORKER.read_text(encoding="utf-8")
    start = text.index(f"{function_name}() {{")
    match = re.search(r"<<'PY'[^\n]*\n(.*?)\nPY\n", text[start:], flags=re.DOTALL)
    assert match is not None
    return match.group(1)


def _create_fixed_manifest(campaign: Path) -> tuple[Path, dict]:
    manifest_path = campaign / "screen" / "fixed_scan_manifest.json"
    subprocess.run(
        [
            sys.executable,
            "-",
            str(manifest_path),
            "google/gemma-3-4b-pt",
            "c" * 40,
            "d" * 40,
        ],
        input=_embedded_python("create_fixed_scan_manifest"),
        text=True,
        check=True,
    )
    return manifest_path, json.loads(manifest_path.read_text(encoding="utf-8"))


def _standard_derived_registry(campaign: Path) -> tuple[Path, Path, dict]:
    """Build the standard 8 fixed + 2 C0 x 5 rho selector fixture."""

    fixed_manifest_path, fixed_manifest = _create_fixed_manifest(campaign)
    rhos = (0.91, 0.93, 0.95, 0.97, 0.99)
    slaclip = []
    for c0 in (0.5, 1.5):
        c_slug = format(c0, ".12g").replace(".", "p")
        for rho in rhos:
            rho_slug = format(rho, ".12g").replace(".", "p")
            candidate_id = f"sla-c{c_slug}-r{rho_slug}-e0p15"
            root = Path("screen") / "runs" / candidate_id
            runs = {
                str(seed): {
                    "run_status": str(
                        root / f"seed-{seed}" / "adapter" / "run_status.json"
                    ),
                    "validation_metrics": str(
                        root
                        / f"seed-{seed}"
                        / "results"
                        / "validation"
                        / "validation_metrics.json"
                    ),
                    "split_manifest": str(
                        root
                        / f"seed-{seed}"
                        / "results"
                        / "validation"
                        / "split_manifest.json"
                    ),
                }
                for seed in (42, 43, 44)
            }
            slaclip.append(
                {
                    "id": candidate_id,
                    "family": "slaclip",
                    "method": "slaclip",
                    "params": {
                        "dp_max_grad_norm": c0,
                        "slaclip_target_non_small_clip_fraction": rho,
                        "slaclip_eta": 0.15,
                        "slaclip_num_slots": 15,
                        "slaclip_c_min": 0.1,
                        "slaclip_c_max": 15.0,
                    },
                    "runs": runs,
                }
            )
    registry = {
        "schema_version": 1,
        "selection_protocol": fixed_manifest["selection_protocol"],
        "candidates": [*fixed_manifest["fixed_candidates"], *slaclip],
    }
    registry_path = campaign / "screen" / "candidate_registry.json"
    registry_path.write_text(
        json.dumps(registry, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return fixed_manifest_path, registry_path, registry


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
    assert 'WALLTIME="${PRISM_WALLTIME:-2-12:00:00}"' in wrapper
    assert 'HOST_MEMORY="${PRISM_HOST_MEMORY:-256G}"' in wrapper
    assert 'EXCLUDE_NODES="${PRISM_SLURM_EXCLUDE:-}"' in wrapper
    assert 'SBATCH_ARGS+=(--exclude="${EXCLUDE_NODES}")' in wrapper
    assert 'echo "the worker reserves two concurrent lanes of 128G each"' in wrapper
    assert "--resume-submit" in wrapper
    assert 'SUBMISSION_RECEIPT="${CAMPAIGN_ROOT}/submission_receipt.json"' in combined
    assert 'echo "submitted_job_id=${SUBMITTED_JOB_ID}"' in wrapper
    assert "scripts/cuda_step_guard.py" in combined
    assert "scripts/derive_slaclip_target_grid.py" in combined
    assert "run_dual_cuda_guard_pair" in worker
    assert "max_attempts=3" in worker
    assert 'if [[ -e "${marker_path}" ]]' in worker
    assert "must never be auto-repeated" in worker


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
    assert 'locked_files = [Path(item).resolve() for item in sys.argv[3:11]]' in text
    assert "output = Path(sys.argv[11]).resolve()" in text
    assert 'temporary="$(mktemp "${output}.tmp.XXXXXX")"' in text
    assert 'install_immutable_file "${temporary}" "${output}"' in text
    assert "environment_freeze.txt" in text
    assert "public_numeric_predictions.json" in text
    assert "telemetry_steps.csv" in text
    assert "adapter_config.json" in text
    assert "adapter_model.safetensors" in text
    assert "run_index.csv" in text
    assert "paired_final_accuracy.csv" in text
    assert "paired_final_mechanism.csv" in text
    assert '"evaluation_config": arm_root / "results" / "evaluation_config.json"' in text
    assert '"raw_training_loss_mean"' in text
    assert '"step_level_source": "each_run/results/research_raw/telemetry_steps.csv"' in text
    assert '"analysis_status": "LOCKED_FRESH_SEED_FINAL_COMPLETE"' in text
    assert '"campaign_primary_metric": "clean_three_task_macro_accuracy"' in text
    assert 'sum(row["is_primary"] is True for row in paired_rows) != 1' in text
    assert '"confidence_interval": "two_sided_paired_t_95_df4"' in text
    assert '--data-path "${REPO_ROOT}/LLM-Adapters/ft-training_set/math_10k.json"' in text
    assert "lock_final_evaluation_assets" in text
    assert text.index("create_fixed_scan_manifest\n") < text.index(
        'create_plan fixed-scan "${FIXED_SCAN_PLAN}"'
    ) < text.index("derive_locked_slaclip_grid\n") < text.index(
        'create_plan slaclip-screen "${SLACLIP_SCREEN_PLAN}"'
    ) < text.index('run_formal_selector stage1 "${STAGE1_SELECTION}"')
    assert text.index('run_formal_selector stage2 "${LOCKED_SELECTION}"') < text.index(
        'create_plan schedule-source "${SCHEDULE_SOURCE_PLAN}"'
    ) < text.index("build_replay_schedule_and_controls\n") < text.index(
        "lock_final_evaluation_assets\n"
    ) < text.index('create_plan final "${FINAL_PLAN}"')
    assert 'root / "plans" / "fixed-scan.tsv"' in text
    assert 'root / "plans" / "slaclip-screen.tsv"' in text
    assert 'root / "plans" / "schedule-source.tsv"' in text
    assert 'root / "plans" / "final.tsv"' in text
    assert '--top-slaclip 3' in text
    assert '--top-fixed 2' in text
    assert 'stage1_run_count = 18' in text
    assert 'fixed_scan_run_count != 8' in text
    assert 'slaclip_screen_run_count != 10' in text
    assert '"NON_PRIVATE_data_dependent_fixed_scan_to_full_slaclip_grid;_"' in text
    assert '"canonical-full-slaclip-paper-default"' in text
    assert 'for seed in (17, 29, 71, 101, 137):' in text
    assert '--protocol_stage final' in text
    assert '--run_eval true' in text
    assert 'campaign_worker_model_evaluation_begins_after_stage2_selection_lock' in text
    assert 'test_assets_and_prior_test_results_were_accessed_before_campaign_design' in text
    assert 'DP_DERIVED_VIA_' in text
    assert 'screen_run_count' in text
    assert 'all_screen_and_fresh_final_dp_outputs' in text
    assert 'DP_DERIVED_FROM_3_SLACLIP_RUNS' not in text
    assert '"epsilon": 78.0' not in text


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


def _locked_evaluation_asset_manifest(tmp_path: Path) -> tuple[Path, dict]:
    result = subprocess.run(
        [sys.executable, "-", str(ROOT), str(ROOT / "README.md")],
        input=_embedded_python("lock_final_evaluation_assets"),
        text=True,
        check=True,
        capture_output=True,
    )
    manifest_path = tmp_path / "final-evaluation-assets.json"
    manifest_path.write_text(result.stdout, encoding="utf-8")
    return manifest_path, json.loads(result.stdout)


def test_locked_evaluation_assets_use_conservative_auditable_decontamination(
    tmp_path: Path,
) -> None:
    _, manifest = _locked_evaluation_asset_manifest(tmp_path)
    assert manifest["schema_version"] == 3
    assert manifest["access_policy"] == (
        "campaign_worker_model_evaluation_begins_after_stage2_selection_lock"
    )
    assert manifest["not_an_untouched_test_set"] is True
    assert manifest["decontamination"]["normalization"]["id"] == (
        "instruction_input_nfkc_casefold_whitespace_collapse_v1"
    )
    assets = {item["task"]: item for item in manifest["assets"]}
    assert {
        task: (
            asset["prompt_overlap"]["overlap_count"],
            asset["prompt_overlap"]["clean_count"],
        )
        for task, asset in assets.items()
    } == {
        "gsm8k": (0, 1319),
        "AQuA": (0, 254),
        "mawps": (53, 185),
        "SVAMP": (0, 1000),
    }
    mawps_overlap = assets["mawps"]["prompt_overlap"]
    assert mawps_overlap["answer_string_exact_overlap_count"] == 38
    assert mawps_overlap["answer_evaluator_equivalent_overlap_count"] == 53
    assert len(mawps_overlap["overlap_training_index_mapping"]) == 53
    assert len(mawps_overlap["overlap_training_index_mapping_sha256"]) == 64


def _make_final_validation_fixture(
    tmp_path: Path,
    *,
    forge_summary: bool,
) -> tuple[Path, Path]:
    asset_manifest_path, manifest = _locked_evaluation_asset_manifest(tmp_path)
    run_root = tmp_path / "final-run"
    adapter = run_root / "adapter"
    results = run_root / "results"
    adapter.mkdir(parents=True)
    results.mkdir(parents=True)
    status = {
        "state": "completed",
        "update_steps": 300,
        "config_fingerprint": "fixture-fingerprint",
        "model_revision": "fixture-requested-revision",
        "resolved_model_revision": "fixture-resolved-revision",
        "config": {
            "method": "baseline",
            "seed": 17,
            "protocol_stage": "final",
            "val_set_size": 0,
            "run_eval": True,
        },
        "privacy_accounting": {"epsilon_spent": 6.0},
    }
    adapter.joinpath("run_status.json").write_text(
        json.dumps(status), encoding="utf-8"
    )
    assets = {item["task"]: item for item in manifest["assets"]}
    evaluation_config = {
        "dataset": "math10k",
        "base_model": "google/gemma-3-4b-pt",
        "batch_size": 8,
        "num_beams": 4,
        "max_new_tokens": 256,
        "max_input_length": 1024,
        "tasks": ["gsm8k", "AQuA", "mawps", "SVAMP"],
        "config_fingerprint": status["config_fingerprint"],
        "requested_model_revision": status["model_revision"],
        "resolved_model_revision": status["resolved_model_revision"],
        "test_assets": {
            task: {"rows": asset["rows"], "sha256": asset["sha256"]}
            for task, asset in assets.items()
        },
    }
    results.joinpath("evaluation_config.json").write_text(
        json.dumps(evaluation_config), encoding="utf-8"
    )
    for task, asset in assets.items():
        source = ROOT / asset["path"]
        rows = json.loads(source.read_text(encoding="utf-8"))
        predictions = [
            {**row, "output_pred": "", "pred": "", "flag": False}
            for row in rows
        ]
        results.joinpath(f"{task}.json").write_text(
            json.dumps(predictions), encoding="utf-8"
        )
    gsm8k_accuracy = 0.1 if forge_summary else 0.0
    average = gsm8k_accuracy / 4.0
    results.joinpath("summary.csv").write_text(
        "gsm8k,AQuA,mawps,SVAMP,Average\n"
        f"{gsm8k_accuracy},0.0,0.0,0.0,{average}\n",
        encoding="utf-8",
    )
    detail_lines = ["dataset,accuracy,n,json"]
    for task, asset in assets.items():
        accuracy = gsm8k_accuracy if task == "gsm8k" else 0.0
        detail_lines.append(
            f"{task},{accuracy},{asset['rows']},{results / f'{task}.json'}"
        )
    results.joinpath("details.csv").write_text(
        "\n".join(detail_lines) + "\n", encoding="utf-8"
    )
    return run_root, asset_manifest_path


def test_final_validator_recomputes_complete_consistent_metrics(tmp_path: Path) -> None:
    run_root, asset_manifest = _make_final_validation_fixture(
        tmp_path, forge_summary=False
    )
    subprocess.run(
        [sys.executable, "-", str(run_root), "baseline", "17", str(asset_manifest)],
        input=_embedded_python("validate_final_run"),
        text=True,
        check=True,
    )
    metrics = json.loads(
        (run_root / "results" / "decontaminated_metrics.json").read_text(
            encoding="utf-8"
        )
    )
    assert metrics["primary_metric"] == "clean_three_task_macro_accuracy"
    assert metrics["clean_mawps_records"] == 185


def test_final_validator_rejects_forged_summary_before_postprocessing(
    tmp_path: Path,
) -> None:
    run_root, asset_manifest = _make_final_validation_fixture(
        tmp_path, forge_summary=True
    )
    result = subprocess.run(
        [sys.executable, "-", str(run_root), "baseline", "17", str(asset_manifest)],
        input=_embedded_python("validate_final_run"),
        text=True,
        capture_output=True,
    )
    assert result.returncode != 0
    assert "does not match summary" in result.stderr
    assert not (run_root / "results" / "decontaminated_metrics.json").exists()


def test_schedule_source_validator_requires_full_data_split_and_no_test_outputs(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "schedule-source"
    adapter = run_root / "adapter"
    validation = run_root / "results" / "validation"
    adapter.mkdir(parents=True)
    validation.mkdir(parents=True)
    split = {
        "protocol_stage": "final",
        "source_rows": 9919,
        "train_rows": 9919,
        "validation_rows": 0,
        "validation_data_is_public": False,
        "manifest_sha256": "a" * 64,
    }
    validation.joinpath("split_manifest.json").write_text(
        json.dumps(split), encoding="utf-8"
    )
    adapter.joinpath("run_status.json").write_text(
        json.dumps(
            {
                "state": "completed",
                "update_steps": 300,
                "config": {
                    "method": "slaclip",
                    "seed": 42,
                    "protocol_stage": "final",
                    "val_set_size": 0,
                    "run_eval": False,
                },
                "privacy_accounting": {"epsilon_spent": 6.0},
                "data_split": split,
            }
        ),
        encoding="utf-8",
    )
    adapter.joinpath("train_log.jsonl").write_text(
        "".join(
            json.dumps({"step": step, "dp_clip_threshold": 1.0}) + "\n"
            for step in range(1, 301)
        ),
        encoding="utf-8",
    )
    subprocess.run(
        [sys.executable, "-", str(run_root), "42"],
        input=_embedded_python("validate_schedule_source_run"),
        text=True,
        check=True,
    )

    run_root.joinpath("results", "summary.csv").write_text(
        "forbidden\n", encoding="utf-8"
    )
    rejected = subprocess.run(
        [sys.executable, "-", str(run_root), "42"],
        input=_embedded_python("validate_schedule_source_run"),
        text=True,
        capture_output=True,
    )
    assert rejected.returncode != 0
    assert "must not access" in rejected.stderr


def test_generated_registry_matches_formal_selector_schema(tmp_path: Path) -> None:
    fixed_manifest_path, registry_path, payload = _standard_derived_registry(
        tmp_path / "campaign"
    )
    fixed_manifest = json.loads(fixed_manifest_path.read_text(encoding="utf-8"))
    assert fixed_manifest["schema_version"] == 1
    assert set(fixed_manifest) == {
        "schema_version",
        "selection_protocol",
        "fixed_candidates",
    }
    assert len(fixed_manifest["fixed_candidates"]) == 8
    assert len(payload["candidates"]) == 18
    assert sum(item["family"] == "fixed" for item in payload["candidates"]) == 8
    assert sum(item["family"] == "slaclip" for item in payload["candidates"]) == 10
    assert {
        item["params"]["slaclip_target_non_small_clip_fraction"]
        for item in payload["candidates"]
        if item["family"] == "slaclip"
    } == {0.91, 0.93, 0.95, 0.97, 0.99}
    assert {
        item["params"]["slaclip_eta"]
        for item in payload["candidates"]
        if item["family"] == "slaclip"
    } == {0.15}
    assert {
        item["params"]["dp_max_grad_norm"]
        for item in payload["candidates"]
        if item["family"] == "slaclip"
    } == {0.5, 1.5}
    assert {
        item["params"]["dp_max_grad_norm"]
        for item in payload["candidates"]
        if item["family"] == "fixed"
    } == {0.1, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 15.0}
    assert not any(
        item["family"] == "slaclip"
        and item["params"]["slaclip_target_non_small_clip_fraction"] == 0.5
        and item["params"]["slaclip_eta"] == 0.2
        for item in payload["candidates"]
    )
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
    assert len(candidates) == 18


def test_plans_lock_three_stage_matrix_and_fresh_final_seeds(tmp_path: Path) -> None:
    campaign = tmp_path / "campaign"
    fixed_manifest_path, registry_path, registry = _standard_derived_registry(campaign)
    fixed = [item for item in registry["candidates"] if item["family"] == "fixed"]
    slaclip = [item for item in registry["candidates"] if item["family"] == "slaclip"]
    canonical = next(item for item in fixed if item["params"]["dp_max_grad_norm"] == 1.0)
    noncanonical = [item for item in fixed if item["id"] != canonical["id"]]
    stage1_selection = campaign / "selection" / "stage1-selection.json"
    locked_selection = campaign / "selection" / "selection.json"
    control_manifest = campaign / "schedule" / "control_manifest.json"
    replay_schedule = campaign / "schedule" / "replay_schedule.json"
    stage1_selection.parent.mkdir(parents=True)
    control_manifest.parent.mkdir(parents=True)
    stage1_selection.write_text(
        json.dumps(
            {
                "top_slaclip": [
                    {"candidate_id": item["id"]} for item in slaclip[:3]
                ],
                # Exclude C=1 deliberately: stage2 must still add it as the
                # preregistered paper anchor.
                "top_fixed": [
                    {"candidate_id": item["id"]} for item in noncanonical[:2]
                ],
            }
        ),
        encoding="utf-8",
    )
    def lock_final_choices(selected: dict, best_fixed: dict) -> None:
        selected_c0 = float(selected["params"]["dp_max_grad_norm"])
        best_c = float(best_fixed["params"]["dp_max_grad_norm"])
        if math.isclose(selected_c0, best_c, abs_tol=1e-12):
            initial_role = "best-fixed"
        elif math.isclose(selected_c0, 1.0, abs_tol=1e-12):
            initial_role = "canonical-fixed-c1"
        else:
            initial_role = "initial-C-matched-fixed"
        locked_selection.write_text(
            json.dumps(
                {
                    "selected_slaclip": {
                        "candidate_id": selected["id"],
                        "params": selected["params"],
                    },
                    "best_fixed": {
                        "candidate_id": best_fixed["id"],
                        "params": best_fixed["params"],
                    },
                }
            ),
            encoding="utf-8",
        )
        control_manifest.write_text(
            json.dumps(
                {
                    "selected_slaclip_candidate_id": selected["id"],
                    "schedule_initial_clip": selected_c0,
                    "matched_fixed_noise_rms_clip": 2.5,
                    "initial_c_matched_fixed_C": selected_c0,
                    "initial_c_matched_fixed_reference_role": initial_role,
                    "canonical_full_slaclip_reference_role": (
                        "canonical-full-slaclip"
                    ),
                }
            ),
            encoding="utf-8",
        )

    lock_final_choices(slaclip[0], noncanonical[0])

    def plan(mode: str) -> list[list[str]]:
        source_manifest = fixed_manifest_path if mode == "fixed-scan" else registry_path
        result = subprocess.run(
            [
                sys.executable,
                "-",
                mode,
                str(source_manifest),
                str(campaign),
                str(stage1_selection),
                str(locked_selection),
                str(control_manifest),
                str(replay_schedule),
                "gemma-3-4b-pt",
            ],
            input=_embedded_python("create_plan"),
            text=True,
            check=True,
            capture_output=True,
        )
        return [line.split("|") for line in result.stdout.splitlines() if line]

    fixed_scan = plan("fixed-scan")
    slaclip_screen = plan("slaclip-screen")
    stage2 = plan("stage2")
    schedule_source = plan("schedule-source")
    final = plan("final")
    assert len(fixed_scan) == 8
    assert {fields[1] for fields in fixed_scan} == {"fixed-scan"}
    assert {int(fields[4]) for fields in fixed_scan} == {42}
    assert len(slaclip_screen) == 10
    assert {fields[1] for fields in slaclip_screen} == {"slaclip-screen"}
    assert {int(fields[4]) for fields in slaclip_screen} == {42}
    assert len(stage2) == 12
    assert {int(fields[4]) for fields in stage2} == {43, 44}
    assert canonical["id"] in {fields[2] for fields in stage2}
    assert len(schedule_source) == 3
    assert {int(fields[4]) for fields in schedule_source} == {42, 43, 44}
    assert {fields[3] for fields in schedule_source} == {"schedule-source-full-data"}
    assert all("/schedule-source/" in fields[10] for fields in schedule_source)
    assert len(final) == 35
    assert {int(fields[4]) for fields in final} == {17, 29, 71, 101, 137}
    assert {fields[3] for fields in final} == {
        "selected-slaclip",
        "best-fixed",
        "replay",
        "matched-fixed-noise-energy",
        "canonical-fixed-c1",
        "initial-C-matched-fixed",
        "canonical-full-slaclip",
    }
    assert all("/final/" in fields[10] for fields in final)
    assert "canonical-full-slaclip-paper-default" in {
        fields[2] for fields in final
    }
    assert "canonical-full-slaclip-paper-default" not in {
        item["id"] for item in registry["candidates"]
    }

    # Dynamic controls collapse to six or five roles when best fixed and/or
    # the selected initial C already supply the corresponding references.
    lock_final_choices(slaclip[0], canonical)
    final_30 = plan("final")
    assert len(final_30) == 30
    assert "canonical-fixed-c1" not in {fields[3] for fields in final_30}

    selected_c1 = {
        **slaclip[0],
        "id": "sla-fixture-c1-r0p95-e0p15",
        "params": {
            **slaclip[0]["params"],
            "dp_max_grad_norm": 1.0,
        },
    }
    registry["candidates"].append(selected_c1)
    registry_path.write_text(json.dumps(registry), encoding="utf-8")
    lock_final_choices(selected_c1, canonical)
    final_25 = plan("final")
    assert len(final_25) == 25
    assert {fields[3] for fields in final_25} == {
        "selected-slaclip",
        "best-fixed",
        "replay",
        "matched-fixed-noise-energy",
        "canonical-full-slaclip",
    }


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
                    "params": {
                        "dp_max_grad_norm": 1.0,
                        "slaclip_target_non_small_clip_fraction": 0.5,
                        "slaclip_eta": 0.2,
                    },
                },
                "best_fixed": {
                    "candidate_id": "fixed-c0p5",
                    "params": {"dp_max_grad_norm": 0.5},
                },
            }
        ),
        encoding="utf-8",
    )
    for seed, offset in ((42, 0.0), (43, 0.3), (44, 0.6)):
        root = (
            campaign
            / "schedule-source"
            / "runs"
            / candidate_id
            / f"seed-{seed}"
            / "adapter"
        )
        root.mkdir(parents=True)
        root.joinpath("run_status.json").write_text(
            json.dumps(
                {
                    "state": "completed",
                    "config": {
                        "method": "slaclip",
                        "protocol_stage": "final",
                        "val_set_size": 0,
                        "run_eval": False,
                    },
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
        [
            sys.executable,
            "-",
            str(campaign),
            str(selection),
            str(schedule),
            str(controls),
            "12",
        ],
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
    assert control["schedule_privacy_class"] == (
        "DP_DERIVED_VIA_30_RUN_SELECTION_PLUS_3_FULL_DATA_SOURCES"
    )
    assert replay["schedule_privacy_class"] == (
        "DP_DERIVED_VIA_30_RUN_SELECTION_PLUS_3_FULL_DATA_SOURCES"
    )
    privacy = replay["privacy_accounting"]
    assert privacy["stage1_run_count"] == 18
    assert privacy["stage2_run_count"] == 12
    assert privacy["screen_run_count"] == 30
    assert privacy["pre_final_dp_run_count"] == 33
    assert privacy["final_run_count"] == 25
    replay_matched = privacy[
        "selection_plus_five_replay_plus_five_matched_fixed"
    ]
    assert replay_matched["release_count"] == 43
    assert replay_matched["epsilon"] == 258.0
    assert replay_matched["delta"] == pytest.approx(43e-5)
    full_bundle = privacy["all_screen_and_fresh_final_dp_outputs"]
    assert full_bundle["release_count"] == 58
    assert full_bundle["epsilon"] == 348.0
    assert full_bundle["delta"] == pytest.approx(58e-5)
    assert control["schedule_source_seeds"] == [42, 43, 44]
    assert control["final_control_seeds"] == [17, 29, 71, 101, 137]


def test_wrapper_receipt_blocks_duplicates_and_requires_explicit_resume(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    required_files = (
        "train_eval.py",
        "scripts/cuda_step_guard.py",
        "scripts/derive_slaclip_target_grid.py",
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
    sbatch_arguments = tmp_path / "fake-sbatch-arguments"
    commands = {
        "sbatch": """#!/usr/bin/env bash
set -euo pipefail
printf '%s\n' "$*" >>"${FAKE_SBATCH_ARGUMENTS:?}"
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
            "FAKE_SBATCH_ARGUMENTS": str(sbatch_arguments),
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
            "PRISM_SLURM_EXCLUDE": "blossom03",
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
    assert first["resources"]["walltime"] == "2-12:00:00"
    assert first["resources"]["exclude_nodes"] == "blossom03"
    assert first["latest_resources"]["exclude_nodes"] == "blossom03"
    assert first["attempts"][0]["resources"]["exclude_nodes"] == "blossom03"

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
    assert second["attempts"][1]["resources"]["exclude_nodes"] == "blossom03"
    submitted_argument_lines = sbatch_arguments.read_text(encoding="utf-8").splitlines()
    assert len(submitted_argument_lines) == 3
    assert all("--exclude=blossom03" in line for line in submitted_argument_lines)
