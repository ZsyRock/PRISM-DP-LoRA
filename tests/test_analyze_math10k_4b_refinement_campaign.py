from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path

import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "analyze_math10k_4b_refinement_campaign.py"
)
SPEC = importlib.util.spec_from_file_location(
    "analyze_math10k_4b_refinement_campaign", SCRIPT
)
assert SPEC is not None and SPEC.loader is not None
analyzer = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = analyzer
SPEC.loader.exec_module(analyzer)

BUILDER_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "build_math10k_4b_refinement_registry.py"
)
BUILDER_SPEC = importlib.util.spec_from_file_location(
    "build_math10k_4b_refinement_registry_for_analyzer_test", BUILDER_SCRIPT
)
assert BUILDER_SPEC is not None and BUILDER_SPEC.loader is not None
builder = importlib.util.module_from_spec(BUILDER_SPEC)
BUILDER_SPEC.loader.exec_module(builder)


def _plan_spec(
    tmp_path: Path,
    *,
    seed: int,
    arm: str,
    candidate_id: str,
    method: str,
    clip: float,
    rho: float | None = None,
    eta: float | None = None,
) -> analyzer.PlanSpec:
    return analyzer.PlanSpec(
        phase="final",
        candidate_id=candidate_id,
        role=arm,
        seed=seed,
        method=method,
        clip=clip,
        rho=rho,
        eta=eta,
        schedule_path="NA",
        arm_root=tmp_path / "final" / "gemma-3-4b-pt" / f"seed-{seed}" / arm,
        source_path=tmp_path / "plans" / "final.tsv",
        line_number=1,
    )


def _final_plan(tmp_path: Path) -> list[analyzer.PlanSpec]:
    result = []
    for seed in analyzer.FINAL_SEEDS:
        result.extend(
            (
                _plan_spec(
                    tmp_path,
                    seed=seed,
                    arm="slaclip",
                    candidate_id="sla-selected",
                    method="slaclip",
                    clip=1.75,
                    rho=0.995,
                    eta=0.05,
                ),
                _plan_spec(
                    tmp_path,
                    seed=seed,
                    arm="baseline",
                    candidate_id="fixed-c2",
                    method="baseline",
                    clip=2.0,
                ),
            )
        )
    return result


def _registry_candidates() -> dict[str, dict]:
    return {
        "fixed-c2": {
            "id": "fixed-c2",
            "family": "fixed",
            "method": "baseline",
            "params": {"dp_max_grad_norm": 2.0},
        },
        "sla-selected": {
            "id": "sla-selected",
            "family": "slaclip",
            "method": "slaclip",
            "params": {
                "dp_max_grad_norm": 1.75,
                "slaclip_target_non_small_clip_fraction": 0.995,
                "slaclip_eta": 0.05,
                "slaclip_num_slots": 15,
                "slaclip_c_min": 0.1,
                "slaclip_c_max": 15.0,
            },
        },
    }


def test_final_plan_is_exactly_five_fixed_c2_slaclip_pairs(tmp_path: Path) -> None:
    analyzer._validate_final_plan(
        _final_plan(tmp_path),
        "sla-selected",
        "fixed-c2",
        _registry_candidates(),
    )


def test_final_plan_rejects_non_c2_reference(tmp_path: Path) -> None:
    plan = _final_plan(tmp_path)
    baseline = next(item for item in plan if item.arm_root.name == "baseline")
    plan[plan.index(baseline)] = _plan_spec(
        tmp_path,
        seed=baseline.seed,
        arm="baseline",
        candidate_id="fixed-c1",
        method="baseline",
        clip=1.0,
    )
    with pytest.raises(analyzer.AnalysisError, match="fixed C=2"):
        analyzer._validate_final_plan(
            plan, "sla-selected", "fixed-c2", _registry_candidates()
        )


def test_paired_ci_uses_df4_seed_level_inference() -> None:
    candidate = [0.51, 0.52, 0.54, 0.55, 0.57]
    reference = [0.50, 0.50, 0.51, 0.51, 0.52]
    stats = analyzer._paired_statistics(candidate, reference)
    deltas = [left - right for left, right in zip(candidate, reference, strict=True)]
    mean = sum(deltas) / 5
    sample_sd = __import__("statistics").stdev(deltas)
    half_width = analyzer.T95_DF4 * sample_sd / math.sqrt(5)

    assert stats["t95_critical_df4"] == pytest.approx(2.7764451051977987)
    assert stats["paired_mean_delta"] == pytest.approx(mean)
    assert stats["t95_ci_low"] == pytest.approx(mean - half_width)
    assert stats["t95_ci_high"] == pytest.approx(mean + half_width)
    assert stats["wins"] == 5


def test_decontaminated_accuracy_uses_selection_file_sha_and_raw_macro(
    tmp_path: Path,
) -> None:
    selection_bytes = b'{\n  "stage": "stage2"\n}\n'
    selection_sha = hashlib.sha256(selection_bytes).hexdigest()
    tasks = {"gsm8k": 0.4, "AQuA": 0.5, "mawps": 0.6, "SVAMP": 0.7}
    payload = {
        "primary_metric": "clean_three_task_macro_accuracy",
        "selection_sha256": selection_sha,
        "decontaminated_task_accuracy": tasks,
        "clean_mawps_accuracy": 0.6,
        "clean_three_task_macro_accuracy": (0.4 + 0.5 + 0.7) / 3,
        "decontaminated_four_task_macro_accuracy": 0.55,
        "paper_raw_four_task_macro_accuracy": 0.56,
    }
    path = tmp_path / "decontaminated_metrics.json"
    path.write_text(__import__("json").dumps(payload), encoding="utf-8")

    values = analyzer._read_decontaminated(path, selection_sha)

    assert values["clean_three_task_macro_accuracy"] == pytest.approx(1.6 / 3)
    assert values["paper_raw_four_task_macro_accuracy"] == pytest.approx(0.56)
    with pytest.raises(analyzer.AnalysisError, match="locked selection"):
        analyzer._read_decontaminated(path, "0" * 64)


def test_plan_count_is_fail_closed(tmp_path: Path) -> None:
    campaign = tmp_path / "campaign"
    plan = campaign / "plans" / "stage1.tsv"
    plan.parent.mkdir(parents=True)
    line = (
        "train|stage1|fixed-c1|local-grid-screen|42|baseline|1.0|NA|NA|NA|"
        f"{campaign / 'screen' / 'runs' / 'fixed-c1' / 'seed-42'}\n"
    )
    plan.write_text(line, encoding="utf-8")

    with pytest.raises(analyzer.AnalysisError, match="exactly 27 arms"):
        analyzer._read_plan(campaign.resolve(), "stage1")


def test_selector_best_fixed_can_be_c1_while_final_reference_stays_c2(
    tmp_path: Path,
) -> None:
    campaign = (tmp_path / "campaign").resolve()
    campaign.mkdir()
    registry = builder._build_payload(
        model_id="google/gemma-3-4b-pt",
        model_revision="c" * 40,
        code_sha="d" * 40,
    )
    registry_path = campaign / "screen" / "candidate_registry.json"
    registry_path.parent.mkdir()
    registry_path.write_text(json.dumps(registry, indent=2, sort_keys=True) + "\n")
    candidates = {item["id"]: item for item in registry["candidates"]}

    def spec(candidate_id: str, phase: str, seed: int) -> analyzer.PlanSpec:
        candidate = candidates[candidate_id]
        params = candidate["params"]
        method = candidate["method"]
        return analyzer.PlanSpec(
            phase=phase,
            candidate_id=candidate_id,
            role="screen",
            seed=seed,
            method=method,
            clip=params["dp_max_grad_norm"],
            rho=(
                params.get("slaclip_target_non_small_clip_fraction")
                if method == "slaclip"
                else None
            ),
            eta=params.get("slaclip_eta") if method == "slaclip" else None,
            schedule_path="NA",
            arm_root=campaign / "screen" / "runs" / candidate_id / f"seed-{seed}",
            source_path=campaign / "plans" / f"{phase}.tsv",
            line_number=1,
        )

    stage1 = [spec(candidate_id, "stage1", 42) for candidate_id in candidates]
    sla_ids = [
        candidate_id
        for candidate_id, candidate in candidates.items()
        if candidate["family"] == "slaclip"
    ][:3]
    promoted = [*sla_ids, "fixed-c1", "fixed-c2"]
    stage2 = [
        spec(candidate_id, "stage2", seed)
        for seed in (43, 44, 45, 46)
        for candidate_id in promoted
    ]
    ranking = []
    for rank, candidate_id in enumerate(promoted, 1):
        candidate = candidates[candidate_id]
        ranking.append(
            {
                "rank": rank,
                "candidate_id": candidate_id,
                "family": candidate["family"],
                "method": candidate["method"],
                "params": candidate["params"],
            }
        )
    selection = {
        "stage": "stage2",
        "registry_sha256": analyzer._sha256_bytes(analyzer._canonical_json(registry)),
        "required_seeds": [42, 43, 44, 45, 46],
        "ranking": ranking,
        "selected_slaclip": ranking[0],
        "best_fixed": ranking[-2],
    }

    _required, selected_id, fixed_c2_id, best_fixed_id, _ranks = (
        analyzer._validate_selection_structure(
            selection,
            registry_path,
            candidates,
            registry["selection_protocol"],
            stage1,
            stage2,
        )
    )

    assert selected_id == sla_ids[0]
    assert best_fixed_id == "fixed-c1"
    assert fixed_c2_id == "fixed-c2"
