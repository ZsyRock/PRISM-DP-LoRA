#!/usr/bin/env python3
"""Fail-closed analysis for the focused Math-10K 4B refinement campaign.

The campaign is deliberately small and rigid: 27 one-seed stage-1 arms,
20 additional stage-2 arms, and five fresh paired final seeds comparing the
locked full-SlaClip candidate with fixed C=2.  This script treats a seed (not a
training step) as the independent unit, validates every planned run before
writing anything, and records SHA256 provenance for every consumed artifact.

Research telemetry is exact and therefore NON_PRIVATE.  The generated tables
must not be described as a differentially private release.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import statistics
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


ANALYSIS_SCHEMA_VERSION = 1
EXPECTED_UPDATE_STEPS = 300
EXPECTED_PHASE_COUNTS = {"stage1": 27, "stage2": 20, "final": 10}
FINAL_SEEDS = (191, 223, 257, 293, 331)
T95_DF4 = 2.7764451051977987
RAW_WARNING = (
    "NON_PRIVATE: contains exact training-example telemetry and is not a "
    "differentially private release"
)

ACCURACY_METRICS = (
    ("clean_three_task_macro_accuracy", "primary", True),
    ("decontaminated_four_task_macro_accuracy", "sensitivity", False),
    ("paper_raw_four_task_macro_accuracy", "paper_comparability", False),
    ("gsm8k_accuracy", "task", False),
    ("AQuA_accuracy", "task", False),
    ("clean_mawps_accuracy", "decontaminated_task", False),
    ("SVAMP_accuracy", "task", False),
)

MECHANISM_METRICS = {
    "loss_mean": "lower_is_better",
    "raw_clip_fraction": "target_dependent_diagnostic",
    "dp_clip_threshold": "target_dependent_diagnostic",
    "raw_signal_to_noise_ratio": "higher_is_better",
    "raw_clipping_bias_norm": "lower_is_better",
    "raw_realized_noise_norm": "lower_is_better",
    "raw_signal_retention_ratio": "higher_is_better",
    "raw_clipping_bias_to_noise_ratio": "lower_is_better",
    "raw_bias_noise_squared_error_proxy": "lower_is_better",
    "raw_unclipped_clipped_cosine": "higher_is_better",
    "raw_clipped_noisy_cosine": "higher_is_better",
}


class AnalysisError(RuntimeError):
    """The campaign is incomplete, inconsistent, or unsafe to aggregate."""


@dataclass(frozen=True)
class PlanSpec:
    phase: str
    candidate_id: str
    role: str
    seed: int
    method: str
    clip: float
    rho: float | None
    eta: float | None
    schedule_path: str
    arm_root: Path
    source_path: Path
    line_number: int

    @property
    def key(self) -> tuple[str, str, int]:
        return (self.phase, self.candidate_id, self.seed)


@dataclass
class RunEvidence:
    spec: PlanSpec
    status: dict[str, Any]
    telemetry: dict[str, float]
    validation_accuracy: float | None
    validation_loss: float | None
    final_accuracy: dict[str, float]
    input_paths: tuple[Path, ...]


def _canonical_json(payload: Any) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _pretty_json(payload: Any) -> bytes:
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size == 0:
        raise AnalysisError(f"missing or empty {label}: {path}")
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise AnalysisError(f"could not read {label} {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise AnalysisError(f"{label} must be a JSON object: {path}")
    return payload


def _finite(
    value: Any,
    *,
    label: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool):
        raise AnalysisError(f"{label} must be numeric, not boolean")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise AnalysisError(f"{label} must be numeric, got {value!r}") from exc
    if not math.isfinite(number):
        raise AnalysisError(f"{label} must be finite, got {value!r}")
    if minimum is not None and number < minimum:
        raise AnalysisError(f"{label} must be >= {minimum}, got {number}")
    if maximum is not None and number > maximum:
        raise AnalysisError(f"{label} must be <= {maximum}, got {number}")
    return number


def _integer(value: Any, *, label: str, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise AnalysisError(f"{label} must be an integer, got {value!r}")
    number = int(value)
    if minimum is not None and number < minimum:
        raise AnalysisError(f"{label} must be >= {minimum}, got {number}")
    return number


def _same(expected: Any, actual: Any) -> bool:
    if isinstance(expected, bool) or isinstance(actual, bool):
        return expected is actual
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        return math.isclose(
            float(expected), float(actual), rel_tol=1e-12, abs_tol=1e-12
        )
    return expected == actual


def _resolve_inside(root: Path, path: Path, *, label: str) -> Path:
    candidate = path if path.is_absolute() else root / path
    resolved = candidate.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise AnalysisError(f"{label} escapes campaign root: {path}") from exc
    return resolved


def _relative(root: Path, path: Path) -> str:
    try:
        return str(path.resolve().relative_to(root))
    except ValueError as exc:
        raise AnalysisError(f"artifact escapes campaign root: {path}") from exc


def _read_plan(campaign_root: Path, phase: str) -> list[PlanSpec]:
    path = campaign_root / "plans" / f"{phase}.tsv"
    if not path.is_file() or path.stat().st_size == 0:
        raise AnalysisError(f"missing or empty {phase} plan: {path}")
    specs: list[PlanSpec] = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        fields = raw.split("|")
        if len(fields) != 11 or fields[0] != "train":
            raise AnalysisError(
                f"{path}:{line_number}: expected an 11-field train plan record"
            )
        if fields[1] != phase:
            raise AnalysisError(
                f"{path}:{line_number}: phase={fields[1]!r}, expected={phase!r}"
            )
        candidate_id, role, seed_text, method = fields[2:6]
        if not candidate_id or not role:
            raise AnalysisError(f"{path}:{line_number}: empty candidate/role")
        try:
            seed = int(seed_text)
        except ValueError as exc:
            raise AnalysisError(f"{path}:{line_number}: invalid seed={seed_text!r}") from exc
        if str(seed) != seed_text or seed < 0:
            raise AnalysisError(f"{path}:{line_number}: invalid seed={seed_text!r}")
        if method not in {"baseline", "slaclip"}:
            raise AnalysisError(f"{path}:{line_number}: unsupported method={method!r}")
        clip = _finite(fields[6], label=f"{path}:{line_number}:C", minimum=1e-300)
        if method == "baseline":
            if fields[7] != "NA" or fields[8] != "NA":
                raise AnalysisError(
                    f"{path}:{line_number}: baseline must use NA rho/eta"
                )
            rho = eta = None
        else:
            rho = _finite(
                fields[7],
                label=f"{path}:{line_number}:rho",
                minimum=0.0,
                maximum=1.0,
            )
            eta = _finite(fields[8], label=f"{path}:{line_number}:eta", minimum=0.0)
        if fields[9] != "NA":
            raise AnalysisError(
                f"{path}:{line_number}: refinement campaign forbids replay schedules"
            )
        arm_root = _resolve_inside(
            campaign_root, Path(fields[10]), label=f"{path}:{line_number}:arm_root"
        )
        specs.append(
            PlanSpec(
                phase=phase,
                candidate_id=candidate_id,
                role=role,
                seed=seed,
                method=method,
                clip=clip,
                rho=rho,
                eta=eta,
                schedule_path=fields[9],
                arm_root=arm_root,
                source_path=path,
                line_number=line_number,
            )
        )
    expected = EXPECTED_PHASE_COUNTS[phase]
    if len(specs) != expected:
        raise AnalysisError(f"{phase} plan must contain exactly {expected} arms, found {len(specs)}")
    return specs


def _candidate_target(params: Mapping[str, Any]) -> Any:
    canonical = params.get("slaclip_target_non_small_clip_fraction")
    legacy = params.get("slaclip_beta")
    if canonical is not None and legacy is not None and not _same(canonical, legacy):
        raise AnalysisError("candidate has conflicting canonical and legacy full-SlaClip targets")
    return canonical if canonical is not None else legacy


def _validate_candidate_spec(candidate: Mapping[str, Any], spec: PlanSpec) -> None:
    method = candidate.get("method")
    family = candidate.get("family")
    params = candidate.get("params")
    if method != spec.method or family not in {"fixed", "slaclip"} or not isinstance(params, dict):
        raise AnalysisError(
            f"registry candidate identity mismatch for {spec.candidate_id} in {spec.source_path}"
        )
    expected_family = "fixed" if spec.method == "baseline" else "slaclip"
    if family != expected_family:
        raise AnalysisError(f"candidate {spec.candidate_id} family/method mismatch")
    if not _same(params.get("dp_max_grad_norm"), spec.clip):
        raise AnalysisError(f"candidate {spec.candidate_id} plan C does not match registry")
    if spec.method == "slaclip":
        if not _same(_candidate_target(params), spec.rho):
            raise AnalysisError(f"candidate {spec.candidate_id} plan rho does not match registry")
        if not _same(params.get("slaclip_eta"), spec.eta):
            raise AnalysisError(f"candidate {spec.candidate_id} plan eta does not match registry")


def _validate_registry(
    campaign_root: Path,
    registry: Mapping[str, Any],
    selection_specs: Sequence[PlanSpec],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    candidates = registry.get("candidates")
    protocol = registry.get("selection_protocol")
    if not isinstance(candidates, list) or len(candidates) != 27:
        raise AnalysisError("candidate registry must contain exactly 27 candidates")
    if not isinstance(protocol, dict):
        raise AnalysisError("candidate registry lacks selection_protocol")
    indexed: dict[str, dict[str, Any]] = {}
    for index, item in enumerate(candidates):
        if not isinstance(item, dict) or not isinstance(item.get("id"), str):
            raise AnalysisError(f"registry candidates[{index}] has no valid id")
        candidate_id = item["id"]
        if not candidate_id or candidate_id in indexed:
            raise AnalysisError(f"duplicate/empty registry candidate id={candidate_id!r}")
        if not isinstance(item.get("runs"), dict):
            raise AnalysisError(f"registry candidate {candidate_id} lacks runs")
        indexed[candidate_id] = item
    stage1_ids = [spec.candidate_id for spec in selection_specs if spec.phase == "stage1"]
    if len(set(stage1_ids)) != 27 or set(stage1_ids) != set(indexed):
        raise AnalysisError("stage1 plan must cover every registry candidate exactly once")
    for spec in selection_specs:
        candidate = indexed.get(spec.candidate_id)
        if candidate is None:
            raise AnalysisError(f"plan references unknown candidate={spec.candidate_id}")
        _validate_candidate_spec(candidate, spec)
        run = candidate["runs"].get(str(spec.seed))
        if not isinstance(run, dict):
            raise AnalysisError(
                f"registry candidate={spec.candidate_id} lacks seed={spec.seed} run mapping"
            )
        expected = {
            "run_status": spec.arm_root / "adapter" / "run_status.json",
            "validation_metrics": (
                spec.arm_root / "results" / "validation" / "validation_metrics.json"
            ),
            "split_manifest": (
                spec.arm_root / "results" / "validation" / "split_manifest.json"
            ),
        }
        for key, expected_path in expected.items():
            supplied = run.get(key)
            if not isinstance(supplied, str):
                raise AnalysisError(
                    f"registry candidate={spec.candidate_id} seed={spec.seed} lacks {key}"
                )
            resolved = _resolve_inside(campaign_root, Path(supplied), label=f"registry {key}")
            if resolved != expected_path:
                raise AnalysisError(
                    f"registry {key} does not match plan root for "
                    f"candidate={spec.candidate_id} seed={spec.seed}"
                )
    return indexed, dict(protocol)


def _selection_ranking(selection: Mapping[str, Any]) -> list[dict[str, Any]]:
    ranking = selection.get("ranking")
    if not isinstance(ranking, list) or not ranking:
        raise AnalysisError("selection lock has no non-empty ranking")
    if not all(isinstance(item, dict) for item in ranking):
        raise AnalysisError("selection ranking entries must be objects")
    return list(ranking)


def _validate_selection_structure(
    selection: Mapping[str, Any],
    registry_path: Path,
    registry_candidates: Mapping[str, Mapping[str, Any]],
    protocol: Mapping[str, Any],
    stage1: Sequence[PlanSpec],
    stage2: Sequence[PlanSpec],
) -> tuple[list[int], str, str, str, dict[str, int]]:
    if selection.get("stage") != "stage2":
        raise AnalysisError("selection/selection.json must be a stage2 lock")
    registry_payload = _read_json(registry_path, label="candidate registry provenance")
    registry_canonical_sha = _sha256_bytes(_canonical_json(registry_payload))
    if selection.get("registry_sha256") != registry_canonical_sha:
        raise AnalysisError("selection registry_sha256 does not match candidate registry")
    required = selection.get("required_seeds")
    if (
        not isinstance(required, list)
        or len(required) != 5
        or any(isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 for seed in required)
        or required != sorted(set(required))
    ):
        raise AnalysisError("selection required_seeds must be five sorted unique integers")
    stage1_seed = protocol.get("stage1_seed")
    stage2_seeds = protocol.get("stage2_seeds")
    if stage1_seed != required[0] or stage2_seeds != required:
        raise AnalysisError("registry selection seeds do not match locked selection")
    if {spec.seed for spec in stage1} != {stage1_seed}:
        raise AnalysisError("all stage1 candidates must use the preregistered stage1 seed")

    ranking = _selection_ranking(selection)
    ranked_ids = [item.get("candidate_id") for item in ranking]
    if any(not isinstance(value, str) or value not in registry_candidates for value in ranked_ids):
        raise AnalysisError("selection ranking references an unknown candidate")
    if len(set(ranked_ids)) != 5:
        raise AnalysisError("selection ranking must contain exactly five promoted candidates")
    stage2_ids = {spec.candidate_id for spec in stage2}
    if stage2_ids != set(ranked_ids):
        raise AnalysisError("stage2 plan candidates do not match locked selection ranking")
    expected_stage2_seeds = set(required) - {stage1_seed}
    for candidate_id in ranked_ids:
        seeds = {spec.seed for spec in stage2 if spec.candidate_id == candidate_id}
        if seeds != expected_stage2_seeds:
            raise AnalysisError(
                f"stage2 candidate={candidate_id} does not cover the four additional seeds"
            )

    selected = selection.get("selected_slaclip")
    best_fixed = selection.get("best_fixed")
    if not isinstance(selected, dict) or selected.get("method") != "slaclip":
        raise AnalysisError("selection lock has no selected full-SlaClip candidate")
    if not isinstance(best_fixed, dict) or best_fixed.get("method") != "baseline":
        raise AnalysisError("selection lock has no fixed baseline")
    selected_id = selected.get("candidate_id")
    fixed_id = best_fixed.get("candidate_id")
    if selected_id not in ranked_ids or fixed_id not in ranked_ids:
        raise AnalysisError("selected candidates are absent from the locked ranking")
    for locked, candidate_id in ((selected, selected_id), (best_fixed, fixed_id)):
        candidate = registry_candidates[str(candidate_id)]
        if locked.get("family") != candidate.get("family") or locked.get("params") != candidate.get("params"):
            raise AnalysisError(f"locked candidate={candidate_id} differs from registry")

    fixed_c2 = [
        candidate_id
        for candidate_id, candidate in registry_candidates.items()
        if candidate.get("method") == "baseline"
        and isinstance(candidate.get("params"), dict)
        and _same(candidate["params"].get("dp_max_grad_norm"), 2.0)
    ]
    if len(fixed_c2) != 1 or fixed_c2[0] not in ranked_ids:
        raise AnalysisError(
            "registry/stage2 selection must contain one unique fixed C=2 candidate"
        )
    ranks: dict[str, int] = {}
    for position, item in enumerate(ranking, 1):
        declared = item.get("rank")
        if declared is not None and declared != position:
            raise AnalysisError("selection ranking has inconsistent rank values")
        ranks[str(item["candidate_id"])] = position
    # The refinement's predeclared reference is fixed C=2, independent of
    # whether noisy public validation ranks fixed C=1 or C=2 first.
    return list(required), str(selected_id), fixed_c2[0], str(fixed_id), ranks


def _validate_final_plan(
    final: Sequence[PlanSpec],
    selected_id: str,
    fixed_id: str,
    registry_candidates: Mapping[str, Mapping[str, Any]],
) -> None:
    if tuple(sorted({spec.seed for spec in final})) != FINAL_SEEDS:
        raise AnalysisError(f"final seeds must be exactly {list(FINAL_SEEDS)}")
    selected_candidate = registry_candidates[selected_id]
    for seed in FINAL_SEEDS:
        pair = [spec for spec in final if spec.seed == seed]
        if len(pair) != 2:
            raise AnalysisError(f"final seed={seed} must have exactly two arms")
        by_arm = {spec.arm_root.name: spec for spec in pair}
        if set(by_arm) != {"baseline", "slaclip"}:
            raise AnalysisError(
                f"final seed={seed} must contain baseline and slaclip artifact arms exactly"
            )
        baseline = by_arm["baseline"]
        slaclip = by_arm["slaclip"]
        if (
            baseline.method != "baseline"
            or baseline.candidate_id != fixed_id
            or not _same(baseline.clip, 2.0)
            or baseline.role != "baseline"
            or baseline.arm_root.name != "baseline"
        ):
            raise AnalysisError(f"final seed={seed} baseline is not locked fixed C=2")
        if (
            slaclip.method != "slaclip"
            or slaclip.candidate_id != selected_id
            or slaclip.role != "slaclip"
            or slaclip.arm_root.name != "slaclip"
        ):
            raise AnalysisError(f"final seed={seed} does not use the locked SlaClip candidate")
        _validate_candidate_spec(selected_candidate, slaclip)


def _validate_status(spec: PlanSpec, status: Mapping[str, Any]) -> dict[str, Any]:
    path = spec.arm_root / "adapter" / "run_status.json"
    if status.get("state") != "completed":
        raise AnalysisError(f"run is not completed: {path}")
    if _integer(status.get("update_steps"), label=f"{path}:update_steps") != EXPECTED_UPDATE_STEPS:
        raise AnalysisError(f"run must complete exactly 300 update steps: {path}")
    config = status.get("config")
    if not isinstance(config, dict):
        raise AnalysisError(f"run status lacks config: {path}")
    expected_stage = "final" if spec.phase == "final" else "selection"
    expected_public = spec.phase != "final"
    expected = {
        "method": spec.method,
        "seed": spec.seed,
        "total_update_steps": EXPECTED_UPDATE_STEPS,
        "protocol_stage": expected_stage,
        "validation_data_is_public": expected_public,
        "dp_max_grad_norm": spec.clip,
    }
    for key, value in expected.items():
        if not _same(value, config.get(key)):
            raise AnalysisError(
                f"run config mismatch {key}={config.get(key)!r}, expected={value!r}: {path}"
            )
    if spec.phase == "final":
        if config.get("run_eval") is not True or config.get("val_set_size") != 0:
            raise AnalysisError(f"final run must use full data and evaluation: {path}")
    if spec.method == "slaclip":
        target = config.get("slaclip_target_non_small_clip_fraction")
        if target is None:
            target = config.get("slaclip_beta")
        if not _same(target, spec.rho) or not _same(config.get("slaclip_eta"), spec.eta):
            raise AnalysisError(f"SlaClip config does not match plan: {path}")
    fingerprint = status.get("config_fingerprint")
    if not isinstance(fingerprint, str) or not fingerprint:
        raise AnalysisError(f"run status lacks config_fingerprint: {path}")
    if config.get("config_fingerprint") not in {None, fingerprint}:
        raise AnalysisError(f"run status/config fingerprint mismatch: {path}")
    accounting = status.get("privacy_accounting")
    if not isinstance(accounting, dict):
        raise AnalysisError(f"run status lacks privacy_accounting: {path}")
    if accounting.get("completed_update_steps") not in {None, EXPECTED_UPDATE_STEPS}:
        raise AnalysisError(f"privacy accounting does not cover 300 steps: {path}")
    _finite(accounting.get("epsilon_spent"), label=f"{path}:epsilon_spent", minimum=0.0)
    return config


def _read_telemetry_means(
    path: Path,
    *,
    method: str,
    fingerprint: str,
) -> dict[str, float]:
    payload = _read_json(path, label="telemetry summary")
    if payload.get("NON_PRIVATE_TELEMETRY") is not True:
        raise AnalysisError(f"telemetry is not explicitly marked NON_PRIVATE: {path}")
    steps = payload.get("steps")
    expected_steps = {
        "count": EXPECTED_UPDATE_STEPS,
        "first": 1,
        "last": EXPECTED_UPDATE_STEPS,
        "missing_count": 0,
    }
    if not isinstance(steps, dict) or any(steps.get(key) != value for key, value in expected_steps.items()):
        raise AnalysisError(f"telemetry does not cover steps 1..300 exactly: {path}")
    if steps.get("missing") not in (None, []):
        raise AnalysisError(f"telemetry reports missing steps: {path}")
    identity = payload.get("run_identity")
    if not isinstance(identity, dict):
        raise AnalysisError(f"telemetry lacks run_identity: {path}")
    if identity.get("method") != method or identity.get("config_fingerprint") != fingerprint:
        raise AnalysisError(f"telemetry/run status identity mismatch: {path}")
    metrics = payload.get("metrics")
    if not isinstance(metrics, dict):
        raise AnalysisError(f"telemetry summary lacks metrics: {path}")
    means: dict[str, float] = {}
    for name in MECHANISM_METRICS:
        metric = metrics.get(name)
        if not isinstance(metric, dict):
            raise AnalysisError(f"telemetry summary lacks metric={name}: {path}")
        if metric.get("count") != EXPECTED_UPDATE_STEPS or metric.get("missing") != 0:
            raise AnalysisError(f"telemetry metric={name} is incomplete: {path}")
        means[name] = _finite(metric.get("mean"), label=f"{path}:{name}.mean")
    return means


def _read_validation(path: Path) -> tuple[float, float]:
    payload = _read_json(path, label="public validation metrics")
    if payload.get("protocol_stage") != "selection" or payload.get("validation_data_is_public") is not True:
        raise AnalysisError(f"selection metrics are not declared public validation: {path}")
    if payload.get("selection_metric") != "public_math10k_numeric_exact_match_accuracy":
        raise AnalysisError(f"unexpected selection metric: {path}")
    accuracy = _finite(
        payload.get("numeric_exact_accuracy"),
        label=f"{path}:numeric_exact_accuracy",
        minimum=0.0,
        maximum=1.0,
    )
    loss = _finite(payload.get("loss_mean"), label=f"{path}:loss_mean", minimum=0.0)
    return accuracy, loss


def _read_decontaminated(path: Path, selection_sha: str) -> dict[str, float]:
    payload = _read_json(path, label="decontaminated final metrics")
    if payload.get("primary_metric") != "clean_three_task_macro_accuracy":
        raise AnalysisError(f"unexpected primary final metric: {path}")
    if payload.get("selection_sha256") != selection_sha:
        raise AnalysisError(f"final metrics do not match locked selection: {path}")
    tasks = payload.get("decontaminated_task_accuracy")
    if not isinstance(tasks, dict) or set(tasks) != {"gsm8k", "AQuA", "mawps", "SVAMP"}:
        raise AnalysisError(f"decontaminated task accuracy schema mismatch: {path}")
    values = {
        f"{task}_accuracy" if task != "mawps" else "clean_mawps_accuracy": _finite(
            tasks[task], label=f"{path}:{task}", minimum=0.0, maximum=1.0
        )
        for task in ("gsm8k", "AQuA", "mawps", "SVAMP")
    }
    clean_three = _finite(
        payload.get("clean_three_task_macro_accuracy"),
        label=f"{path}:clean_three_task_macro_accuracy",
        minimum=0.0,
        maximum=1.0,
    )
    decont_four = _finite(
        payload.get("decontaminated_four_task_macro_accuracy"),
        label=f"{path}:decontaminated_four_task_macro_accuracy",
        minimum=0.0,
        maximum=1.0,
    )
    expected_three = math.fsum(
        values[key] for key in ("gsm8k_accuracy", "AQuA_accuracy", "SVAMP_accuracy")
    ) / 3.0
    expected_four = math.fsum(values.values()) / 4.0
    if not math.isclose(clean_three, expected_three, rel_tol=0.0, abs_tol=1e-12):
        raise AnalysisError(f"clean three-task macro is inconsistent: {path}")
    if not math.isclose(decont_four, expected_four, rel_tol=0.0, abs_tol=1e-12):
        raise AnalysisError(f"decontaminated four-task macro is inconsistent: {path}")
    if not math.isclose(
        _finite(payload.get("clean_mawps_accuracy"), label=f"{path}:clean_mawps_accuracy"),
        values["clean_mawps_accuracy"],
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise AnalysisError(f"clean MAWPS accuracy is inconsistent: {path}")
    return {
        **values,
        "clean_three_task_macro_accuracy": clean_three,
        "decontaminated_four_task_macro_accuracy": decont_four,
        "paper_raw_four_task_macro_accuracy": _finite(
            payload.get("paper_raw_four_task_macro_accuracy"),
            label=f"{path}:paper_raw_four_task_macro_accuracy",
            minimum=0.0,
            maximum=1.0,
        ),
    }


def _validate_run(
    spec: PlanSpec,
    *,
    selection_sha: str,
) -> RunEvidence:
    status_path = spec.arm_root / "adapter" / "run_status.json"
    telemetry_path = spec.arm_root / "results" / "research_raw" / "telemetry_summary.json"
    status = _read_json(status_path, label="run status")
    _validate_status(spec, status)
    telemetry = _read_telemetry_means(
        telemetry_path,
        method=spec.method,
        fingerprint=str(status["config_fingerprint"]),
    )
    input_paths = [status_path, telemetry_path]
    validation_accuracy = validation_loss = None
    final_accuracy: dict[str, float] = {}
    if spec.phase == "final":
        metrics_path = spec.arm_root / "results" / "decontaminated_metrics.json"
        final_accuracy = _read_decontaminated(metrics_path, selection_sha)
        input_paths.append(metrics_path)
    else:
        metrics_path = spec.arm_root / "results" / "validation" / "validation_metrics.json"
        validation_accuracy, validation_loss = _read_validation(metrics_path)
        split_path = spec.arm_root / "results" / "validation" / "split_manifest.json"
        split = _read_json(split_path, label="selection split manifest")
        if split.get("protocol_stage") != "selection" or split.get("validation_data_is_public") is not True:
            raise AnalysisError(f"invalid public selection split: {split_path}")
        input_paths.extend((metrics_path, split_path))
    return RunEvidence(
        spec=spec,
        status=status,
        telemetry=telemetry,
        validation_accuracy=validation_accuracy,
        validation_loss=validation_loss,
        final_accuracy=final_accuracy,
        input_paths=tuple(input_paths),
    )


def _validate_locked_ranking(
    selection: Mapping[str, Any],
    evidence: Sequence[RunEvidence],
    required_seeds: Sequence[int],
) -> None:
    by_candidate_seed = {
        (run.spec.candidate_id, run.spec.seed): run
        for run in evidence
        if run.spec.phase in {"stage1", "stage2"}
    }
    for item in _selection_ranking(selection):
        candidate_id = str(item["candidate_id"])
        seeds = item.get("seeds")
        if seeds != list(required_seeds):
            raise AnalysisError(f"selection candidate={candidate_id} has wrong seeds")
        accuracies = {
            str(seed): by_candidate_seed[(candidate_id, seed)].validation_accuracy
            for seed in required_seeds
        }
        losses = {
            str(seed): by_candidate_seed[(candidate_id, seed)].validation_loss
            for seed in required_seeds
        }
        for field, actual in (("accuracy_by_seed", accuracies), ("loss_by_seed", losses)):
            declared = item.get(field)
            if not isinstance(declared, dict) or set(declared) != set(actual):
                raise AnalysisError(f"selection {candidate_id} {field} schema mismatch")
            if any(not _same(declared[key], value) for key, value in actual.items()):
                raise AnalysisError(f"selection {candidate_id} {field} differs from run artifacts")
        mean_accuracy = math.fsum(float(value) for value in accuracies.values()) / len(accuracies)
        mean_loss = math.fsum(float(value) for value in losses.values()) / len(losses)
        if not _same(item.get("mean_validation_accuracy"), mean_accuracy):
            raise AnalysisError(f"selection {candidate_id} mean accuracy is inconsistent")
        if not _same(item.get("mean_validation_loss"), mean_loss):
            raise AnalysisError(f"selection {candidate_id} mean loss is inconsistent")


def _csv_bytes(rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(fields), lineterminator="\n", extrasaction="raise")
    writer.writeheader()
    for row in rows:
        writer.writerow({field: "" if row.get(field) is None else row.get(field) for field in fields})
    return buffer.getvalue().encode("utf-8")


def _atomic_write_consistent(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if not path.is_file() or path.read_bytes() != content:
            raise AnalysisError(f"refusing to overwrite inconsistent analysis artifact: {path}")
        return
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        os.chmod(path, 0o600)
    finally:
        temporary.unlink(missing_ok=True)


def _selection_rows(
    registry_candidates: Mapping[str, Mapping[str, Any]],
    evidence: Sequence[RunEvidence],
    selected_id: str,
    fixed_id: str,
    locked_best_fixed_id: str,
    ranks: Mapping[str, int],
) -> list[dict[str, Any]]:
    grouped: dict[str, list[RunEvidence]] = {}
    for run in evidence:
        if run.spec.phase in {"stage1", "stage2"}:
            grouped.setdefault(run.spec.candidate_id, []).append(run)
    rows: list[dict[str, Any]] = []
    for candidate_id in sorted(grouped):
        runs = sorted(grouped[candidate_id], key=lambda item: item.spec.seed)
        candidate = registry_candidates[candidate_id]
        params = candidate["params"]
        accuracies = {str(run.spec.seed): run.validation_accuracy for run in runs}
        losses = {str(run.spec.seed): run.validation_loss for run in runs}
        rows.append(
            {
                "candidate_id": candidate_id,
                "family": candidate["family"],
                "method": candidate["method"],
                "initial_or_fixed_C": params["dp_max_grad_norm"],
                "slaclip_target_non_small_clip_fraction": (
                    _candidate_target(params) if candidate["method"] == "slaclip" else None
                ),
                "slaclip_eta": params.get("slaclip_eta") if candidate["method"] == "slaclip" else None,
                "n_selection_runs": len(runs),
                "seeds": ",".join(str(run.spec.seed) for run in runs),
                "mean_validation_accuracy": math.fsum(float(value) for value in accuracies.values()) / len(runs),
                "mean_validation_loss": math.fsum(float(value) for value in losses.values()) / len(runs),
                "accuracy_by_seed_json": json.dumps(accuracies, sort_keys=True, separators=(",", ":")),
                "loss_by_seed_json": json.dumps(losses, sort_keys=True, separators=(",", ":")),
                "promoted_to_stage2": candidate_id in ranks,
                "locked_rank": ranks.get(candidate_id),
                "selected_slaclip": candidate_id == selected_id,
                "locked_fixed_c2": candidate_id == fixed_id,
                "locked_best_fixed": candidate_id == locked_best_fixed_id,
                "params_json": json.dumps(params, sort_keys=True, separators=(",", ":")),
            }
        )
    return rows


def _run_index_rows(evidence: Sequence[RunEvidence], campaign_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run in evidence:
        accounting = run.status["privacy_accounting"]
        row: dict[str, Any] = {
            "NON_PRIVATE_TELEMETRY": True,
            "phase": run.spec.phase,
            "candidate_id": run.spec.candidate_id,
            "role": run.spec.role,
            "seed": run.spec.seed,
            "method": run.spec.method,
            "initial_or_fixed_C": run.spec.clip,
            "slaclip_target_non_small_clip_fraction": run.spec.rho,
            "slaclip_eta": run.spec.eta,
            "update_steps": EXPECTED_UPDATE_STEPS,
            "epsilon_spent": accounting["epsilon_spent"],
            "validation_numeric_exact_accuracy": run.validation_accuracy,
            "validation_loss": run.validation_loss,
            **{name: run.final_accuracy.get(name) for name, _role, _primary in ACCURACY_METRICS},
            **{f"{name}_mean": value for name, value in run.telemetry.items()},
            "config_fingerprint": run.status["config_fingerprint"],
            "arm_root": _relative(campaign_root, run.spec.arm_root),
        }
        rows.append(row)
    phase_order = {"stage1": 0, "stage2": 1, "final": 2}
    return sorted(rows, key=lambda row: (phase_order[str(row["phase"])], int(row["seed"]), str(row["candidate_id"])))


def _paired_statistics(
    candidate: Sequence[float], reference: Sequence[float]
) -> dict[str, float | int]:
    if len(candidate) != 5 or len(reference) != 5:
        raise AnalysisError("paired inference requires exactly five seeds")
    deltas = [left - right for left, right in zip(candidate, reference, strict=True)]
    mean_delta = math.fsum(deltas) / len(deltas)
    sample_sd = statistics.stdev(deltas)
    half_width = T95_DF4 * sample_sd / math.sqrt(len(deltas))
    tolerance = 1e-15
    return {
        "candidate_mean": math.fsum(candidate) / len(candidate),
        "reference_mean": math.fsum(reference) / len(reference),
        "paired_mean_delta": mean_delta,
        "sample_sd": sample_sd,
        "standard_error": sample_sd / math.sqrt(len(deltas)),
        "t95_critical_df4": T95_DF4,
        "t95_ci_low": mean_delta - half_width,
        "t95_ci_high": mean_delta + half_width,
        "wins": sum(value > tolerance for value in deltas),
        "ties": sum(abs(value) <= tolerance for value in deltas),
        "losses": sum(value < -tolerance for value in deltas),
    }


def _paired_accuracy_rows(final: Sequence[RunEvidence]) -> list[dict[str, Any]]:
    indexed = {(run.spec.seed, run.spec.arm_root.name): run for run in final}
    rows: list[dict[str, Any]] = []
    for metric, metric_role, is_primary in ACCURACY_METRICS:
        candidate = [indexed[(seed, "slaclip")].final_accuracy[metric] for seed in FINAL_SEEDS]
        reference = [indexed[(seed, "baseline")].final_accuracy[metric] for seed in FINAL_SEEDS]
        rows.append(
            {
                "comparison": "slaclip_vs_fixed-c2",
                "candidate_role": "slaclip",
                "reference_role": "baseline",
                "metric": metric,
                "metric_role": metric_role,
                "is_primary": is_primary,
                "n_paired_seeds": 5,
                **_paired_statistics(candidate, reference),
                "seeds": ",".join(str(seed) for seed in FINAL_SEEDS),
            }
        )
    return rows


def _paired_mechanism_rows(final: Sequence[RunEvidence]) -> list[dict[str, Any]]:
    indexed = {(run.spec.seed, run.spec.arm_root.name): run for run in final}
    rows: list[dict[str, Any]] = []
    for metric, direction in MECHANISM_METRICS.items():
        candidate = [indexed[(seed, "slaclip")].telemetry[metric] for seed in FINAL_SEEDS]
        reference = [indexed[(seed, "baseline")].telemetry[metric] for seed in FINAL_SEEDS]
        stats = _paired_statistics(candidate, reference)
        if direction == "higher_is_better":
            better, worse = stats["wins"], stats["losses"]
        elif direction == "lower_is_better":
            better, worse = stats["losses"], stats["wins"]
        else:
            better = worse = None
        rows.append(
            {
                "comparison": "slaclip_vs_fixed-c2",
                "candidate_role": "slaclip",
                "reference_role": "baseline",
                "metric": metric,
                "preferred_direction": direction,
                "n_paired_seeds": 5,
                **stats,
                "candidate_better_wins": better,
                "candidate_worse_losses": worse,
                "seeds": ",".join(str(seed) for seed in FINAL_SEEDS),
            }
        )
    return rows


def _provenance_entry(campaign_root: Path, path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise AnalysisError(f"provenance input disappeared during analysis: {path}")
    return {
        "path": _relative(campaign_root, path),
        "sha256": _sha256_file(path),
        "bytes": path.stat().st_size,
    }


def analyze(campaign_root: Path, output_dir: Path = Path("artifacts")) -> dict[str, Any]:
    campaign_root = campaign_root.resolve(strict=True)
    if not campaign_root.is_dir():
        raise AnalysisError(f"campaign root is not a directory: {campaign_root}")
    output_dir = _resolve_inside(campaign_root, output_dir, label="output directory")
    output_dir.mkdir(parents=True, exist_ok=True)

    plans = {phase: _read_plan(campaign_root, phase) for phase in EXPECTED_PHASE_COUNTS}
    all_specs = [*plans["stage1"], *plans["stage2"], *plans["final"]]
    keys = [spec.key for spec in all_specs]
    roots = [spec.arm_root for spec in all_specs]
    if len(set(keys)) != len(keys) or len(set(roots)) != len(roots):
        raise AnalysisError("plans contain duplicate run identities or arm roots")

    registry_path = campaign_root / "screen" / "candidate_registry.json"
    selection_path = campaign_root / "selection" / "selection.json"
    registry = _read_json(registry_path, label="candidate registry")
    selection = _read_json(selection_path, label="locked selection")
    registry_candidates, protocol = _validate_registry(
        campaign_root, registry, [*plans["stage1"], *plans["stage2"]]
    )
    required_seeds, selected_id, fixed_id, locked_best_fixed_id, ranks = _validate_selection_structure(
        selection,
        registry_path,
        registry_candidates,
        protocol,
        plans["stage1"],
        plans["stage2"],
    )
    _validate_final_plan(plans["final"], selected_id, fixed_id, registry_candidates)
    selection_sha = _sha256_bytes(_canonical_json(selection))
    selection_file_sha = _sha256_file(selection_path)

    evidence = [_validate_run(spec, selection_sha=selection_file_sha) for spec in all_specs]
    _validate_locked_ranking(selection, evidence, required_seeds)

    selection_rows = _selection_rows(
        registry_candidates,
        evidence,
        selected_id,
        fixed_id,
        locked_best_fixed_id,
        ranks,
    )
    run_rows = _run_index_rows(evidence, campaign_root)
    final_evidence = [run for run in evidence if run.spec.phase == "final"]
    accuracy_rows = _paired_accuracy_rows(final_evidence)
    mechanism_rows = _paired_mechanism_rows(final_evidence)

    selection_fields = (
        "candidate_id",
        "family",
        "method",
        "initial_or_fixed_C",
        "slaclip_target_non_small_clip_fraction",
        "slaclip_eta",
        "n_selection_runs",
        "seeds",
        "mean_validation_accuracy",
        "mean_validation_loss",
        "accuracy_by_seed_json",
        "loss_by_seed_json",
        "promoted_to_stage2",
        "locked_rank",
        "selected_slaclip",
        "locked_fixed_c2",
        "locked_best_fixed",
        "params_json",
    )
    run_fields = (
        "NON_PRIVATE_TELEMETRY",
        "phase",
        "candidate_id",
        "role",
        "seed",
        "method",
        "initial_or_fixed_C",
        "slaclip_target_non_small_clip_fraction",
        "slaclip_eta",
        "update_steps",
        "epsilon_spent",
        "validation_numeric_exact_accuracy",
        "validation_loss",
        *(name for name, _role, _primary in ACCURACY_METRICS),
        *(f"{name}_mean" for name in MECHANISM_METRICS),
        "config_fingerprint",
        "arm_root",
    )
    paired_accuracy_fields = (
        "comparison",
        "candidate_role",
        "reference_role",
        "metric",
        "metric_role",
        "is_primary",
        "n_paired_seeds",
        "candidate_mean",
        "reference_mean",
        "paired_mean_delta",
        "sample_sd",
        "standard_error",
        "t95_critical_df4",
        "t95_ci_low",
        "t95_ci_high",
        "wins",
        "ties",
        "losses",
        "seeds",
    )
    paired_mechanism_fields = (
        "comparison",
        "candidate_role",
        "reference_role",
        "metric",
        "preferred_direction",
        "n_paired_seeds",
        "candidate_mean",
        "reference_mean",
        "paired_mean_delta",
        "sample_sd",
        "standard_error",
        "t95_critical_df4",
        "t95_ci_low",
        "t95_ci_high",
        "wins",
        "ties",
        "losses",
        "candidate_better_wins",
        "candidate_worse_losses",
        "seeds",
    )

    outputs = {
        "selection_results": (output_dir / "selection_results.csv", _csv_bytes(selection_rows, selection_fields)),
        "run_index": (output_dir / "run_index.csv", _csv_bytes(run_rows, run_fields)),
        "paired_final_accuracy": (
            output_dir / "paired_final_accuracy.csv",
            _csv_bytes(accuracy_rows, paired_accuracy_fields),
        ),
        "paired_final_mechanism": (
            output_dir / "paired_final_mechanism.csv",
            _csv_bytes(mechanism_rows, paired_mechanism_fields),
        ),
        "final_manifest": (
            campaign_root / "final" / "manifest.json",
            _pretty_json({"expected_update_steps": EXPECTED_UPDATE_STEPS}),
        ),
    }
    for _label, (path, content) in outputs.items():
        _atomic_write_consistent(path, content)

    input_paths = {
        *(spec.source_path for spec in all_specs),
        registry_path,
        selection_path,
        *(path for run in evidence for path in run.input_paths),
    }
    provenance_inputs = [
        _provenance_entry(campaign_root, path)
        for path in sorted(input_paths, key=lambda item: _relative(campaign_root, item))
    ]
    provenance_outputs = {
        label: {
            "path": _relative(campaign_root, path),
            "sha256": _sha256_file(path),
            "bytes": path.stat().st_size,
        }
        for label, (path, _content) in outputs.items()
    }
    manifest = {
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "analysis_status": "LOCKED_REFINEMENT_CAMPAIGN_COMPLETE",
        "warning": RAW_WARNING,
        "raw_exact_telemetry_is_non_private": True,
        "expected_update_steps": EXPECTED_UPDATE_STEPS,
        "phase_counts": EXPECTED_PHASE_COUNTS,
        "total_run_count": sum(EXPECTED_PHASE_COUNTS.values()),
        "selection": {
            "registry_path": _relative(campaign_root, registry_path),
            "registry_file_sha256": _sha256_file(registry_path),
            "registry_canonical_sha256": _sha256_bytes(_canonical_json(registry)),
            "selection_path": _relative(campaign_root, selection_path),
            "selection_file_sha256": _sha256_file(selection_path),
            "selection_canonical_sha256": selection_sha,
            "required_seeds": required_seeds,
            "selected_slaclip_candidate_id": selected_id,
            "fixed_c2_candidate_id": fixed_id,
            "selector_best_fixed_candidate_id": locked_best_fixed_id,
        },
        "final": {
            "seeds": list(FINAL_SEEDS),
            "arms": ["baseline", "slaclip"],
            "paired_t_degrees_of_freedom": 4,
            "paired_t95_critical": T95_DF4,
            "inference_unit": "training_seed",
        },
        "inputs": provenance_inputs,
        "outputs": provenance_outputs,
    }
    manifest_path = output_dir / "manifest.json"
    _atomic_write_consistent(manifest_path, _pretty_json(manifest))
    manifest_sha = _sha256_file(manifest_path)
    _atomic_write_consistent(
        output_dir / "manifest.json.sha256",
        f"{manifest_sha}  manifest.json\n".encode("ascii"),
    )
    return manifest


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        manifest = analyze(args.campaign_root, args.output_dir)
    except AnalysisError as exc:
        raise SystemExit(f"refinement analysis refused: {exc}") from exc
    print(
        "refinement analysis complete "
        f"runs={manifest['total_run_count']} "
        f"selected={manifest['selection']['selected_slaclip_candidate_id']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
