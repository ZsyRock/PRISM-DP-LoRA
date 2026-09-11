#!/usr/bin/env python3
"""Fail-closed aggregation for the formal PRISM + SlaClip journal campaign."""

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
from pathlib import Path
from typing import Any, Iterable, Mapping


ANALYSIS_SCHEMA_VERSION = 1
FINAL_MANIFEST_SCHEMA_VERSION = 1
TASKS = ("gsm8k", "AQuA", "mawps", "SVAMP")
METRICS = (*TASKS, "macro")
RAW_WARNING = "NON_PRIVATE: exact training telemetry is not a DP release"
T95 = {
    1: 12.706204736432095,
    2: 4.302652729911275,
    3: 3.182446305284263,
    4: 2.7764451051977987,
}
PAIR_EXCLUDED_CONFIG_KEYS = {
    "config_fingerprint",
    "dp_max_grad_norm",
    "force_eval",
    "force_train",
    "method",
    "output_dir",
    "repeat_id",
    "result_dir",
    "run_id",
    "run_name",
}
FINAL_COMMON_EXCLUDED_KEYS = {
    "base_model",
    "model_revision",
    "protocol_stage",
    "val_set_size",
    "validation_batch_size",
    "validation_data_is_public",
    "validation_seed",
}


class AnalysisError(RuntimeError):
    """Formal campaign evidence is malformed or inconsistent."""


class IncompleteRun(AnalysisError):
    """An expected arm has not produced all three required artifacts yet."""


def _canonical_json(payload: Any) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path, *, label: str, incomplete_if_missing: bool = False) -> dict[str, Any]:
    if not path.is_file():
        error = IncompleteRun if incomplete_if_missing else AnalysisError
        raise error(f"missing {label}: {path}")
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise AnalysisError(f"could not read {label} {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise AnalysisError(f"{label} must be a JSON object: {path}")
    return payload


def _finite(value: Any, *, label: str, lower: float | None = None, upper: float | None = None) -> float:
    if isinstance(value, bool):
        raise AnalysisError(f"{label} must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise AnalysisError(f"{label} must be numeric") from exc
    if not math.isfinite(result):
        raise AnalysisError(f"{label} must be finite")
    if lower is not None and result < lower:
        raise AnalysisError(f"{label} must be >= {lower}")
    if upper is not None and result > upper:
        raise AnalysisError(f"{label} must be <= {upper}")
    return result


def _integer(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise AnalysisError(f"{label} must be an integer")
    return int(value)


def _same(expected: Any, actual: Any) -> bool:
    if isinstance(expected, bool) or isinstance(actual, bool):
        return expected is actual
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        return math.isclose(float(expected), float(actual), rel_tol=1e-12, abs_tol=1e-12)
    return expected == actual


def _safe_child(root: Path, *parts: str) -> Path:
    path = root.joinpath(*parts)
    resolved_parent = path.parent.resolve()
    try:
        resolved_parent.relative_to(root.resolve())
    except ValueError as exc:
        raise AnalysisError(f"artifact path escapes campaign final root: {path}") from exc
    return resolved_parent / path.name


def _parse_selected_env(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise AnalysisError(f"missing immutable selection environment: {path}")
    values: dict[str, str] = {}
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise AnalysisError(f"invalid selected.env line {line_number}")
        key, value = line.split("=", 1)
        if not key or key in values:
            raise AnalysisError(f"duplicate/invalid selected.env key on line {line_number}")
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        values[key] = value
    return values


def _validate_selection(campaign_root: Path, selection_path: Path) -> tuple[dict[str, Any], str]:
    selection = _read_json(selection_path, label="stage2 selection")
    if selection.get("stage") != "stage2":
        raise AnalysisError("formal analysis requires a stage2 selection lock")
    selected = selection.get("selected_slaclip")
    best_fixed = selection.get("best_fixed")
    protocol = selection.get("selection_protocol")
    if not isinstance(protocol, dict) or not isinstance(protocol.get("common_config"), dict):
        raise AnalysisError("selection lock has no validated common_config protocol")
    if not isinstance(selected, dict) or selected.get("method") != "slaclip":
        raise AnalysisError("selection has no locked SlaClip candidate")
    if not isinstance(best_fixed, dict) or best_fixed.get("method") != "baseline":
        raise AnalysisError("selection has no locked fixed-C candidate")
    for label, candidate in (("selected_slaclip", selected), ("best_fixed", best_fixed)):
        if not isinstance(candidate.get("candidate_id"), str) or not isinstance(candidate.get("params"), dict):
            raise AnalysisError(f"selection {label} identity is incomplete")
    selection_sha = _sha256_bytes(_canonical_json(selection))
    selected_env = _parse_selected_env(campaign_root / "selection" / "selected.env")
    if selected_env.get("PRISM_SELECTION_SHA256") != selection_sha:
        raise AnalysisError("selected.env does not match selection.json SHA256")
    if selected_env.get("PRISM_SELECTED_CANDIDATE_ID") != selected["candidate_id"]:
        raise AnalysisError("selected.env SlaClip candidate does not match selection.json")
    if selected_env.get("PRISM_SELECTED_BEST_FIXED_CANDIDATE_ID") != best_fixed["candidate_id"]:
        raise AnalysisError("selected.env best-fixed candidate does not match selection.json")
    return selection, selection_sha


def _validate_final_manifest(
    payload: Mapping[str, Any], selection: Mapping[str, Any], selection_sha: str
) -> dict[str, Any]:
    required = {
        "schema_version",
        "selection_sha256",
        "expected_update_steps",
        "target_epsilon",
        "target_delta",
        "fresh_seeds",
        "best_fixed_seeds",
        "models",
    }
    missing = sorted(required - set(payload))
    unknown = sorted(set(payload) - required)
    if missing or unknown:
        raise AnalysisError(f"final manifest fields mismatch: missing={missing}, unknown={unknown}")
    if payload.get("schema_version") != FINAL_MANIFEST_SCHEMA_VERSION:
        raise AnalysisError(f"final manifest schema_version must be {FINAL_MANIFEST_SCHEMA_VERSION}")
    if payload.get("selection_sha256") != selection_sha:
        raise AnalysisError("final manifest selection SHA does not match selection.json")
    if _integer(payload.get("expected_update_steps"), label="expected_update_steps") != 300:
        raise AnalysisError("formal campaign must complete exactly 300 update steps")
    target_epsilon = _finite(payload.get("target_epsilon"), label="target_epsilon", lower=0.0)
    target_delta = _finite(payload.get("target_delta"), label="target_delta", lower=0.0, upper=1.0)
    if target_epsilon <= 0.0 or not 0.0 < target_delta < 1.0:
        raise AnalysisError("invalid target privacy budget")
    fresh = payload.get("fresh_seeds")
    best = payload.get("best_fixed_seeds")
    if not isinstance(fresh, list) or len(fresh) != 5 or len(set(fresh)) != 5:
        raise AnalysisError("fresh_seeds must contain exactly five unique seeds")
    if not isinstance(best, list) or len(best) != 3 or len(set(best)) != 3:
        raise AnalysisError("best_fixed_seeds must contain exactly three unique seeds")
    for seed in [*fresh, *best]:
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise AnalysisError("all formal seeds must be non-negative integers")
    if not set(best).issubset(fresh):
        raise AnalysisError("best_fixed_seeds must be a subset of fresh_seeds")
    selection_seeds = selection.get("required_seeds")
    if not isinstance(selection_seeds, list) or set(fresh).intersection(selection_seeds):
        raise AnalysisError("formal fresh_seeds must be disjoint from stage2 selection seeds")
    models = payload.get("models")
    if not isinstance(models, list) or len(models) != 2:
        raise AnalysisError("formal manifest must identify exactly two models")
    seen_slugs: set[str] = set()
    for index, model in enumerate(models):
        if not isinstance(model, dict) or set(model) != {"slug", "base_model", "model_revision"}:
            raise AnalysisError(f"models[{index}] must contain slug/base_model/model_revision")
        if not all(isinstance(model[key], str) and model[key] for key in model):
            raise AnalysisError(f"models[{index}] contains an empty identity")
        if "/" in model["slug"] or model["slug"] in {".", ".."} or model["slug"] in seen_slugs:
            raise AnalysisError(f"invalid/duplicate model slug: {model['slug']!r}")
        seen_slugs.add(model["slug"])
    return dict(payload)


def _read_accuracy(path: Path) -> dict[str, float]:
    if not path.is_file():
        raise IncompleteRun(f"missing task accuracy summary: {path}")
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    except OSError as exc:
        raise AnalysisError(f"could not read task accuracy summary {path}: {exc}") from exc
    expected = [*TASKS, "Average"]
    if len(rows) != 1 or list(rows[0]) != expected:
        raise AnalysisError(f"summary.csv must contain exactly {expected}: {path}")
    values = {task: _finite(rows[0][task], label=f"{path}:{task}", lower=0.0, upper=1.0) for task in TASKS}
    macro = _finite(rows[0]["Average"], label=f"{path}:Average", lower=0.0, upper=1.0)
    calculated = math.fsum(values.values()) / len(TASKS)
    if not math.isclose(macro, calculated, rel_tol=0.0, abs_tol=1e-12):
        raise AnalysisError(f"summary macro is not the equal-task mean: {path}")
    return {**values, "macro": macro}


def _validate_config_params(config: Mapping[str, Any], params: Mapping[str, Any], *, label: str) -> None:
    for key, expected in params.items():
        if key not in config or not _same(expected, config[key]):
            raise AnalysisError(
                f"{label} config {key}={config.get(key)!r} does not match selection {expected!r}"
            )


def _pair_signature(config: Mapping[str, Any]) -> bytes:
    retained = {
        key: value
        for key, value in config.items()
        if key not in PAIR_EXCLUDED_CONFIG_KEYS and not key.startswith("slaclip_")
    }
    return _canonical_json(retained)


def _across_seed_signature(config: Mapping[str, Any]) -> bytes:
    retained = {
        key: value
        for key, value in config.items()
        if key not in PAIR_EXCLUDED_CONFIG_KEYS - {"dp_max_grad_norm", "method"}
        and key != "seed"
    }
    return _canonical_json(retained)


def _flatten_telemetry(telemetry: Mapping[str, Any]) -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    for section in ("metrics", "boolean_metrics"):
        values = telemetry.get(section)
        if not isinstance(values, dict):
            raise AnalysisError(f"telemetry {section} must be an object")
        for metric, stats in sorted(values.items()):
            if not isinstance(stats, dict):
                raise AnalysisError(f"telemetry {section}.{metric} must be an object")
            for statistic, value in sorted(stats.items()):
                key = f"{section}__{metric}__{statistic}"
                if isinstance(value, (dict, list)):
                    try:
                        flattened[key] = _canonical_json(value).decode("utf-8")
                    except (TypeError, ValueError) as exc:
                        raise AnalysisError(f"telemetry {key} is not finite/canonical JSON") from exc
                elif isinstance(value, (int, float)) and not isinstance(value, bool):
                    flattened[key] = _finite(value, label=f"telemetry {key}")
                else:
                    flattened[key] = value
    return flattened


def _read_run(
    *,
    final_root: Path,
    model: Mapping[str, str],
    seed: int,
    arm: str,
    artifact_arm: str,
    selection: Mapping[str, Any],
    manifest: Mapping[str, Any],
    alias: bool,
) -> dict[str, Any]:
    run_root = _safe_child(final_root, model["slug"], f"seed-{seed}", artifact_arm)
    status_path = run_root / "adapter" / "run_status.json"
    summary_path = run_root / "results" / "summary.csv"
    telemetry_path = run_root / "results" / "research_raw" / "telemetry_summary.json"
    status = _read_json(status_path, label="completed run status", incomplete_if_missing=True)
    if status.get("state") != "completed":
        raise IncompleteRun(f"run is not completed: {status_path}")
    steps = int(manifest["expected_update_steps"])
    if status.get("update_steps") != steps:
        raise AnalysisError(f"run did not complete exactly {steps} steps: {status_path}")
    if status.get("privacy") != "dp" or status.get("dataset") != "math10k":
        raise AnalysisError(f"run method/privacy/dataset mismatch: {status_path}")
    config = status.get("config")
    if not isinstance(config, dict):
        raise AnalysisError(f"run status has no config: {status_path}")
    expected_method = "slaclip" if arm == "slaclip" else "baseline"
    if status.get("method") != expected_method or config.get("method") != expected_method:
        raise AnalysisError(f"run method does not match arm {arm}: {status_path}")
    expected_identity = {
        "seed": seed,
        "base_model": model["base_model"],
        "model_revision": model["model_revision"],
        "protocol_stage": "final",
        "val_set_size": 0,
        "total_update_steps": steps,
    }
    for key, expected in expected_identity.items():
        if not _same(expected, config.get(key)):
            raise AnalysisError(f"run {key} identity mismatch: {status_path}")
    for key, expected in selection["selection_protocol"]["common_config"].items():
        if key in FINAL_COMMON_EXCLUDED_KEYS:
            continue
        if key not in config or not _same(expected, config[key]):
            raise AnalysisError(
                f"formal run common config {key}={config.get(key)!r} does not match "
                f"locked selection protocol {expected!r}: {status_path}"
            )
    if status.get("base_model") != model["base_model"] or status.get("model_revision") != model["model_revision"]:
        raise AnalysisError(f"top-level model identity mismatch: {status_path}")
    if arm == "slaclip":
        candidate = selection["selected_slaclip"]
    elif arm == "best-fixed":
        candidate = selection["best_fixed"]
    else:
        candidate = {"candidate_id": "canonical-fixed-c1", "params": {"dp_max_grad_norm": 1.0}}
    _validate_config_params(config, candidate["params"], label=f"{model['slug']} seed {seed} {arm}")
    accounting = status.get("privacy_accounting")
    if not isinstance(accounting, dict) or accounting.get("completed_update_steps") != steps:
        raise AnalysisError(f"invalid privacy accountant completion: {status_path}")
    target_epsilon = float(manifest["target_epsilon"])
    epsilon = _finite(accounting.get("epsilon_spent"), label="epsilon_spent", lower=0.0)
    if abs(epsilon - target_epsilon) > 0.02:
        raise AnalysisError(f"epsilon is not within 0.02 of target: {status_path}")
    if not _same(accounting.get("target_epsilon"), target_epsilon) or not _same(
        accounting.get("target_delta"), manifest["target_delta"]
    ):
        raise AnalysisError(f"privacy target mismatch: {status_path}")
    accuracy = _read_accuracy(summary_path)
    telemetry = _read_json(telemetry_path, label="NON_PRIVATE telemetry summary", incomplete_if_missing=True)
    if telemetry.get("NON_PRIVATE_TELEMETRY") is not True or "NON-PRIVATE" not in str(telemetry.get("warning", "")):
        raise AnalysisError(f"telemetry summary lacks NON_PRIVATE warning: {telemetry_path}")
    telemetry_steps = telemetry.get("steps")
    if not isinstance(telemetry_steps, dict) or any(
        telemetry_steps.get(key) != expected
        for key, expected in (("count", steps), ("first", 1), ("last", steps), ("missing_count", 0))
    ):
        raise AnalysisError(f"telemetry does not cover all {steps} steps: {telemetry_path}")
    identity = telemetry.get("run_identity")
    if not isinstance(identity, dict) or identity.get("config_fingerprint") != status.get("config_fingerprint"):
        raise AnalysisError(f"telemetry/status fingerprint mismatch: {telemetry_path}")
    for key, expected in (
        ("method", expected_method),
        ("base_model", model["base_model"]),
        ("model_revision", model["model_revision"]),
    ):
        if identity.get(key) != expected:
            raise AnalysisError(f"telemetry {key} identity mismatch: {telemetry_path}")
    source = telemetry.get("source")
    if not isinstance(source, dict) or not isinstance(source.get("raw_sha256"), str):
        raise AnalysisError(f"telemetry has no raw source hash: {telemetry_path}")
    return {
        "model_slug": model["slug"],
        "base_model": model["base_model"],
        "model_revision": model["model_revision"],
        "seed": seed,
        "arm": arm,
        "artifact_arm": artifact_arm,
        "artifact_alias": alias,
        "candidate_id": candidate["candidate_id"],
        "config": config,
        "config_fingerprint": status.get("config_fingerprint"),
        "epsilon_spent": epsilon,
        "accuracy": accuracy,
        "telemetry": telemetry,
        "telemetry_flat": _flatten_telemetry(telemetry),
        "status_sha256": _sha256_file(status_path),
        "summary_sha256": _sha256_file(summary_path),
        "telemetry_summary_sha256": _sha256_file(telemetry_path),
        "raw_telemetry_sha256": source["raw_sha256"],
    }


def _validate_pairs(runs: Iterable[Mapping[str, Any]]) -> None:
    indexed = {(run["model_slug"], run["seed"], run["arm"]): run for run in runs}
    for (model, seed, arm), run in indexed.items():
        if arm != "baseline":
            continue
        slaclip = indexed.get((model, seed, "slaclip"))
        if slaclip is None:
            continue
        if _pair_signature(run["config"]) != _pair_signature(slaclip["config"]):
            raise AnalysisError(f"baseline/SlaClip paired settings differ for model={model}, seed={seed}")
    signatures: dict[tuple[str, str], bytes] = {}
    for run in indexed.values():
        key = (run["model_slug"], run["arm"])
        signature = _across_seed_signature(run["config"])
        existing = signatures.setdefault(key, signature)
        if existing != signature:
            raise AnalysisError(
                f"formal settings vary across seeds for model={run['model_slug']}, arm={run['arm']}"
            )


def _accuracy_rows(runs: Iterable[Mapping[str, Any]], analysis_status: str, selection_sha: str) -> list[dict[str, Any]]:
    rows = []
    for run in sorted(runs, key=lambda item: (item["model_slug"], item["seed"], item["arm"])):
        for metric in METRICS:
            rows.append(
                {
                    "analysis_status": analysis_status,
                    "raw_exact_telemetry_status": RAW_WARNING,
                    "selection_sha256": selection_sha,
                    "model_slug": run["model_slug"],
                    "base_model": run["base_model"],
                    "model_revision": run["model_revision"],
                    "seed": run["seed"],
                    "arm": run["arm"],
                    "artifact_arm": run["artifact_arm"],
                    "artifact_alias": run["artifact_alias"],
                    "candidate_id": run["candidate_id"],
                    "config_fingerprint": run["config_fingerprint"],
                    "metric": metric,
                    "accuracy": run["accuracy"][metric],
                }
            )
    return rows


def _paired_rows(runs: Iterable[Mapping[str, Any]], analysis_status: str, selection_sha: str) -> list[dict[str, Any]]:
    indexed = {(run["model_slug"], run["seed"], run["arm"]): run for run in runs}
    models = sorted({run["model_slug"] for run in runs})
    comparisons = (
        ("slaclip_vs_baseline", "slaclip", "baseline"),
        ("best_fixed_vs_baseline", "best-fixed", "baseline"),
        ("slaclip_vs_best_fixed", "slaclip", "best-fixed"),
    )
    rows = []
    for model in models:
        for comparison, candidate_arm, reference_arm in comparisons:
            paired_seeds = sorted(
                seed
                for candidate_model, seed, arm in indexed
                if candidate_model == model
                and arm == candidate_arm
                and (model, seed, reference_arm) in indexed
            )
            if not paired_seeds:
                continue
            for metric in METRICS:
                deltas = [
                    indexed[(model, seed, candidate_arm)]["accuracy"][metric]
                    - indexed[(model, seed, reference_arm)]["accuracy"][metric]
                    for seed in paired_seeds
                ]
                n = len(deltas)
                mean = math.fsum(deltas) / n
                sample_sd = statistics.stdev(deltas) if n >= 2 else None
                if sample_sd is not None and n - 1 in T95:
                    half_width = T95[n - 1] * sample_sd / math.sqrt(n)
                    ci_low, ci_high = mean - half_width, mean + half_width
                else:
                    ci_low = ci_high = None
                rows.append(
                    {
                        "analysis_status": analysis_status,
                        "raw_exact_telemetry_status": RAW_WARNING,
                        "selection_sha256": selection_sha,
                        "model_slug": model,
                        "comparison": comparison,
                        "candidate_arm": candidate_arm,
                        "reference_arm": reference_arm,
                        "metric": metric,
                        "n": n,
                        "seeds": ",".join(str(seed) for seed in paired_seeds),
                        "paired_mean_delta": mean,
                        "sample_sd": sample_sd,
                        "t95_ci_low": ci_low,
                        "t95_ci_high": ci_high,
                        "wins": sum(delta > 0.0 for delta in deltas),
                        "ties": sum(delta == 0.0 for delta in deltas),
                        "losses": sum(delta < 0.0 for delta in deltas),
                    }
                )
    return rows


def _telemetry_rows(runs: Iterable[Mapping[str, Any]], analysis_status: str, selection_sha: str) -> list[dict[str, Any]]:
    rows = []
    for run in sorted(runs, key=lambda item: (item["model_slug"], item["seed"], item["arm"])):
        rows.append(
            {
                "analysis_status": analysis_status,
                "NON_PRIVATE_TELEMETRY": True,
                "raw_exact_telemetry_status": RAW_WARNING,
                "selection_sha256": selection_sha,
                "model_slug": run["model_slug"],
                "base_model": run["base_model"],
                "model_revision": run["model_revision"],
                "seed": run["seed"],
                "arm": run["arm"],
                "artifact_arm": run["artifact_arm"],
                "artifact_alias": run["artifact_alias"],
                "candidate_id": run["candidate_id"],
                "config_fingerprint": run["config_fingerprint"],
                "epsilon_spent": run["epsilon_spent"],
                "raw_telemetry_sha256": run["raw_telemetry_sha256"],
                "telemetry_summary_sha256": run["telemetry_summary_sha256"],
                **run["telemetry_flat"],
            }
        )
    return rows


def _csv_text(rows: list[Mapping[str, Any]], *, preferred: list[str] | None = None) -> str:
    if not rows:
        fieldnames = preferred or ["analysis_status", "raw_exact_telemetry_status"]
    else:
        all_fields = set().union(*(row.keys() for row in rows))
        fieldnames = [field for field in (preferred or []) if field in all_fields]
        fieldnames.extend(sorted(all_fields - set(fieldnames)))
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, lineterminator="\n", extrasaction="raise")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        directory_fd = os.open(path.parent, flags)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def analyze_campaign(
    *,
    campaign_root: Path,
    selection_path: Path,
    final_root: Path,
    output_dir: Path,
    allow_incomplete: bool,
) -> dict[str, Any]:
    campaign_root = campaign_root.resolve(strict=True)
    selection_path = selection_path if selection_path.is_absolute() else campaign_root / selection_path
    final_root = final_root if final_root.is_absolute() else campaign_root / final_root
    output_dir = output_dir if output_dir.is_absolute() else campaign_root / output_dir
    selection_path = selection_path.resolve(strict=True)
    final_root = final_root.resolve(strict=True)
    for label, path in (("selection", selection_path), ("final", final_root), ("journal output", output_dir)):
        comparison = path if label != "journal output" else path.parent.resolve()
        try:
            comparison.relative_to(campaign_root)
        except ValueError as exc:
            raise AnalysisError(f"{label} path must remain inside campaign root") from exc
    selection, selection_sha = _validate_selection(campaign_root, selection_path)
    final_manifest_path = final_root / "manifest.json"
    final_manifest = _validate_final_manifest(
        _read_json(final_manifest_path, label="formal final manifest"), selection, selection_sha
    )
    best_is_baseline = (
        selection.get("canonical_fixed_candidate_id") == selection["best_fixed"]["candidate_id"]
        and _same(selection["best_fixed"]["params"].get("dp_max_grad_norm"), 1.0)
    )
    runs: list[dict[str, Any]] = []
    incomplete: list[dict[str, Any]] = []
    for model in final_manifest["models"]:
        for seed in final_manifest["fresh_seeds"]:
            for arm in ("baseline", "slaclip"):
                try:
                    runs.append(
                        _read_run(
                            final_root=final_root,
                            model=model,
                            seed=seed,
                            arm=arm,
                            artifact_arm=arm,
                            selection=selection,
                            manifest=final_manifest,
                            alias=False,
                        )
                    )
                except IncompleteRun as exc:
                    if not allow_incomplete:
                        raise
                    incomplete.append({"model_slug": model["slug"], "seed": seed, "arm": arm, "reason": str(exc)})
        for seed in final_manifest["best_fixed_seeds"]:
            artifact_arm = "baseline" if best_is_baseline else "best-fixed"
            try:
                runs.append(
                    _read_run(
                        final_root=final_root,
                        model=model,
                        seed=seed,
                        arm="best-fixed",
                        artifact_arm=artifact_arm,
                        selection=selection,
                        manifest=final_manifest,
                        alias=best_is_baseline,
                    )
                )
            except IncompleteRun as exc:
                if not allow_incomplete:
                    raise
                incomplete.append(
                    {"model_slug": model["slug"], "seed": seed, "arm": "best-fixed", "reason": str(exc)}
                )
    expected_runs = len(final_manifest["models"]) * (
        2 * len(final_manifest["fresh_seeds"]) + len(final_manifest["best_fixed_seeds"])
    )
    complete = not incomplete and len(runs) == expected_runs
    if not complete and not allow_incomplete:
        raise AnalysisError(f"formal campaign is incomplete: observed={len(runs)}, expected={expected_runs}")
    _validate_pairs(runs)
    analysis_status = "FORMAL_COMPLETE" if complete else "INCOMPLETE_SNAPSHOT_DO_NOT_REPORT"
    accuracy_rows = _accuracy_rows(runs, analysis_status, selection_sha)
    paired_rows = _paired_rows(runs, analysis_status, selection_sha)
    telemetry_rows = _telemetry_rows(runs, analysis_status, selection_sha)
    accuracy_text = _csv_text(
        accuracy_rows,
        preferred=[
            "analysis_status",
            "raw_exact_telemetry_status",
            "selection_sha256",
            "model_slug",
            "seed",
            "arm",
            "metric",
            "accuracy",
        ],
    )
    paired_text = _csv_text(
        paired_rows,
        preferred=[
            "analysis_status",
            "raw_exact_telemetry_status",
            "model_slug",
            "comparison",
            "metric",
            "n",
            "paired_mean_delta",
            "sample_sd",
            "t95_ci_low",
            "t95_ci_high",
            "wins",
            "ties",
            "losses",
            "seeds",
        ],
    )
    telemetry_text = _csv_text(
        telemetry_rows,
        preferred=[
            "analysis_status",
            "NON_PRIVATE_TELEMETRY",
            "raw_exact_telemetry_status",
            "model_slug",
            "seed",
            "arm",
        ],
    )
    output_payloads = {
        "accuracy_by_run.csv": accuracy_text.encode("utf-8"),
        "paired_accuracy_summary.csv": paired_text.encode("utf-8"),
        "telemetry_by_run.csv": telemetry_text.encode("utf-8"),
    }
    script_path = Path(__file__).resolve()
    analysis_manifest = {
        "analysis_schema_version": ANALYSIS_SCHEMA_VERSION,
        "analysis_status": analysis_status,
        "complete": complete,
        "allow_incomplete": bool(allow_incomplete),
        "warning": RAW_WARNING,
        "raw_exact_telemetry_is_non_private": True,
        "selection_sha256": selection_sha,
        "selection_candidate_id": selection["selected_slaclip"]["candidate_id"],
        "best_fixed_candidate_id": selection["best_fixed"]["candidate_id"],
        "best_fixed_uses_baseline_alias": best_is_baseline,
        "final_manifest_sha256": _sha256_file(final_manifest_path),
        "analyzer_sha256": _sha256_file(script_path),
        "expected_runs": expected_runs,
        "observed_runs": len(runs),
        "missing_or_incomplete_runs": incomplete,
        "fresh_seeds": final_manifest["fresh_seeds"],
        "best_fixed_seeds": final_manifest["best_fixed_seeds"],
        "models": final_manifest["models"],
        "run_inputs": [
            {
                "model_slug": run["model_slug"],
                "seed": run["seed"],
                "arm": run["arm"],
                "artifact_alias": run["artifact_alias"],
                "config_fingerprint": run["config_fingerprint"],
                "status_sha256": run["status_sha256"],
                "summary_sha256": run["summary_sha256"],
                "telemetry_summary_sha256": run["telemetry_summary_sha256"],
                "raw_telemetry_sha256": run["raw_telemetry_sha256"],
            }
            for run in sorted(runs, key=lambda item: (item["model_slug"], item["seed"], item["arm"]))
        ],
        "outputs": {name: _sha256_bytes(content) for name, content in output_payloads.items()},
    }
    for name, content in output_payloads.items():
        _atomic_write(output_dir / name, content)
    _atomic_write(
        output_dir / "manifest.json",
        (json.dumps(analysis_manifest, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8"),
    )
    return analysis_manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--selection", type=Path, default=Path("selection/selection.json"))
    parser.add_argument("--final-root", type=Path, default=Path("final"))
    parser.add_argument("--output-dir", type=Path, default=Path("journal"))
    parser.add_argument("--allow-incomplete", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = analyze_campaign(
            campaign_root=args.campaign_root,
            selection_path=args.selection,
            final_root=args.final_root,
            output_dir=args.output_dir,
            allow_incomplete=args.allow_incomplete,
        )
    except AnalysisError as exc:
        raise SystemExit(f"journal analysis refused: {exc}") from exc
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
