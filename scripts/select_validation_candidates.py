#!/usr/bin/env python3
"""Deterministically select PRISM/SlaClip candidates using validation only.

The selector deliberately has no directory-discovery logic.  It reads the
three explicitly registered artifacts for each screen run and refuses paths
outside ``<campaign>/screen``.  In particular, task-test/evaluation outputs
cannot become inputs to hyperparameter selection.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import re
import shlex
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping


REGISTRY_SCHEMA_VERSION = 1
SELECTION_SCHEMA_VERSION = 1
EXPECTED_METRIC = "response_only_mean_per_record_causal_lm_loss"
NUMERIC_EXACT_METRIC = "public_math10k_numeric_exact_match_accuracy"
SUPPORTED_METRICS = {EXPECTED_METRIC, NUMERIC_EXACT_METRIC}
EXPECTED_LOSS_DEFINITION = "response_only_per_record_mean_of_nonignored_next_token_losses"
FIXED_PARAM_KEYS = {"dp_max_grad_norm"}
SLACLIP_COMMON_PARAM_KEYS = {
    "dp_max_grad_norm",
    "slaclip_eta",
    "slaclip_num_slots",
    "slaclip_c_min",
    "slaclip_c_max",
}
SLACLIP_TARGET_PARAM_KEY = "slaclip_target_non_small_clip_fraction"
SLACLIP_LEGACY_TARGET_PARAM_KEY = "slaclip_beta"
SLACLIP_PARAM_KEYS = SLACLIP_COMMON_PARAM_KEYS | {SLACLIP_TARGET_PARAM_KEY}
SLACLIP_LEGACY_PARAM_KEYS = SLACLIP_COMMON_PARAM_KEYS | {
    SLACLIP_LEGACY_TARGET_PARAM_KEY
}
FORBIDDEN_PATH_PARTS = {
    "test",
    "tests",
    "evaluation",
    "evaluations",
    "eval",
    "gsm8k",
    "aqua",
    "mawps",
    "svamp",
}
ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class SelectionError(RuntimeError):
    """Raised when a candidate set is unsafe, incomplete, or inconsistent."""


def _canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_payload(payload: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise SelectionError(f"could not read {label} {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise SelectionError(f"{label} must be a JSON object: {path}")
    return payload


def _require_exact_keys(payload: Mapping[str, Any], allowed: set[str], *, label: str) -> None:
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise SelectionError(f"{label} contains unsupported fields: {unknown}")


def _resolve_screen_input(campaign_root: Path, value: Any, *, filename: str, label: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise SelectionError(f"{label} must be a non-empty path string")
    supplied = Path(value)
    path = supplied if supplied.is_absolute() else campaign_root / supplied
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise SelectionError(f"missing {label}: {path}") from exc
    screen_root = (campaign_root / "screen").resolve(strict=True)
    try:
        screen_relative = resolved.relative_to(screen_root)
    except ValueError as exc:
        raise SelectionError(f"{label} must remain inside {screen_root}: {resolved}") from exc
    if resolved.name != filename:
        raise SelectionError(f"{label} must be named {filename}, got: {resolved.name}")
    forbidden = FORBIDDEN_PATH_PARTS.intersection(part.casefold() for part in screen_relative.parts)
    if forbidden:
        raise SelectionError(f"{label} path contains forbidden test/evaluation component(s): {sorted(forbidden)}")
    return resolved


def _finite_number(value: Any, *, label: str, minimum: float | None = None) -> float:
    if isinstance(value, bool):
        raise SelectionError(f"{label} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise SelectionError(f"{label} must be a finite number") from exc
    if not math.isfinite(result) or (minimum is not None and result < minimum):
        raise SelectionError(f"{label} must be finite and >= {minimum}, got {value!r}")
    return result


def _integer(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SelectionError(f"{label} must be an integer, got {value!r}")
    return int(value)


def _values_match(expected: Any, actual: Any) -> bool:
    if isinstance(expected, bool) or isinstance(actual, bool):
        return expected is actual
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        return math.isclose(float(expected), float(actual), rel_tol=1e-12, abs_tol=1e-12)
    return expected == actual


def _validate_protocol(registry: Mapping[str, Any]) -> dict[str, Any]:
    _require_exact_keys(
        registry,
        {"schema_version", "selection_protocol", "candidates"},
        label="candidate registry",
    )
    if registry.get("schema_version") != REGISTRY_SCHEMA_VERSION:
        raise SelectionError(f"candidate registry schema_version must be {REGISTRY_SCHEMA_VERSION}")
    protocol = registry.get("selection_protocol")
    if not isinstance(protocol, dict):
        raise SelectionError("selection_protocol must be a JSON object")
    required = {
        "name",
        "protocol_stage",
        "validation_data_is_public",
        "selection_metric",
        "loss_definition",
        "required_update_steps",
        "target_epsilon",
        "epsilon_tolerance",
        "stage1_seed",
        "stage2_seeds",
        "common_config",
    }
    _require_exact_keys(protocol, required, label="selection_protocol")
    missing = sorted(required - set(protocol))
    if missing:
        raise SelectionError(f"selection_protocol is missing fields: {missing}")
    if not isinstance(protocol.get("name"), str) or not protocol["name"].strip():
        raise SelectionError("selection_protocol.name must be non-empty")
    if protocol.get("protocol_stage") != "selection":
        raise SelectionError("selection_protocol.protocol_stage must be 'selection'")
    if protocol.get("validation_data_is_public") is not True:
        raise SelectionError("selection requires validation_data_is_public=true")
    if protocol.get("selection_metric") not in SUPPORTED_METRICS:
        raise SelectionError(
            f"selection_metric must be one of {sorted(SUPPORTED_METRICS)!r}"
        )
    if protocol.get("loss_definition") != EXPECTED_LOSS_DEFINITION:
        raise SelectionError(f"loss_definition must be {EXPECTED_LOSS_DEFINITION!r}")
    required_steps = _integer(protocol.get("required_update_steps"), label="required_update_steps")
    if required_steps != 300:
        raise SelectionError("journal selection protocol requires exactly 300 update steps")
    target_epsilon = _finite_number(protocol.get("target_epsilon"), label="target_epsilon", minimum=0.0)
    tolerance = _finite_number(protocol.get("epsilon_tolerance"), label="epsilon_tolerance", minimum=0.0)
    if target_epsilon <= 0.0 or tolerance > max(0.1, target_epsilon * 0.02):
        raise SelectionError("target_epsilon must be positive and epsilon_tolerance must be reasonably small")
    stage1_seed = _integer(protocol.get("stage1_seed"), label="stage1_seed")
    if stage1_seed < 0:
        raise SelectionError("stage1_seed must be non-negative")
    stage2_seeds = protocol.get("stage2_seeds")
    if not isinstance(stage2_seeds, list):
        raise SelectionError("stage2_seeds must be a preregistered list of integers")
    if len(stage2_seeds) < 3:
        raise SelectionError("stage2_seeds must contain at least three seeds")
    validated_stage2_seeds = [
        _integer(seed, label=f"stage2_seeds[{index}]")
        for index, seed in enumerate(stage2_seeds)
    ]
    if any(seed < 0 for seed in validated_stage2_seeds):
        raise SelectionError("stage2_seeds must contain only non-negative seeds")
    if len(set(validated_stage2_seeds)) != len(validated_stage2_seeds):
        raise SelectionError("stage2_seeds must contain unique seeds")
    if validated_stage2_seeds != sorted(validated_stage2_seeds):
        raise SelectionError("stage2_seeds must be strictly increasing")
    if stage1_seed not in stage2_seeds:
        raise SelectionError("stage1_seed must also be present in stage2_seeds")
    common_config = protocol.get("common_config")
    if not isinstance(common_config, dict) or not common_config:
        raise SelectionError("selection_protocol.common_config must be a non-empty object")
    for key, value in common_config.items():
        if not isinstance(key, str) or not key or isinstance(value, (dict, list)):
            raise SelectionError("selection_protocol.common_config must contain scalar RunConfig fields")
    forbidden_common = (
        FIXED_PARAM_KEYS
        | (SLACLIP_PARAM_KEYS - {"dp_max_grad_norm"})
        | (SLACLIP_LEGACY_PARAM_KEYS - {"dp_max_grad_norm"})
        | {"method", "seed"}
    )
    overlap = sorted(set(common_config).intersection(forbidden_common))
    if overlap:
        raise SelectionError(f"common_config contains candidate-specific fields: {overlap}")
    return dict(protocol)


def _validate_split_manifest(payload: Mapping[str, Any], *, path: Path) -> str:
    claimed = payload.get("manifest_sha256")
    if not isinstance(claimed, str) or not re.fullmatch(r"[0-9a-f]{64}", claimed):
        raise SelectionError(f"split manifest has no valid manifest_sha256: {path}")
    unhashed = dict(payload)
    unhashed.pop("manifest_sha256", None)
    calculated = _sha256_payload(unhashed)
    if calculated != claimed:
        raise SelectionError(
            f"split manifest content hash mismatch: claimed={claimed}, calculated={calculated}, path={path}"
        )
    if payload.get("protocol_stage") != "selection" or payload.get("validation_data_is_public") is not True:
        raise SelectionError(f"split manifest is not an explicit public selection split: {path}")
    validation_rows = _integer(payload.get("validation_rows"), label=f"validation_rows in {path}")
    if validation_rows <= 0:
        raise SelectionError(f"selection split must contain validation rows: {path}")
    return claimed


def _candidate_map(registry: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    raw = registry.get("candidates")
    if not isinstance(raw, list) or not raw:
        raise SelectionError("candidate registry must contain a non-empty candidates list")
    candidates: dict[str, dict[str, Any]] = {}
    for index, candidate in enumerate(raw):
        if not isinstance(candidate, dict):
            raise SelectionError(f"candidate[{index}] must be a JSON object")
        _require_exact_keys(candidate, {"id", "family", "method", "params", "runs"}, label=f"candidate[{index}]")
        candidate_id = candidate.get("id")
        if not isinstance(candidate_id, str) or not ID_RE.fullmatch(candidate_id):
            raise SelectionError(f"candidate[{index}].id is invalid: {candidate_id!r}")
        if candidate_id in candidates:
            raise SelectionError(f"duplicate candidate id: {candidate_id}")
        family = candidate.get("family")
        method = candidate.get("method")
        if family not in {"fixed", "slaclip"}:
            raise SelectionError(f"candidate {candidate_id} family must be fixed or slaclip")
        expected_method = "baseline" if family == "fixed" else "slaclip"
        if method != expected_method:
            raise SelectionError(f"candidate {candidate_id} method must be {expected_method!r}")
        params = candidate.get("params")
        runs = candidate.get("runs")
        if not isinstance(params, dict) or not params:
            raise SelectionError(f"candidate {candidate_id} params must be a non-empty object")
        if not isinstance(runs, dict):
            raise SelectionError(f"candidate {candidate_id} runs must be an object")
        for key, value in params.items():
            if not isinstance(key, str) or not key or isinstance(value, (dict, list)):
                raise SelectionError(f"candidate {candidate_id} params must contain scalar config fields")
        accepted_param_key_sets = (
            (FIXED_PARAM_KEYS,)
            if family == "fixed"
            else (SLACLIP_PARAM_KEYS, SLACLIP_LEGACY_PARAM_KEYS)
        )
        if not any(set(params) == keys for keys in accepted_param_key_sets):
            expected_text = (
                sorted(FIXED_PARAM_KEYS)
                if family == "fixed"
                else [
                    sorted(SLACLIP_PARAM_KEYS),
                    sorted(SLACLIP_LEGACY_PARAM_KEYS),
                ]
            )
            raise SelectionError(
                f"candidate {candidate_id} params must contain exactly transferable fields "
                f"{expected_text}, got {sorted(params)}"
            )
        candidates[candidate_id] = dict(candidate)
    if not any(item["family"] == "fixed" for item in candidates.values()):
        raise SelectionError("registry contains no fixed-C candidate")
    if not any(item["family"] == "slaclip" for item in candidates.values()):
        raise SelectionError("registry contains no SlaClip candidate")
    return candidates


def _registered_run(candidate: Mapping[str, Any], seed: int) -> Mapping[str, Any]:
    run = candidate["runs"].get(str(seed))
    if not isinstance(run, dict):
        raise SelectionError(f"candidate {candidate['id']} has no registered run for seed {seed}")
    _require_exact_keys(
        run,
        {"run_status", "validation_metrics", "split_manifest"},
        label=f"candidate {candidate['id']} seed {seed} run",
    )
    if set(run) != {"run_status", "validation_metrics", "split_manifest"}:
        raise SelectionError(f"candidate {candidate['id']} seed {seed} run must register exactly three artifacts")
    return run


def _validate_run(
    campaign_root: Path,
    candidate: Mapping[str, Any],
    seed: int,
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    run = _registered_run(candidate, seed)
    status_path = _resolve_screen_input(
        campaign_root, run["run_status"], filename="run_status.json", label="run status"
    )
    metrics_path = _resolve_screen_input(
        campaign_root,
        run["validation_metrics"],
        filename="validation_metrics.json",
        label="validation metrics",
    )
    split_path = _resolve_screen_input(
        campaign_root,
        run["split_manifest"],
        filename="split_manifest.json",
        label="split manifest",
    )
    status = _read_json(status_path, label="run status")
    metrics = _read_json(metrics_path, label="validation metrics")
    split = _read_json(split_path, label="split manifest")
    split_sha = _validate_split_manifest(split, path=split_path)

    required_steps = int(protocol["required_update_steps"])
    if status.get("state") != "completed":
        raise SelectionError(f"candidate {candidate['id']} seed {seed} is not completed")
    if status.get("privacy") != "dp" or status.get("method") != candidate["method"]:
        raise SelectionError(f"candidate {candidate['id']} seed {seed} method/privacy mismatch")
    if _integer(status.get("update_steps"), label="run_status.update_steps") != required_steps:
        raise SelectionError(f"candidate {candidate['id']} seed {seed} did not complete {required_steps} steps")
    config = status.get("config")
    if not isinstance(config, dict):
        raise SelectionError(f"candidate {candidate['id']} seed {seed} status has no config")
    if config.get("protocol_stage") != "selection" or config.get("validation_data_is_public") is not True:
        raise SelectionError(f"candidate {candidate['id']} seed {seed} was not run under public selection protocol")
    if config.get("seed") != seed or config.get("method") != candidate["method"]:
        raise SelectionError(f"candidate {candidate['id']} seed {seed} config identity mismatch")
    if config.get("total_update_steps") != required_steps:
        raise SelectionError(f"candidate {candidate['id']} seed {seed} config does not request 300 steps")
    for key, expected in protocol["common_config"].items():
        if key not in config or not _values_match(expected, config[key]):
            raise SelectionError(
                f"candidate {candidate['id']} seed {seed} common config {key}={config.get(key)!r} "
                f"does not match registry value {expected!r}"
            )
    for key, expected in candidate["params"].items():
        if key not in config or not _values_match(expected, config[key]):
            raise SelectionError(
                f"candidate {candidate['id']} seed {seed} config {key}={config.get(key)!r} "
                f"does not match registry value {expected!r}"
            )

    status_split = status.get("data_split")
    if not isinstance(status_split, dict) or status_split.get("manifest_sha256") != split_sha:
        raise SelectionError(f"candidate {candidate['id']} seed {seed} status split does not match manifest")
    if metrics.get("manifest_sha256") != split_sha:
        raise SelectionError(f"candidate {candidate['id']} seed {seed} metric split does not match manifest")
    if metrics.get("protocol_stage") != "selection" or metrics.get("validation_data_is_public") is not True:
        raise SelectionError(f"candidate {candidate['id']} seed {seed} metrics are not public-selection metrics")
    if metrics.get("NON_PRIVATE_SELECTION_METRIC") is not True:
        raise SelectionError(f"candidate {candidate['id']} seed {seed} lacks NON_PRIVATE metric acknowledgement")
    if metrics.get("selection_metric") != protocol["selection_metric"]:
        raise SelectionError(f"candidate {candidate['id']} seed {seed} selection metric mismatch")
    if metrics.get("loss_definition") != protocol["loss_definition"]:
        raise SelectionError(f"candidate {candidate['id']} seed {seed} loss definition mismatch")
    records = _integer(metrics.get("records"), label="validation metric records")
    if records != split.get("validation_rows"):
        raise SelectionError(f"candidate {candidate['id']} seed {seed} validation row count mismatch")
    loss = _finite_number(metrics.get("loss_mean"), label="validation response-only loss", minimum=0.0)
    accuracy: float | None = None
    if protocol["selection_metric"] == NUMERIC_EXACT_METRIC:
        accuracy = _finite_number(
            metrics.get("numeric_exact_accuracy"),
            label="validation numeric exact accuracy",
            minimum=0.0,
        )
        if accuracy > 1.0:
            raise SelectionError(
                f"validation numeric exact accuracy must be <= 1, got {accuracy!r}"
            )
        correct = _integer(
            metrics.get("numeric_exact_correct"),
            label="validation numeric exact correct",
        )
        parse_failures = _integer(
            metrics.get("numeric_parse_failures"),
            label="validation numeric parse failures",
        )
        if not 0 <= correct <= records or not 0 <= parse_failures <= records:
            raise SelectionError(
                f"candidate {candidate['id']} seed {seed} has invalid numeric counts"
            )
        expected_accuracy = float(correct) / float(records)
        if not math.isclose(accuracy, expected_accuracy, rel_tol=0.0, abs_tol=1e-12):
            raise SelectionError(
                f"candidate {candidate['id']} seed {seed} numeric accuracy/count mismatch"
            )
    embedded = status.get("validation")
    if not isinstance(embedded, dict):
        raise SelectionError(f"candidate {candidate['id']} seed {seed} status has no embedded validation result")
    embedded_loss = _finite_number(embedded.get("loss_mean"), label="embedded validation loss", minimum=0.0)
    if not math.isclose(loss, embedded_loss, rel_tol=0.0, abs_tol=1e-12):
        raise SelectionError(f"candidate {candidate['id']} seed {seed} embedded validation loss mismatch")
    if accuracy is not None:
        embedded_accuracy = _finite_number(
            embedded.get("numeric_exact_accuracy"),
            label="embedded validation numeric exact accuracy",
            minimum=0.0,
        )
        if not math.isclose(accuracy, embedded_accuracy, rel_tol=0.0, abs_tol=1e-12):
            raise SelectionError(
                f"candidate {candidate['id']} seed {seed} embedded validation numeric accuracy mismatch"
            )
    if embedded.get("manifest_sha256") != split_sha:
        raise SelectionError(f"candidate {candidate['id']} seed {seed} embedded validation split mismatch")

    accounting = status.get("privacy_accounting")
    if not isinstance(accounting, dict):
        raise SelectionError(f"candidate {candidate['id']} seed {seed} has no privacy accounting")
    if accounting.get("completed_update_steps") != required_steps:
        raise SelectionError(f"candidate {candidate['id']} seed {seed} privacy accountant step mismatch")
    target = _finite_number(accounting.get("target_epsilon"), label="accountant target epsilon", minimum=0.0)
    spent = _finite_number(accounting.get("epsilon_spent"), label="epsilon spent", minimum=0.0)
    registry_target = float(protocol["target_epsilon"])
    tolerance = float(protocol["epsilon_tolerance"])
    if not math.isclose(target, registry_target, rel_tol=0.0, abs_tol=1e-12):
        raise SelectionError(f"candidate {candidate['id']} seed {seed} target epsilon mismatch")
    if abs(spent - registry_target) > tolerance or spent > registry_target + tolerance:
        raise SelectionError(
            f"candidate {candidate['id']} seed {seed} epsilon {spent} is outside "
            f"{registry_target} +/- {tolerance}"
        )
    return {
        "seed": seed,
        "loss": loss,
        "accuracy": accuracy,
        "selection_metric": protocol["selection_metric"],
        "epsilon_spent": spent,
        "split_manifest_sha256": split_sha,
        "config_fingerprint": status.get("config_fingerprint"),
    }


def _summary(candidate: Mapping[str, Any], runs: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    ordered = sorted(runs, key=lambda item: int(item["seed"]))
    losses = [float(item["loss"]) for item in ordered]
    metrics = {str(item["selection_metric"]) for item in ordered}
    if len(metrics) != 1:
        raise SelectionError(
            f"candidate {candidate['id']} mixes selection metrics: {sorted(metrics)}"
        )
    selection_metric = next(iter(metrics))
    result = {
        "candidate_id": candidate["id"],
        "family": candidate["family"],
        "method": candidate["method"],
        "params": candidate["params"],
        "selection_metric": selection_metric,
        "seeds": [int(item["seed"]) for item in ordered],
        "loss_by_seed": {str(item["seed"]): float(item["loss"]) for item in ordered},
        "mean_validation_loss": math.fsum(losses) / len(losses),
    }
    if selection_metric == NUMERIC_EXACT_METRIC:
        accuracies = [float(item["accuracy"]) for item in ordered]
        result["accuracy_by_seed"] = {
            str(item["seed"]): float(item["accuracy"]) for item in ordered
        }
        result["mean_validation_accuracy"] = math.fsum(accuracies) / len(accuracies)
    return result


def _rank(summaries: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries = list(summaries)
    metrics = {str(item["selection_metric"]) for item in summaries}
    if len(metrics) != 1:
        raise SelectionError(f"cannot rank mixed selection metrics: {sorted(metrics)}")
    selection_metric = next(iter(metrics))
    if selection_metric == NUMERIC_EXACT_METRIC:
        ranked = sorted(
            summaries,
            key=lambda item: (
                -float(item["mean_validation_accuracy"]),
                float(item["mean_validation_loss"]),
                item["candidate_id"],
            ),
        )
    else:
        ranked = sorted(
            summaries,
            key=lambda item: (
                float(item["mean_validation_loss"]),
                item["candidate_id"],
            ),
        )
    return [{**item, "rank": index} for index, item in enumerate(ranked, start=1)]


def _require_common_split(runs: Iterable[Mapping[str, Any]]) -> str:
    splits = {str(run["split_manifest_sha256"]) for run in runs}
    if len(splits) != 1:
        raise SelectionError(f"all compared runs must share one split manifest SHA, got: {sorted(splits)}")
    return next(iter(splits))


def _selection_csv(payload: Mapping[str, Any]) -> str:
    buffer = io.StringIO(newline="")
    fieldnames = [
        "stage",
        "rank",
        "ranking_group",
        "group_rank",
        "candidate_id",
        "family",
        "method",
        "seeds",
        "selection_metric",
        "mean_validation_loss",
        "mean_validation_accuracy",
        "selected_role",
        "params_json",
        "split_manifest_sha256",
    ]
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    best_fixed = (payload.get("best_fixed") or {}).get("candidate_id")
    top_ids = {item["candidate_id"] for item in payload.get("top_slaclip", [])}
    top_fixed_ids = {
        item["candidate_id"] for item in payload.get("top_fixed", [])
    }
    selected = (payload.get("selected_slaclip") or {}).get("candidate_id")
    fixed_ranks = {
        item["candidate_id"]: item["rank"] for item in payload.get("fixed_ranking", [])
    }
    slaclip_ranks = {
        item["candidate_id"]: item["rank"] for item in payload.get("slaclip_ranking", [])
    }
    for item in payload["ranking"]:
        roles = []
        if item["candidate_id"] == best_fixed:
            roles.append("best_fixed")
        if item["candidate_id"] in top_fixed_ids:
            roles.append("top_fixed")
        if item["candidate_id"] in top_ids:
            roles.append("top_slaclip")
        if item["candidate_id"] == selected:
            roles.append("selected_slaclip")
        if item["family"] == "fixed":
            ranking_group = "fixed"
            group_rank = fixed_ranks[item["candidate_id"]]
        else:
            ranking_group = "slaclip"
            group_rank = slaclip_ranks[item["candidate_id"]]
        writer.writerow(
            {
                "stage": payload["stage"],
                "rank": item["rank"],
                "ranking_group": ranking_group,
                "group_rank": group_rank,
                "candidate_id": item["candidate_id"],
                "family": item["family"],
                "method": item["method"],
                "seeds": ",".join(str(seed) for seed in item["seeds"]),
                "selection_metric": item["selection_metric"],
                "mean_validation_loss": format(float(item["mean_validation_loss"]), ".17g"),
                "mean_validation_accuracy": (
                    format(float(item["mean_validation_accuracy"]), ".17g")
                    if "mean_validation_accuracy" in item
                    else ""
                ),
                "selected_role": "+".join(roles),
                "params_json": _canonical_json_bytes(item["params"]).decode("utf-8"),
                "split_manifest_sha256": payload["split_manifest_sha256"],
            }
        )
    return buffer.getvalue()


def _selected_env(
    selected: Mapping[str, Any],
    *,
    best_fixed: Mapping[str, Any],
    selection_sha256: str,
) -> str:
    variables: dict[str, Any] = {
        "PRISM_SELECTED_CANDIDATE_ID": selected["candidate_id"],
        "PRISM_SELECTED_METHOD": selected["method"],
        "PRISM_SELECTION_SHA256": selection_sha256,
        "PRISM_SELECTED_BEST_FIXED_CANDIDATE_ID": best_fixed["candidate_id"],
    }
    for key, value in sorted(selected["params"].items()):
        env_key = "PRISM_SELECTED_" + re.sub(r"[^A-Za-z0-9]", "_", key).upper()
        variables[env_key] = value
    for key, value in sorted(best_fixed["params"].items()):
        env_key = "PRISM_SELECTED_BEST_FIXED_" + re.sub(r"[^A-Za-z0-9]", "_", key).upper()
        variables[env_key] = value
    return "".join(f"{key}={shlex.quote(str(value))}\n" for key, value in sorted(variables.items()))


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_immutable_atomic(path: Path, content: str) -> None:
    """Create an immutable result atomically, accepting only byte-identical reruns."""
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = content.encode("utf-8")
    if path.exists():
        if path.read_bytes() != encoded:
            raise SelectionError(f"refusing to overwrite inconsistent selection artifact: {path}")
        return
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != encoded:
                raise SelectionError(f"concurrent inconsistent selection artifact exists: {path}")
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _resolve_output(campaign_root: Path, value: Path) -> Path:
    supplied = value if value.is_absolute() else campaign_root / value
    resolved_parent = supplied.parent.resolve()
    selection_root = (campaign_root / "selection").resolve()
    try:
        resolved_parent.relative_to(selection_root)
    except ValueError as exc:
        raise SelectionError(f"selection output must remain inside {selection_root}: {supplied}") from exc
    if supplied.suffix != ".json":
        raise SelectionError("--output must name a .json file")
    return resolved_parent / supplied.name


def select_candidates(
    *,
    campaign_root: Path,
    registry_path: Path,
    stage: str,
    output_path: Path,
    top_slaclip: int,
    stage1_selection_path: Path | None = None,
    top_fixed: int = 2,
) -> dict[str, Any]:
    campaign_root = campaign_root.resolve(strict=True)
    registry_path = _resolve_screen_input(
        campaign_root, str(registry_path), filename="candidate_registry.json", label="candidate registry"
    )
    registry = _read_json(registry_path, label="candidate registry")
    protocol = _validate_protocol(registry)
    candidates = _candidate_map(registry)
    registry_sha = _sha256_payload(registry)
    protocol_sha = _sha256_payload(protocol)
    stage = stage.casefold()
    if top_slaclip <= 0:
        raise SelectionError("--top-slaclip must be positive")
    if top_fixed <= 0:
        raise SelectionError("--top-fixed must be positive")

    if stage == "stage1":
        seed = int(protocol["stage1_seed"])
        run_records: list[dict[str, Any]] = []
        summaries = []
        for candidate in candidates.values():
            validated = _validate_run(campaign_root, candidate, seed, protocol)
            run_records.append(validated)
            summaries.append(_summary(candidate, [validated]))
        split_sha = _require_common_split(run_records)
        fixed = _rank(item for item in summaries if item["family"] == "fixed")
        slaclip = _rank(item for item in summaries if item["family"] == "slaclip")
        if len(slaclip) < top_slaclip:
            raise SelectionError(f"requested top {top_slaclip} SlaClip candidates but only {len(slaclip)} exist")
        if len(fixed) < top_fixed:
            raise SelectionError(
                f"requested top {top_fixed} fixed-C candidates but only {len(fixed)} exist"
            )
        combined = _rank(summaries)
        payload: dict[str, Any] = {
            "schema_version": SELECTION_SCHEMA_VERSION,
            "stage": "stage1",
            "ranking_rule": (
                "descending_validation_numeric_exact_accuracy_then_ascending_validation_loss_"
                "then_ascending_candidate_id"
                if protocol["selection_metric"] == NUMERIC_EXACT_METRIC
                else "ascending_validation_loss_then_ascending_candidate_id"
            ),
            "registry_sha256": registry_sha,
            "selection_protocol_sha256": protocol_sha,
            "selection_protocol": protocol,
            "split_manifest_sha256": split_sha,
            "ranking": combined,
            "fixed_ranking": fixed,
            "slaclip_ranking": slaclip,
            "best_fixed": fixed[0],
            "top_fixed": fixed[:top_fixed],
            "top_slaclip": slaclip[:top_slaclip],
        }
    elif stage == "stage2":
        selection_path = stage1_selection_path or output_path.parent / "stage1-selection.json"
        if not selection_path.is_absolute():
            selection_path = campaign_root / selection_path
        try:
            selection_path = selection_path.resolve(strict=True)
            selection_path.relative_to((campaign_root / "selection").resolve())
        except (OSError, ValueError) as exc:
            raise SelectionError("stage1 selection must exist inside the campaign selection directory") from exc
        stage1 = _read_json(selection_path, label="stage1 selection")
        if stage1.get("stage") != "stage1" or stage1.get("registry_sha256") != registry_sha:
            raise SelectionError("stage1 selection does not belong to the current candidate registry")
        top = stage1.get("top_slaclip")
        if not isinstance(top, list) or len(top) != top_slaclip:
            raise SelectionError("stage1 selection has an unexpected top_slaclip set")
        top_ids = [item.get("candidate_id") for item in top if isinstance(item, dict)]
        if len(top_ids) != top_slaclip or len(set(top_ids)) != top_slaclip:
            raise SelectionError("stage1 top_slaclip candidate identities are invalid")
        top_fixed_items = stage1.get("top_fixed")
        if not isinstance(top_fixed_items, list) or len(top_fixed_items) != top_fixed:
            raise SelectionError("stage1 selection has an unexpected top_fixed set")
        top_fixed_ids = [
            item.get("candidate_id")
            for item in top_fixed_items
            if isinstance(item, dict)
        ]
        if len(top_fixed_ids) != top_fixed or len(set(top_fixed_ids)) != top_fixed:
            raise SelectionError("stage1 top_fixed candidate identities are invalid")
        seeds = [int(seed) for seed in protocol["stage2_seeds"]]
        all_runs: list[dict[str, Any]] = []
        slaclip_summaries = []
        for candidate_id in top_ids:
            candidate = candidates.get(candidate_id)
            if candidate is None or candidate["family"] != "slaclip":
                raise SelectionError(f"stage1 selected unknown/non-SlaClip candidate: {candidate_id}")
            validated_runs = [_validate_run(campaign_root, candidate, seed, protocol) for seed in seeds]
            all_runs.extend(validated_runs)
            slaclip_summaries.append(_summary(candidate, validated_runs))

        canonical_fixed_ids = [
            candidate_id
            for candidate_id, candidate in candidates.items()
            if candidate["family"] == "fixed"
            and _values_match(candidate["params"].get("dp_max_grad_norm"), 1.0)
        ]
        if len(canonical_fixed_ids) != 1:
            raise SelectionError(
                "registry must identify exactly one canonical fixed-C=1 candidate using params.dp_max_grad_norm=1.0"
            )
        canonical_fixed_id = canonical_fixed_ids[0]
        # Re-rank the top one-seed fixed candidates using every preregistered
        # selection seed.  C=1 remains a paper anchor even when it did not
        # survive the one-seed fixed-candidate filter.
        fixed_ids = list(dict.fromkeys([*top_fixed_ids, canonical_fixed_id]))
        fixed_summaries = []
        for candidate_id in fixed_ids:
            candidate = candidates.get(candidate_id)
            if candidate is None or candidate["family"] != "fixed":
                raise SelectionError(f"stage1 selected unknown/non-fixed candidate: {candidate_id}")
            validated_runs = [_validate_run(campaign_root, candidate, seed, protocol) for seed in seeds]
            all_runs.extend(validated_runs)
            fixed_summaries.append(_summary(candidate, validated_runs))
        split_sha = _require_common_split(all_runs)
        if split_sha != stage1.get("split_manifest_sha256"):
            raise SelectionError("stage2 runs do not use the stage1 split manifest")
        slaclip_ranked = _rank(slaclip_summaries)
        fixed_ranked = _rank(fixed_summaries)
        combined_ranked = _rank([*fixed_summaries, *slaclip_summaries])
        payload = {
            "schema_version": SELECTION_SCHEMA_VERSION,
            "stage": "stage2",
            "ranking_rule": (
                "descending_arithmetic_mean_validation_numeric_exact_accuracy_then_"
                "ascending_arithmetic_mean_validation_loss_then_ascending_candidate_id"
                if protocol["selection_metric"] == NUMERIC_EXACT_METRIC
                else "ascending_arithmetic_mean_validation_loss_then_ascending_candidate_id"
            ),
            "required_seeds": seeds,
            "registry_sha256": registry_sha,
            "selection_protocol_sha256": protocol_sha,
            "selection_protocol": protocol,
            "stage1_selection_sha256": _sha256_payload(stage1),
            "split_manifest_sha256": split_sha,
            "ranking": combined_ranked,
            "fixed_ranking": fixed_ranked,
            "slaclip_ranking": slaclip_ranked,
            "canonical_fixed_candidate_id": canonical_fixed_id,
            "stage1_best_fixed_candidate_id": stage1["best_fixed"]["candidate_id"],
            "stage1_top_fixed_candidate_ids": top_fixed_ids,
            "best_fixed": fixed_ranked[0],
            "top_fixed": fixed_ranked,
            "top_slaclip": slaclip_ranked,
            "selected_slaclip": slaclip_ranked[0],
        }
    else:
        raise SelectionError("--stage must be stage1 or stage2")

    json_text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    csv_text = _selection_csv(payload)
    _write_immutable_atomic(output_path, json_text)
    _write_immutable_atomic(output_path.with_suffix(".csv"), csv_text)
    if stage == "stage2":
        selection_sha = hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()
        _write_immutable_atomic(
            output_path.parent / "selected.env",
            _selected_env(
                payload["selected_slaclip"],
                best_fixed=payload["best_fixed"],
                selection_sha256=selection_sha,
            ),
        )
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--registry", type=Path, default=Path("screen/candidate_registry.json"))
    parser.add_argument("--stage", type=str.lower, choices=("stage1", "stage2"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-slaclip", type=int, default=2)
    parser.add_argument("--top-fixed", type=int, default=2)
    parser.add_argument("--stage1-selection", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        campaign_root = args.campaign_root.resolve(strict=True)
        output = _resolve_output(campaign_root, args.output)
        registry = args.registry if args.registry.is_absolute() else campaign_root / args.registry
        payload = select_candidates(
            campaign_root=campaign_root,
            registry_path=registry,
            stage=args.stage,
            output_path=output,
            top_slaclip=args.top_slaclip,
            top_fixed=args.top_fixed,
            stage1_selection_path=args.stage1_selection,
        )
    except SelectionError as exc:
        raise SystemExit(f"selection refused: {exc}") from exc
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
