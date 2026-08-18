#!/usr/bin/env python3
"""Build and summarize the preregistered PRISM paper-coverage screen.

This is a breadth screen, not a multi-seed confirmatory experiment.  It covers
the highest-value settings from the PRISM paper that were not already tested
systematically with Full SlaClip: GLUE8 at both paper privacy budgets,
Math-10K on Gemma-2-9B, and the rank-8/rank-32 Math-10K ablations.  Every
setting compares the paper fixed C=1 control with two conservative Full
SlaClip targets in one immutable, two-lane Slurm allocation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1
SEED = 42
MODEL_4B = "google/gemma-3-4b-pt"
MODEL_9B = "google/gemma-2-9b"
FULL_SHA_LENGTH = 40

SETTINGS = (
    {
        "id": "glue8-4b-eps6-r16",
        "lane": 0,
        "paper_reference": "Table 2 / Table 7",
        "dataset": "glue8",
        "model_slug": "gemma-3-4b-pt",
        "model_id": MODEL_4B,
        "epsilon": 6.0,
        "lora_r": 16,
        "steps": 500,
        "learning_rate": 0.0002,
        "cutoff_len": 384,
        "train_on_inputs": False,
    },
    {
        "id": "glue8-4b-eps3-r16",
        "lane": 1,
        "paper_reference": "Table 2 / Table 7",
        "dataset": "glue8",
        "model_slug": "gemma-3-4b-pt",
        "model_id": MODEL_4B,
        "epsilon": 3.0,
        "lora_r": 16,
        "steps": 500,
        "learning_rate": 0.0002,
        "cutoff_len": 384,
        "train_on_inputs": False,
    },
    {
        "id": "math10k-9b-eps6-r16",
        "lane": 0,
        "paper_reference": "Table 3",
        "dataset": "math10k",
        "model_slug": "gemma-2-9b",
        "model_id": MODEL_9B,
        "epsilon": 6.0,
        "lora_r": 16,
        "steps": 300,
        "learning_rate": 0.0003,
        "cutoff_len": 256,
        "train_on_inputs": True,
    },
    {
        "id": "math10k-4b-eps6-r8",
        "lane": 1,
        "paper_reference": "Table 4",
        "dataset": "math10k",
        "model_slug": "gemma-3-4b-pt",
        "model_id": MODEL_4B,
        "epsilon": 6.0,
        "lora_r": 8,
        "steps": 300,
        "learning_rate": 0.0003,
        "cutoff_len": 256,
        "train_on_inputs": True,
    },
    {
        "id": "math10k-4b-eps6-r32",
        "lane": 1,
        "paper_reference": "Table 4",
        "dataset": "math10k",
        "model_slug": "gemma-3-4b-pt",
        "model_id": MODEL_4B,
        "epsilon": 6.0,
        "lora_r": 32,
        "steps": 300,
        "learning_rate": 0.0003,
        "cutoff_len": 256,
        "train_on_inputs": True,
    },
)

CANDIDATES = (
    {
        "id": "fixed-c1",
        "method": "baseline",
        "initial_c": 1.0,
        "rho": None,
        "eta": None,
    },
    {
        "id": "full-sla-rho090",
        "method": "slaclip",
        "initial_c": 1.0,
        "rho": 0.90,
        "eta": 0.05,
    },
    {
        "id": "full-sla-rho098",
        "method": "slaclip",
        "initial_c": 1.0,
        "rho": 0.98,
        "eta": 0.05,
    },
)


class CampaignError(RuntimeError):
    """The campaign plan or output is incomplete or inconsistent."""


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _write_immutable(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != data:
            raise CampaignError(f"refusing to overwrite immutable artifact: {path}")
        return
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        os.chmod(path, 0o600)
    except FileExistsError:
        if path.read_bytes() != data:
            raise CampaignError(f"concurrent inconsistent writer: {path}")
    finally:
        temporary.unlink(missing_ok=True)


def _with_sha(path: Path, data: bytes) -> None:
    _write_immutable(path, data)
    digest = hashlib.sha256(data).hexdigest()
    _write_immutable(path.with_name(path.name + ".sha256"), f"{digest}  {path.name}\n".encode())


def build_manifest(code_sha: str, model_4b_revision: str, model_9b_revision: str) -> dict[str, Any]:
    for label, value in (
        ("code_sha", code_sha),
        ("model_4b_revision", model_4b_revision),
        ("model_9b_revision", model_9b_revision),
    ):
        if len(value) != FULL_SHA_LENGTH or any(ch not in "0123456789abcdef" for ch in value):
            raise CampaignError(f"{label} must be a full lowercase commit SHA")
    revisions = {MODEL_4B: model_4b_revision, MODEL_9B: model_9b_revision}
    arms = []
    for setting in SETTINGS:
        for candidate in CANDIDATES:
            arm_id = f"{setting['id']}--{candidate['id']}--seed{SEED}"
            arms.append(
                {
                    **setting,
                    **candidate,
                    "setting_id": setting["id"],
                    "candidate_id": candidate["id"],
                    "arm_id": arm_id,
                    "seed": SEED,
                    "model_revision": revisions[setting["model_id"]],
                    "relative_root": f"runs/{setting['id']}/{candidate['id']}/seed-{SEED}",
                }
            )
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol": "prism_paper_coverage_full_slaclip_breadth_screen_v1",
        "inference_class": "single_seed_exploratory_breadth_screen_requires_fresh_seed_confirmation",
        "code_sha": code_sha,
        "seed": SEED,
        "privacy": {"delta": 1e-5, "accountant": "prv", "secure_mode": False},
        "full_slaclip": {
            "K": 15,
            "C_min": 0.1,
            "C_max": 15.0,
            "target_semantics": "p_star_t=rho*(1-z_t); rho is conditional on residual non-small mass",
        },
        "selection_warning": (
            "Task-test metrics are descriptive only. Any promising setting must be "
            "repeated on fresh seeds with a locked tuned-fixed comparator."
        ),
        "arms": arms,
    }


PLAN_FIELDS = (
    "lane", "arm_id", "setting_id", "dataset", "model_slug", "model_id",
    "model_revision", "epsilon", "lora_r", "method", "initial_c", "rho",
    "eta", "steps", "learning_rate", "cutoff_len", "train_on_inputs",
    "seed", "relative_root",
)


def _plan_bytes(manifest: dict[str, Any], lane: int) -> bytes:
    rows = []
    for arm in manifest["arms"]:
        if arm["lane"] != lane:
            continue
        values = {
            "lane": lane,
            "arm_id": arm["arm_id"],
            "setting_id": arm["setting_id"],
            "dataset": arm["dataset"],
            "model_slug": arm["model_slug"],
            "model_id": arm["model_id"],
            "model_revision": arm["model_revision"],
            "epsilon": arm["epsilon"],
            "lora_r": arm["lora_r"],
            "method": arm["method"],
            "initial_c": arm["initial_c"],
            "rho": "NA" if arm["rho"] is None else arm["rho"],
            "eta": "NA" if arm["eta"] is None else arm["eta"],
            "steps": arm["steps"],
            "learning_rate": arm["learning_rate"],
            "cutoff_len": arm["cutoff_len"],
            "train_on_inputs": str(arm["train_on_inputs"]).lower(),
            "seed": arm["seed"],
            "relative_root": arm["relative_root"],
        }
        rows.append("|".join(str(values[field]) for field in PLAN_FIELDS))
    return ("\n".join(rows) + "\n").encode()


def prepare(root: Path, code_sha: str, model_4b_revision: str, model_9b_revision: str) -> None:
    manifest = build_manifest(code_sha, model_4b_revision, model_9b_revision)
    _with_sha(root / "plans" / "manifest.json", _json_bytes(manifest))
    _with_sha(root / "plans" / "lane-0.tsv", _plan_bytes(manifest, 0))
    _with_sha(root / "plans" / "lane-1.tsv", _plan_bytes(manifest, 1))
    print(f"prepared_arms={len(manifest['arms'])} lane0=6 lane1=9")


def _finite(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise CampaignError(f"{label} is not numeric") from exc
    if not math.isfinite(result):
        raise CampaignError(f"{label} is not finite")
    return result


def _task_average(summary_path: Path) -> float:
    try:
        rows = list(csv.DictReader(summary_path.open(encoding="utf-8", newline="")))
    except OSError as exc:
        raise CampaignError(f"cannot read evaluation summary: {summary_path}") from exc
    if len(rows) != 1 or "Average" not in rows[0]:
        raise CampaignError(f"invalid evaluation summary: {summary_path}")
    return _finite(rows[0]["Average"], f"{summary_path}:Average")


def analyze(root: Path) -> None:
    manifest_path = root / "plans" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    results = []
    for arm in manifest.get("arms", []):
        arm_root = root / arm["relative_root"]
        status_path = arm_root / "adapter" / "run_status.json"
        telemetry_path = arm_root / "results" / "research_raw" / "telemetry_summary.json"
        summary_path = arm_root / "results" / "summary.csv"
        try:
            status = json.loads(status_path.read_text(encoding="utf-8"))
            telemetry = json.loads(telemetry_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise CampaignError(f"incomplete arm {arm['arm_id']}: {exc}") from exc
        if status.get("state") != "completed" or status.get("config", {}).get("implementation_git_sha") != manifest["code_sha"]:
            raise CampaignError(f"arm is not completed at the locked SHA: {arm['arm_id']}")
        numeric = telemetry.get("numeric_metrics", {})
        metric = lambda name, key="mean": _finite(numeric.get(name, {}).get(key), f"{arm['arm_id']}:{name}.{key}")
        results.append(
            {
                "setting_id": arm["setting_id"],
                "paper_reference": arm["paper_reference"],
                "dataset": arm["dataset"],
                "model": arm["model_id"],
                "epsilon": arm["epsilon"],
                "lora_r": arm["lora_r"],
                "candidate": arm["candidate_id"],
                "method": arm["method"],
                "rho": arm["rho"],
                "task_average": _task_average(summary_path),
                "loss_last": metric("loss_mean", "last"),
                "clip_fraction_mean": metric("raw_clip_fraction"),
                "clip_fraction_last": metric("raw_clip_fraction", "last"),
                "clip_threshold_mean": metric("dp_clip_threshold"),
                "clip_threshold_last": metric("dp_clip_threshold", "last"),
                "signal_to_noise_mean": metric("raw_signal_to_noise_ratio"),
                "bias_to_noise_mean": metric("raw_clipping_bias_to_noise_ratio"),
            }
        )
    out = root / "artifacts"
    out.mkdir(parents=True, exist_ok=True)
    fields = list(results[0])
    buffer = []
    buffer.append(",".join(fields))
    for row in results:
        buffer.append(",".join("" if row[k] is None else str(row[k]) for k in fields))
    _with_sha(out / "paper_coverage_summary.csv", ("\n".join(buffer) + "\n").encode())
    _with_sha(out / "paper_coverage_summary.json", _json_bytes({"schema_version": 1, "rows": results}))
    print(f"analyzed_arms={len(results)}")


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    sub = value.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--campaign-root", required=True, type=Path)
    prep.add_argument("--code-sha", required=True)
    prep.add_argument("--model-4b-revision", required=True)
    prep.add_argument("--model-9b-revision", required=True)
    report = sub.add_parser("analyze")
    report.add_argument("--campaign-root", required=True, type=Path)
    return value


def main() -> None:
    args = parser().parse_args()
    if args.command == "prepare":
        prepare(args.campaign_root, args.code_sha, args.model_4b_revision, args.model_9b_revision)
    else:
        analyze(args.campaign_root)


if __name__ == "__main__":
    main()
