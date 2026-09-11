"""Export public-benchmark, NON_PRIVATE baseline telemetry without changing sources.

The output is research data, not a DP-sanitized release. Quantities unavailable
in an older telemetry schema remain blank. Existing differing outputs are never
overwritten. Raw-file hashes are checked against inventory/source-landscape pins
when supplied, and always included in the exported rows.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
SCALARS = (
    "loss_mean", "raw_clip_coefficient_mean", "raw_clip_coefficient_min",
    "raw_global_norm_mean", "raw_global_norm_std", "raw_global_norm_max",
    "raw_clipping_bias_norm", "raw_realized_noise_norm", "raw_unclipped_signal_norm",
    "raw_clipped_signal_norm", "raw_signal_to_noise_ratio",
    "raw_unclipped_clipped_cosine", "raw_clipped_noisy_cosine", "eps_spent",
    "raw_realized_batch_size",
)
QUANTILES = ("0.1", "0.25", "0.5", "0.75", "0.9", "0.95", "0.99")
FIELDS = (
    "NON_PRIVATE_TELEMETRY", "setting_id", "dataset", "model", "epsilon",
    "rank", "seed", "step", "C", "raw_clip_fraction", *SCALARS,
    *(f"raw_global_norm_q{int(float(q) * 100)}" for q in QUANTILES),
    "signal_retention_norm_ratio", "raw_telemetry_sha256",
)


def _number(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label}: boolean is not a numeric measurement")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label}: expected finite number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{label}: non-finite number")
    return result


def _inside(parent: Path, value: str, label: str) -> Path:
    relative = Path(value)
    if not value or relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{label}: unsafe relative path")
    candidate = (parent / relative).resolve()
    if not candidate.is_relative_to(parent.resolve()):
        raise ValueError(f"{label}: path escapes its source root")
    return candidate


def _same(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise ValueError(f"{label}: identity/configuration mismatch")


def export_rows(campaigns_root: Path, inventory: Path,
                source_landscape: Path | None = None) -> list[dict[str, Any]]:
    """Validate complete fixed runs and return ordered, scalar-only plot rows."""
    campaigns_root = campaigns_root.resolve(strict=True)
    pins = {}
    if source_landscape is not None:
        entries = json.loads(source_landscape.read_text(encoding="utf-8"))["landscape"]
        for source in entries:
            key = (source["setting_id"], Path(source["campaign_root"]).name)
            if key in pins:
                raise ValueError("duplicate source-landscape identity")
            pins[key] = source["raw_telemetry_sha256"]
    with inventory.open(newline="", encoding="utf-8") as stream:
        entries = list(csv.DictReader(stream))
    if not entries:
        raise ValueError("inventory is empty")
    output, seen = [], set()
    for item in entries:
        setting = item["setting_id"]
        if setting in seen:
            raise ValueError(f"{setting}: duplicate inventory setting")
        seen.add(setting)
        campaign = _inside(campaigns_root, item["campaign_id"], "campaign_id")
        if len(Path(item["campaign_id"]).parts) != 1:
            raise ValueError("campaign_id must be a single directory name")
        root = _inside(campaign, item["relative_run_root"], "relative_run_root")
        status_path = _inside(root, "adapter/run_status.json", "status")
        raw_path = _inside(root, "results/research_raw/NON_PRIVATE_train_log.jsonl", "raw")
        status = json.loads(status_path.read_text(encoding="utf-8"))
        config = status["config"]
        steps, rank = int(item["steps"]), int(item["rank"])
        epsilon, threshold = _number(item["epsilon"], "epsilon"), _number(item["C"], "C")
        if steps <= 0 or rank <= 0 or threshold <= 0 or epsilon <= 0:
            raise ValueError(f"{setting}: invalid expected configuration")
        _same(status.get("state"), "completed", f"{setting}:state")
        _same(status.get("update_steps"), steps, f"{setting}:steps")
        for key, expected in (("dataset", item["dataset"]), ("base_model", item["model"]),
                              ("method", "baseline"), ("privacy", "dp")):
            _same(status.get(key), expected, f"{setting}:status:{key}")
            _same(config.get(key), expected, f"{setting}:config:{key}")
        for key, expected in (("lora_r", rank), ("dp_epsilon", epsilon),
                              ("dp_max_grad_norm", threshold), ("total_update_steps", steps)):
            _same(config.get(key), expected, f"{setting}:{key}")
        seed = config.get("seed")
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise ValueError(f"{setting}: invalid seed")
        if item.get("seed"):
            _same(seed, int(item["seed"]), f"{setting}:seed")
        if item.get("raw_records"):
            _same(int(item["raw_records"]), steps, f"{setting}:inventory record count")
        identity = {key: status.get(key) for key in ("run_id", "config_fingerprint")}
        if not all(isinstance(value, str) and value for value in identity.values()):
            raise ValueError(f"{setting}: missing source identities")
        for key, value in identity.items():
            if config.get(key) is not None:
                _same(config[key], value, f"{setting}:config:{key}")
        content = raw_path.read_bytes()
        digest = hashlib.sha256(content).hexdigest()
        expected_digest = pins.get((setting, item["campaign_id"]))
        if source_landscape is not None and expected_digest is None:
            raise ValueError(f"{setting}: absent from supplied source landscape")
        for pinned in (item.get("raw_telemetry_sha256"), expected_digest):
            if pinned:
                _same(digest, pinned, f"{setting}:raw SHA256")
        raw_rows = [json.loads(line) for line in content.splitlines() if line.strip()]
        if len(raw_rows) != steps:
            raise ValueError(f"{setting}: incomplete raw trajectory")
        for expected_step, raw in enumerate(raw_rows, start=1):
            _same(raw.get("step"), expected_step, f"{setting}:contiguous step")
            _same(raw.get("NON_PRIVATE_TELEMETRY"), True, f"{setting}:raw disclosure label")
            for key, value in {**identity, "dataset": item["dataset"],
                               "base_model": item["model"], "method": "baseline",
                               "privacy": "dp"}.items():
                _same(raw.get(key), value, f"{setting}:raw:{key}")
            if "seed" in raw:
                _same(raw["seed"], seed, f"{setting}:raw:seed")
            clipping = _number(raw.get("raw_clip_fraction"), "raw_clip_fraction")
            if not 0 <= clipping <= 1:
                raise ValueError(f"{setting}: invalid clipping fraction")
            for key in ("dp_clip_threshold", "dp_next_clip_threshold"):
                if key == "dp_clip_threshold" or key in raw:
                    _same(_number(raw.get(key), key), threshold, f"{setting}:fixed C")
            row = dict(NON_PRIVATE_TELEMETRY=True, setting_id=setting,
                       dataset=item["dataset"], model=item["model"], epsilon=epsilon,
                       rank=rank, seed=seed, step=expected_step, C=threshold,
                       raw_clip_fraction=clipping, raw_telemetry_sha256=digest)
            for key in SCALARS:
                row[key] = "" if raw.get(key) is None else _number(raw[key], key)
            quantiles = raw.get("raw_global_norm_quantiles") or {}
            for q in QUANTILES:
                value = quantiles.get(q)
                row[f"raw_global_norm_q{int(float(q) * 100)}"] = (
                    "" if value is None else _number(value, f"norm quantile {q}"))
            before, after = row["raw_unclipped_signal_norm"], row["raw_clipped_signal_norm"]
            # Norm ratio, not retained sample fraction: cancellation can make it >1.
            row["signal_retention_norm_ratio"] = (
                after / before if before != "" and before > 0 and after != "" else "")
            output.append(row)
    return output


def write_export(rows: list[dict[str, Any]], output: Path) -> bool:
    """Return True for a new file, False for an identical existing export."""
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    content = stream.getvalue().encode("utf-8")
    if output.exists():
        if output.read_bytes() != content:
            raise ValueError(f"refusing to overwrite differing export: {output}")
        return False
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as target:
        target.write(content)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    configured_root = os.environ.get("PRISM_CAMPAIGNS_ROOT")
    if configured_root is None and os.environ.get("PRISM_RUN_ROOT"):
        configured_root = str(Path(os.environ["PRISM_RUN_ROOT"]) / "campaigns")
    parser.add_argument("--campaigns-root", type=Path, default=configured_root,
                        help="Explicit root, PRISM_CAMPAIGNS_ROOT, or PRISM_RUN_ROOT/campaigns")
    parser.add_argument("--inventory", type=Path,
                        default=REPO / "docs/baseline_audit_2026-09-07.csv")
    parser.add_argument("--source-landscape", type=Path,
                        help="Optional existing landscape containing raw SHA256 pins")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--acknowledge-public-research-telemetry", action="store_true",
                        help="Acknowledge research-only public-benchmark raw data, not a DP release")
    args = parser.parse_args()
    if args.campaigns_root is None:
        parser.error("set --campaigns-root or PRISM_CAMPAIGNS_ROOT/PRISM_RUN_ROOT")
    if not args.acknowledge_public_research_telemetry:
        parser.error("raw export requires --acknowledge-public-research-telemetry")
    rows = export_rows(args.campaigns_root, args.inventory, args.source_landscape)
    created = write_export(rows, args.output)
    print(json.dumps({"rows": len(rows), "output": str(args.output.resolve()),
                      "bytes": args.output.stat().st_size, "created": created,
                      "NON_PRIVATE_TELEMETRY": True}, sort_keys=True))


if __name__ == "__main__":
    main()
