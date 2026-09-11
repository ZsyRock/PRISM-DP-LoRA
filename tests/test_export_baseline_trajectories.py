from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


SPEC = importlib.util.spec_from_file_location(
    "export_baseline_trajectories",
    Path(__file__).resolve().parents[1] / "scripts/export_baseline_trajectories.py",
)
assert SPEC and SPEC.loader
exporter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(exporter)


@pytest.fixture
def source(tmp_path):
    campaigns = tmp_path / "campaigns"
    root = campaigns / "campaign1/runs/fixed/seed-42"
    (root / "adapter").mkdir(parents=True)
    (root / "results/research_raw").mkdir(parents=True)
    identity = dict(dataset="glue8", base_model="model/4b", method="baseline",
                    privacy="dp", run_id="run1", config_fingerprint="f" * 64)
    config = dict(identity, seed=42, lora_r=16, dp_epsilon=6.0,
                  dp_max_grad_norm=1.0, total_update_steps=3)
    status = dict(identity, config=config, state="completed", update_steps=3)
    status_path = root / "adapter/run_status.json"
    status_path.write_text(json.dumps(status))
    rows = [dict(identity, step=i, NON_PRIVATE_TELEMETRY=True,
                 raw_clip_fraction=.25 * i, dp_clip_threshold=1.0,
                 dp_next_clip_threshold=1.0, loss_mean=2.0 / i,
                 raw_unclipped_signal_norm=2.0, raw_clipped_signal_norm=1.0,
                 raw_global_norm_quantiles={"0.5": 4.0}) for i in (1, 2, 3)]
    raw_path = root / "results/research_raw/NON_PRIVATE_train_log.jsonl"
    raw_path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    item = dict(setting_id="test-setting", dataset="glue8", model="model/4b",
                epsilon=6, rank=16, C=1, steps=3, raw_records=3,
                campaign_id="campaign1", relative_run_root="runs/fixed/seed-42")
    inventory = tmp_path / "inventory.csv"
    def save_inventory():
        with inventory.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(item))
            writer.writeheader()
            writer.writerow(item)
    save_inventory()
    return campaigns, inventory, raw_path, rows, item, save_inventory, status_path


def rewrite(path, rows):
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")


def test_export_complete_and_honest_missing_values(source, tmp_path):
    campaigns, inventory, _, _, _, _, _ = source
    rows = exporter.export_rows(campaigns, inventory)
    assert [r["step"] for r in rows] == [1, 2, 3]
    assert rows[0]["NON_PRIVATE_TELEMETRY"] is True
    assert rows[0]["signal_retention_norm_ratio"] == .5
    assert rows[0]["raw_global_norm_q50"] == 4
    assert rows[0]["raw_global_norm_q25"] == ""
    assert rows[0]["raw_realized_noise_norm"] == ""
    target = tmp_path / "exports/plot.csv"
    assert exporter.write_export(rows, target)
    assert not exporter.write_export(rows, target)
    rows[0]["loss_mean"] = 100
    with pytest.raises(ValueError, match="overwrite"):
        exporter.write_export(rows, target)


@pytest.mark.parametrize("mode", ["missing", "duplicate", "out_of_order"])
def test_reject_incomplete_steps(source, mode):
    campaigns, inventory, raw, rows, *_ = source
    if mode == "missing":
        rows.pop(1)
    elif mode == "duplicate":
        rows[1]["step"] = 1
    else:
        rows.reverse()
    rewrite(raw, rows)
    with pytest.raises(ValueError, match="incomplete|contiguous"):
        exporter.export_rows(campaigns, inventory)


@pytest.mark.parametrize("value", [-.01, 1.01, float("nan"), float("inf"), True, None])
def test_reject_invalid_fraction(source, value):
    campaigns, inventory, raw, rows, *_ = source
    rows[0]["raw_clip_fraction"] = value
    rewrite(raw, rows)
    with pytest.raises(ValueError):
        exporter.export_rows(campaigns, inventory)


@pytest.mark.parametrize("field,value", [("campaign_id", "../outside"),
                                        ("relative_run_root", "../../outside"),
                                        ("relative_run_root", "/tmp/outside")])
def test_reject_path_traversal(source, field, value):
    campaigns, inventory, _, _, item, save, _ = source
    item[field] = value
    save()
    with pytest.raises(ValueError, match="unsafe"):
        exporter.export_rows(campaigns, inventory)


def test_reject_symlink_escape(source, tmp_path):
    campaigns, inventory, _, _, item, save, _ = source
    outside = tmp_path / "outside"
    outside.mkdir()
    (campaigns / "escape").symlink_to(outside, target_is_directory=True)
    item["campaign_id"] = "escape"
    save()
    with pytest.raises(ValueError, match="escapes"):
        exporter.export_rows(campaigns, inventory)


@pytest.mark.parametrize("field,value", [("run_id", "wrong"),
                                        ("config_fingerprint", "wrong"),
                                        ("dataset", "wrong"),
                                        ("dp_clip_threshold", 2),
                                        ("dp_next_clip_threshold", 2),
                                        ("NON_PRIVATE_TELEMETRY", False)])
def test_reject_wrong_identity_or_adaptive_source(source, field, value):
    campaigns, inventory, raw, rows, *_ = source
    rows[0][field] = value
    rewrite(raw, rows)
    with pytest.raises(ValueError):
        exporter.export_rows(campaigns, inventory)


def test_source_hash_pins(source, tmp_path):
    campaigns, inventory, raw, _, item, save, _ = source
    pin = hashlib.sha256(raw.read_bytes()).hexdigest()
    item["raw_telemetry_sha256"] = pin
    save()
    landscape = tmp_path / "landscape.json"
    landscape.write_text(json.dumps({"landscape": [{"setting_id": "test-setting",
        "campaign_root": "/old/account/campaign1", "raw_telemetry_sha256": pin}]}))
    assert len(exporter.export_rows(campaigns, inventory, landscape)) == 3
    item["raw_telemetry_sha256"] = "0" * 64
    save()
    with pytest.raises(ValueError, match="SHA256"):
        exporter.export_rows(campaigns, inventory, landscape)


def test_reject_unfinished_status(source):
    campaigns, inventory, _, _, _, _, path = source
    status = json.loads(path.read_text())
    status["state"] = "running"
    path.write_text(json.dumps(status))
    with pytest.raises(ValueError, match="state"):
        exporter.export_rows(campaigns, inventory)
