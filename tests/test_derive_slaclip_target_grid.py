from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import stat
from pathlib import Path

import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "derive_slaclip_target_grid.py"
)
SPEC = importlib.util.spec_from_file_location("derive_slaclip_target_grid", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
grid = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(grid)


def _payload_sha(payload) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _split(discriminator: str = "shared") -> dict:
    payload = {
        "schema_version": 2,
        "protocol_stage": "selection",
        "validation_data_is_public": True,
        "validation_rows": 100,
        "discriminator": discriminator,
    }
    payload["manifest_sha256"] = _payload_sha(payload)
    return payload


def _raw_clip(c_value: float, step: int) -> float:
    if c_value < 1.0:
        return 1.0
    if math.isclose(c_value, 1.0):
        return 0.995
    if math.isclose(c_value, 1.5):
        return 0.62 + 0.16 * (step - 1) / 299.0
    return max(0.52, 0.76 - 0.02 * c_value)


def _build_fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path, dict]:
    campaign = tmp_path / "campaign"
    screen = campaign / "screen"
    screen.mkdir(parents=True)
    shared_split = _split()
    common_config = {
        "dataset": "math10k",
        "privacy": "dp",
        "base_model": "google/gemma-3-4b-pt",
        "model_revision": "a" * 40,
        "total_update_steps": 300,
        "batch_size": 64,
        "micro_batch_size": 4,
        "val_set_size": 100,
        "protocol_stage": "selection",
        "validation_data_is_public": True,
        "telemetry_mode": "research_raw",
        "allow_non_private_telemetry": True,
    }
    protocol = {
        "name": "fixed_scan_dynamic_rho_fixture_v1",
        "protocol_stage": "selection",
        "validation_data_is_public": True,
        "selection_metric": grid.selector.NUMERIC_EXACT_METRIC,
        "loss_definition": grid.selector.EXPECTED_LOSS_DEFINITION,
        "required_update_steps": 300,
        "target_epsilon": 6.0,
        "epsilon_tolerance": 0.02,
        "stage1_seed": 42,
        "stage2_seeds": [42, 43, 44],
        "common_config": common_config,
    }
    accuracies = {
        0.1: 0.70,
        0.5: 0.95,
        1.0: 0.90,
        1.5: 0.85,
        2.0: 0.80,
        3.0: 0.75,
        5.0: 0.65,
        15.0: 0.55,
    }
    candidates = []
    for index, c_value in enumerate(grid.FIXED_C_VALUES):
        candidate_id = f"fixed-c{grid._slug(c_value)}"
        arm = screen / "runs" / candidate_id / "seed-42"
        status_path = arm / "adapter" / "run_status.json"
        validation = arm / "results" / "validation"
        metrics_path = validation / "validation_metrics.json"
        split_path = validation / "split_manifest.json"
        raw_path = arm / "results" / "research_raw" / grid.RAW_LOG_NAME
        accuracy = accuracies[c_value]
        correct = round(accuracy * shared_split["validation_rows"])
        loss = 1.0 + index / 10.0
        config_fingerprint = f"fixture-{candidate_id}-42"
        config = {
            **common_config,
            "seed": 42,
            "method": "baseline",
            "dp_max_grad_norm": c_value,
        }
        metrics = {
            "manifest_sha256": shared_split["manifest_sha256"],
            "protocol_stage": "selection",
            "validation_data_is_public": True,
            "NON_PRIVATE_SELECTION_METRIC": True,
            "selection_metric": grid.selector.NUMERIC_EXACT_METRIC,
            "loss_definition": grid.selector.EXPECTED_LOSS_DEFINITION,
            "records": shared_split["validation_rows"],
            "loss_mean": loss,
            "numeric_exact_accuracy": accuracy,
            "numeric_exact_correct": correct,
            "numeric_parse_failures": 0,
        }
        status = {
            "state": "completed",
            "privacy": "dp",
            "method": "baseline",
            "update_steps": 300,
            "config": config,
            "config_fingerprint": config_fingerprint,
            "data_split": shared_split,
            "validation": metrics,
            "privacy_accounting": {
                "target_epsilon": 6.0,
                "epsilon_spent": 5.999,
                "completed_update_steps": 300,
            },
        }
        _write_json(status_path, status)
        _write_json(metrics_path, metrics)
        _write_json(split_path, shared_split)
        raw_path.parent.mkdir(parents=True)
        raw_path.write_text(
            "".join(
                json.dumps(
                    {
                        "NON_PRIVATE_TELEMETRY": True,
                        "config_fingerprint": config_fingerprint,
                        "step": step,
                        "raw_clip_fraction": _raw_clip(c_value, step),
                        "raw_reference_small_gradient_proxy": 0.02,
                        "raw_reference_remaining_mass_proxy": 0.98,
                        "raw_reference_conditional_clip_fraction_valid": True,
                    },
                    sort_keys=True,
                )
                + "\n"
                for step in range(1, 301)
            ),
            encoding="utf-8",
        )
        runs = {}
        for seed in (42, 43, 44):
            root = Path("screen") / "runs" / candidate_id / f"seed-{seed}"
            runs[str(seed)] = {
                "run_status": str(root / "adapter" / "run_status.json"),
                "validation_metrics": str(
                    root / "results" / "validation" / "validation_metrics.json"
                ),
                "split_manifest": str(
                    root / "results" / "validation" / "split_manifest.json"
                ),
            }
        candidates.append(
            {
                "id": candidate_id,
                "family": "fixed",
                "method": "baseline",
                "params": {"dp_max_grad_norm": c_value},
                "runs": runs,
            }
        )
    manifest = {
        "schema_version": 1,
        "selection_protocol": protocol,
        "fixed_candidates": candidates,
    }
    manifest_path = screen / "fixed_scan_manifest.json"
    output = campaign / "selection" / "slaclip_target_grid.json"
    registry = screen / "candidate_registry.json"
    _write_json(manifest_path, manifest)
    return campaign, manifest_path, output, registry, manifest


def _derive(tmp_path: Path):
    campaign, manifest, output, registry, payload = _build_fixture(tmp_path)
    result = grid.derive_target_grid(
        campaign_root=campaign,
        fixed_manifest=manifest,
        output=output,
        registry_out=registry,
    )
    return campaign, manifest, output, registry, payload, result


def _rewrite_raw(path: Path, mutation) -> None:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    rows = mutation(rows)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _raw_path(campaign: Path, candidate: dict) -> Path:
    status = campaign / candidate["runs"]["42"]["run_status"]
    return status.parent.parent / "results" / "research_raw" / grid.RAW_LOG_NAME


def test_derives_locked_standard_registry_and_complete_provenance(tmp_path: Path) -> None:
    campaign, manifest, output, registry_path, _, artifact = _derive(tmp_path)
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    assert artifact["candidate_counts"] == {"fixed": 8, "slaclip": 10, "total": 18}
    assert artifact["best_fixed"]["dp_max_grad_norm"] == pytest.approx(0.5)
    assert artifact["transition_fixed"]["dp_max_grad_norm"] == pytest.approx(1.5)
    assert [item["value"] for item in artifact["selected_initial_clip_thresholds"]] == [
        0.5,
        1.5,
    ]
    rhos = [item["rho"] for item in artifact["target_points"]]
    assert len(rhos) == 5
    assert all(0.5 <= rho <= 0.995 for rho in rhos)
    assert all(right > left for left, right in zip(rhos, rhos[1:]))
    assert len(registry["candidates"]) == 18
    assert sum(item["family"] == "fixed" for item in registry["candidates"]) == 8
    slaclip = [item for item in registry["candidates"] if item["family"] == "slaclip"]
    assert len(slaclip) == 10
    assert {item["params"]["dp_max_grad_norm"] for item in slaclip} == {0.5, 1.5}
    assert {item["params"]["slaclip_eta"] for item in slaclip} == {0.15}
    assert {
        item["params"]["slaclip_target_non_small_clip_fraction"] for item in slaclip
    } == set(rhos)
    assert not any(
        item["params"]["slaclip_target_non_small_clip_fraction"] == 0.5
        and item["params"]["slaclip_eta"] == 0.2
        for item in slaclip
    )
    grid.selector._validate_protocol(registry)
    assert len(grid.selector._candidate_map(registry)) == 18
    unsigned = dict(artifact)
    expected_hash = unsigned.pop("manifest_sha256")
    assert grid._payload_sha256(unsigned) == expected_hash
    assert artifact["provenance"]["fixed_manifest_file_sha256"] == grid._file_sha256(
        manifest
    )
    assert artifact["provenance"]["candidate_registry_file_sha256"] == grid._file_sha256(
        registry_path
    )
    assert stat.S_IMODE(output.stat().st_mode) == 0o600
    assert stat.S_IMODE(registry_path.stat().st_mode) == 0o600

    # Byte-identical reruns are accepted.
    rerun = grid.derive_target_grid(
        campaign_root=campaign,
        fixed_manifest=manifest,
        output=output,
        registry_out=registry_path,
    )
    assert rerun == artifact


def test_rejects_fully_censored_100_percent_fixed_scan(tmp_path: Path) -> None:
    campaign, manifest_path, output, registry, manifest = _build_fixture(tmp_path)
    for candidate in manifest["fixed_candidates"]:
        path = _raw_path(campaign, candidate)
        _rewrite_raw(
            path,
            lambda rows: [{**row, "raw_clip_fraction": 1.0} for row in rows],
        )
    with pytest.raises(grid.DerivationError, match="100% clipped"):
        grid.derive_target_grid(
            campaign_root=campaign,
            fixed_manifest=manifest_path,
            output=output,
            registry_out=registry,
        )


def test_rejects_missing_raw_step(tmp_path: Path) -> None:
    campaign, manifest_path, output, registry, manifest = _build_fixture(tmp_path)
    path = _raw_path(campaign, manifest["fixed_candidates"][3])
    _rewrite_raw(path, lambda rows: rows[:-1])
    with pytest.raises(grid.DerivationError, match="steps 1..300"):
        grid.derive_target_grid(
            campaign_root=campaign,
            fixed_manifest=manifest_path,
            output=output,
            registry_out=registry,
        )


def test_rejects_mismatched_public_validation_split(tmp_path: Path) -> None:
    campaign, manifest_path, output, registry, manifest = _build_fixture(tmp_path)
    candidate = manifest["fixed_candidates"][-1]
    run = candidate["runs"]["42"]
    status_path = campaign / run["run_status"]
    metrics_path = campaign / run["validation_metrics"]
    split_path = campaign / run["split_manifest"]
    different = _split("different")
    status = json.loads(status_path.read_text(encoding="utf-8"))
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    status["data_split"] = different
    status["validation"]["manifest_sha256"] = different["manifest_sha256"]
    metrics["manifest_sha256"] = different["manifest_sha256"]
    _write_json(status_path, status)
    _write_json(metrics_path, metrics)
    _write_json(split_path, different)
    with pytest.raises(grid.DerivationError, match="share one public validation split"):
        grid.derive_target_grid(
            campaign_root=campaign,
            fixed_manifest=manifest_path,
            output=output,
            registry_out=registry,
        )


def test_rejects_registered_path_escape(tmp_path: Path) -> None:
    campaign, manifest_path, output, registry, manifest = _build_fixture(tmp_path)
    outside = tmp_path / "outside" / "run_status.json"
    _write_json(outside, {})
    manifest["fixed_candidates"][0]["runs"]["42"]["run_status"] = str(outside)
    _write_json(manifest_path, manifest)
    with pytest.raises(grid.DerivationError, match="must remain inside"):
        grid.derive_target_grid(
            campaign_root=campaign,
            fixed_manifest=manifest_path,
            output=output,
            registry_out=registry,
        )


def test_rejects_output_path_escape_without_creating_parent(tmp_path: Path) -> None:
    campaign, manifest_path, _, registry, _ = _build_fixture(tmp_path)
    outside = tmp_path / "outside" / "nested" / "target.json"
    with pytest.raises(grid.DerivationError, match="must remain inside"):
        grid.derive_target_grid(
            campaign_root=campaign,
            fixed_manifest=manifest_path,
            output=outside,
            registry_out=registry,
        )
    assert not outside.parent.exists()


@pytest.mark.parametrize("field,value", [("small", 0.40), ("invalid_flag", 1.1)])
def test_rejects_duplicate_or_invalid_rho_derivation(
    tmp_path: Path, field: str, value: float
) -> None:
    campaign, manifest_path, output, registry, manifest = _build_fixture(tmp_path)
    transition = manifest["fixed_candidates"][3]
    path = _raw_path(campaign, transition)

    def mutate(rows):
        if field == "small":
            return [
                {
                    **row,
                    "raw_reference_small_gradient_proxy": value,
                    "raw_reference_remaining_mass_proxy": 1.0 - value,
                }
                for row in rows
            ]
        rows[0]["raw_reference_small_gradient_proxy"] = value
        rows[0]["raw_reference_remaining_mass_proxy"] = 1.0 - value
        return rows

    _rewrite_raw(path, mutate)
    match = "strictly unique" if field == "small" else "is inconsistent"
    with pytest.raises(grid.DerivationError, match=match):
        grid.derive_target_grid(
            campaign_root=campaign,
            fixed_manifest=manifest_path,
            output=output,
            registry_out=registry,
        )


def test_accepts_out_of_range_proxy_on_an_unselected_fixed_trajectory(
    tmp_path: Path,
) -> None:
    campaign, manifest_path, output, registry, manifest = _build_fixture(tmp_path)
    path = _raw_path(campaign, manifest["fixed_candidates"][0])

    def mutate(rows):
        rows[0]["raw_reference_small_gradient_proxy"] = 1.1
        rows[0]["raw_reference_remaining_mass_proxy"] = -0.1
        rows[0]["raw_reference_conditional_clip_fraction"] = None
        rows[0]["raw_reference_conditional_clip_fraction_valid"] = False
        return rows

    _rewrite_raw(path, mutate)
    artifact = grid.derive_target_grid(
        campaign_root=campaign,
        fixed_manifest=manifest_path,
        output=output,
        registry_out=registry,
    )
    assert artifact["transition_fixed"]["candidate_id"] != (
        manifest["fixed_candidates"][0]["id"]
    )


@pytest.mark.parametrize("tampered", ["output", "registry"])
def test_refuses_immutable_artifact_conflicts(tmp_path: Path, tampered: str) -> None:
    campaign, manifest, output, registry, _, _ = _derive(tmp_path)
    target = output if tampered == "output" else registry
    target.write_text('{"tampered":true}\n', encoding="utf-8")
    with pytest.raises(grid.DerivationError, match="refusing to overwrite inconsistent"):
        grid.derive_target_grid(
            campaign_root=campaign,
            fixed_manifest=manifest,
            output=output,
            registry_out=registry,
        )
