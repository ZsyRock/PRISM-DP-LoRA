"""Immutable, direct clipping-trajectory targets for a four-setting GLUE screen.

The supplied ``builder`` owns the established campaign validators and I/O.
There is deliberately no import back into that module, allowing CLI loading.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from statistics import fmean
from typing import Any


PROFILE = "glue-weighted-target-screen"
STAGE1_SEED = 47
STAGE2_SEED = 48
STAGE1_STEPS = 200
STAGE2_STEPS = 150
WINDOW = 25
EARLY_WEIGHT = 0.3
LATE_WEIGHT = 0.7
ETA = 0.05
C_MAX = 100.0
SETTING_ORDER = (
    "glue8-4b-eps6-r8", "glue8-4b-eps6-r16",
    "glue8-4b-eps6-r32", "glue8-4b-eps3-r16",
)
STAGE1_GRIDS = {
    "glue8-4b-eps6-r8": (15.0, 40.0, 80.0),
    "glue8-4b-eps6-r16": (15.0, 30.0, 60.0),
}
MODEL_REVISION = "cc012e0a6d0787b4adcc0fa2c4da74402494554d"
DATA_SHA256 = "281bad3305e8be7a7660b64b22704a9462f983b90df5d6f0c1c4abb649d6d091"
DEFAULT_CAMPAIGN = "paper-coverage-8495ac8f0c07-baseline-reproduction-cached-v2"
DEFAULT_CODE = "8495ac8f0c07addf910d8f3a2e7eec6a88884a92"
DEFAULT_MANIFEST_SHA = "933507b20afbe734f88addee33c710d3bc1834e3816a955e75d59bfc726ef664"
GAP_CAMPAIGN = "paper-coverage-b53ed134fa42-baseline-gap-fill-all-cached-v2"
GAP_CODE = "b53ed134fa424e48e1d2475cb604091375e3c327"
GAP_MANIFEST_SHA = "cfddbd9c8bf614f38a18d1fcd487ffa5566f6157eaa048942ef8dfe556a3b990"
DEFAULT_HASHES = {
    "glue8-4b-eps6-r8": (
        "2034563a132b334fcca293729a9cc42b3949c2cb54b8b9caf5306bb4e5efd99a",
        "ab48ea38ed3cf6d1e599e07a5f5004787ba62f19341c90141de36fa291a8e864",
        "5141126a0cff0e4865b53bf5ce8b27a49dc7810dede68de8f228cd0ddf3fb179",
    ),
    "glue8-4b-eps6-r16": (
        "0c0b407b1af4e94d0104bf28ff963288c52452fa5bcf28235e9168e293b74c01",
        "51a37be6392e9085a10482c8eceae41535e6318ce8c7133b1c0dfffae2c33d36",
        "ebe0085350f0d3dc4b7cbe90cbc18dd3a9179056cc9e6f899fe99785260312e8",
    ),
    "glue8-4b-eps6-r32": (
        "38c283a9f6d84eee2d600a3829c8025a1cbf6d61a114d2bc13febd68092eae76",
        "7024efa1265ea7988283b9c823d2087fc635bfc972bd901c01a5f70d24171051",
        "1844ec98378f94bac4d29186a2e0ae3c78ecfd6a94181ffb2ab8ce319b983725",
    ),
    "glue8-4b-eps3-r16": (
        "38234503aad9ca8f3ec67acb5f1058128c052021fd3a0d91e455aebad33df4d3",
        "e5e5cfeee555a2d01bb5a79846ef0cdacd4ff8287db0cdd4926258290120703f",
        "db81c9b4831978a2669c280c6075f29173e715cae1f5bd5c723cb068b92b5a94",
    ),
}
PRIOR_CAMPAIGN = "paper-coverage-6d931e37eaa3-glue-target-baseline-screen-v2"
PRIOR_CODE = "6d931e37eaa37c8f7b038de986b668599b2c9a31"
PRIOR_MANIFEST_SHA = "53d29d7f3bb5d1c0c2bb23d86c8a044794112071866cf85adfbe22655a5e5ad5"
PRIOR_WINNERS = {"glue8-4b-eps6-r32": 15.0, "glue8-4b-eps3-r16": 30.0}
PRIOR_HASHES = {
    "glue8-4b-eps6-r32": (
        "88c0a7075df6661fd813179f996d8a0aa8ba5ae0738aca1f0a01f675ff138686",
        "ed9f6460ba5938697b19c04441a958de228eac16b8a0c6dd5f1b58164791c054",
        "422372126b7a33e8b660e5369e0fa1b285721f411d9ab259872073386e95f947",
        "b9032b1ea329de3296843fbf62625acc32fdbdde002d68393a443c800fac0079",
        "2a107241bef3f7ab189bb259cdd7af22b6b4c87c6d8127aaf10f5a12f2e6b582",
        "2641c734670dd49f18b84123300215d3368c5163da02ebcb52fc6700e278aa2f",
    ),
    "glue8-4b-eps3-r16": (
        "e9fdce6e9af1039f1c720356176b9aee2b2ba97d1c97190e37b8bb8c00abd182",
        "b16378f49e5583e3640d6603867b6a7dd53fdd5e919be5c8ef1fee3306103312",
        "652616f399efb32d22c33414f205d9d2fe88d8608565d4c5e0a54fb3efadfde5",
        "b9032b1ea329de3296843fbf62625acc32fdbdde002d68393a443c800fac0079",
        "1fc3e3873c1a2da11f7b2be4a28e251c47d74a6687f810dad8d4390749ba6737",
        "dd4db57906d1513c790086a141c945fa6ef1e1808a8a395276935b0fbde0a627",
    ),
}
SOURCE_PATHS = (
    "adapter/run_status.json", "results/research_raw/telemetry_summary.json",
    "results/research_raw/NON_PRIVATE_train_log.jsonl",
    "results/validation/split_manifest.json", "results/validation/validation_metrics.json",
    "results/validation/validation_curve.jsonl",
)


def derive_target(records: dict[int, dict[str, Any]]) -> dict[str, Any]:
    """Map realized clipping to rho directly, without an inverse small-mass map."""
    if len(records) < 2 * WINDOW or set(records) != set(range(1, len(records) + 1)):
        raise ValueError("weighted target needs at least 50 consecutive steps starting at 1")
    values = [float(records[step]["raw_clip_fraction"]) for step in sorted(records)]
    if any(not math.isfinite(x) or not 0.0 <= x <= 1.0 for x in values):
        raise ValueError("realized clipping fractions must be finite and in [0,1]")
    early, late = fmean(values[:WINDOW]), fmean(values[-WINDOW:])
    rho = EARLY_WEIGHT * early + LATE_WEIGHT * late
    return {
        "rho": rho, "early_mean": early, "late_mean": late,
        "early_steps": [1, WINDOW], "late_steps": [len(values) - WINDOW + 1, len(values)],
        "source_steps": len(values), "realized_clip_min": min(values),
        "realized_clip_max": max(values),
        "formula": "rho=0.3*mean(first25 realized_clip_fraction)+0.7*mean(last25 realized_clip_fraction)",
        "inverse_small_mass_adjustment": False,
        "controller_global_target": "p_star_t=rho*(1-z_t)",
        "saturated_source": min(values) >= 0.99,
        "degenerate_target": rho <= 0.0 or rho >= 1.0,
    }


def _arm(setting: dict[str, Any], candidate: str, c: float, seed: int,
         steps: int, role: str, rho: float | None = None) -> dict[str, Any]:
    sid = setting.get("setting_id", setting["id"])
    return {
        **setting, "id": candidate, "candidate_id": candidate, "setting_id": sid,
        "arm_id": f"{sid}--{candidate}--seed{seed}", "lane": 0,
        "relative_root": f"runs/{sid}/{candidate}/seed-{seed}",
        "method": "baseline" if rho is None else "slaclip",
        "initial_c": c, "rho": rho, "eta": None if rho is None else ETA,
        "seed": seed, "steps": steps, "eval_limit": 0, "role": role,
        "stage": 1 if role == "weighted_stage1_fixed" else 2,
        "c_max": C_MAX, "raw_hist_bins": 512, "raw_hist_max": 200.0,
    }


def build_manifest(builder: Any, code_sha: str, model4revision: str,
                   model9revision: str, model12revision: str | None = None) -> dict[str, Any]:
    manifest = builder.build_manifest(code_sha, model4revision, model9revision,
                                     profile="glue-target-baseline-screen",
                                     model_12b_revision=model12revision)
    if model4revision != MODEL_REVISION:
        raise builder.CampaignError("weighted sources require their pinned 4B model revision")
    settings = {s["id"]: {**s, "model_revision": model4revision} for s in builder.REGIME_SETTINGS}
    manifest.update({
        "profile": PROFILE, "protocol": "prism_glue_weighted_target_screen_v1",
        "seed": STAGE1_SEED,
        "inference_class": "fresh_seed_exploratory_weighted_target_screen_requires_full_length_multi_seed_confirmation",
        "arms": [_arm(settings[sid], f"weighted-fixed-c{c:g}", c, STAGE1_SEED,
                      STAGE1_STEPS, "weighted_stage1_fixed")
                 for sid, grid in STAGE1_GRIDS.items() for c in grid],
    })
    manifest["full_slaclip"].update({"C_max": C_MAX, "eta": ETA})
    manifest["glue_target_baseline_screen"] = {"enabled": False}
    manifest["baseline_reproduction"].update({"covered_settings": 4, "full_length": False})
    manifest["regime_map"].update({
        "exploratory": True, "screen_steps": STAGE1_STEPS,
        "fixed_C_grid": {k: list(v) for k, v in STAGE1_GRIDS.items()},
        "conditional_rho_grid": "direct_weighted_realized_clipping_default_and_tuned_sources",
    })
    manifest["weighted_target_screen"] = {
        "enabled": True, "settings": list(SETTING_ORDER), "stage1_arms": 6,
        "stage2_arms": 12, "total_planned_arms": 18,
        "stage1_steps": STAGE1_STEPS, "stage2_steps": STAGE2_STEPS,
        "stage1_seed": STAGE1_SEED, "stage2_seed": STAGE2_SEED,
        "stage2_recipe": ["fresh_tuned_fixed", "full_default_weighted_rho", "full_tuned_weighted_rho"],
        "target_formula": "rho=0.3*mean(first25 raw_clip_fraction)+0.7*mean(last25 raw_clip_fraction)",
        "inverse_small_mass_adjustment": False, "C0": "selected_tuned_fixed_C",
        "boundary_winner_blocks_confirmation": True, "boundary_winner_allows_exploratory_stage2": True,
        "journal_confirmation": False, "official_task_evaluation": False,
        "end_to_end_dp_claim": False, "NON_PRIVATE_CALIBRATION": True,
        "source_specs": source_specs(),
        "public_holdout": {
            "rows": builder.GLUE_SLACLIP_VALIDATION_ROWS,
            "seed": builder.GLUE_SLACLIP_VALIDATION_SEED,
            "indices_sha256": builder.GLUE_SLACLIP_VALIDATION_INDICES_SHA256,
            "records_sha256": builder.GLUE_SLACLIP_VALIDATION_RECORDS_SHA256,
        },
        "resources": {"gpus": 1, "gpu_type": "a100", "memory": "80G", "time": "24:00:00", "lanes": 1},
    }
    return manifest


def source_specs() -> dict[str, Any]:
    defaults = {}
    for sid, hashes in DEFAULT_HASHES.items():
        gap = sid.endswith("r32")
        defaults[sid] = {
            "campaign_id": GAP_CAMPAIGN if gap else DEFAULT_CAMPAIGN,
            "code_sha": GAP_CODE if gap else DEFAULT_CODE,
            "manifest_sha256": GAP_MANIFEST_SHA if gap else DEFAULT_MANIFEST_SHA,
            "relative_root": f"runs/{sid}/fixed-c1-paper-default/seed-42",
            "hashes": dict(zip(SOURCE_PATHS, hashes)),
        }
    tuned = {
        sid: {"campaign_id": PRIOR_CAMPAIGN, "code_sha": PRIOR_CODE,
              "manifest_sha256": PRIOR_MANIFEST_SHA, "C": c,
              "relative_root": f"runs/{sid}/fixed-c{c:.1f}".replace(".0", "p0") + "/seed-47",
              "hashes": dict(zip(SOURCE_PATHS, PRIOR_HASHES[sid]))}
        for sid, c in PRIOR_WINNERS.items()
    }
    return {"default": defaults, "tuned": tuned}


def _read(builder: Any, path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise builder.CampaignError(f"cannot read weighted source {path}: {exc}") from exc


def _check_hash(builder: Any, path: Path, expected: str) -> None:
    try:
        actual = builder._file_sha256(path)
    except OSError as exc:
        raise builder.CampaignError(f"missing weighted source {path}") from exc
    if actual != expected:
        raise builder.CampaignError(f"weighted source hash mismatch: {path}")


def _load_default(builder: Any, root: Path, sid: str, spec: dict[str, Any]) -> dict[int, dict[str, Any]]:
    arm_root = root / spec["relative_root"]
    status = _read(builder, arm_root / SOURCE_PATHS[0])
    summary = _read(builder, arm_root / SOURCE_PATHS[1])
    config = status.get("config", {})
    expected = {
        "state": "completed", "update_steps": 500, "dataset": "glue8", "method": "baseline",
        "privacy": "dp", "base_model": builder.MODEL_4B, "model_revision": MODEL_REVISION,
        "data_content_sha256": DATA_SHA256,
    }
    if any(status.get(k) != v for k, v in expected.items()):
        raise builder.CampaignError(f"weighted default source status mismatch: {sid}")
    wanted = {
        "seed": 42, "total_update_steps": 500, "dp_max_grad_norm": 1.0,
        "lora_r": int(sid.rsplit("r", 1)[1]), "dp_epsilon": float(sid.split("eps")[1].split("-")[0]),
        "implementation_git_sha": spec["code_sha"], "implementation_git_dirty": False,
        "batch_size": 64, "micro_batch_size": 4, "val_set_size": 0,
        "learning_rate": 0.0002, "cutoff_len": 384, "train_on_inputs": False,
        "dp_delta": 1e-5, "dp_accountant": "prv",
    }
    if any(config.get(k) != v for k, v in wanted.items()):
        raise builder.CampaignError(f"weighted default source config mismatch: {sid}")
    raw_path = arm_root / SOURCE_PATHS[2]
    raw_hash = builder._file_sha256(raw_path)
    if summary.get("source", {}).get("raw_sha256") != raw_hash:
        raise builder.CampaignError(f"weighted default summary does not bind raw data: {sid}")
    records = {}
    try:
        for line in raw_path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            step = int(row["step"])
            if step in records or any(row.get(k) != status.get(k) for k in (
                "run_id", "config_fingerprint", "method", "privacy", "dataset", "base_model", "model_revision",
            )) or row.get("NON_PRIVATE_TELEMETRY") is not True or row.get("dp_clip_threshold") != 1.0:
                raise ValueError("duplicate step or mismatched raw identity/threshold")
            records[step] = row
        if len(records) != 500:
            raise ValueError("expected 500 raw steps")
        derive_target(records)
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise builder.CampaignError(f"invalid weighted default telemetry: {sid}: {exc}") from exc
    return records


def verify_sources(builder: Any, root: Path) -> dict[str, dict[str, Any]]:
    """Reverify content pins at prepare and immediately before Stage-2 locking."""
    checked: dict[str, dict[str, Any]] = {"default": {}, "tuned": {}}
    for kind, entries in source_specs().items():
        for sid, spec in entries.items():
            source_root = root.parent / spec["campaign_id"]
            _check_hash(builder, source_root / "plans/manifest.json", spec["manifest_sha256"])
            for path, digest in spec["hashes"].items():
                _check_hash(builder, source_root / spec["relative_root"] / path, digest)
            manifest = _read(builder, source_root / "plans/manifest.json")
            matching = [a for a in manifest["arms"] if a["relative_root"] == spec["relative_root"]]
            if len(matching) != 1 or matching[0]["setting_id"] != sid:
                raise builder.CampaignError(f"weighted source manifest arm mismatch: {sid}")
            arm = matching[0]
            if kind == "default":
                records = _load_default(builder, source_root, sid, spec)
                endpoint = None
            else:
                status, records, _, _ = builder._load_focused_arm(source_root, arm, spec["code_sha"])
                if arm["initial_c"] != spec["C"] or arm["steps"] != STAGE1_STEPS or arm["seed"] != STAGE1_SEED:
                    raise builder.CampaignError(f"weighted tuned source parameters mismatch: {sid}")
                endpoint = float(status["validation"]["loss_mean"])
            checked[kind][sid] = {"arm": arm, "records": records, "provenance": spec,
                                  "target": derive_target(records), "endpoint_loss": endpoint}
    return checked


def lock(builder: Any, root: Path) -> dict[str, Any]:
    sources = verify_sources(builder, root)
    manifest_path = root / "plans/manifest.json"
    manifest = _read(builder, manifest_path)
    if manifest.get("profile") != PROFILE:
        raise builder.CampaignError("weighted lock requires weighted profile")
    # Verify the immutable plan against this implementation before consuming it.
    expected = build_manifest(builder, manifest["code_sha"], MODEL_REVISION, "0" * 40)
    if manifest["arms"] != expected["arms"] or manifest.get("weighted_target_screen") != expected["weighted_target_screen"]:
        raise builder.CampaignError("weighted stage1 plan or recipe changed")
    stage1 = {}
    all_artifacts = {}
    for arm in manifest["arms"]:
        status, records, _, hashes = builder._load_focused_arm(root, arm, manifest["code_sha"])
        result = {"arm": arm, "records": records, "hashes": hashes,
                  "loss": float(status["validation"]["loss_mean"])}
        if not math.isfinite(result["loss"]):
            raise builder.CampaignError("non-finite weighted stage1 validation loss")
        stage1.setdefault(arm["setting_id"], []).append(result)
        all_artifacts[arm["arm_id"]] = hashes
    selections, stage2 = {}, []
    for sid in SETTING_ORDER:
        if sid in stage1:
            candidates = sorted(stage1[sid], key=lambda r: (r["loss"], r["arm"]["initial_c"], r["arm"]["candidate_id"]))
            winner = candidates[0]
            arm, records = winner["arm"], winner["records"]
            tuned = {"target": derive_target(records), "provenance": {
                "campaign_id": root.name, "relative_root": arm["relative_root"],
                "code_sha": manifest["code_sha"], "hashes": winner["hashes"],
            }}
            c = float(arm["initial_c"])
            boundary = c in (min(STAGE1_GRIDS[sid]), max(STAGE1_GRIDS[sid]))
            ranking = [{"C": r["arm"]["initial_c"], "validation_loss": r["loss"],
                        "arm_id": r["arm"]["arm_id"]} for r in candidates]
        else:
            tuned = sources["tuned"][sid]
            arm = tuned["arm"]
            c = float(arm["initial_c"])
            boundary = False  # Pinned C15/C30 are interior to prior {1,5,15,30,50}.
            ranking = [{"C": c, "validation_loss": tuned["endpoint_loss"], "source": PRIOR_CAMPAIGN}]
        default = sources["default"][sid]
        selections[sid] = {
            "selected_C": c, "boundary_winner": boundary,
            "journal_confirmation": False, "fixed_ranking": ranking,
            "default_target": default["target"], "tuned_target": tuned["target"],
            "default_source": default["provenance"], "tuned_source": tuned["provenance"],
        }
        for candidate, rho, role in (
            ("weighted-fresh-fixed", None, "fresh_fixed_comparator"),
            ("weighted-full-default-rho", default["target"]["rho"], "weighted_default_target"),
            ("weighted-full-tuned-rho", tuned["target"]["rho"], "weighted_tuned_target"),
        ):
            stage2.append(_arm(arm, candidate, c, STAGE2_SEED, STAGE2_STEPS, role, rho))
    plan = builder._plan_bytes({"profile": PROFILE, "arms": stage2}, 0, include_all=True)
    source_bytes = builder._json_bytes(source_specs())
    result = {
        "schema_version": 1, "profile": PROFILE, "code_sha": manifest["code_sha"],
        "manifest_sha256": builder._file_sha256(manifest_path),
        "source_lock_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "stage2_plan_sha256": hashlib.sha256(plan).hexdigest(),
        "stage1_artifacts": all_artifacts,
        "settings": [{"setting_id": sid, **selections[sid]} for sid in SETTING_ORDER],
        "stage2_arms": stage2,
        "stage1_seed": STAGE1_SEED, "stage2_seed": STAGE2_SEED,
        "stage2_steps": STAGE2_STEPS, "total_planned_arms": 18,
        "public_holdout": manifest["weighted_target_screen"]["public_holdout"],
        "target_formula": manifest["weighted_target_screen"]["target_formula"],
        "inverse_small_mass_adjustment": False, "journal_confirmation": False,
        "official_task_evaluation": False, "NON_PRIVATE_CALIBRATION": True,
        "end_to_end_dp_claim": False,
    }
    builder._with_sha(root / "selection/weighted-sources.lock.json", source_bytes)
    builder._with_sha(root / "selection/weighted-targets.lock.json", builder._json_bytes(result))
    builder._with_sha(root / "plans/stage2-weighted.tsv", plan)
    return result
