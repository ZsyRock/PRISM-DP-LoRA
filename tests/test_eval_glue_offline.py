from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from prism_cli.eval_glue import (
    GLUE_TASKS,
    _compute_metric,
    _load_glue_asset_manifest,
    _maybe_limit,
)


ROOT = Path(__file__).resolve().parents[1]
PREP_SPEC = importlib.util.spec_from_file_location(
    "prepare_glue_eval_assets",
    ROOT / "scripts" / "prepare_glue_eval_assets.py",
)
assert PREP_SPEC and PREP_SPEC.loader
prepare_assets = importlib.util.module_from_spec(PREP_SPEC)
PREP_SPEC.loader.exec_module(prepare_assets)


def test_local_glue_metrics_match_expected_definitions() -> None:
    assert _compute_metric('sst2', [0, 1, 1], [0, 1, 0])['accuracy'] == pytest.approx(2 / 3)
    mrpc = _compute_metric('mrpc', [1, 1, 0, 0], [1, 0, 0, 0])
    assert mrpc['accuracy'] == pytest.approx(0.75)
    assert mrpc['f1'] == pytest.approx(2 / 3)
    assert _compute_metric('cola', [0, 1, 1, 0], [0, 1, 1, 0])['matthews_correlation'] == 1.0
    stsb = _compute_metric('stsb', [0.0, 1.0, 2.0], [0.0, 1.0, 2.0])
    assert stsb == {'pearson': pytest.approx(1.0), 'spearmanr': pytest.approx(1.0)}


def test_asset_manifest_identity_is_content_bound(tmp_path: Path) -> None:
    payload = {
        'schema_version': 1,
        'dataset_id': 'nyu-mll/glue',
        'dataset_revision': '1' * 40,
        'tasks': GLUE_TASKS,
        'content_sha256': '2' * 64,
    }
    raw = (json.dumps(payload, sort_keys=True) + '\n').encode()
    (tmp_path / 'manifest.json').write_bytes(raw)
    loaded = _load_glue_asset_manifest(tmp_path)
    assert loaded['manifest_sha256'] == hashlib.sha256(raw).hexdigest()


def test_asset_manifest_rejects_task_mismatch(tmp_path: Path) -> None:
    (tmp_path / 'manifest.json').write_text(
        json.dumps({'schema_version': 1, 'tasks': ['sst2']}), encoding='utf-8'
    )
    with pytest.raises(RuntimeError, match='task order'):
        _load_glue_asset_manifest(tmp_path)


def test_asset_digest_ignores_derived_dataset_cache_files(tmp_path: Path) -> None:
    task = tmp_path / "sst2" / "validation"
    task.mkdir(parents=True)
    (task / "data-00000-of-00001.arrow").write_bytes(b"locked source")
    before = prepare_assets._content_digest_without_manifest(tmp_path)
    (task / "cache-derived.arrow").write_bytes(b"transient filter output")
    assert prepare_assets._content_digest_without_manifest(tmp_path) == before
    (task / "data-00000-of-00001.arrow").write_bytes(b"changed source")
    assert prepare_assets._content_digest_without_manifest(tmp_path) != before


def test_eval_subset_shuffle_stays_in_memory() -> None:
    class FakeDataset:
        def __init__(self) -> None:
            self.shuffle_kwargs = None
            self.selected = None

        def __len__(self) -> int:
            return 5

        def shuffle(self, **kwargs):
            self.shuffle_kwargs = kwargs
            return self

        def select(self, indices):
            self.selected = list(indices)
            return self

    dataset = FakeDataset()
    assert _maybe_limit(dataset, 2) is dataset
    assert dataset.shuffle_kwargs == {
        "seed": 1729,
        "keep_in_memory": True,
    }
    assert dataset.selected == [0, 1]
