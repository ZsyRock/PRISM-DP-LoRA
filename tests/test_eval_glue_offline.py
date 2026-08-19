from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from prism_cli.eval_glue import GLUE_TASKS, _compute_metric, _load_glue_asset_manifest


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
