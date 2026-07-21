from __future__ import annotations

import json
from pathlib import Path

import pytest

from prism_cli.evaluation_identity import prepare_evaluation_cache


def _payload(beams: int = 1) -> dict:
    return {
        'config_fingerprint': 'abc',
        'dataset': 'glue8',
        'num_beams': beams,
        'fast_dev_run': 0,
    }


def test_matching_evaluation_manifest_keeps_cached_predictions(tmp_path: Path) -> None:
    prepare_evaluation_cache(
        tmp_path,
        _payload(),
        artifact_names=['sst2.json'],
        force=False,
    )
    cached = tmp_path / 'sst2.json'
    cached.write_text('{}', encoding='utf-8')

    prepare_evaluation_cache(
        tmp_path,
        _payload(),
        artifact_names=['sst2.json'],
        force=False,
    )
    assert cached.exists()


def test_mismatched_evaluation_manifest_requires_force(tmp_path: Path) -> None:
    prepare_evaluation_cache(
        tmp_path,
        _payload(beams=1),
        artifact_names=['sst2.json'],
        force=False,
    )
    cached = tmp_path / 'sst2.json'
    cached.write_text('{}', encoding='utf-8')

    with pytest.raises(RuntimeError, match='do not match'):
        prepare_evaluation_cache(
            tmp_path,
            _payload(beams=4),
            artifact_names=['sst2.json'],
            force=False,
        )
    prepare_evaluation_cache(
        tmp_path,
        _payload(beams=4),
        artifact_names=['sst2.json'],
        force=True,
    )
    assert not cached.exists()
    manifest = json.loads((tmp_path / 'evaluation_config.json').read_text(encoding='utf-8'))
    assert manifest['num_beams'] == 4


def test_legacy_cache_without_manifest_is_rejected(tmp_path: Path) -> None:
    (tmp_path / 'sst2.json').write_text('{}', encoding='utf-8')
    with pytest.raises(RuntimeError, match='Legacy evaluation cache'):
        prepare_evaluation_cache(
            tmp_path,
            _payload(),
            artifact_names=['sst2.json'],
            force=False,
        )
