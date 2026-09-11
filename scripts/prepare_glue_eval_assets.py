#!/usr/bin/env python3
"""Materialize pinned GLUE validation splits for offline compute nodes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

from datasets import DatasetDict, load_dataset


SCHEMA_VERSION = 1
DATASET_ID = "nyu-mll/glue"
DATASET_REVISION = "bcdcba79d07bc864c1c254ccfcedcce55bcc9a8c"
TASKS = ['cola', 'sst2', 'mrpc', 'stsb', 'qqp', 'mnli', 'qnli', 'rte']
SPLITS = {
    **{task: ['validation'] for task in TASKS if task != 'mnli'},
    'mnli': ['validation_matched', 'validation_mismatched'],
}


def _validate_existing(root: Path) -> bool:
    manifest_path = root / 'manifest.json'
    if not manifest_path.is_file():
        return False
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if (
        manifest.get('schema_version') != SCHEMA_VERSION
        or manifest.get('dataset_id') != DATASET_ID
        or manifest.get('dataset_revision') != DATASET_REVISION
        or manifest.get('tasks') != TASKS
    ):
        return False
    for task in TASKS:
        if not (root / task / 'dataset_dict.json').is_file():
            return False
    expected = manifest.get('content_sha256')
    actual = _content_digest_without_manifest(root)
    if expected != actual:
        raise RuntimeError(f'GLUE asset checksum mismatch: expected {expected}, got {actual}')
    return True


def _content_digest_without_manifest(root: Path) -> str:
    digest = hashlib.sha256()
    # Hugging Face Dataset transformations may create cache-*.arrow files next
    # to a dataset loaded from disk. They are derived evaluation scratch, not
    # part of the materialized source snapshot, and must not invalidate its
    # content identity. The evaluator also requests in-memory transforms, but
    # ignoring legacy cache files keeps older valid snapshots usable.
    for path in sorted(
        p for p in root.rglob('*')
        if p.is_file()
        and p.name != 'manifest.json'
        and not (p.name.startswith('cache-') and p.suffix == '.arrow')
    ):
        relative = path.relative_to(root).as_posix().encode()
        digest.update(len(relative).to_bytes(8, 'big'))
        digest.update(relative)
        with path.open('rb') as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b''):
                digest.update(chunk)
    return digest.hexdigest()


def prepare(root: Path, cache_dir: Path | None) -> None:
    root = root.resolve()
    if root.exists() and _validate_existing(root):
        print(f'glue_eval_assets=ready root={root}')
        return
    root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f'.{root.name}.', dir=root.parent))
    try:
        counts = {}
        for task in TASKS:
            source = load_dataset(
                DATASET_ID,
                task,
                revision=DATASET_REVISION,
                cache_dir=str(cache_dir) if cache_dir else None,
            )
            selected = DatasetDict({split: source[split] for split in SPLITS[task]})
            selected.save_to_disk(str(temporary / task))
            counts[task] = {split: len(selected[split]) for split in SPLITS[task]}
        content_sha256 = _content_digest_without_manifest(temporary)
        manifest = {
            'schema_version': SCHEMA_VERSION,
            'dataset_id': DATASET_ID,
            'dataset_revision': DATASET_REVISION,
            'tasks': TASKS,
            'splits': SPLITS,
            'row_counts': counts,
            'content_sha256': content_sha256,
        }
        manifest_path = temporary / 'manifest.json'
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n', encoding='utf-8')
        os.chmod(manifest_path, 0o600)
        if root.exists():
            if _validate_existing(root):
                return
            raise RuntimeError(f'refusing to replace incompatible GLUE assets: {root}')
        os.rename(temporary, root)
        print(f'glue_eval_assets=prepared root={root} content_sha256={content_sha256}')
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', required=True, type=Path)
    parser.add_argument('--cache-dir', type=Path)
    args = parser.parse_args()
    prepare(args.output_root, args.cache_dir)


if __name__ == '__main__':
    main()
