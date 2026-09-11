"""Evaluation-cache identities tied to one completed training run."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from .utils import write_json_atomic


EVALUATION_SCHEMA_VERSION = 1
MANIFEST_NAME = "evaluation_config.json"


def prepare_evaluation_cache(
    result_dir: Path,
    payload: Mapping[str, Any],
    *,
    artifact_names: Iterable[str],
    force: bool,
) -> Path:
    """Validate or initialize an evaluation cache and return its manifest.

    Cached predictions are valid only for the exact adapter fingerprint and
    decoding/evaluation parameters.  Legacy or mismatched caches are rejected
    unless ``force`` explicitly authorizes deleting the known eval artifacts.
    """

    result_dir = Path(result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    manifest = result_dir / MANIFEST_NAME
    expected = {
        "evaluation_schema_version": EVALUATION_SCHEMA_VERSION,
        **dict(payload),
    }
    names = sorted({str(name) for name in artifact_names} | {"summary.csv", "details.csv"})
    artifacts = [result_dir / name for name in names]

    existing = None
    if manifest.exists():
        try:
            existing = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            if not force:
                raise RuntimeError(
                    f"Evaluation manifest is unreadable: {manifest}; use --force_eval to rebuild"
                ) from exc
    elif any(path.exists() for path in artifacts) and not force:
        raise RuntimeError(
            f"Legacy evaluation cache has no verifiable manifest in {result_dir}; "
            "use --force_eval to rebuild it"
        )

    if existing is not None and existing != expected and not force:
        raise RuntimeError(
            f"Evaluation settings do not match cached predictions in {result_dir}; "
            "use --force_eval to rebuild them"
        )

    if force:
        for path in artifacts:
            if path.is_file() or path.is_symlink():
                path.unlink()
    write_json_atomic(manifest, expected)
    return manifest
