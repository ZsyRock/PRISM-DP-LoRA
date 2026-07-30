"""Stable experiment identities and artifact compatibility checks."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional


FINGERPRINT_SCHEMA_VERSION = 7


def git_worktree_identity(
    root: Path,
) -> tuple[Optional[str], Optional[bool], Optional[str]]:
    """Return commit, dirty flag, and a content digest of dirty changes.

    Formal runs are expected to use a clean commit.  Including this identity in
    the experiment fingerprint prevents a completed adapter produced by an
    older implementation from being silently reused after the code changes.
    """

    root = Path(root)
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain=v1", "-z"],
            cwd=root,
            check=True,
            capture_output=True,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None, None, None
    if not status:
        return (commit or None), False, None

    digest = hashlib.sha256()
    digest.update(status)
    try:
        tracked_diff = subprocess.run(
            ["git", "diff", "--binary", "HEAD", "--"],
            cwd=root,
            check=True,
            capture_output=True,
        ).stdout
        untracked = subprocess.run(
            ["git", "ls-files", "--others", "--exclude-standard", "-z"],
            cwd=root,
            check=True,
            capture_output=True,
        ).stdout.split(b"\0")
        digest.update(tracked_diff)
        for encoded_path in sorted(value for value in untracked if value):
            relative = encoded_path.decode("utf-8", errors="surrogateescape")
            path = root / relative
            if not path.is_file():
                continue
            digest.update(len(encoded_path).to_bytes(8, "big"))
            digest.update(encoded_path)
            _update_from_file(digest, path)
    except (OSError, subprocess.SubprocessError):
        # The status itself still distinguishes clean from dirty; refusing to
        # invent a partial content hash is safer than claiming exact identity.
        return (commit or None), True, None
    return (commit or None), True, digest.hexdigest()


def content_sha256(path: Path) -> Optional[str]:
    """Hash a file or directory without including its machine-specific path."""
    path = Path(path)
    if not path.exists():
        return None

    digest = hashlib.sha256()
    if path.is_file():
        _update_from_file(digest, path)
        return digest.hexdigest()

    files: Iterable[Path] = sorted(p for p in path.rglob("*") if p.is_file())
    for child in files:
        relative = child.relative_to(path).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        _update_from_file(digest, child)
    return digest.hexdigest()


def _update_from_file(digest: "hashlib._Hash", path: Path) -> None:
    with open(path, "rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)


def config_fingerprint(payload: Mapping[str, Any]) -> str:
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def readable_float(value: float) -> str:
    text = format(float(value), ".10g")
    return text.replace("-", "m").replace("+", "").replace(".", "p")


def safe_run_label(value: str) -> str:
    label = re.sub(r"[^A-Za-z0-9._-]+", "-", str(value).strip())
    label = label.strip("-._")
    if not label:
        raise ValueError("run_name must contain at least one letter or digit")
    return label[:96]


def make_run_id(
    *,
    label: str,
    clip_threshold: float,
    fingerprint: str,
    repeat_id: Optional[int] = None,
) -> str:
    base = safe_run_label(label)
    if repeat_id is not None:
        base += f"_repeat{int(repeat_id)}"
    return f"{base}_C{readable_float(clip_threshold)}_{fingerprint[:10]}"


def extract_status_fingerprint(payload: Mapping[str, Any]) -> Optional[str]:
    direct = payload.get("config_fingerprint")
    if direct:
        return str(direct)
    config = payload.get("config")
    if isinstance(config, Mapping) and config.get("config_fingerprint"):
        return str(config["config_fingerprint"])
    return None


def load_json_object(path: Path) -> Dict[str, Any]:
    path = Path(path)
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except FileNotFoundError as exc:
        raise ValueError(f"Config file does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON config {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Config file must contain one JSON object: {path}")
    return payload
