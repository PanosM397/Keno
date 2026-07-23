"""SHA256 helpers for the paper-freeze U-Net checkpoint."""

from __future__ import annotations

import hashlib
from pathlib import Path

# app/core/checkpoint_hash.py -> parents[3] is the repo root.
_FREEZE_HASH_FILE = (
    Path(__file__).resolve().parents[3] / "docs" / "freeze" / "current" / "checkpoint.sha256"
)


def expected_freeze_sha256() -> str | None:
    if _FREEZE_HASH_FILE.is_file():
        return _FREEZE_HASH_FILE.read_text(encoding="utf-8").strip()
    return None


def file_sha256(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_hash_status(path: Path) -> tuple[str | None, bool | None]:
    """Return (sha256 or None, matches_freeze or None if unknown/missing)."""
    if not path.is_file():
        return None, None
    actual = file_sha256(path)
    expected = expected_freeze_sha256()
    if expected is None:
        return actual, None
    return actual, actual == expected
