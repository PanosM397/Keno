"""Tests for freeze checkpoint hash helpers."""

from pathlib import Path

from app.core.checkpoint_hash import checkpoint_hash_status, expected_freeze_sha256, file_sha256


def test_expected_freeze_sha256_is_pinned():
    expected = expected_freeze_sha256()
    assert expected is not None
    assert expected.startswith("55ce7637")
    assert len(expected) == 64


def test_file_sha256_roundtrip(tmp_path: Path):
    path = tmp_path / "tiny.bin"
    path.write_bytes(b"keno")
    assert file_sha256(path) == file_sha256(path)


def test_missing_checkpoint_status(tmp_path: Path):
    sha, matches = checkpoint_hash_status(tmp_path / "missing.pt")
    assert sha is None
    assert matches is None
