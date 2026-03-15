"""Tests for freshness tracking module."""

import tempfile
from datetime import datetime
from pathlib import Path

from src.freshness import compute_source_hash, compute_freshness_fields


def test_compute_source_hash_deterministic():
    """Same file content produces same hash."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as f:
        f.write(b"test content for hashing")
        path = Path(f.name)
    try:
        h1 = compute_source_hash(path)
        h2 = compute_source_hash(path)
        assert h1 == h2
        assert len(h1) == 64  # SHA-256 hex digest
    finally:
        path.unlink()


def test_compute_source_hash_different_content():
    """Different content produces different hash."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as f1:
        f1.write(b"content A")
        path1 = Path(f1.name)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as f2:
        f2.write(b"content B")
        path2 = Path(f2.name)
    try:
        assert compute_source_hash(path1) != compute_source_hash(path2)
    finally:
        path1.unlink()
        path2.unlink()


def test_compute_freshness_fields():
    """All freshness fields are populated with correct types."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
        f.write(b"fake pdf content")
        path = Path(f.name)
    try:
        fields = compute_freshness_fields(path)
        assert "source_path" in fields
        assert "source_hash" in fields
        assert "source_mtime" in fields
        assert "extracted_at" in fields
        assert len(fields["source_hash"]) == 64
        # Verify ISO format parsing
        datetime.fromisoformat(fields["source_mtime"])
        datetime.fromisoformat(fields["extracted_at"])
        # Path uses forward slashes
        assert "\\" not in fields["source_path"]
    finally:
        path.unlink()
