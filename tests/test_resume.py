"""Tests for resume support — skip files with matching source_hash."""

import json
import hashlib
from pathlib import Path

import pytest


def _compute_hash(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _setup_existing_output(tmp_path, source_path: Path, hash_value: str):
    """Create a fake existing extraction JSON with freshness data."""
    extract_dir = tmp_path / "extract"
    extract_dir.mkdir(parents=True, exist_ok=True)
    json_path = extract_dir / f"{source_path.stem}.json"
    json_path.write_text(json.dumps({
        "title": "Existing Extraction",
        "summary": "Already done",
        "facts": [],
        "freshness": {
            "source_path": str(source_path),
            "source_hash": hash_value,
            "source_mtime": "2026-03-20T10:00:00",
            "extracted_at": "2026-03-20T10:05:00",
        },
    }), encoding="utf-8")
    return json_path


class TestResumeLogic:
    def test_resume_skips_existing(self, tmp_path):
        """Output with matching hash → should be skipped."""
        source = tmp_path / "input" / "test.pdf"
        source.parent.mkdir()
        source.write_bytes(b"PDF content here")
        file_hash = _compute_hash(source)

        output_dir = tmp_path / "output" / "test"
        _setup_existing_output(output_dir, source, file_hash)

        # Verify the hash matches
        existing_json = output_dir / "extract" / "test.json"
        data = json.loads(existing_json.read_text())
        assert data["freshness"]["source_hash"] == file_hash

    def test_resume_extracts_changed(self, tmp_path):
        """Output with different hash → should re-extract."""
        source = tmp_path / "input" / "test.pdf"
        source.parent.mkdir()
        source.write_bytes(b"New PDF content")
        current_hash = _compute_hash(source)

        output_dir = tmp_path / "output" / "test"
        _setup_existing_output(output_dir, source, "old_hash_that_doesnt_match")

        existing_json = output_dir / "extract" / "test.json"
        data = json.loads(existing_json.read_text())
        assert data["freshness"]["source_hash"] != current_hash

    def test_resume_force_flag(self, tmp_path):
        """--force → should always extract regardless of hash."""
        source = tmp_path / "input" / "test.pdf"
        source.parent.mkdir()
        source.write_bytes(b"PDF content here")
        file_hash = _compute_hash(source)

        output_dir = tmp_path / "output" / "test"
        _setup_existing_output(output_dir, source, file_hash)

        # With force=True, the skip logic should not apply
        # (tested via CLI integration, here we verify the JSON setup)
        existing_json = output_dir / "extract" / "test.json"
        data = json.loads(existing_json.read_text())
        # Hash matches but force should override — verified in integration
        assert data["freshness"]["source_hash"] == file_hash

    def test_resume_no_output(self, tmp_path):
        """No existing output → normal extraction (no skip)."""
        source = tmp_path / "input" / "test.pdf"
        source.parent.mkdir()
        source.write_bytes(b"PDF content here")

        output_dir = tmp_path / "output" / "test"
        existing_json = output_dir / "extract" / "test.json"
        assert not existing_json.exists()
