"""Tests for manifest loading, status tracking, and batch processor."""
import json
import pytest
from pathlib import Path
from src.manifest import Manifest, ManifestEntry, FileStatus, load_status, save_status


def test_load_manifest(tmp_path):
    manifest_file = tmp_path / "manifest.json"
    manifest_file.write_text(json.dumps({
        "schema_version": 1,
        "project": "test-project",
        "output_dir": str(tmp_path / "output"),
        "files": [
            {"id": "file-1", "path": str(tmp_path / "test.pdf"), "doc_type": "document", "name": "Test PDF"},
            {"id": "file-2", "path": str(tmp_path / "test.pptx"), "doc_type": "presentation"},
        ]
    }))

    manifest = Manifest.from_file(manifest_file)
    assert manifest.project == "test-project"
    assert len(manifest.files) == 2
    assert manifest.files[0].doc_type == "document"
    assert manifest.files[0].name == "Test PDF"
    assert manifest.files[1].name == "file-2"  # defaults to id
    assert manifest.files[1].doc_type == "presentation"


def test_manifest_output_dir(tmp_path):
    manifest_file = tmp_path / "manifest.json"
    manifest_file.write_text(json.dumps({
        "schema_version": 1,
        "project": "proj",
        "output_dir": str(tmp_path / "out"),
        "files": [],
    }))
    manifest = Manifest.from_file(manifest_file)
    assert manifest.output_dir == tmp_path / "out"


def test_invalid_schema_version(tmp_path):
    manifest_file = tmp_path / "manifest.json"
    manifest_file.write_text(json.dumps({"schema_version": 99, "files": []}))
    with pytest.raises(ValueError, match="Unsupported manifest schema version"):
        Manifest.from_file(manifest_file)


def test_manifest_config_passthrough(tmp_path):
    manifest_file = tmp_path / "manifest.json"
    manifest_file.write_text(json.dumps({
        "schema_version": 1,
        "project": "proj",
        "output_dir": str(tmp_path),
        "config": {"custom_key": "custom_value"},
        "files": [],
    }))
    manifest = Manifest.from_file(manifest_file)
    assert manifest.config == {"custom_key": "custom_value"}


def test_manifest_default_config(tmp_path):
    manifest_file = tmp_path / "manifest.json"
    manifest_file.write_text(json.dumps({
        "schema_version": 1,
        "output_dir": str(tmp_path),
        "files": [],
    }))
    manifest = Manifest.from_file(manifest_file)
    assert manifest.config == {}
    assert manifest.project == "unknown"


def test_status_save_and_load(tmp_path):
    statuses = {
        "file-1": {"status": "done", "processed_at": "2026-03-05T12:00:00"},
        "file-2": {"status": "error", "error": "File not found"},
    }
    save_status(tmp_path, statuses)
    loaded = load_status(tmp_path)
    assert loaded["file-1"] == FileStatus.DONE
    assert loaded["file-2"] == FileStatus.ERROR


def test_status_resume_empty(tmp_path):
    """No status file = empty dict (all files pending)."""
    loaded = load_status(tmp_path)
    assert loaded == {}


def test_status_creates_directory(tmp_path):
    """save_status should create output_dir if it doesn't exist."""
    out = tmp_path / "nested" / "output"
    save_status(out, {"f1": {"status": "done"}})
    assert (out / "status.json").exists()


def test_manifest_entry_defaults():
    entry = ManifestEntry(id="test", path=Path("test.pdf"), doc_type="document", name="Test")
    assert entry.status == FileStatus.PENDING
    assert entry.error is None


def test_file_status_enum_values():
    assert FileStatus.PENDING.value == "pending"
    assert FileStatus.DONE.value == "done"
    assert FileStatus.ERROR.value == "error"
    assert FileStatus.SKIPPED.value == "skipped"


def test_manifest_entry_client_project_defaults():
    """Client and project default to None."""
    entry = ManifestEntry(id="test", path=Path("test.pdf"), doc_type="document", name="Test")
    assert entry.client is None
    assert entry.project is None


def test_manifest_parses_client_project(tmp_path):
    """Client and project should be parsed from manifest JSON."""
    manifest_file = tmp_path / "manifest.json"
    manifest_file.write_text(json.dumps({
        "schema_version": 1,
        "project": "Lenzing_Planning",
        "output_dir": str(tmp_path / "output"),
        "files": [
            {
                "id": "file-1",
                "path": str(tmp_path / "test.pdf"),
                "doc_type": "document",
                "name": "Test",
                "client": "Lenzing AG",
                "project": "Lenzing_Planning",
            }
        ]
    }))
    manifest = Manifest.from_file(manifest_file)
    assert manifest.files[0].client == "Lenzing AG"
    assert manifest.files[0].project == "Lenzing_Planning"


def test_manifest_client_project_optional(tmp_path):
    """Client and project are optional in manifest JSON."""
    manifest_file = tmp_path / "manifest.json"
    manifest_file.write_text(json.dumps({
        "schema_version": 1,
        "project": "proj",
        "output_dir": str(tmp_path / "output"),
        "files": [
            {"id": "file-1", "path": str(tmp_path / "test.pdf"), "doc_type": "document"}
        ]
    }))
    manifest = Manifest.from_file(manifest_file)
    assert manifest.files[0].client is None
    assert manifest.files[0].project is None
