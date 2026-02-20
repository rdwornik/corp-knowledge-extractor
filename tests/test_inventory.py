"""Unit tests for src/inventory.py — file classification and scanning."""

import pytest
from pathlib import Path

from src.inventory import scan_input, FileType, SourceFile


# Minimal config matching settings.yaml file_types section
MINIMAL_CONFIG = {
    "file_types": {
        "video": [".mkv", ".mp4", ".webm", ".avi", ".mov", ".wmv"],
        "audio": [".mp3", ".wav", ".m4a", ".ogg", ".flac"],
        "document": [".pdf", ".docx"],
        "slides": [".pptx"],
        "spreadsheet": [".xlsx", ".csv"],
        "note": [".md", ".txt"],
        "transcript": [".srt", ".vtt"],
    }
}


def _make_file(tmp_path: Path, name: str, content: str = "x") -> Path:
    p = tmp_path / name
    p.write_text(content)
    return p


# ---------------------------------------------------------------------------
# Classification tests
# ---------------------------------------------------------------------------

def test_classify_video(tmp_path):
    p = _make_file(tmp_path, "meeting.mp4")
    files = scan_input(p, MINIMAL_CONFIG)
    assert len(files) == 1
    assert files[0].type == FileType.VIDEO


def test_classify_audio(tmp_path):
    p = _make_file(tmp_path, "audio.mp3")
    files = scan_input(p, MINIMAL_CONFIG)
    assert files[0].type == FileType.AUDIO


def test_classify_document(tmp_path):
    p = _make_file(tmp_path, "report.pdf")
    files = scan_input(p, MINIMAL_CONFIG)
    assert files[0].type == FileType.DOCUMENT


def test_classify_slides(tmp_path):
    p = _make_file(tmp_path, "deck.pptx")
    files = scan_input(p, MINIMAL_CONFIG)
    assert files[0].type == FileType.SLIDES


def test_classify_note(tmp_path):
    p = _make_file(tmp_path, "readme.md")
    files = scan_input(p, MINIMAL_CONFIG)
    assert files[0].type == FileType.NOTE


def test_classify_transcript(tmp_path):
    p = _make_file(tmp_path, "captions.srt")
    files = scan_input(p, MINIMAL_CONFIG)
    assert files[0].type == FileType.TRANSCRIPT


def test_classify_unknown(tmp_path):
    p = _make_file(tmp_path, "mystery.xyz")
    # Unknown files are skipped when scanning a folder,
    # but a direct single-file call should return UNKNOWN
    files = scan_input(p, MINIMAL_CONFIG)
    assert files[0].type == FileType.UNKNOWN


# ---------------------------------------------------------------------------
# Folder scan tests
# ---------------------------------------------------------------------------

def test_scan_folder(tmp_path):
    _make_file(tmp_path, "session.mp4")
    _make_file(tmp_path, "slides.pptx")
    _make_file(tmp_path, "notes.md")

    files = scan_input(tmp_path, MINIMAL_CONFIG)
    assert len(files) == 3
    types = {f.type for f in files}
    assert FileType.VIDEO in types
    assert FileType.SLIDES in types
    assert FileType.NOTE in types


def test_skip_hidden_file(tmp_path):
    _make_file(tmp_path, "session.mp4")
    _make_file(tmp_path, ".hidden.mp4")

    files = scan_input(tmp_path, MINIMAL_CONFIG)
    names = [f.name for f in files]
    assert "session" in names
    assert ".hidden" not in names


def test_skip_hidden_dir(tmp_path):
    hidden_dir = tmp_path / ".git"
    hidden_dir.mkdir()
    _make_file(hidden_dir, "config.mp4")
    _make_file(tmp_path, "visible.mp4")

    files = scan_input(tmp_path, MINIMAL_CONFIG)
    names = [f.name for f in files]
    assert "visible" in names
    assert "config" not in names


def test_scan_unknown_skipped_in_folder(tmp_path):
    _make_file(tmp_path, "video.mp4")
    _make_file(tmp_path, "mystery.xyz")

    files = scan_input(tmp_path, MINIMAL_CONFIG)
    # .xyz is unknown → should be skipped in folder scan
    names = [f.name for f in files]
    assert "video" in names
    assert "mystery" not in names


def test_source_file_fields(tmp_path):
    p = _make_file(tmp_path, "training.mp4", content="fake")
    files = scan_input(p, MINIMAL_CONFIG)
    f = files[0]

    assert isinstance(f.path, Path)
    assert f.name == "training"
    assert f.size_bytes == p.stat().st_size
    assert f.type == FileType.VIDEO


def test_invalid_path_raises():
    with pytest.raises(ValueError):
        scan_input(Path("/nonexistent/path/file.mp4"), MINIMAL_CONFIG)
