"""Tests for the cke scan command and scan module."""

import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.scan import (
    scan_file,
    scan_path,
    results_to_json,
    _sha256_file,
    _scan_csv,
    _scan_text,
    _determine_tier,
    FileScanResult,
)


class TestSha256:
    def test_small_file(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("hello world")
        h = _sha256_file(f)
        assert h is not None
        assert h.startswith("sha256:")
        assert len(h) == 71  # "sha256:" + 64 hex chars

    def test_deterministic(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("same content")
        h1 = _sha256_file(f)
        h2 = _sha256_file(f)
        assert h1 == h2

    def test_different_content(self, tmp_path):
        f1 = tmp_path / "a.txt"
        f2 = tmp_path / "b.txt"
        f1.write_text("content A")
        f2.write_text("content B")
        assert _sha256_file(f1) != _sha256_file(f2)

    def test_skip_large_file(self, tmp_path):
        f = tmp_path / "big.bin"
        f.write_bytes(b"x")
        with patch("src.scan.MAX_HASH_SIZE", 0):
            assert _sha256_file(f) is None


class TestScanText:
    def test_basic(self, tmp_path):
        f = tmp_path / "notes.txt"
        f.write_text("Line 1\nLine 2\nLine 3")
        meta = _scan_text(f)
        assert meta["text_chars"] == 20
        assert meta["line_count"] == 3
        assert "Line 1" in meta["text_preview"]

    def test_empty_file(self, tmp_path):
        f = tmp_path / "empty.txt"
        f.write_text("")
        meta = _scan_text(f)
        assert meta["text_chars"] == 0


class TestScanCsv:
    def test_basic(self, tmp_path):
        f = tmp_path / "data.csv"
        f.write_text("name,age,city\nAlice,30,NYC\nBob,25,LA\nCharlie,35,CHI\nDave,40,SF")
        meta = _scan_csv(f)
        assert meta["row_count"] == 5
        assert meta["headers"] == ["name", "age", "city"]
        assert len(meta["sample_rows"]) == 3


class TestDetermineTier:
    def test_video_tier3(self):
        assert _determine_tier(".mp4", {}) == 3

    def test_audio_tier3(self):
        assert _determine_tier(".mp3", {}) == 3

    def test_text_with_content_tier1(self):
        assert _determine_tier(".pdf", {"text_chars": 500}) == 1

    def test_text_no_content_tier2(self):
        assert _determine_tier(".pdf", {"text_chars": 0}) == 2

    def test_unknown_extension_tier1(self):
        assert _determine_tier(".yaml", {}) == 1


class TestScanFile:
    def test_text_file(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("Hello world content here")
        result = scan_file(f)
        assert result.filename == "test.txt"
        assert result.extension == ".txt"
        assert result.size_bytes > 0
        assert result.file_hash is not None
        assert result.metadata["text_chars"] > 0
        assert result.tier == 1
        assert result.error is None

    def test_csv_file(self, tmp_path):
        f = tmp_path / "data.csv"
        f.write_text("a,b\n1,2\n3,4")
        result = scan_file(f)
        assert result.extension == ".csv"
        assert result.metadata["row_count"] == 3
        assert result.metadata["headers"] == ["a", "b"]

    def test_relative_path(self, tmp_path):
        f = tmp_path / "subdir" / "test.md"
        f.parent.mkdir()
        f.write_text("# Hello")
        result = scan_file(f, base_path=tmp_path)
        assert result.path == "subdir/test.md"

    def test_unsupported_extension(self, tmp_path):
        f = tmp_path / "data.xyz"
        f.write_bytes(b"binary")
        result = scan_file(f)
        assert result.metadata == {}
        assert result.error is None

    def test_pptx_mock(self, tmp_path):
        f = tmp_path / "test.pptx"
        f.write_bytes(b"fake")

        mock_prs = MagicMock()
        mock_slide = MagicMock()
        mock_shape = MagicMock()
        mock_shape.has_text_frame = True
        mock_para = MagicMock()
        mock_para.text = "Slide Title"
        mock_shape.text_frame.paragraphs = [mock_para]
        mock_slide.shapes = [mock_shape]
        mock_slide.has_notes_slide = False
        mock_prs.slides = [mock_slide]

        with patch("src.scan.Presentation", mock_prs.__class__, create=True):
            with patch("pptx.Presentation", return_value=mock_prs):
                result = scan_file(f)
        assert result.extension == ".pptx"

    def test_extraction_error_handled(self, tmp_path):
        f = tmp_path / "bad.pdf"
        f.write_bytes(b"not a pdf")
        result = scan_file(f)
        # Should not crash — returns result with error
        assert result.filename == "bad.pdf"
        assert result.error is not None or result.metadata == {}


class TestScanPath:
    def test_single_file(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("hello")
        results = scan_path(f)
        assert len(results) == 1
        assert results[0].filename == "test.txt"

    def test_directory(self, tmp_path):
        (tmp_path / "a.txt").write_text("file a")
        (tmp_path / "b.txt").write_text("file b")
        results = scan_path(tmp_path)
        assert len(results) == 2

    def test_recursive(self, tmp_path):
        (tmp_path / "top.txt").write_text("top")
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "deep.txt").write_text("deep")
        results = scan_path(tmp_path, recursive=True)
        assert len(results) == 2

    def test_no_recursive(self, tmp_path):
        (tmp_path / "top.txt").write_text("top")
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "deep.txt").write_text("deep")
        results = scan_path(tmp_path, recursive=False)
        assert len(results) == 1

    def test_exclude_folders(self, tmp_path):
        (tmp_path / "keep.txt").write_text("keep")
        skip = tmp_path / "node_modules"
        skip.mkdir()
        (skip / "lib.txt").write_text("skip me")
        results = scan_path(tmp_path, exclude=("node_modules",))
        assert len(results) == 1
        assert results[0].filename == "keep.txt"

    def test_skip_hidden(self, tmp_path):
        (tmp_path / "visible.txt").write_text("yes")
        hidden = tmp_path / ".hidden"
        hidden.mkdir()
        (hidden / "secret.txt").write_text("no")
        results = scan_path(tmp_path)
        assert len(results) == 1

    def test_nonexistent_path(self):
        with pytest.raises(ValueError, match="does not exist"):
            scan_path(Path("/nonexistent/path"))


class TestResultsToJson:
    def test_structure(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("content")
        results = scan_path(f)
        data = results_to_json(results)
        assert "scan_date" in data
        assert data["total_files"] == 1
        assert len(data["results"]) == 1
        r = data["results"][0]
        assert "path" in r
        assert "filename" in r
        assert "extension" in r
        assert "size_bytes" in r
        assert "file_hash" in r
        assert "tier" in r
        assert "metadata" in r

    def test_json_serializable(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("hello")
        results = scan_path(f)
        data = results_to_json(results)
        # Should not raise
        json_str = json.dumps(data, default=str)
        parsed = json.loads(json_str)
        assert parsed["total_files"] == 1
