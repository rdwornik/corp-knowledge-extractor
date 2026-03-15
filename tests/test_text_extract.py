"""Tests for local text extraction (Tier 1)."""

import pytest
from pathlib import Path
from src.text_extract import extract_text, TextExtractionResult, _assess_quality


def test_assess_quality_good():
    assert _assess_quality("x" * 500) == "good"


def test_assess_quality_partial_short():
    assert _assess_quality("x" * 100) == "partial"


def test_assess_quality_partial_sparse_pages():
    assert _assess_quality("x" * 200, page_count=10) == "partial"


def test_assess_quality_none():
    assert _assess_quality("") == "none"


def test_extract_plaintext(tmp_path):
    """Plain text files should extract with good quality."""
    f = tmp_path / "note.txt"
    f.write_text("This is a test note with enough content to be considered good quality. " * 10)
    result = extract_text(f)
    assert result.extraction_quality == "good"
    assert result.extractor == "plaintext"
    assert "test note" in result.text
    assert result.char_count > 200


def test_extract_markdown(tmp_path):
    """Markdown files should use plaintext extractor."""
    f = tmp_path / "doc.md"
    f.write_text("# Title\n\nSome content here that is long enough for extraction. " * 10)
    result = extract_text(f)
    assert result.extraction_quality == "good"
    assert result.extractor == "plaintext"


def test_extract_csv(tmp_path):
    """CSV files should extract with good quality."""
    f = tmp_path / "data.csv"
    f.write_text(
        "name,value,description\n"
        + "\n".join(f"item_{i},{i},Description of item {i} with enough text" for i in range(20))
    )
    result = extract_text(f)
    assert result.extraction_quality == "good"
    assert result.extractor == "csv"
    assert "item_0" in result.text


def test_extract_unsupported():
    """Unsupported extensions should return none quality."""
    result = extract_text(Path("video.mp4"))
    assert result.extraction_quality == "none"
    assert result.extractor == "unsupported"


def test_extract_nonexistent_file(tmp_path):
    """Missing file should return error, not crash."""
    result = extract_text(tmp_path / "missing.txt")
    assert result.extraction_quality == "none"
    assert result.error is not None


def test_extract_empty_file(tmp_path):
    """Empty file should return none quality."""
    f = tmp_path / "empty.txt"
    f.write_text("")
    result = extract_text(f)
    assert result.extraction_quality == "none"


def test_result_fields():
    """TextExtractionResult should have all expected fields."""
    r = TextExtractionResult(text="test", char_count=4, extraction_quality="good", extractor="test")
    assert r.page_count == 0
    assert r.slide_count == 0
    assert r.has_images is False
    assert r.error is None
