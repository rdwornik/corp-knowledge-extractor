"""Tests for tier routing logic."""
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.inventory import SourceFile, FileType
from src.tier_router import route_tier, Tier, TierDecision, estimate_batch_cost, TIER_COSTS
from src.text_extract import TextExtractionResult


def _make_file(path: str = "test.pdf", file_type: FileType = FileType.DOCUMENT, size: int = 1000):
    return SourceFile(path=Path(path), type=file_type, size_bytes=size, name=Path(path).stem)


def test_video_always_tier3():
    """Video files always route to Tier 3 multimodal."""
    f = _make_file("meeting.mp4", FileType.VIDEO)
    decision = route_tier(f)
    assert decision.tier == Tier.MULTIMODAL
    assert "video" in decision.reason.lower()


def test_audio_always_tier3():
    """Audio files always route to Tier 3 multimodal."""
    f = _make_file("call.mp3", FileType.AUDIO)
    decision = route_tier(f)
    assert decision.tier == Tier.MULTIMODAL


def test_force_tier_override():
    """force_tier should override automatic routing."""
    f = _make_file("meeting.mp4", FileType.VIDEO)
    decision = route_tier(f, force_tier=1)
    assert decision.tier == Tier.LOCAL
    assert "forced" in decision.reason


@patch("src.tier_router.extract_text")
def test_good_text_routes_tier2(mock_extract):
    """Documents with good text extraction → Tier 2."""
    mock_extract.return_value = TextExtractionResult(
        text="x" * 5000, char_count=5000, extraction_quality="good", extractor="pdfplumber"
    )
    f = _make_file("report.pdf", FileType.DOCUMENT)
    decision = route_tier(f)
    assert decision.tier == Tier.TEXT_AI
    assert decision.text_result is not None


@patch("src.tier_router.extract_text")
def test_partial_text_routes_tier3(mock_extract):
    """Documents with partial text → Tier 3 multimodal."""
    mock_extract.return_value = TextExtractionResult(
        text="x" * 100, char_count=100, extraction_quality="partial", extractor="pdfplumber"
    )
    f = _make_file("scanned.pdf", FileType.DOCUMENT)
    decision = route_tier(f)
    assert decision.tier == Tier.MULTIMODAL


@patch("src.tier_router.extract_text")
def test_no_text_routes_tier3(mock_extract):
    """Documents with no extractable text → Tier 3."""
    mock_extract.return_value = TextExtractionResult(
        text="", char_count=0, extraction_quality="none", extractor="pdfplumber", error="empty"
    )
    f = _make_file("image.pdf", FileType.DOCUMENT)
    decision = route_tier(f)
    assert decision.tier == Tier.MULTIMODAL


@patch("src.tier_router.extract_text")
def test_pptx_always_tier2(mock_extract):
    """PPTX always routes to Tier 2 — Gemini rejects PPTX MIME type."""
    mock_extract.return_value = TextExtractionResult(
        text="x" * 5000, char_count=5000, extraction_quality="good",
        extractor="python-pptx", has_images=True, slide_count=20
    )
    f = _make_file("deck.pptx", FileType.SLIDES)
    decision = route_tier(f)
    assert decision.tier == Tier.TEXT_AI
    assert "blocked" in decision.reason.lower()


@patch("src.tier_router.extract_text")
def test_docx_always_tier2(mock_extract):
    """DOCX always routes to Tier 2 — Gemini rejects DOCX MIME type."""
    mock_extract.return_value = TextExtractionResult(
        text="x" * 5000, char_count=5000, extraction_quality="good",
        extractor="python-docx"
    )
    f = _make_file("report.docx", FileType.DOCUMENT)
    decision = route_tier(f)
    assert decision.tier == Tier.TEXT_AI


@patch("src.tier_router.extract_text")
def test_xlsx_always_tier2(mock_extract):
    """XLSX always routes to Tier 2 — Gemini rejects XLSX MIME type."""
    mock_extract.return_value = TextExtractionResult(
        text="x" * 5000, char_count=5000, extraction_quality="good",
        extractor="openpyxl"
    )
    f = _make_file("data.xlsx", FileType.SPREADSHEET)
    decision = route_tier(f)
    assert decision.tier == Tier.TEXT_AI


@patch("src.tier_router.extract_text")
def test_pptx_tier2_even_with_no_text(mock_extract):
    """PPTX with failed extraction still caps at Tier 2, not Tier 3."""
    mock_extract.return_value = TextExtractionResult(
        text="", char_count=0, extraction_quality="none",
        extractor="python-pptx", error="empty"
    )
    f = _make_file("empty.pptx", FileType.SLIDES)
    decision = route_tier(f)
    assert decision.tier == Tier.TEXT_AI


@patch("src.tier_router.extract_text")
def test_small_note_tier1(mock_extract):
    """Small text notes → Tier 1 (local only)."""
    mock_extract.return_value = TextExtractionResult(
        text="x" * 2000, char_count=2000, extraction_quality="good", extractor="plaintext"
    )
    f = _make_file("notes.txt", FileType.NOTE)
    decision = route_tier(f)
    assert decision.tier == Tier.LOCAL
    assert decision.estimated_cost == 0.0


@patch("src.tier_router.extract_text")
def test_estimate_batch_cost(mock_extract):
    """Batch cost estimation sums per-file costs."""
    mock_extract.return_value = TextExtractionResult(
        text="x" * 5000, char_count=5000, extraction_quality="good", extractor="pdfplumber"
    )
    files = [
        _make_file("a.mp4", FileType.VIDEO),
        _make_file("b.pdf", FileType.DOCUMENT),
        _make_file("c.pdf", FileType.DOCUMENT),
    ]
    estimate = estimate_batch_cost(files)
    assert estimate["file_count"] == 3
    assert estimate["tier_counts"][3] >= 1  # video is always tier 3
    assert estimate["total_cost"] > 0


def test_tier_costs_defined():
    """All tiers should have costs defined."""
    assert TIER_COSTS[Tier.LOCAL] == 0.0
    assert TIER_COSTS[Tier.TEXT_AI] > 0
    assert TIER_COSTS[Tier.MULTIMODAL] > TIER_COSTS[Tier.TEXT_AI]


def test_tier_enum_values():
    """Tier enum should have expected int values."""
    assert Tier.LOCAL == 1
    assert Tier.TEXT_AI == 2
    assert Tier.MULTIMODAL == 3
