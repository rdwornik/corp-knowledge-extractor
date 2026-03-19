"""Tests for Haiku enrichment pass — additional fact extraction from source text."""

import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from src.extract import _haiku_enrichment
from src.inventory import SourceFile, FileType
from src.text_extract import TextExtractionResult
from src.providers.base import ExtractionResponse


def _make_source_file(name="test.pptx"):
    return SourceFile(path=Path(name), name=name, type=FileType.SLIDES, size_bytes=100)


def _make_text_result(text="Source text here", char_count=15):
    return TextExtractionResult(text=text, char_count=char_count, extractor="python-pptx", slide_count=5)


def _mock_haiku_response(facts: list[str]) -> ExtractionResponse:
    return ExtractionResponse(
        text=json.dumps(facts),
        input_tokens=500,
        output_tokens=100,
        model="claude-haiku-4-5-20251001",
        provider="anthropic",
        cost_estimate=0.001,
    )


class TestHaikuEnrichment:
    def test_enrichment_adds_new_facts(self):
        """Haiku returns 3 new facts → added to result."""
        existing = ["Fact A", "Fact B", "Fact C", "Fact D", "Fact E"]
        new_facts = ["New fact 1 with 42 items", "New fact 2 about SAP", "New fact 3 in 2025"]
        source_text = "Source has 42 items and SAP integration since 2025."

        mock_response = _mock_haiku_response(new_facts)
        mock_provider = MagicMock()
        mock_provider.extract.return_value = mock_response

        with patch("src.providers.router.get_provider", return_value=mock_provider):
            result = _haiku_enrichment(
                existing_facts=existing,
                source_text=source_text,
                source_file=_make_source_file(),
                source_date="2026-01-01",
                text_result=_make_text_result(source_text),
            )

        assert len(result) == 3
        assert result[0]["fact"] == "New fact 1 with 42 items"

    def test_enrichment_empty_response(self):
        """Haiku returns [] → no new facts."""
        mock_response = _mock_haiku_response([])
        mock_provider = MagicMock()
        mock_provider.extract.return_value = mock_response

        with patch("src.providers.router.get_provider", return_value=mock_provider):
            result = _haiku_enrichment(
                existing_facts=["Existing fact"],
                source_text="Some source text",
                source_file=_make_source_file(),
                source_date="2026-01-01",
                text_result=_make_text_result(),
            )

        assert result == []

    def test_enrichment_failure_continues(self):
        """Haiku throws exception → empty list, no crash."""
        mock_provider = MagicMock()
        mock_provider.extract.side_effect = RuntimeError("API error")

        with patch("src.providers.router.get_provider", return_value=mock_provider):
            result = _haiku_enrichment(
                existing_facts=["Existing fact"],
                source_text="Some source text",
                source_file=_make_source_file(),
                source_date="2026-01-01",
                text_result=_make_text_result(),
            )

        assert result == []

    def test_enrichment_tagged(self):
        """New facts have source_extractor='haiku_enrichment'."""
        new_facts = ["Revenue grew 15% in Q4"]
        source_text = "Revenue grew 15% in Q4 compared to prior year."

        mock_response = _mock_haiku_response(new_facts)
        mock_provider = MagicMock()
        mock_provider.extract.return_value = mock_response

        with patch("src.providers.router.get_provider", return_value=mock_provider):
            result = _haiku_enrichment(
                existing_facts=["Existing fact"],
                source_text=source_text,
                source_file=_make_source_file(),
                source_date="2026-01-01",
                text_result=_make_text_result(source_text),
            )

        assert len(result) == 1
        assert result[0]["source_extractor"] == "haiku_enrichment"

    def test_enrichment_validated(self):
        """New facts get verification_status from fact validation."""
        new_facts = ["1706 customers served"]
        source_text = "The platform serves 1,706 customers across regions."

        mock_response = _mock_haiku_response(new_facts)
        mock_provider = MagicMock()
        mock_provider.extract.return_value = mock_response

        with patch("src.providers.router.get_provider", return_value=mock_provider):
            result = _haiku_enrichment(
                existing_facts=["Existing fact"],
                source_text=source_text,
                source_file=_make_source_file(),
                source_date="2026-01-01",
                text_result=_make_text_result(source_text),
            )

        assert len(result) == 1
        assert "verification_status" in result[0]
        assert result[0]["verification_status"] == "verified"

    def test_enrichment_no_source_text(self):
        """No source text → skip enrichment."""
        result = _haiku_enrichment(
            existing_facts=["Fact"],
            source_text="",
            source_file=_make_source_file(),
            source_date=None,
            text_result=None,
        )
        assert result == []

    def test_enrichment_no_existing_facts(self):
        """No existing facts → skip enrichment."""
        result = _haiku_enrichment(
            existing_facts=[],
            source_text="Some text",
            source_file=_make_source_file(),
            source_date=None,
            text_result=None,
        )
        assert result == []
