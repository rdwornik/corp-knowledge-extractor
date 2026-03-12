"""Tests for locator building and fact enrichment."""

import pytest
from unittest.mock import MagicMock
from pathlib import Path

from src.extract import _build_locator, _enrich_facts
from src.inventory import SourceFile, FileType
from src.text_extract import TextExtractionResult


class TestBuildLocator:
    def test_pdf_page(self):
        loc = _build_locator(3, ".pdf", max_pages=10)
        assert loc == {"type": "pdf", "page": 3}

    def test_pptx_slide(self):
        loc = _build_locator(5, ".pptx", max_pages=20)
        assert loc == {"type": "slide", "number": 5}

    def test_docx_section(self):
        loc = _build_locator(2, ".docx", max_pages=10)
        assert loc == {"type": "section", "number": 2}

    def test_none_page_ref(self):
        assert _build_locator(None, ".pdf", max_pages=10) is None

    def test_invalid_string(self):
        assert _build_locator("abc", ".pdf", max_pages=10) is None

    def test_zero_page(self):
        assert _build_locator(0, ".pdf", max_pages=10) is None

    def test_negative_page(self):
        assert _build_locator(-1, ".pdf", max_pages=10) is None

    def test_exceeds_max_pages(self):
        assert _build_locator(11, ".pdf", max_pages=10) is None

    def test_at_max_pages(self):
        loc = _build_locator(10, ".pdf", max_pages=10)
        assert loc == {"type": "pdf", "page": 10}

    def test_no_max_pages_constraint(self):
        # max_pages=0 means no constraint
        loc = _build_locator(999, ".pdf", max_pages=0)
        assert loc == {"type": "pdf", "page": 999}

    def test_unsupported_extension(self):
        assert _build_locator(1, ".txt", max_pages=10) is None

    def test_string_number(self):
        loc = _build_locator("7", ".pdf", max_pages=10)
        assert loc == {"type": "pdf", "page": 7}


class TestEnrichFacts:
    def _make_source_file(self, name="test.pdf"):
        sf = MagicMock(spec=SourceFile)
        sf.path = Path(f"/tmp/{name}")
        return sf

    def _make_text_result(self, page_count=10, slide_count=0):
        return TextExtractionResult(
            text="sample text",
            char_count=100,
            page_count=page_count,
            slide_count=slide_count,
        )

    def test_structured_facts_with_pages(self):
        sf = self._make_source_file("report.pdf")
        data = {
            "facts": [
                {"fact": "WMS supports REST API", "page": 3},
                {"fact": "System does not support legacy mode", "page": 7},
            ]
        }
        result = _enrich_facts(data, sf, "2025-03", self._make_text_result())
        assert len(result) == 2
        assert result[0]["fact"] == "WMS supports REST API"
        assert result[0]["source_date"] == "2025-03"
        assert result[0]["locator"] == {"type": "pdf", "page": 3}
        assert result[0]["polarity"] == "positive"
        # "does not support" → negative + positive overlap → unknown
        assert result[1]["polarity"] == "unknown"

    def test_structured_facts_with_slides(self):
        sf = self._make_source_file("deck.pptx")
        data = {
            "facts": [
                {"fact": "Platform provides multi-tenant support", "slide": 2},
            ]
        }
        tr = self._make_text_result(page_count=0, slide_count=15)
        result = _enrich_facts(data, sf, "2025-01", tr)
        assert result[0]["locator"] == {"type": "slide", "number": 2}
        assert result[0]["polarity"] == "positive"

    def test_fallback_to_key_points(self):
        sf = self._make_source_file("notes.pdf")
        data = {
            "key_points": [
                "WMS supports batch processing",
                "Data goes through validation pipeline",
            ]
        }
        result = _enrich_facts(data, sf, None, self._make_text_result())
        assert len(result) == 2
        assert result[0]["fact"] == "WMS supports batch processing"
        assert result[0]["source_date"] is None
        assert result[0]["locator"] is None  # no page info in key_points
        assert result[0]["polarity"] == "positive"
        assert result[1]["polarity"] == "unknown"

    def test_empty_facts_and_key_points(self):
        sf = self._make_source_file("empty.pdf")
        data = {}
        result = _enrich_facts(data, sf, "2025-03", self._make_text_result())
        assert result == []

    def test_invalid_page_ref_filtered(self):
        sf = self._make_source_file("report.pdf")
        data = {
            "facts": [
                {"fact": "Some fact", "page": 999},  # exceeds max
            ]
        }
        result = _enrich_facts(data, sf, "2025-03", self._make_text_result(page_count=10))
        assert result[0]["locator"] is None  # invalid page filtered out

    def test_no_text_result(self):
        sf = self._make_source_file("video.mp4")
        data = {
            "facts": [
                {"fact": "Platform offers real-time analytics", "page": 1},
            ]
        }
        result = _enrich_facts(data, sf, "2025-06", None)
        assert len(result) == 1
        assert result[0]["source_date"] == "2025-06"
        # No max_pages validation without text_result (max_pages=0)
        assert result[0]["polarity"] == "positive"

    def test_fact_text_field_alternative(self):
        sf = self._make_source_file("doc.pdf")
        data = {
            "facts": [
                {"text": "System provides audit trail", "page": 2},
            ]
        }
        result = _enrich_facts(data, sf, "2025-03", self._make_text_result())
        assert result[0]["fact"] == "System provides audit trail"
        assert result[0]["polarity"] == "positive"
