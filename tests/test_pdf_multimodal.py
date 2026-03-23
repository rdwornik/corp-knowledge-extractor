"""Tests for PDF multimodal extraction pipeline."""

import json
from pathlib import Path
from unittest.mock import patch, MagicMock, PropertyMock


# ---------------------------------------------------------------------------
# STEP 1: Router — all PDFs → Pro
# ---------------------------------------------------------------------------


class TestSelectModelPdf:
    def test_pdf_always_pro(self):
        """Any .pdf → gemini-3.1-pro-preview regardless of size or images."""
        from src.providers.router import select_model

        model, reason = select_model(Path("report.pdf"), file_size=100_000)
        assert model == "gemini-3.1-pro-preview"

    def test_pdf_reason(self):
        """routing_reason for PDF is pdf_multimodal."""
        from src.providers.router import select_model

        _, reason = select_model(Path("report.pdf"), file_size=100_000)
        assert reason == "pdf_multimodal"

    def test_pdf_no_images_flag_still_pro(self):
        """PDF routes to Pro even without has_images flag."""
        from src.providers.router import select_model

        model, _ = select_model(Path("report.pdf"), file_size=500_000, has_images=False)
        assert model == "gemini-3.1-pro-preview"

    def test_pdf_override_wins(self):
        """Manual model override still takes precedence."""
        from src.providers.router import select_model

        model, reason = select_model(Path("report.pdf"), file_size=100_000, model_override="flash")
        assert model == "gemini-3-flash-preview"
        assert reason == "manual_override"

    def test_small_pdf_still_local(self):
        """PDFs under 5000 bytes still route to local (free)."""
        from src.providers.router import select_model

        model, reason = select_model(Path("tiny.pdf"), file_size=3000)
        assert model == "free"
        assert reason == "small_file_local"


# ---------------------------------------------------------------------------
# STEP 2: PDF multimodal extraction
# ---------------------------------------------------------------------------


def _make_fake_gemini_response(text: str, tokens: int = 500):
    """Create a mock Gemini response object."""
    resp = MagicMock()
    resp.text = text
    resp.usage_metadata = MagicMock()
    resp.usage_metadata.total_token_count = tokens
    resp.usage_metadata.prompt_token_count = 400
    resp.usage_metadata.candidates_token_count = 100
    return resp


def _mock_pdf_extraction_deps():
    """Return a dict of common patches for _try_pdf_multimodal."""
    return {
        "src.doc_type_classifier.classify_doc_type": MagicMock(return_value="general"),
        "src.doc_type_classifier.should_extract_deep": MagicMock(return_value=False),
        "src.freshness.compute_freshness_fields": MagicMock(return_value={}),
        "src.extract.extract_source_date": MagicMock(return_value=None),
        "src.extract._enrich_facts": MagicMock(return_value=[]),
        "src.extract._haiku_enrichment": MagicMock(return_value=[]),
        "src.extract.get_taxonomy_for_prompt": MagicMock(return_value="TAXONOMY"),
    }


class TestPdfMultimodalUpload:
    def test_pdf_multimodal_returns_result(self, tmp_path):
        """Mock Gemini upload + generate → returns ExtractionResult."""
        from src.inventory import SourceFile, FileType
        from src.text_extract import TextExtractionResult
        from src.extract import _try_pdf_multimodal

        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 dummy content for testing " * 200)

        source = SourceFile(path=pdf_file, type=FileType.DOCUMENT, size_bytes=pdf_file.stat().st_size, name="test")
        text_result = TextExtractionResult(text="Sample PDF text content.", char_count=24, extractor="pdfplumber")

        response_json = json.dumps({
            "title": "Test PDF Report",
            "summary": "A test document.",
            "topics": ["Testing"],
            "products": [],
            "people": [],
            "key_points": ["Point one"],
        })

        mock_client = MagicMock()
        mock_uploaded = MagicMock()
        mock_uploaded.uri = "gs://test/file"
        mock_uploaded.mime_type = "application/pdf"
        mock_uploaded.state.name = "ACTIVE"
        mock_client.files.upload.return_value = mock_uploaded
        mock_client.files.get.return_value = mock_uploaded
        mock_client.models.generate_content.return_value = _make_fake_gemini_response(response_json)

        patches = _mock_pdf_extraction_deps()
        patches["src.extract._get_client"] = MagicMock(return_value=mock_client)
        patches["src.providers.router.select_model"] = MagicMock(return_value=("gemini-3.1-pro-preview", "pdf_multimodal"))

        mock_fitz = MagicMock()
        mock_fitz.open.return_value.__len__ = MagicMock(return_value=5)

        patch_stack = [patch(t, m) for t, m in patches.items()]
        fitz_patch = patch.dict("sys.modules", {"fitz": mock_fitz})

        for p in patch_stack:
            p.start()
        fitz_patch.start()
        try:
            result = _try_pdf_multimodal(source, {"prompts": {"extract": "Extract."}}, text_result)
        finally:
            fitz_patch.stop()
            for p in reversed(patch_stack):
                p.stop()

        assert result is not None
        assert result.title == "Test PDF Report"
        assert result.model_used == "gemini-3.1-pro-preview"
        assert result.routing_reason == "pdf_multimodal"


class TestPdfMultimodalFallback:
    def test_fallback_to_text_on_failure(self):
        """When PDF multimodal fails, extract_from_text falls back to text-only."""
        from src.extract import extract_from_text
        from src.inventory import SourceFile, FileType
        from src.text_extract import TextExtractionResult
        from src.providers.base import ExtractionResponse

        source = SourceFile(path=Path("report.pdf"), type=FileType.DOCUMENT, size_bytes=50000, name="report")
        text_result = TextExtractionResult(text="Content", char_count=7, extractor="pdfplumber")

        class FakeProvider:
            def extract(self, request):
                return ExtractionResponse(
                    text='{"title":"T","summary":"S","topics":[],"products":[],"people":[]}',
                    input_tokens=100, output_tokens=50,
                    model="claude-haiku-4-5-20251001", provider="anthropic",
                    cost_estimate=0.001,
                )

        with (
            # _try_pdf_multimodal raises → fallback
            patch("src.extract._try_pdf_multimodal", side_effect=Exception("upload failed")),
            patch("src.doc_type_classifier.classify_doc_type", return_value="general"),
            patch("src.doc_type_classifier.should_extract_deep", return_value=False),
            patch("src.freshness.compute_freshness_fields", return_value={}),
            patch("src.extract.extract_source_date", return_value=None),
            patch("src.extract._enrich_facts", return_value=[]),
            patch("src.extract.get_taxonomy_for_prompt", return_value="TAX"),
            patch("src.providers.router.route_model", return_value=("claude-haiku-4-5-20251001", "text_default")),
            patch("src.providers.router.get_provider", return_value=FakeProvider()),
            patch("src.providers.router.has_anthropic_key", return_value=True),
            patch("src.providers.validator.validate_and_retry", side_effect=lambda r, req: (r, False)),
        ):
            result = extract_from_text(source, {"prompts": {"extract": "Extract."}}, text_result)

        assert result is not None
        assert result.title == "T"


class TestPdfLargeTruncation:
    def test_truncation_flag_set(self, tmp_path):
        """100-page PDF → multimodal_truncated: true in raw_json."""
        from src.extract import _try_pdf_multimodal
        from src.inventory import SourceFile, FileType
        from src.text_extract import TextExtractionResult

        pdf_file = tmp_path / "big.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 big " * 5000)

        source = SourceFile(path=pdf_file, type=FileType.DOCUMENT, size_bytes=pdf_file.stat().st_size, name="big")
        text_result = TextExtractionResult(text="Big doc text.", char_count=13, extractor="pdfplumber")

        response_json = json.dumps({
            "title": "Big PDF", "summary": "Big.", "topics": [], "products": [], "people": [],
        })

        mock_client = MagicMock()
        mock_uploaded = MagicMock()
        mock_uploaded.uri = "gs://test/big"
        mock_uploaded.state.name = "ACTIVE"
        mock_client.files.upload.return_value = mock_uploaded
        mock_client.files.get.return_value = mock_uploaded
        mock_client.models.generate_content.return_value = _make_fake_gemini_response(response_json)

        # Mock fitz to report 100 pages and support truncation
        mock_doc = MagicMock()
        mock_doc.__len__ = MagicMock(return_value=100)
        mock_doc.__getitem__ = MagicMock(return_value=MagicMock())

        mock_truncated = MagicMock()
        mock_fitz = MagicMock()
        mock_fitz.open.side_effect = [mock_doc, mock_truncated, mock_doc]  # page count, truncate, cover
        mock_fitz.Matrix.return_value = MagicMock()

        # Mock cover pixmap
        mock_pix = MagicMock()
        mock_doc.__getitem__.return_value.get_pixmap.return_value = mock_pix

        patches = _mock_pdf_extraction_deps()
        patches["src.extract._get_client"] = MagicMock(return_value=mock_client)
        patches["src.providers.router.select_model"] = MagicMock(return_value=("gemini-3.1-pro-preview", "pdf_multimodal"))

        patch_stack = [patch(t, m) for t, m in patches.items()]
        # fitz needs special handling since it's imported inside the function
        fitz_patch = patch.dict("sys.modules", {"fitz": mock_fitz})

        for p in patch_stack:
            p.start()
        fitz_patch.start()
        try:
            result = _try_pdf_multimodal(source, {"prompts": {"extract": "Extract."}}, text_result)
        finally:
            fitz_patch.stop()
            for p in reversed(patch_stack):
                p.stop()

        assert result is not None
        assert result.raw_json.get("multimodal_truncated") is True


class TestPdfTextGrounding:
    def test_pdfplumber_text_in_prompt(self, tmp_path):
        """Text from pdfplumber is included in the prompt as grounding."""
        from src.extract import _try_pdf_multimodal
        from src.inventory import SourceFile, FileType
        from src.text_extract import TextExtractionResult

        pdf_file = tmp_path / "grounded.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 content " * 200)

        source = SourceFile(path=pdf_file, type=FileType.DOCUMENT, size_bytes=pdf_file.stat().st_size, name="grounded")
        text_result = TextExtractionResult(text="Pdfplumber extracted this text.", char_count=31, extractor="pdfplumber")

        response_json = json.dumps({
            "title": "Grounded", "summary": "S", "topics": [], "products": [], "people": [],
        })

        captured_prompt = {}
        mock_client = MagicMock()
        mock_uploaded = MagicMock()
        mock_uploaded.uri = "gs://test/g"
        mock_uploaded.state.name = "ACTIVE"
        mock_client.files.upload.return_value = mock_uploaded
        mock_client.files.get.return_value = mock_uploaded

        def capture_generate(**kwargs):
            contents = kwargs.get("contents", [])
            for part in contents:
                if hasattr(part, "text") and part.text:
                    captured_prompt["text"] = part.text
            return _make_fake_gemini_response(response_json)

        mock_client.models.generate_content.side_effect = capture_generate

        patches = _mock_pdf_extraction_deps()
        patches["src.extract._get_client"] = MagicMock(return_value=mock_client)
        patches["src.providers.router.select_model"] = MagicMock(return_value=("gemini-3.1-pro-preview", "pdf_multimodal"))

        mock_fitz = MagicMock()
        mock_fitz.open.return_value.__len__ = MagicMock(return_value=5)
        mock_fitz.open.return_value.__enter__ = MagicMock(return_value=mock_fitz.open.return_value)
        mock_fitz.open.return_value.__exit__ = MagicMock(return_value=False)

        patch_stack = [patch(t, m) for t, m in patches.items()]
        fitz_patch = patch.dict("sys.modules", {"fitz": mock_fitz})

        for p in patch_stack:
            p.start()
        fitz_patch.start()
        try:
            result = _try_pdf_multimodal(source, {"prompts": {"extract": "Extract."}}, text_result)
        finally:
            fitz_patch.stop()
            for p in reversed(patch_stack):
                p.stop()

        assert "pdfplumber" in captured_prompt.get("text", "").lower() or "Pdfplumber extracted this text" in captured_prompt.get("text", "")


# ---------------------------------------------------------------------------
# STEP 3: Cover page (tested via mock fitz in TestPdfLargeTruncation)
# ---------------------------------------------------------------------------

class TestPdfCoverRendered:
    def test_cover_in_slide_image_paths(self, tmp_path):
        """PDF extraction sets slide_image_paths with cover PNG."""
        from src.extract import _try_pdf_multimodal
        from src.inventory import SourceFile, FileType
        from src.text_extract import TextExtractionResult
        import fitz as real_fitz

        # Create a real single-page PDF with PyMuPDF
        try:
            doc = real_fitz.open()
            page = doc.new_page()
            page.insert_text((72, 72), "Cover page test")
            pdf_file = tmp_path / "cover_test.pdf"
            doc.save(str(pdf_file))
            doc.close()
        except Exception:
            import pytest
            pytest.skip("PyMuPDF not available for real PDF creation")

        source = SourceFile(path=pdf_file, type=FileType.DOCUMENT, size_bytes=pdf_file.stat().st_size, name="cover_test")
        text_result = TextExtractionResult(text="Cover page test", char_count=15, extractor="pdfplumber")

        response_json = json.dumps({
            "title": "Cover Test", "summary": "S", "topics": [], "products": [], "people": [],
        })

        mock_client = MagicMock()
        mock_uploaded = MagicMock()
        mock_uploaded.uri = "gs://test/cover"
        mock_uploaded.state.name = "ACTIVE"
        mock_client.files.upload.return_value = mock_uploaded
        mock_client.files.get.return_value = mock_uploaded
        mock_client.models.generate_content.return_value = _make_fake_gemini_response(response_json)

        patches = _mock_pdf_extraction_deps()
        patches["src.extract._get_client"] = MagicMock(return_value=mock_client)
        patches["src.providers.router.select_model"] = MagicMock(return_value=("gemini-3.1-pro-preview", "pdf_multimodal"))

        patch_stack = [patch(t, m) for t, m in patches.items()]
        for p in patch_stack:
            p.start()
        try:
            result = _try_pdf_multimodal(source, {"prompts": {"extract": "Extract."}}, text_result)
        finally:
            for p in reversed(patch_stack):
                p.stop()

        assert result is not None
        assert len(result.slide_image_paths) == 1
        assert result.slide_image_paths[0].name == "cover_001.png"
        assert result.slide_image_paths[0].exists()


# ---------------------------------------------------------------------------
# STEP 4: Haiku enrichment on PDF
# ---------------------------------------------------------------------------

class TestPdfHaikuEnrichment:
    def test_haiku_enrichment_called_for_pdf(self, tmp_path):
        """PDF multimodal path calls _haiku_enrichment."""
        from src.extract import _try_pdf_multimodal
        from src.inventory import SourceFile, FileType
        from src.text_extract import TextExtractionResult

        pdf_file = tmp_path / "enriched.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 enriched " * 200)

        source = SourceFile(path=pdf_file, type=FileType.DOCUMENT, size_bytes=pdf_file.stat().st_size, name="enriched")
        text_result = TextExtractionResult(text="Source text for enrichment.", char_count=27, extractor="pdfplumber")

        response_json = json.dumps({
            "title": "Enriched", "summary": "S", "topics": [], "products": [], "people": [],
        })

        mock_client = MagicMock()
        mock_uploaded = MagicMock()
        mock_uploaded.uri = "gs://test/e"
        mock_uploaded.state.name = "ACTIVE"
        mock_client.files.upload.return_value = mock_uploaded
        mock_client.files.get.return_value = mock_uploaded
        mock_client.models.generate_content.return_value = _make_fake_gemini_response(response_json)

        mock_haiku = MagicMock(return_value=[{"fact": "Extra fact", "verification_status": "verified", "source_extractor": "haiku_enrichment"}])

        patches = _mock_pdf_extraction_deps()
        patches["src.extract._get_client"] = MagicMock(return_value=mock_client)
        patches["src.providers.router.select_model"] = MagicMock(return_value=("gemini-3.1-pro-preview", "pdf_multimodal"))
        patches["src.extract._haiku_enrichment"] = mock_haiku

        mock_fitz = MagicMock()
        mock_fitz.open.return_value.__len__ = MagicMock(return_value=5)

        patch_stack = [patch(t, m) for t, m in patches.items()]
        fitz_patch = patch.dict("sys.modules", {"fitz": mock_fitz})

        for p in patch_stack:
            p.start()
        fitz_patch.start()
        try:
            result = _try_pdf_multimodal(source, {"prompts": {"extract": "Extract."}}, text_result)
        finally:
            fitz_patch.stop()
            for p in reversed(patch_stack):
                p.stop()

        mock_haiku.assert_called_once()
        # Verify source_text kwarg was passed (from pdfplumber)
        call_args = mock_haiku.call_args
        # _haiku_enrichment is called with keyword args
        source_text = call_args.kwargs.get("source_text", "")
        assert "Source text for enrichment" in source_text


# ---------------------------------------------------------------------------
# STEP 5: Prompt has PDF instructions
# ---------------------------------------------------------------------------

class TestPromptPdfInstructions:
    def test_deep_multimodal_has_pdf_instructions(self):
        """deep_multimodal.txt contains PDF-specific instructions."""
        prompt_path = Path(__file__).parent.parent / "config" / "prompts" / "deep_multimodal.txt"
        content = prompt_path.read_text(encoding="utf-8")
        assert "PDF document pages" in content

    def test_deep_multimodal_has_diagram_instruction(self):
        prompt_path = Path(__file__).parent.parent / "config" / "prompts" / "deep_multimodal.txt"
        content = prompt_path.read_text(encoding="utf-8")
        assert "architecture diagrams" in content

    def test_deep_multimodal_has_table_instruction(self):
        prompt_path = Path(__file__).parent.parent / "config" / "prompts" / "deep_multimodal.txt"
        content = prompt_path.read_text(encoding="utf-8")
        assert "table structures" in content
