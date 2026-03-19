"""Tests for PPTX → PDF converter cascade and multimodal routing."""

import subprocess
from pathlib import Path
from unittest.mock import patch, MagicMock, ANY

import fitz
import pytest

from src.slides.pdf_converter import convert_pptx_to_pdf, _convert_via_com
from src.extract import _render_pdf_to_slides


class TestComScriptContent:
    """Verify the inline COM script has required safety flags."""

    def test_com_script_has_with_window_false(self, tmp_path):
        """Inline script must open without window (PowerPoint rejects Visible=0)."""
        from src.slides.pdf_converter import _convert_via_com
        import inspect
        source = inspect.getsource(_convert_via_com)
        assert "WithWindow=False" in source

    def test_com_script_has_display_alerts_0(self, tmp_path):
        """Inline script must suppress dialogs."""
        from src.slides.pdf_converter import _convert_via_com
        import inspect
        source = inspect.getsource(_convert_via_com)
        assert "DisplayAlerts = 0" in source

    def test_com_script_has_quit(self, tmp_path):
        """Inline script must call ppt.Quit()."""
        from src.slides.pdf_converter import _convert_via_com
        import inspect
        source = inspect.getsource(_convert_via_com)
        assert "Quit()" in source


class TestConverterCascade:
    def test_converter_cascade_com_success(self, tmp_path):
        """COM succeeds → returns PDF path."""
        pptx = tmp_path / "deck.pptx"
        pptx.touch()
        out_dir = tmp_path / "pdf_out"

        def fake_com(pptx_path, pdf_path):
            pdf_path.parent.mkdir(parents=True, exist_ok=True)
            pdf_path.write_bytes(b"%PDF-1.4 fake")
            return pdf_path

        with patch("src.slides.pdf_converter._convert_via_com", side_effect=fake_com):
            result = convert_pptx_to_pdf(pptx, out_dir)

        assert result is not None
        assert result.suffix == ".pdf"
        assert result.exists()

    def test_converter_cascade_com_timeout(self, tmp_path):
        """COM hangs → falls to LibreOffice."""
        pptx = tmp_path / "deck.pptx"
        pptx.touch()
        out_dir = tmp_path / "pdf_out"

        def fake_lo(pptx_path, output_dir):
            pdf = output_dir / f"{pptx_path.stem}.pdf"
            pdf.parent.mkdir(parents=True, exist_ok=True)
            pdf.write_bytes(b"%PDF-1.4 fake")
            return pdf

        with (
            patch("src.slides.pdf_converter._convert_via_com", side_effect=subprocess.TimeoutExpired("cmd", 30)),
            patch("src.slides.pdf_converter._convert_via_libreoffice", side_effect=fake_lo),
        ):
            result = convert_pptx_to_pdf(pptx, out_dir)

        assert result is not None
        assert result.exists()

    def test_converter_cascade_libreoffice(self, tmp_path):
        """COM unavailable → LibreOffice succeeds."""
        pptx = tmp_path / "deck.pptx"
        pptx.touch()
        out_dir = tmp_path / "pdf_out"

        def fake_lo(pptx_path, output_dir):
            pdf = output_dir / f"{pptx_path.stem}.pdf"
            pdf.parent.mkdir(parents=True, exist_ok=True)
            pdf.write_bytes(b"%PDF-1.4 fake")
            return pdf

        with (
            patch("src.slides.pdf_converter._convert_via_com", side_effect=ImportError("No comtypes")),
            patch("src.slides.pdf_converter._convert_via_libreoffice", side_effect=fake_lo),
        ):
            result = convert_pptx_to_pdf(pptx, out_dir)

        assert result is not None

    def test_converter_cascade_all_fail(self, tmp_path):
        """Both fail → returns None."""
        pptx = tmp_path / "deck.pptx"
        pptx.touch()
        out_dir = tmp_path / "pdf_out"

        with (
            patch("src.slides.pdf_converter._convert_via_com", side_effect=ImportError("No comtypes")),
            patch("src.slides.pdf_converter._convert_via_libreoffice", side_effect=FileNotFoundError("No LO")),
        ):
            result = convert_pptx_to_pdf(pptx, out_dir)

        assert result is None


class TestPptxMultimodalRouting:
    """Test that extract_from_text routes PPTX through PDF multimodal."""

    def test_pptx_routes_multimodal_with_pdf(self, tmp_path):
        """PDF available → Tier 3 multimodal extraction used."""
        from unittest.mock import patch, MagicMock
        from src.inventory import SourceFile, FileType
        from src.text_extract import TextExtractionResult
        from src.extract import ExtractionResult

        pptx = tmp_path / "deck.pptx"
        pptx.write_bytes(b"PK\x03\x04fake")
        sf = SourceFile(path=pptx, name="deck.pptx", type=FileType.SLIDES, size_bytes=1000)
        text_result = TextExtractionResult(text="Slide content here", char_count=18, extractor="python-pptx", slide_count=10)

        fake_result = MagicMock(spec=ExtractionResult)
        fake_result.title = "Test Deck"

        with patch("src.extract._try_pptx_pdf_multimodal", return_value=fake_result) as mock_multi:
            from src.extract import extract_from_text
            result = extract_from_text(sf, {"gemini": {"model": "test"}}, text_result)

        mock_multi.assert_called_once()
        assert result == fake_result

    def test_pptx_falls_back_text_only(self, tmp_path):
        """No PDF → Tier 2 text-only with warning."""
        from unittest.mock import patch, MagicMock
        from src.inventory import SourceFile, FileType
        from src.text_extract import TextExtractionResult

        pptx = tmp_path / "deck.pptx"
        pptx.write_bytes(b"PK\x03\x04fake")
        sf = SourceFile(path=pptx, name="deck.pptx", type=FileType.SLIDES, size_bytes=1000)
        text_result = TextExtractionResult(text="Slide content", char_count=13, extractor="python-pptx", slide_count=5)

        # Make PDF multimodal return None (conversion failed)
        with (
            patch("src.extract._try_pptx_pdf_multimodal", return_value=None),
            patch("src.providers.router.route_model", return_value="gemini-3-flash-preview"),
            patch("src.providers.router.get_provider") as mock_provider_fn,
            patch("src.providers.validator.validate_and_retry") as mock_validate,
            patch("src.freshness.compute_freshness_fields", return_value={}),
            patch("src.extract.extract_source_date", return_value=None),
            patch("src.doc_type_classifier.classify_doc_type", return_value="presentation"),
            patch("src.doc_type_classifier.should_extract_deep", return_value=False),
            patch("src.extract.post_process_extraction") as mock_pp,
        ):
            # Set up provider mock
            mock_response = MagicMock()
            mock_response.text = '{"title": "Test", "summary": "Test summary", "topics": [], "type": "presentation", "date": "2026-03-01"}'
            mock_response.input_tokens = 100
            mock_response.output_tokens = 50
            mock_response.cost_estimate = 0.001
            mock_provider = MagicMock()
            mock_provider.extract.return_value = mock_response
            mock_provider_fn.return_value = mock_provider
            mock_validate.return_value = (mock_response, False)

            # Set up post_process mock
            mock_pp_result = MagicMock()
            mock_pp_result.data = {"title": "Test", "summary": "Test summary", "topics": [], "type": "presentation", "date": "2026-03-01", "source_tool": "knowledge-extractor", "source_file": str(pptx)}
            mock_pp_result.links_line = ""
            mock_pp_result.validation_result = MagicMock()
            mock_pp_result.validation_result.value = "valid"
            mock_pp_result.changes = []
            mock_pp.return_value = mock_pp_result

            from src.extract import extract_from_text
            result = extract_from_text(sf, {"gemini": {"model": "test"}, "prompts": {"extract": "Extract knowledge"}}, text_result)

        # Should still produce a result via text-only path
        assert result is not None
        assert result.title == "Test"


class TestRenderPdfToSlides:
    """Test PDF → slide PNG rendering via PyMuPDF."""

    def test_render_pdf_to_slides(self, tmp_path):
        """Create a 1-page PDF, render to PNG, verify PNG exists."""
        pdf_path = tmp_path / "test.pdf"
        doc = fitz.open()
        page = doc.new_page(width=1920, height=1080)
        page.insert_text((100, 100), "Test slide content")
        doc.save(str(pdf_path))
        doc.close()

        slides = _render_pdf_to_slides(pdf_path, tmp_path)

        assert len(slides) == 1
        assert slides[0].exists()
        assert slides[0].name == "slide_001.png"
        assert slides[0].stat().st_size > 0

    def test_slides_dir_created(self, tmp_path):
        """Verify source/slides/ directory is created."""
        pdf_path = tmp_path / "test.pdf"
        doc = fitz.open()
        doc.new_page()
        doc.save(str(pdf_path))
        doc.close()

        _render_pdf_to_slides(pdf_path, tmp_path)

        slides_dir = tmp_path / "slides"
        assert slides_dir.is_dir()

    def test_render_multipage_pdf(self, tmp_path):
        """Multi-page PDF produces one PNG per page."""
        pdf_path = tmp_path / "multi.pdf"
        doc = fitz.open()
        for i in range(3):
            page = doc.new_page()
            page.insert_text((100, 100), f"Page {i + 1}")
        doc.save(str(pdf_path))
        doc.close()

        slides = _render_pdf_to_slides(pdf_path, tmp_path)

        assert len(slides) == 3
        for i, s in enumerate(slides, 1):
            assert s.name == f"slide_{i:03d}.png"
            assert s.exists()
