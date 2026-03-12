"""Tests for source date extraction from file metadata."""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from src.text_extract import extract_source_date


class TestPdfSourceDate:
    def test_creation_date(self, tmp_path):
        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(b"fake")

        mock_pdf = MagicMock()
        mock_pdf.metadata = {"CreationDate": "D:20250315120000"}
        mock_pdf.__enter__ = lambda s: s
        mock_pdf.__exit__ = MagicMock(return_value=False)

        with patch("pdfplumber.open", return_value=mock_pdf):
            result = extract_source_date(pdf_path)
        assert result == "2025-03"

    def test_mod_date_fallback(self, tmp_path):
        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(b"fake")

        mock_pdf = MagicMock()
        mock_pdf.metadata = {"ModDate": "D:20241201000000"}
        mock_pdf.__enter__ = lambda s: s
        mock_pdf.__exit__ = MagicMock(return_value=False)

        with patch("pdfplumber.open", return_value=mock_pdf):
            result = extract_source_date(pdf_path)
        assert result == "2024-12"

    def test_no_metadata(self, tmp_path):
        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(b"fake")

        mock_pdf = MagicMock()
        mock_pdf.metadata = {}
        mock_pdf.__enter__ = lambda s: s
        mock_pdf.__exit__ = MagicMock(return_value=False)

        with patch("pdfplumber.open", return_value=mock_pdf):
            result = extract_source_date(pdf_path)
        assert result is None

    def test_none_metadata(self, tmp_path):
        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(b"fake")

        mock_pdf = MagicMock()
        mock_pdf.metadata = None
        mock_pdf.__enter__ = lambda s: s
        mock_pdf.__exit__ = MagicMock(return_value=False)

        with patch("pdfplumber.open", return_value=mock_pdf):
            result = extract_source_date(pdf_path)
        assert result is None


class TestPptxSourceDate:
    def test_modified_date(self, tmp_path):
        from datetime import datetime

        pptx_path = tmp_path / "test.pptx"
        pptx_path.write_bytes(b"fake")

        mock_prs = MagicMock()
        mock_prs.core_properties.modified = datetime(2025, 6, 15, 10, 30)

        with patch("pptx.Presentation", return_value=mock_prs):
            result = extract_source_date(pptx_path)
        assert result == "2025-06"

    def test_no_modified_date(self, tmp_path):
        pptx_path = tmp_path / "test.pptx"
        pptx_path.write_bytes(b"fake")

        mock_prs = MagicMock()
        mock_prs.core_properties.modified = None

        with patch("pptx.Presentation", return_value=mock_prs):
            result = extract_source_date(pptx_path)
        assert result is None


class TestDocxSourceDate:
    def test_modified_date(self, tmp_path):
        from datetime import datetime

        docx_path = tmp_path / "test.docx"
        docx_path.write_bytes(b"fake")

        mock_doc = MagicMock()
        mock_doc.core_properties.modified = datetime(2024, 11, 1)

        with patch("docx.Document", return_value=mock_doc):
            result = extract_source_date(docx_path)
        assert result == "2024-11"


class TestXlsxSourceDate:
    def test_modified_date(self, tmp_path):
        from datetime import datetime

        xlsx_path = tmp_path / "test.xlsx"
        xlsx_path.write_bytes(b"fake")

        mock_wb = MagicMock()
        mock_wb.properties.modified = datetime(2025, 1, 20)

        with patch("openpyxl.load_workbook", return_value=mock_wb):
            result = extract_source_date(xlsx_path)
        assert result == "2025-01"


class TestUnsupportedFormats:
    def test_txt_returns_none(self, tmp_path):
        txt_path = tmp_path / "test.txt"
        txt_path.write_text("hello")
        assert extract_source_date(txt_path) is None

    def test_csv_returns_none(self, tmp_path):
        csv_path = tmp_path / "test.csv"
        csv_path.write_text("a,b,c")
        assert extract_source_date(csv_path) is None

    def test_mp4_returns_none(self, tmp_path):
        mp4_path = tmp_path / "test.mp4"
        mp4_path.write_bytes(b"fake")
        assert extract_source_date(mp4_path) is None


class TestExceptionHandling:
    def test_corrupted_file_returns_none(self, tmp_path):
        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(b"not a real pdf")

        with patch("pdfplumber.open", side_effect=Exception("corrupt")):
            result = extract_source_date(pdf_path)
        assert result is None
