"""Tests for normalized output filename convention (Council Decision #7)."""

from src.synthesize import normalize_output_filename


class TestNormalizeOutputFilename:
    def test_normalize_filename_basic(self):
        result = normalize_output_filename(
            "SGDBF Architecural requirements.pptx",
            "2026-03-22T10:00:00",
            "a7b2c3d4",
        )
        assert result == "2026-03-22_sgdbf_architecural_requirements_a7b2"

    def test_normalize_filename_hyphen_stripped(self):
        """Hyphens in filename are stripped (not alphanumeric/space)."""
        result = normalize_output_filename(
            "SGDBF-Architecural requirements.pptx",
            "2026-03-22T10:00:00",
            "a7b2c3d4",
        )
        # Hyphen stripped → sgdbfarchitecural merges
        assert result == "2026-03-22_sgdbfarchitecural_requirements_a7b2"

    def test_normalize_filename_truncate(self):
        long_name = "a b " * 30 + ".pdf"  # spaces → underscores, very long
        result = normalize_output_filename(long_name, "2026-03-22T10:00:00", "b3c4d5e6")
        # Extract stem part between date_ and _hash
        after_date = result[len("2026-03-22_"):]
        stem_part = after_date[:after_date.rfind("_")]
        assert len(stem_part) <= 64

    def test_normalize_filename_special_chars(self):
        result = normalize_output_filename(
            "Platform  Architecture v2.pdf",
            "2026-03-22T10:00:00",
            "b3c4d5e6",
        )
        assert result == "2026-03-22_platform_architecture_v2_b3c4"

    def test_normalize_filename_no_hash(self):
        result = normalize_output_filename(
            "document.pdf",
            "2026-03-22T10:00:00",
            "",
        )
        assert result.endswith("_0000")
        assert result == "2026-03-22_document_0000"

    def test_normalize_filename_no_extracted_at(self):
        result = normalize_output_filename("doc.pdf", "", "abcd1234")
        assert result.endswith("_abcd")
        # Date is YYYY-MM-DD format
        date_part = result[:10]
        assert len(date_part.split("-")) == 3

    def test_normalize_filename_preserves_numbers(self):
        result = normalize_output_filename(
            "report2026q1.pdf",
            "2026-01-15T00:00:00",
            "ffff",
        )
        assert result == "2026-01-15_report2026q1_ffff"
