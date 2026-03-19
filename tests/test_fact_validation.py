"""Tests for fact validation — number normalizer, source cross-reference, anomaly detection."""

from unittest.mock import patch, MagicMock

import pytest

from src.fact_validation import (
    normalize_number,
    extract_numbers_from_text,
    validate_fact_against_source,
    check_anomalies,
)
from src.extract import _enrich_facts


class TestNormalizeNumber:
    def test_normalize_number_plain(self):
        assert normalize_number("1706") == 1706.0

    def test_normalize_number_commas(self):
        assert normalize_number("1,706") == 1706.0

    def test_normalize_number_k_suffix(self):
        assert normalize_number("$950K") == 950_000.0

    def test_normalize_number_m_suffix(self):
        assert normalize_number("$2M") == 2_000_000.0

    def test_normalize_number_b_suffix(self):
        assert normalize_number("$1.7B") == 1_700_000_000.0

    def test_normalize_number_percentage(self):
        assert normalize_number("10.8%") == 10.8

    def test_normalize_number_word_million(self):
        assert normalize_number("1.7 million") == 1_700_000.0

    def test_normalize_number_word_thousand(self):
        assert normalize_number("5 thousand") == 5_000.0

    def test_normalize_number_word_billion(self):
        assert normalize_number("2.5 billion") == 2_500_000_000.0

    def test_normalize_number_range(self):
        """Ranges return first value."""
        assert normalize_number("5-8%") == 5.0

    def test_normalize_number_none(self):
        assert normalize_number("hello world") is None

    def test_normalize_number_empty(self):
        assert normalize_number("") is None

    def test_normalize_number_euro(self):
        assert normalize_number("€500K") == 500_000.0

    def test_normalize_number_lowercase_m(self):
        assert normalize_number("$2m") == 2_000_000.0


class TestExtractNumbers:
    def test_extract_numbers(self):
        result = extract_numbers_from_text("Reduced from $2M to $950K in 15 weeks")
        assert result == {2_000_000, 950_000, 15}

    def test_extract_numbers_with_percentages(self):
        result = extract_numbers_from_text("10.8% reduction saving $1.5M")
        assert 10.8 in result
        assert 1_500_000 in result

    def test_extract_numbers_empty(self):
        result = extract_numbers_from_text("No numbers here")
        assert result == set()

    def test_extract_numbers_commas(self):
        result = extract_numbers_from_text("1,706 customers and 2,500 orders")
        assert 1706 in result
        assert 2500 in result


class TestValidateFactAgainstSource:
    def test_validate_fact_verified(self):
        """Fact numbers found in source → verified."""
        result = validate_fact_against_source(
            "1706 customers onboarded",
            "The platform has 1,706 active customers across all regions.",
        )
        assert result["status"] == "verified"
        assert 1706 in result["fact_numbers"]

    def test_validate_fact_missing(self):
        """$950M in fact but $950K in source → flagged_mismatch (1000x)."""
        result = validate_fact_against_source(
            "Achieved $950M in savings",
            "Cost reduction achieved $950K in annual savings over 15 weeks.",
        )
        assert result["status"] == "flagged_mismatch"
        assert len(result["missing_numbers"]) > 0
        assert any("Magnitude" in a for a in result["anomalies"])

    def test_validate_fact_no_numbers(self):
        """Non-numeric fact → verified by default."""
        result = validate_fact_against_source(
            "Uses Snowflake for data warehousing",
            "The platform uses Snowflake and Databricks.",
        )
        assert result["status"] == "verified"
        assert result["fact_numbers"] == []

    def test_validate_fact_rounding_tolerance(self):
        """Numbers within 1% tolerance → verified."""
        result = validate_fact_against_source(
            "Approximately 1700 users",
            "System has 1,706 registered users.",
        )
        assert result["status"] == "verified"

    def test_validate_fact_unverified(self):
        """Number not in source but not a magnitude error → unverified."""
        result = validate_fact_against_source(
            "42 integrations available",
            "The platform supports multiple integrations with various systems.",
        )
        assert result["status"] == "unverified"
        assert 42 in result["missing_numbers"]


class TestCheckAnomalies:
    def test_anomaly_percentage_over_100_valid(self):
        """250% increase is valid growth — no anomaly."""
        anomalies = check_anomalies("250% increase in throughput")
        assert not any("Percentage over 100%" in a for a in anomalies)

    def test_anomaly_percentage_over_100_invalid(self):
        """250% completion is suspicious."""
        anomalies = check_anomalies("Project is 250% complete")
        assert any("Percentage over 100%" in a for a in anomalies)

    def test_anomaly_magnitude_error(self):
        """$2M to $950M in same fact → suspicious magnitude spread."""
        anomalies = check_anomalies("Reduced costs from $2M to $950M annually")
        assert any("magnitude spread" in a.lower() for a in anomalies)

    def test_anomaly_future_date(self):
        """Year 2035 → future date anomaly."""
        anomalies = check_anomalies("Expected completion in 2035")
        assert any("Future date" in a for a in anomalies)

    def test_anomaly_near_future_ok(self):
        """Next year is fine."""
        anomalies = check_anomalies("Planned for 2027 deployment")
        assert not any("Future date" in a for a in anomalies)

    def test_no_anomalies_normal_fact(self):
        """Normal fact → no anomalies."""
        anomalies = check_anomalies("Reduced from $2M to $950K in 15 weeks")
        assert anomalies == []


class TestValidationWired:
    """Test that fact validation is wired into the extraction pipeline."""

    def test_validation_wired_pptx(self):
        """Enriched facts include verification_status when source text available."""
        from src.inventory import SourceFile, FileType
        from src.text_extract import TextExtractionResult
        from pathlib import Path

        sf = SourceFile(
            path=Path("test.pptx"), name="test.pptx",
            type=FileType.SLIDES, size_bytes=100,
        )
        text_result = TextExtractionResult(
            text="The platform serves 1,706 customers across 15 regions.",
            char_count=55, extractor="python-pptx", slide_count=5,
        )
        data = {
            "facts": [
                {"fact": "1706 customers onboarded", "slide": 3},
                {"fact": "Uses Snowflake for analytics", "slide": 5},
            ]
        }

        facts = _enrich_facts(data, sf, "2026-01-01", text_result)

        assert len(facts) == 2
        assert facts[0]["verification_status"] == "verified"
        assert facts[1]["verification_status"] == "verified"  # non-numeric

    def test_flagged_fact_has_anomalies(self):
        """Magnitude error fact gets flagged_mismatch with anomalies."""
        from src.inventory import SourceFile, FileType
        from src.text_extract import TextExtractionResult
        from pathlib import Path

        sf = SourceFile(
            path=Path("test.pptx"), name="test.pptx",
            type=FileType.SLIDES, size_bytes=100,
        )
        text_result = TextExtractionResult(
            text="Achieved $950K in annual savings.",
            char_count=32, extractor="python-pptx", slide_count=3,
        )
        data = {
            "facts": [
                {"fact": "Achieved $950M in savings", "slide": 1},
            ]
        }

        facts = _enrich_facts(data, sf, "2026-01-01", text_result)

        assert facts[0]["verification_status"] == "flagged_mismatch"
        assert "anomalies" in facts[0]
        assert any("Magnitude" in a for a in facts[0]["anomalies"])

    def test_flagged_fact_in_template(self):
        """Flagged facts render in the markdown template."""
        from jinja2 import Environment, FileSystemLoader

        env = Environment(loader=FileSystemLoader("templates"))
        env.filters["tojson_raw"] = lambda v: str(v)
        tmpl = env.get_template("extract.md.j2")

        flagged = [
            {
                "fact": "Achieved $950M in savings",
                "verification_status": "flagged_mismatch",
                "anomalies": ["Magnitude error: 950000000 not in source (1000x off)"],
            }
        ]
        output = tmpl.render(
            source_file="test.pptx", content_type="presentation",
            title="Test", date="2026-01-01", topics=[], people=[],
            products=[], language="en", quality="full", tokens_used=0,
            summary="Test summary", links_line="", slides=[],
            key_points=[], transcript_excerpt="", model="test",
            flagged_facts=flagged,
        )

        assert "Flagged Facts" in output
        assert "FLAGGED_MISMATCH" in output
        assert "$950M" in output
        assert "Magnitude error" in output
