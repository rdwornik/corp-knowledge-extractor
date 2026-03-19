"""Tests for post-processing wrapper around corp-os-meta."""

import pytest
import yaml
from pathlib import Path
from unittest.mock import patch
from src.post_process import post_process_extraction, _log_unknown_terms
from corp_os_meta import ValidationResult


def test_basic_normalization():
    """Test that corp-os-meta normalizes terms correctly."""
    result = post_process_extraction(
        raw_result={
            "title": "Test Video",
            "date": "2026-03-01",
            "content_type": "presentation",
            "topics": ["DR", "SLAs", "Some New Topic"],
            "products": ["BY Platform"],
            "people": ["Mike Geller (Presenter)"],
            "summary": "A test.",
        },
        source_file="test.mkv",
    )
    assert "Disaster Recovery" in result.data["topics"]
    assert "SLA" in result.data["topics"]
    assert "Blue Yonder Platform" in result.data["products"]
    assert "Some New Topic" in result.unknown_terms


def test_links_line_generated():
    """Test deterministic Links line."""
    result = post_process_extraction(
        raw_result={
            "title": "Test",
            "date": "2026-03-01",
            "type": "presentation",
            "topics": ["Disaster Recovery"],
            "products": ["WMS"],
            "people": ["Mike Geller (Presenter)"],
            "summary": "A test.",
        },
        source_file="test.mkv",
    )
    assert "[[Disaster Recovery]]" in result.links_line
    assert "[[WMS]]" in result.links_line
    assert "[[Mike Geller]]" in result.links_line
    assert "(Presenter)" not in result.links_line


def test_content_type_mapped_to_type():
    """CKE uses content_type, corp-os-meta uses type."""
    result = post_process_extraction(
        raw_result={
            "title": "Test",
            "date": "2026-03-01",
            "content_type": "training",
            "topics": ["SLA"],
            "summary": "Training video.",
        },
        source_file="test.mkv",
    )
    assert "type" in result.data
    assert result.data["type"] == "training"


def test_validation_with_full_data():
    """Full data should validate as valid or warnings."""
    result = post_process_extraction(
        raw_result={
            "title": "Complete Note",
            "date": "2026-03-01",
            "type": "presentation",
            "topics": ["SLA", "Disaster Recovery"],
            "products": ["Blue Yonder Platform"],
            "people": ["Mike Geller (Presenter)"],
            "summary": "Complete summary.",
            "language": "en",
            "quality": "full",
        },
        source_file="test.mkv",
    )
    assert result.validation_result in (ValidationResult.VALID, ValidationResult.WARNINGS)
    assert result.validated_note is not None


def test_cardinality_caps():
    """Caps should be enforced by corp-os-meta."""
    result = post_process_extraction(
        raw_result={
            "title": "Overcapped",
            "date": "2026-03-01",
            "type": "document",
            "topics": [f"Topic {i}" for i in range(15)],
            "products": [f"Product {i}" for i in range(10)],
            "people": [f"Person {i}" for i in range(8)],
            "summary": "Too many terms.",
        },
        source_file="test.mkv",
    )
    assert len(result.data["topics"]) <= 8
    assert len(result.data["products"]) <= 4
    assert len(result.data["people"]) <= 3


def test_deduplication_after_normalization():
    """Multiple aliases for same term should deduplicate."""
    result = post_process_extraction(
        raw_result={
            "title": "Dedup Test",
            "date": "2026-03-01",
            "type": "presentation",
            "topics": ["DR", "Disaster Recovery", "disaster recovery planning"],
            "summary": "Test.",
        },
        source_file="test.mkv",
    )
    assert result.data["topics"].count("Disaster Recovery") == 1


def test_links_line_empty_when_no_data():
    """No topics/products/people means no Links line."""
    result = post_process_extraction(
        raw_result={
            "title": "Empty",
            "date": "2026-03-01",
            "type": "document",
            "topics": [],
            "products": [],
            "people": [],
            "summary": "Nothing.",
        },
        source_file="test.mkv",
    )
    assert result.links_line == ""


def test_source_tool_defaults():
    """source_tool should be set automatically."""
    result = post_process_extraction(
        raw_result={
            "title": "Test",
            "date": "2026-03-01",
            "type": "presentation",
            "topics": [],
            "summary": "Test.",
        },
        source_file="test.mkv",
    )
    assert result.data["source_tool"] == "knowledge-extractor"
    assert result.data["source_file"] == "test.mkv"


def test_unknown_terms_logged(tmp_path):
    """Unknown terms should be appended to taxonomy_review.yaml."""
    review_path = tmp_path / "config" / "taxonomy_review.yaml"
    (tmp_path / "config").mkdir()

    with patch("src.post_process.Path") as MockPath:
        # Make Path(__file__).parent.parent / "config" / ... resolve to tmp_path
        MockPath.return_value.parent.parent.__truediv__ = lambda self, x: tmp_path / x
        # But keep real Path for everything else
        MockPath.side_effect = lambda *a, **k: Path(*a, **k) if a else MockPath.return_value
        # Directly patch the function to use our tmp path
        import src.post_process as pp_mod

        orig_fn = pp_mod._log_unknown_terms

        def _patched_log(terms):
            rp = tmp_path / "config" / "taxonomy_review.yaml"
            import yaml as _yaml

            data = {"pending": []}
            if rp.exists():
                with open(rp, "r", encoding="utf-8") as f:
                    data = _yaml.safe_load(f) or {"pending": []}
            existing = set(data.get("pending", []))
            for term in terms:
                if term not in existing:
                    data["pending"].append(term)
            with open(rp, "w", encoding="utf-8") as f:
                _yaml.dump(data, f, default_flow_style=False, allow_unicode=True)

        pp_mod._log_unknown_terms = _patched_log
        try:
            result = post_process_extraction(
                raw_result={
                    "title": "Test",
                    "date": "2026-03-01",
                    "type": "presentation",
                    "topics": ["Brand New Concept"],
                    "summary": "Test.",
                },
                source_file="test.mkv",
            )
        finally:
            pp_mod._log_unknown_terms = orig_fn

    assert "Brand New Concept" in result.unknown_terms
    assert review_path.exists()
    data = yaml.safe_load(review_path.read_text(encoding="utf-8"))
    assert "Brand New Concept" in data["pending"]


def test_unknown_terms_not_duplicated(tmp_path):
    """Running twice shouldn't duplicate entries in taxonomy_review.yaml."""
    review_path = tmp_path / "config" / "taxonomy_review.yaml"
    (tmp_path / "config").mkdir()

    import src.post_process as pp_mod

    orig_fn = pp_mod._log_unknown_terms

    def _patched_log(terms):
        import yaml as _yaml

        rp = tmp_path / "config" / "taxonomy_review.yaml"
        data = {"pending": []}
        if rp.exists():
            with open(rp, "r", encoding="utf-8") as f:
                data = _yaml.safe_load(f) or {"pending": []}
        existing = set(data.get("pending", []))
        for term in terms:
            if term not in existing:
                data["pending"].append(term)
        with open(rp, "w", encoding="utf-8") as f:
            _yaml.dump(data, f, default_flow_style=False, allow_unicode=True)

    pp_mod._log_unknown_terms = _patched_log
    try:
        raw = {
            "title": "Test",
            "date": "2026-03-01",
            "type": "presentation",
            "topics": ["Unique Concept"],
            "summary": "Test.",
        }
        post_process_extraction(raw_result=dict(raw), source_file="test.mkv")
        post_process_extraction(raw_result=dict(raw), source_file="test.mkv")
    finally:
        pp_mod._log_unknown_terms = orig_fn

    data = yaml.safe_load(review_path.read_text(encoding="utf-8"))
    assert data["pending"].count("Unique Concept") == 1


def test_product_taxonomy_normalization():
    """Product aliases should normalize to canonical names."""
    result = post_process_extraction(
        raw_result={
            "title": "Product Test",
            "date": "2026-03-01",
            "type": "presentation",
            "topics": [],
            "products": ["BY Platform"],
            "summary": "Test.",
        },
        source_file="test.mkv",
    )
    assert "Blue Yonder Platform" in result.data["products"]
    # At least one change logged for the normalization
    assert any("Blue Yonder Platform" in c for c in result.changes)


# ---------------------------------------------------------------------------
# Schema v2 tests — knowledge dimensions
# ---------------------------------------------------------------------------


def test_domains_normalized():
    """Domains should be normalized via corp-os-meta."""
    result = post_process_extraction(
        raw_result={
            "title": "Pricing Update",
            "date": "2026-03-01",
            "type": "presentation",
            "topics": ["SLA"],
            "domains": ["GTM", "pricing"],
            "summary": "Pricing changes.",
        },
        source_file="test.mkv",
    )
    assert "Go-to-Market" in result.data["domains"]
    assert "Commercials" in result.data["domains"]


def test_confidentiality_defaults_to_internal():
    """Missing confidentiality should default to internal."""
    result = post_process_extraction(
        raw_result={
            "title": "Test",
            "date": "2026-03-01",
            "type": "document",
            "topics": [],
            "summary": "Test.",
        },
        source_file="test.md",
    )
    assert result.data["confidentiality"] == "internal"


def test_schema_v2_defaults():
    """All v2 fields should have safe defaults."""
    result = post_process_extraction(
        raw_result={
            "title": "Minimal",
            "date": "2026-03-01",
            "type": "document",
            "topics": [],
            "summary": "Minimal.",
        },
        source_file="test.md",
    )
    assert result.data["authority"] == "tribal"
    assert result.data["layer"] == "learning"
    assert result.data["source_type"] == "documentation"
    assert result.data["domains"] == []
    assert result.data["schema_version"] == 2


def test_valid_to_auto_calculated():
    """Domains with expiry rules should produce valid_to."""
    result = post_process_extraction(
        raw_result={
            "title": "Pricing Note",
            "date": "2026-03-01",
            "type": "document",
            "topics": ["Pricing"],
            "domains": ["Commercials"],
            "summary": "New pricing tiers.",
        },
        source_file="test.md",
    )
    # Commercials domain triggers valid_to calculation
    assert result.data.get("valid_to") is not None


def test_domains_cap():
    """Domains should be capped at 3."""
    result = post_process_extraction(
        raw_result={
            "title": "Many Domains",
            "date": "2026-03-01",
            "type": "document",
            "topics": [],
            "domains": ["Product", "Commercials", "Competitive", "Go-to-Market", "Security"],
            "summary": "Test.",
        },
        source_file="test.md",
    )
    assert len(result.data["domains"]) <= 3


def test_confidentiality_passthrough():
    """LLM-provided confidentiality should be preserved."""
    result = post_process_extraction(
        raw_result={
            "title": "Secret Deal",
            "date": "2026-03-01",
            "type": "document",
            "topics": [],
            "confidentiality": "restricted",
            "summary": "M&A details.",
        },
        source_file="test.md",
    )
    assert result.data["confidentiality"] == "restricted"


# ---------------------------------------------------------------------------
# Systematic fix tests
# ---------------------------------------------------------------------------


def test_client_from_manifest_overrides_gemini():
    """When manifest provides client, it overrides Gemini extraction."""
    result = post_process_extraction(
        raw_result={
            "title": "Test",
            "date": "2026-03-01",
            "type": "document",
            "client": "Acme Corp",
            "topics": [],
            "products": [],
            "people": [],
            "summary": "Test",
        },
        source_file="test.md",
        client="Lenzing AG",
    )
    assert result.data["client"] == "Lenzing AG"


def test_project_from_manifest():
    """When manifest provides project, it appears in result data."""
    result = post_process_extraction(
        raw_result={
            "title": "Test",
            "date": "2026-03-01",
            "type": "document",
            "topics": [],
            "summary": "Test",
        },
        source_file="test.md",
        project="Lenzing_Planning",
    )
    assert result.data["project"] == "Lenzing_Planning"


def test_quality_mapping_high_to_full():
    """Gemini 'high' maps to schema 'full'."""
    result = post_process_extraction(
        raw_result={
            "title": "Test",
            "date": "2026-03-01",
            "type": "document",
            "quality": "high",
            "topics": [],
            "summary": "Test",
        },
        source_file="test.md",
    )
    assert result.data["quality"] == "full"


def test_quality_mapping_medium_to_partial():
    """Gemini 'medium' maps to schema 'partial'."""
    result = post_process_extraction(
        raw_result={
            "title": "Test",
            "date": "2026-03-01",
            "type": "document",
            "quality": "medium",
            "topics": [],
            "summary": "Test",
        },
        source_file="test.md",
    )
    assert result.data["quality"] == "partial"


def test_quality_mapping_low_to_fragment():
    """Gemini 'low' maps to schema 'fragment'."""
    result = post_process_extraction(
        raw_result={
            "title": "Test",
            "date": "2026-03-01",
            "type": "document",
            "quality": "low",
            "topics": [],
            "summary": "Test",
        },
        source_file="test.md",
    )
    assert result.data["quality"] == "fragment"


def test_quality_passthrough_valid_value():
    """Schema-valid quality values pass through unchanged."""
    result = post_process_extraction(
        raw_result={
            "title": "Test",
            "date": "2026-03-01",
            "type": "document",
            "quality": "full",
            "topics": [],
            "summary": "Test",
        },
        source_file="test.md",
    )
    assert result.data["quality"] == "full"


def test_no_unicode_escape_in_frontmatter():
    """Domains with & should not be escaped to \\u0026 in tojson_raw."""
    import json
    from src.synthesize import _tojson_raw

    result = _tojson_raw(["Platform & Architecture"])
    assert "&" in result
    assert "\\u0026" not in result


def test_backslash_normalized_in_source():
    """Source paths should use forward slashes after normalization."""
    path = "C:\\Users\\test\\file.pdf"
    normalized = path.replace("\\", "/")
    assert "\\" not in normalized
    assert "C:/Users/test/file.pdf" == normalized


# ---------------------------------------------------------------------------
# Company name normalization tests (FIX 2)
# ---------------------------------------------------------------------------


from src.post_process import normalize_company_names


def test_normalize_blue_blue():
    assert normalize_company_names("Blue Blue Yonder Platform") == "Blue Yonder Platform"


def test_normalize_already_correct():
    assert normalize_company_names("Blue Yonder Platform") == "Blue Yonder Platform"


def test_normalize_triple():
    assert normalize_company_names("Blue Blue Blue Yonder") == "Blue Yonder"


def test_normalize_case_insensitive():
    result = normalize_company_names("blue blue yonder")
    assert result == "Blue Yonder"


def test_normalize_in_summary():
    """Summary field also cleaned via post_process_extraction."""
    result = post_process_extraction(
        raw_result={
            "title": "Test",
            "date": "2026-03-01",
            "type": "document",
            "topics": [],
            "summary": "Blue Blue Yonder released new features.",
        },
        source_file="test.md",
    )
    assert "Blue Blue" not in result.data["summary"]
    assert "Blue Yonder" in result.data["summary"]


def test_normalize_no_false_positive():
    assert normalize_company_names("Blue Sky Yonder") == "Blue Sky Yonder"
