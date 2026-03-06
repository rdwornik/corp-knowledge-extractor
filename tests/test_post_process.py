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


def test_unknown_terms_logged(tmp_path, monkeypatch):
    """Unknown terms should be appended to taxonomy_review.yaml."""
    review_file = tmp_path / "taxonomy_review.yaml"
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config").mkdir()

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
    assert "Brand New Concept" in result.unknown_terms
    review_path = tmp_path / "config" / "taxonomy_review.yaml"
    assert review_path.exists()
    data = yaml.safe_load(review_path.read_text(encoding="utf-8"))
    assert "Brand New Concept" in data["pending"]


def test_unknown_terms_not_duplicated(tmp_path, monkeypatch):
    """Running twice shouldn't duplicate entries in taxonomy_review.yaml."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config").mkdir()

    raw = {
        "title": "Test",
        "date": "2026-03-01",
        "type": "presentation",
        "topics": ["Unique Concept"],
        "summary": "Test.",
    }
    post_process_extraction(raw_result=dict(raw), source_file="test.mkv")
    post_process_extraction(raw_result=dict(raw), source_file="test.mkv")

    review_path = tmp_path / "config" / "taxonomy_review.yaml"
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
