"""Tests for post-processing module."""
import pytest
import yaml
from pathlib import Path
from src.post_process import post_process_extraction


def _make_taxonomy(tmp_path: Path) -> Path:
    """Create a minimal taxonomy file for testing."""
    taxonomy = tmp_path / "taxonomy.yaml"
    taxonomy.write_text("""
topics:
  - id: disaster-recovery
    name: "Disaster Recovery"
    aliases: ["DR", "disaster recovery planning"]
  - id: sla
    name: "SLA"
    aliases: ["Service Level Agreement", "SLAs"]
products:
  - id: blue-yonder-platform
    name: "Blue Yonder Platform"
    aliases: ["BY Platform"]
people_vip: []
thresholds:
  people_node_min_mentions: 3
  product_node_min_mentions: 2
""", encoding="utf-8")
    return taxonomy


def test_taxonomy_normalization(tmp_path):
    taxonomy = _make_taxonomy(tmp_path)
    config = {"post_processing": {"term_normalization": {}}}
    result = {"topics": ["DR", "SLAs", "Some New Topic"], "products": [], "people": []}
    pp = post_process_extraction(result, config, taxonomy)
    assert "Disaster Recovery" in pp.data["topics"]
    assert "SLA" in pp.data["topics"]
    assert "Some New Topic" in pp.data["topics"]
    assert "Some New Topic" in pp.unknown_terms


def test_deduplication_after_normalization(tmp_path):
    taxonomy = _make_taxonomy(tmp_path)
    config = {"post_processing": {"term_normalization": {}}}
    result = {
        "topics": ["DR", "Disaster Recovery", "disaster recovery planning"],
        "products": [],
        "people": [],
    }
    pp = post_process_extraction(result, config, taxonomy)
    assert pp.data["topics"].count("Disaster Recovery") == 1


def test_cardinality_caps(tmp_path):
    taxonomy = _make_taxonomy(tmp_path)
    config = {"post_processing": {"term_normalization": {}}}
    result = {
        "topics": [f"Topic {i}" for i in range(15)],
        "products": [f"Product {i}" for i in range(10)],
        "people": [f"Person {i}" for i in range(8)],
    }
    pp = post_process_extraction(result, config, taxonomy)
    assert len(pp.data["topics"]) <= 8
    assert len(pp.data["products"]) <= 4
    assert len(pp.data["people"]) <= 3
    assert "topics" in pp.truncated_fields


def test_links_line_generated(tmp_path):
    taxonomy = _make_taxonomy(tmp_path)
    config = {"post_processing": {"term_normalization": {}}}
    result = {
        "topics": ["Disaster Recovery"],
        "products": ["WMS"],
        "people": ["Mike Geller (Presenter)"],
    }
    pp = post_process_extraction(result, config, taxonomy)
    assert "[[Disaster Recovery]]" in pp.links_line
    assert "[[WMS]]" in pp.links_line
    assert "[[Mike Geller]]" in pp.links_line
    assert "(Presenter)" not in pp.links_line


def test_term_normalization(tmp_path):
    taxonomy = _make_taxonomy(tmp_path)
    config = {"post_processing": {"term_normalization": {"Yonder": "Blue Yonder"}}}
    result = {"title": "Yonder SaaS Platform", "topics": [], "products": [], "people": []}
    pp = post_process_extraction(result, config, taxonomy)
    assert pp.data["title"] == "Blue Yonder SaaS Platform"


def test_unknown_terms_logged(tmp_path):
    taxonomy = _make_taxonomy(tmp_path)
    review_file = tmp_path / "taxonomy_review.yaml"
    config = {"post_processing": {"term_normalization": {}}}
    result = {"topics": ["Brand New Concept"], "products": [], "people": []}
    pp = post_process_extraction(result, config, taxonomy)
    assert "Brand New Concept" in pp.unknown_terms
    assert review_file.exists()
    data = yaml.safe_load(review_file.read_text(encoding="utf-8"))
    assert "Brand New Concept" in data["pending"]


def test_empty_config(tmp_path):
    taxonomy = tmp_path / "taxonomy.yaml"
    taxonomy.write_text("topics: []\nproducts: []\n", encoding="utf-8")
    result = {"title": "Test", "summary": "Nothing changes", "topics": [], "products": [], "people": []}
    pp = post_process_extraction(result, {}, taxonomy)
    assert pp.data["title"] == "Test"


def test_product_taxonomy_normalization(tmp_path):
    taxonomy = _make_taxonomy(tmp_path)
    config = {"post_processing": {"term_normalization": {}}}
    result = {"topics": [], "products": ["BY Platform"], "people": []}
    pp = post_process_extraction(result, config, taxonomy)
    assert "Blue Yonder Platform" in pp.data["products"]
    assert "BY Platform -> Blue Yonder Platform" in pp.normalized_terms


def test_links_line_empty_when_no_data(tmp_path):
    taxonomy = _make_taxonomy(tmp_path)
    config = {}
    result = {"topics": [], "products": [], "people": []}
    pp = post_process_extraction(result, config, taxonomy)
    assert pp.links_line == ""


def test_unknown_terms_not_duplicated(tmp_path):
    """Running twice shouldn't duplicate entries in taxonomy_review.yaml."""
    taxonomy = _make_taxonomy(tmp_path)
    config = {"post_processing": {"term_normalization": {}}}
    result = {"topics": ["Unique Concept"], "products": [], "people": []}
    post_process_extraction(result, config, taxonomy)
    post_process_extraction(result, config, taxonomy)
    review_file = tmp_path / "taxonomy_review.yaml"
    data = yaml.safe_load(review_file.read_text(encoding="utf-8"))
    assert data["pending"].count("Unique Concept") == 1
