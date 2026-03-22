"""Tests for hierarchical tag generation (Council Decision #7)."""

from src.post_process import generate_tags, _normalize_tag


class TestNormalizeTag:
    def test_normalize_tag_special_chars(self):
        assert _normalize_tag("Platform & Architecture") == "platform-architecture"

    def test_normalize_tag_underscores(self):
        assert _normalize_tag("product_doc") == "product-doc"

    def test_normalize_tag_multiple_spaces(self):
        assert _normalize_tag("SaaS  Architecture") == "saas-architecture"

    def test_normalize_tag_already_clean(self):
        assert _normalize_tag("wms") == "wms"


class TestGenerateTags:
    def test_generate_tags_products(self):
        fm = {"products": ["WMS", "Demand Planning"]}
        assert generate_tags(fm) == ["product/wms", "product/demand-planning"]

    def test_generate_tags_topics(self):
        fm = {"topics": ["SaaS Architecture"]}
        assert generate_tags(fm) == ["topic/saas-architecture"]

    def test_generate_tags_domains(self):
        fm = {"domains": ["Platform & Architecture"]}
        assert generate_tags(fm) == ["domain/platform-architecture"]

    def test_generate_tags_client(self):
        fm = {"client": "SGDBF"}
        assert generate_tags(fm) == ["client/sgdbf"]

    def test_generate_tags_doc_type(self):
        fm = {"doc_type": "product_doc"}
        assert generate_tags(fm) == ["type/product-doc"]

    def test_generate_tags_source_type(self):
        fm = {"source_type": "documentation"}
        assert generate_tags(fm) == ["source/documentation"]

    def test_generate_tags_empty(self):
        assert generate_tags({}) == []

    def test_generate_tags_dedup(self):
        # Same normalized value in products twice
        fm = {"products": ["WMS", "WMS"]}
        result = generate_tags(fm)
        assert result == ["product/wms"]

    def test_generate_tags_none_fields(self):
        fm = {"products": None, "topics": None, "client": None}
        assert generate_tags(fm) == []

    def test_full_frontmatter_has_tags(self):
        fm = {
            "products": ["WMS"],
            "topics": ["SaaS Architecture"],
            "domains": ["Platform & Architecture"],
            "client": "SGDBF",
            "doc_type": "product_doc",
            "source_type": "documentation",
        }
        tags = generate_tags(fm)
        assert "product/wms" in tags
        assert "topic/saas-architecture" in tags
        assert "domain/platform-architecture" in tags
        assert "client/sgdbf" in tags
        assert "type/product-doc" in tags
        assert "source/documentation" in tags
        assert len(tags) == 6
