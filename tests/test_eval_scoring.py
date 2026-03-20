"""Tests for doc-type-aware eval scoring and low_content flag."""

import yaml
from pathlib import Path

import pytest

from eval_extraction import evaluate_package


def _write_md(path: Path, frontmatter: dict, body: str = "## Section 1\n\nContent here."):
    """Write a markdown file with YAML frontmatter."""
    fm_str = yaml.dump(frontmatter, default_flow_style=False)
    path.write_text(f"---\n{fm_str}---\n\n{body}", encoding="utf-8")


class TestDocTypeAwareScoring:
    def test_eval_product_doc_overlay(self, tmp_path):
        """product_doc with 5/8 overlay fields → high overlay score."""
        extract_dir = tmp_path / "extract"
        extract_dir.mkdir()

        fm = {
            "title": "Platform Architecture",
            "type": "document",
            "source": "BYPlatform-Architecture.pdf",
            "doc_type": "product_doc",
            "model": "gemini-3-flash-preview",
            "extraction_version": 2,
            "depth": "deep",
            "quality": "full",
            "topics": ["platform", "architecture", "microservices"],
            "people": [],
            "products": ["Platform"],
            "key_facts": [
                "Platform supports 1706 customers across multiple regions globally",
                "99.9% uptime SLA guaranteed with automatic failover capabilities",
                "Microservices architecture with 47 independently deployable services",
                "Supports Snowflake and Azure Data Lake integration natively",
                "Multi-tenant isolation using Kubernetes namespace separation model",
                "API gateway handles 10000 requests per second at peak load",
                "Event-driven architecture with Apache Kafka message streaming",
                "Blue Yonder ICC-2 certified for SOC2 Type II compliance standard",
                "Deployed in 12 Azure regions with cross-region replication enabled",
                "Platform data cloud supports 500TB analytical workloads monthly",
            ],
            "entities_mentioned": ["Blue Yonder", "Snowflake"],
            "source_path": "test.pdf",
            "source_hash": "abc123",
            "extracted_at": "2026-03-20",
            "tokens_used": 1000,
            "language": "en",
            "product_doc_overlay": {
                "target_audience": "Solution Architects",
                "key_capabilities": ["Microservices", "Multi-tenant"],
                "integration_points": ["SAP", "Snowflake"],
                "deployment_model": "Cloud-native",
                "security_considerations": "SOC2 compliant",
                "empty_field_1": None,
                "empty_field_2": [],
                "empty_field_3": "",
            },
        }
        body = "## Architecture\n\n" + "Technical content. " * 200
        _write_md(extract_dir / "test.md", fm, body)

        result = evaluate_package(tmp_path)
        file_eval = list(result["files"].values())[0]

        # 5 populated fields on a static doc (PDF) → should get full overlay score (20)
        assert file_eval["overlay_fields_populated"] == 5
        assert file_eval["is_static_doc"] is True
        # Static doc without slides/frames loses 10 pts visual — 74 is expected ceiling
        assert file_eval["score"] >= 70

    def test_eval_training_pdf_no_penalty(self, tmp_path):
        """PDF with doc_type=training, content_type=document → not penalized for missing attendees."""
        extract_dir = tmp_path / "extract"
        extract_dir.mkdir()

        fm = {
            "title": "Cognitive Demand Planning Help",
            "type": "document",
            "source": "Cognitive_Demand_Planning_Help.pdf",
            "doc_type": "training",
            "model": "gemini-3-flash-preview",
            "extraction_version": 2,
            "depth": "deep",
            "quality": "full",
            "topics": ["demand planning", "cognitive", "forecasting"],
            "people": [],
            "products": ["CDP"],
            "key_facts": [
                "Supports 50+ algorithms for demand forecasting optimization workloads",
                "Real-time forecast refresh every 15 minutes for high-velocity SKUs",
                "Cognitive Demand Planning processes 2 million SKU-location combinations",
                "Machine learning models retrain automatically on weekly data refresh cycles",
                "Integration with SAP APO for legacy migration path supported natively",
                "Forecast accuracy improvement of 15-25% over traditional statistical methods",
                "Supports hierarchical reconciliation across 8 planning levels automatically",
                "Cloud-native deployment on Blue Yonder Platform Data Cloud infrastructure",
            ],
            "entities_mentioned": ["Blue Yonder"],
            "source_path": "test.pdf",
            "source_hash": "def456",
            "extracted_at": "2026-03-20",
            "tokens_used": 2000,
            "language": "en",
            "training_overlay": {
                "learning_objectives": ["Understand CDP forecasting"],
                # No attendees, decisions, action_items — expected for static PDF
            },
        }
        body = "## Overview\n\n" + "Training documentation content. " * 200
        _write_md(extract_dir / "test.md", fm, body)

        result = evaluate_package(tmp_path)
        file_eval = list(result["files"].values())[0]

        # Static doc with 1 overlay field → should NOT be penalized like video
        assert file_eval["is_static_doc"] is True
        # Score should be reasonable — static doc overlay gets 5+5=10 (not 5+3=8 strict)
        # Without slides/frames (-10 visual), 58 is expected; confirm not dragged below 50
        assert file_eval["score"] >= 55
        # Key check: static doc with 1 field scores HIGHER than video with 1 field
        # because static doc gets 5 per field vs video 3 per field

    def test_eval_low_content_flag(self, tmp_path):
        """File with <2000 chars content → low_content flag set."""
        extract_dir = tmp_path / "extract"
        extract_dir.mkdir()

        fm = {
            "title": "Horizon Web Access",
            "type": "document",
            "source": "Horizon_Web.docx",
            "doc_type": "training",
            "model": "gemini-3-flash-preview",
            "extraction_version": 2,
            "depth": "deep",
            "quality": "full",
            "topics": ["horizon"],
            "people": [],
            "products": [],
            "key_facts": ["Login via SSO"],
            "entities_mentioned": [],
            "source_path": "test.docx",
            "source_hash": "ghi789",
            "extracted_at": "2026-03-20",
            "tokens_used": 500,
            "language": "en",
        }
        # Short body — under 2000 chars
        body = "## How to Access\n\nGo to horizon.blueyonder.com and login with SSO."
        _write_md(extract_dir / "test.md", fm, body)

        result = evaluate_package(tmp_path)
        file_eval = list(result["files"].values())[0]

        assert file_eval["low_content"] is True
        assert file_eval["content"]["total_chars"] < 2000

    def test_video_training_strict_scoring(self, tmp_path):
        """MP4 with training overlay → strict scoring (not static doc treatment)."""
        extract_dir = tmp_path / "extract"
        extract_dir.mkdir()

        fm = {
            "title": "MaaS Overview Session",
            "type": "presentation",
            "source": "MaaS_Recording.mp4",
            "doc_type": "training",
            "model": "gemini-3-flash-preview",
            "extraction_version": 2,
            "depth": "deep",
            "quality": "full",
            "topics": ["MaaS", "platform"],
            "people": [],
            "products": ["MaaS"],
            "key_facts": ["MaaS supports 5 chart types"],
            "entities_mentioned": ["Blue Yonder"],
            "source_path": "test.mp4",
            "source_hash": "jkl012",
            "extracted_at": "2026-03-20",
            "tokens_used": 3000,
            "language": "en",
            "training_overlay": {
                "attendees": ["Alice", "Bob"],
                "decisions_made": [],
                "action_items": [],
                "questions_raised": [],
                "concerns_expressed": [],
                "next_steps": [],
            },
        }
        body = "## Session Content\n\n" + "Speaker discussed platform features. " * 200
        _write_md(extract_dir / "test.md", fm, body)

        result = evaluate_package(tmp_path)
        file_eval = list(result["files"].values())[0]

        # MP4 is NOT static doc → strict overlay scoring
        assert file_eval["is_static_doc"] is False
        # Only 1 populated field (attendees) → lower overlay score than static doc treatment
        assert file_eval["overlay_fields_populated"] == 1
