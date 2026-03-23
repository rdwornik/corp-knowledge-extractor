"""Tests for compute_quality_score (0-100 int)."""

import json
from pathlib import Path
from jinja2 import Environment, FileSystemLoader

from src.synthesize import compute_quality_score


class TestComputeQualityScore:
    def test_rich_extraction(self):
        """15 specific facts, 90% verified, 5 overlay fields, 10k chars, 20 entities → 90+."""
        key_facts = [f"This is a detailed fact number {i} with enough length to pass the 30-char filter easily" for i in range(15)]
        facts = (
            [{"fact": f"v{i}", "verification_status": "verified"} for i in range(9)]
            + [{"fact": "u0", "verification_status": "unverified"}]
        )
        score = compute_quality_score(
            key_facts=key_facts,
            facts_with_status=facts,
            overlay_fields_populated=5,
            content_chars=10000,
            entities_count=20,
        )
        assert score >= 90, f"Expected >= 90, got {score}"

    def test_poor_extraction(self):
        """2 short facts, 0 verified, 0 overlay, 500 chars, 1 entity → < 30."""
        key_facts = ["Short fact A that is at least thirty characters long", "Another short fact B at least thirty chars"]
        facts = [
            {"fact": "a", "verification_status": "unverified"},
            {"fact": "b", "verification_status": "unverified"},
        ]
        score = compute_quality_score(
            key_facts=key_facts,
            facts_with_status=facts,
            overlay_fields_populated=0,
            content_chars=500,
            entities_count=1,
        )
        assert score < 30, f"Expected < 30, got {score}"

    def test_max_100(self):
        """Extreme values don't exceed 100."""
        key_facts = [f"Extremely detailed fact {i} that exceeds thirty characters easily and is very specific" for i in range(100)]
        facts = [{"fact": f"f{i}", "verification_status": "verified"} for i in range(100)]
        score = compute_quality_score(
            key_facts=key_facts,
            facts_with_status=facts,
            overlay_fields_populated=50,
            content_chars=100000,
            entities_count=100,
        )
        assert score == 100

    def test_zero_everything(self):
        """Nothing at all → 0."""
        score = compute_quality_score(
            key_facts=[],
            facts_with_status=[],
            overlay_fields_populated=0,
            content_chars=0,
            entities_count=0,
        )
        assert score == 0

    def test_short_facts_filtered(self):
        """Facts shorter than 30 chars don't count."""
        key_facts = ["short", "tiny", "nope"]
        score = compute_quality_score(
            key_facts=key_facts,
            facts_with_status=[],
            overlay_fields_populated=0,
            content_chars=0,
            entities_count=0,
        )
        assert score == 0


class TestQualityScoreInFrontmatter:
    def _render(self, quality_score):
        templates_dir = Path(__file__).parent.parent / "templates"
        env = Environment(
            loader=FileSystemLoader(str(templates_dir)),
            trim_blocks=True,
            lstrip_blocks=True,
        )
        env.filters["tojson_raw"] = lambda v: json.dumps(v, ensure_ascii=False)
        tmpl = env.get_template("extract.md.j2")
        return tmpl.render(
            source_file="C:/test/file.pdf",
            content_type="document",
            title="Test",
            date="2026-03-23",
            summary="Summary.",
            key_points=[],
            topics=["Testing"],
            people=[],
            products=[],
            slides=[],
            language="en",
            quality="full",
            duration_min=None,
            transcript_excerpt="",
            model="gemini-3-flash-preview",
            tokens_used=100,
            links_line="",
            source_tool="knowledge-extractor",
            quality_score=quality_score,
        )

    def test_score_in_frontmatter(self):
        content = self._render(73)
        assert "quality_score: 73" in content

    def test_zero_score_still_rendered(self):
        content = self._render(0)
        assert "quality_score: 0" in content
