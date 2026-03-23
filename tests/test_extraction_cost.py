"""Tests for extraction_cost_usd propagation to frontmatter."""

import json
from pathlib import Path
from unittest.mock import patch
from jinja2 import Environment, FileSystemLoader

from src.extract import ExtractionResult, _estimate_gemini_cost


class TestEstimateGeminiCost:
    def test_flash_cost(self):
        cost = _estimate_gemini_cost("gemini-3-flash-preview", 10000)
        assert cost == 0.01  # 10000 * 1.00 / 1M

    def test_zero_tokens(self):
        assert _estimate_gemini_cost("gemini-3-flash-preview", 0) == 0.0


class TestCostOnExtractionResult:
    def test_default_zero(self):
        from src.inventory import SourceFile, FileType
        result = ExtractionResult(
            source_file=SourceFile(path=Path("t.pdf"), type=FileType.DOCUMENT, size_bytes=100, name="t"),
            title="T", summary="S",
        )
        assert result.extraction_cost_usd == 0.0


class TestCostInExtractFromText:
    @patch("src.extract.get_taxonomy_for_prompt", return_value="TAX")
    @patch("src.doc_type_classifier.classify_doc_type", return_value="general")
    @patch("src.doc_type_classifier.should_extract_deep", return_value=False)
    @patch("src.freshness.compute_freshness_fields", return_value={})
    @patch("src.extract.extract_source_date", return_value=None)
    @patch("src.extract._enrich_facts", return_value=[])
    def test_cost_from_provider(self, *mocks):
        from src.inventory import SourceFile, FileType
        from src.text_extract import TextExtractionResult
        from src.providers.base import ExtractionResponse

        class FakeProvider:
            def extract(self, request):
                return ExtractionResponse(
                    text='{"title":"T","summary":"S","topics":[],"products":[],"people":[]}',
                    input_tokens=5000, output_tokens=1000,
                    model="gemini-3-flash-preview", provider="google",
                    cost_estimate=0.0055,
                )

        with (
            patch("src.providers.router.route_model", return_value=("gemini-3-flash-preview", "text_default")),
            patch("src.providers.router.get_provider", return_value=FakeProvider()),
            patch("src.providers.router.has_anthropic_key", return_value=False),
            patch("src.providers.validator.validate_and_retry", side_effect=lambda r, req: (r, False)),
        ):
            from src.extract import extract_from_text
            source = SourceFile(path=Path("t.docx"), type=FileType.DOCUMENT, size_bytes=1000, name="t")
            text_result = TextExtractionResult(text="content", char_count=7, extractor="python-docx")
            result = extract_from_text(source, {"prompts": {"extract": "Extract."}}, text_result)

        assert result.extraction_cost_usd == 0.0055


class TestCostInFrontmatter:
    def _render(self, cost):
        templates_dir = Path(__file__).parent.parent / "templates"
        env = Environment(loader=FileSystemLoader(str(templates_dir)), trim_blocks=True, lstrip_blocks=True)
        env.filters["tojson_raw"] = lambda v: json.dumps(v, ensure_ascii=False)
        tmpl = env.get_template("extract.md.j2")
        return tmpl.render(
            source_file="C:/test/f.pdf", content_type="document", title="T",
            date="2026-03-23", summary="S.", key_points=[], topics=["X"],
            people=[], products=[], slides=[], language="en", quality="full",
            duration_min=None, transcript_excerpt="", model="gemini-3-flash-preview",
            tokens_used=100, links_line="", source_tool="knowledge-extractor",
            extraction_cost_usd=cost,
        )

    def test_cost_in_frontmatter(self):
        content = self._render(0.0055)
        assert "extraction_cost_usd: 0.0055" in content

    def test_cost_zero_tier1(self):
        content = self._render(0.0)
        assert "extraction_cost_usd: 0.0000" in content
