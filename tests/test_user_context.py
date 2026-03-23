"""Tests for user_context wiring from manifest/CLI to LLM prompt."""

import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.manifest import Manifest, ManifestEntry
from src.extract import _prepend_user_context, ExtractionResult


# ---------------------------------------------------------------------------
# Unit: _prepend_user_context helper
# ---------------------------------------------------------------------------


class TestPrependUserContext:
    def test_with_context(self):
        result = _prepend_user_context("Extract knowledge.", "JLR TMS RFP")
        assert result.startswith("ADDITIONAL CONTEXT FROM USER:\nJLR TMS RFP\n")
        assert "prioritize facts relevant to this context" in result
        assert result.endswith("Extract knowledge.")

    def test_empty_context(self):
        original = "Extract knowledge."
        result = _prepend_user_context(original, "")
        assert result == original

    def test_none_like_empty(self):
        original = "Extract knowledge."
        result = _prepend_user_context(original, "")
        assert result == original


# ---------------------------------------------------------------------------
# Integration: user_context from manifest JSON
# ---------------------------------------------------------------------------


class TestUserContextFromManifest:
    def test_manifest_entry_has_user_context(self, tmp_path):
        manifest_file = tmp_path / "manifest.json"
        manifest_file.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "project": "test",
                    "output_dir": str(tmp_path / "output"),
                    "files": [
                        {
                            "id": "f1",
                            "path": str(tmp_path / "test.pdf"),
                            "doc_type": "document",
                            "user_context": "JLR TMS RFP response",
                        }
                    ],
                }
            )
        )
        manifest = Manifest.from_file(manifest_file)
        assert manifest.files[0].user_context == "JLR TMS RFP response"

    def test_manifest_entry_missing_user_context_defaults_empty(self, tmp_path):
        manifest_file = tmp_path / "manifest.json"
        manifest_file.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "project": "test",
                    "output_dir": str(tmp_path / "output"),
                    "files": [
                        {
                            "id": "f1",
                            "path": str(tmp_path / "test.pdf"),
                            "doc_type": "document",
                        }
                    ],
                }
            )
        )
        manifest = Manifest.from_file(manifest_file)
        assert manifest.files[0].user_context == ""


# ---------------------------------------------------------------------------
# Integration: user_context reaches the LLM prompt in extract_from_text
# ---------------------------------------------------------------------------


class TestUserContextInPrompt:
    @patch("src.extract.get_taxonomy_for_prompt", return_value="TAXONOMY_BLOCK")
    @patch("src.doc_type_classifier.classify_doc_type", return_value="general")
    @patch("src.doc_type_classifier.should_extract_deep", return_value=False)
    @patch("src.freshness.compute_freshness_fields", return_value={})
    @patch("src.extract.extract_source_date", return_value=None)
    @patch("src.extract._enrich_facts", return_value=[])
    def test_user_context_appears_in_prompt(
        self,
        mock_enrich,
        mock_date,
        mock_fresh,
        mock_deep,
        mock_classify,
        mock_taxonomy,
    ):
        """When user_context is provided, it appears in the prompt sent to the LLM."""
        from src.inventory import SourceFile, FileType
        from src.text_extract import TextExtractionResult

        captured_prompt = {}

        class FakeProvider:
            def extract(self, request):
                captured_prompt["user_prompt"] = request.user_prompt
                from src.providers.base import ExtractionResponse
                return ExtractionResponse(
                    text='{"title": "Test", "summary": "Test summary", "topics": [], "products": [], "people": []}',
                    input_tokens=100,
                    output_tokens=50,
                    model="gemini-3-flash-preview",
                    provider="google",
                    cost_estimate=0.001,
                )

        fake_provider = FakeProvider()

        with (
            patch("src.providers.router.route_model", return_value=("gemini-3-flash-preview", "text_default")),
            patch("src.providers.router.get_provider", return_value=fake_provider),
            patch("src.providers.router.has_anthropic_key", return_value=False),
            patch("src.providers.validator.validate_and_retry", side_effect=lambda r, req: (r, False)),
        ):
            from src.extract import extract_from_text

            source = SourceFile(
                path=Path("test.docx"),
                type=FileType.DOCUMENT,
                size_bytes=1000,
                name="test",
            )
            text_result = TextExtractionResult(
                text="Sample document text for testing.",
                char_count=33,
                extractor="python-docx",
            )
            config = {"prompts": {"extract": "Extract knowledge from this file."}}

            extract_from_text(source, config, text_result, user_context="JLR TMS RFP")

        assert "ADDITIONAL CONTEXT FROM USER:" in captured_prompt["user_prompt"]
        assert "JLR TMS RFP" in captured_prompt["user_prompt"]
        assert "prioritize facts relevant to this context" in captured_prompt["user_prompt"]

    @patch("src.extract.get_taxonomy_for_prompt", return_value="TAXONOMY_BLOCK")
    @patch("src.doc_type_classifier.classify_doc_type", return_value="general")
    @patch("src.doc_type_classifier.should_extract_deep", return_value=False)
    @patch("src.freshness.compute_freshness_fields", return_value={})
    @patch("src.extract.extract_source_date", return_value=None)
    @patch("src.extract._enrich_facts", return_value=[])
    def test_empty_context_no_prefix(
        self,
        mock_enrich,
        mock_date,
        mock_fresh,
        mock_deep,
        mock_classify,
        mock_taxonomy,
    ):
        """When user_context is empty, prompt is unchanged."""
        from src.inventory import SourceFile, FileType
        from src.text_extract import TextExtractionResult

        captured_prompt = {}

        class FakeProvider:
            def extract(self, request):
                captured_prompt["user_prompt"] = request.user_prompt
                from src.providers.base import ExtractionResponse
                return ExtractionResponse(
                    text='{"title": "Test", "summary": "Test summary", "topics": [], "products": [], "people": []}',
                    input_tokens=100,
                    output_tokens=50,
                    model="gemini-3-flash-preview",
                    provider="google",
                    cost_estimate=0.001,
                )

        fake_provider = FakeProvider()

        with (
            patch("src.providers.router.route_model", return_value=("gemini-3-flash-preview", "text_default")),
            patch("src.providers.router.get_provider", return_value=fake_provider),
            patch("src.providers.router.has_anthropic_key", return_value=False),
            patch("src.providers.validator.validate_and_retry", side_effect=lambda r, req: (r, False)),
        ):
            from src.extract import extract_from_text

            source = SourceFile(
                path=Path("test.docx"),
                type=FileType.DOCUMENT,
                size_bytes=1000,
                name="test",
            )
            text_result = TextExtractionResult(
                text="Sample document text.",
                char_count=21,
                extractor="python-docx",
            )
            config = {"prompts": {"extract": "Extract knowledge from this file."}}

            extract_from_text(source, config, text_result, user_context="")

        assert "ADDITIONAL CONTEXT FROM USER:" not in captured_prompt["user_prompt"]


# ---------------------------------------------------------------------------
# Integration: --context CLI flag parsed correctly
# ---------------------------------------------------------------------------


class TestUserContextCLIFlag:
    def test_context_option_exists(self):
        """The process command accepts --context."""
        from scripts.run import process
        param_names = [p.name for p in process.params]
        assert "context" in param_names

    def test_context_option_default_empty(self):
        """--context defaults to empty string."""
        from scripts.run import process
        ctx_param = next(p for p in process.params if p.name == "context")
        assert ctx_param.default == ""


# ---------------------------------------------------------------------------
# FIX 2: Escalation model_used reflects actual Sonnet model
# ---------------------------------------------------------------------------


class TestEscalationModelUsed:
    @patch("src.extract.get_taxonomy_for_prompt", return_value="TAXONOMY_BLOCK")
    @patch("src.doc_type_classifier.classify_doc_type", return_value="general")
    @patch("src.doc_type_classifier.should_extract_deep", return_value=False)
    @patch("src.freshness.compute_freshness_fields", return_value={})
    @patch("src.extract.extract_source_date", return_value=None)
    @patch("src.extract._enrich_facts", return_value=[])
    def test_escalation_model_used_is_sonnet(
        self,
        mock_enrich,
        mock_date,
        mock_fresh,
        mock_deep,
        mock_classify,
        mock_taxonomy,
    ):
        """After Haiku→Sonnet escalation, model_used reflects the Sonnet model string."""
        from src.inventory import SourceFile, FileType
        from src.text_extract import TextExtractionResult
        from src.providers.base import ExtractionResponse

        haiku_response = ExtractionResponse(
            text='{"title": "Test", "summary": "Test", "topics": [], "products": [], "people": []}',
            input_tokens=100,
            output_tokens=50,
            model="claude-haiku-4-5-20251001",
            provider="anthropic",
            cost_estimate=0.001,
        )

        sonnet_response = ExtractionResponse(
            text='{"title": "Test", "summary": "Test summary", "topics": [], "products": [], "people": []}',
            input_tokens=100,
            output_tokens=50,
            model="claude-sonnet-4-6-20260310",
            provider="anthropic",
            cost_estimate=0.005,
        )

        class FakeProvider:
            def extract(self, request):
                return haiku_response

        fake_provider = FakeProvider()

        def mock_validate_and_retry(response, request):
            # Simulate: Haiku output fails validation, Sonnet succeeds
            return sonnet_response, True

        with (
            patch("src.providers.router.route_model", return_value=("claude-haiku-4-5-20251001", "text_default")),
            patch("src.providers.router.get_provider", return_value=fake_provider),
            patch("src.providers.router.has_anthropic_key", return_value=True),
            patch("src.providers.validator.validate_and_retry", side_effect=mock_validate_and_retry),
        ):
            from src.extract import extract_from_text

            source = SourceFile(
                path=Path("test.docx"),
                type=FileType.DOCUMENT,
                size_bytes=1000,
                name="test",
            )
            text_result = TextExtractionResult(
                text="Sample text.",
                char_count=12,
                extractor="python-docx",
            )
            config = {"prompts": {"extract": "Extract knowledge."}}

            result = extract_from_text(source, config, text_result)

        # model_used must be the Sonnet model that actually produced the output
        assert result.model_used == "claude-sonnet-4-6-20260310"
        assert result.model_used != "claude-haiku-4-5-20251001"
