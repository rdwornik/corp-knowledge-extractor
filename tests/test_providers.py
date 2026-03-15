"""Tests for the multi-provider extraction abstraction."""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from src.providers.base import ExtractionRequest, ExtractionResponse
from src.providers.router import (
    route_model,
    get_provider,
    ANTHROPIC_MODELS,
    GEMINI_MODELS,
    DEFAULT_TEXT_MODEL,
    DEFAULT_MULTIMODAL_MODEL,
    DEFAULT_LARGE_CONTEXT_MODEL,
    HAIKU_TOKEN_LIMIT,
)


# ── Routing tests ────────────────────────────────────────────────


class TestRouteModel:
    """Test model routing logic."""

    def test_text_default_routes_to_haiku(self):
        """Text extraction routes to Haiku by default."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            model = route_model(tier=2, text_length=10000)
            assert model == DEFAULT_TEXT_MODEL

    def test_multimodal_always_gemini(self):
        """Tier 3 always routes to Gemini regardless of other factors."""
        model = route_model(tier=3, text_length=1000)
        assert model == DEFAULT_MULTIMODAL_MODEL

    def test_large_document_routes_to_gemini(self):
        """Documents exceeding Haiku context limit route to Gemini."""
        # 190K tokens * 4 chars/token = 760K chars
        large_text = HAIKU_TOKEN_LIMIT * 4 + 1000
        model = route_model(tier=2, text_length=large_text)
        assert model == DEFAULT_LARGE_CONTEXT_MODEL

    def test_model_override_takes_precedence(self):
        """--model flag overrides all routing logic."""
        model = route_model(tier=2, text_length=1000, model_override="gemini-3-flash-preview")
        assert model == "gemini-3-flash-preview"

    def test_batch_mode_forces_gemini(self):
        """Batch mode forces Gemini for 50% batch API discount."""
        model = route_model(tier=2, text_length=1000, batch_mode=True)
        assert model == DEFAULT_LARGE_CONTEXT_MODEL

    def test_no_anthropic_key_falls_back_to_gemini(self):
        """Without ANTHROPIC_API_KEY, routes to Gemini with warning."""
        with patch.dict(os.environ, {}, clear=True):
            # Also clear ANTHROPIC_API_KEY if set
            os.environ.pop("ANTHROPIC_API_KEY", None)
            model = route_model(tier=2, text_length=1000)
            assert model in GEMINI_MODELS

    def test_tier1_with_override(self):
        """Tier 1 with model override still returns the override."""
        model = route_model(tier=1, model_override="claude-sonnet-4-6")
        assert model == "claude-sonnet-4-6"


class TestGetProvider:
    """Test provider instantiation."""

    def test_anthropic_model_without_key_falls_back(self):
        """Anthropic model without API key falls back to Gemini provider."""
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("ANTHROPIC_API_KEY", None)
            os.environ["GEMINI_API_KEY"] = "test-gemini-key"
            provider = get_provider("claude-haiku-4-5-20251001")
            # Should be a GeminiProvider (fallback)
            from src.providers.gemini_provider import GeminiProvider

            assert isinstance(provider, GeminiProvider)

    def test_unknown_model_raises(self):
        """Unknown model raises ValueError."""
        with pytest.raises(ValueError, match="Unknown model"):
            get_provider("gpt-4-turbo")

    def test_gemini_model_creates_gemini_provider(self):
        """Gemini model creates GeminiProvider."""
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
            provider = get_provider("gemini-3-flash-preview")
            from src.providers.gemini_provider import GeminiProvider

            assert isinstance(provider, GeminiProvider)


# ── Request/Response dataclass tests ─────────────────────────────


class TestExtractionRequest:
    """Test ExtractionRequest dataclass."""

    def test_defaults(self):
        req = ExtractionRequest(
            system_prompt="Extract facts",
            user_prompt="Document text here",
            model="claude-haiku-4-5-20251001",
        )
        assert req.max_tokens == 4096
        assert req.temperature == 0.2
        assert req.response_format == "json"

    def test_custom_values(self):
        req = ExtractionRequest(
            system_prompt="Extract",
            user_prompt="Text",
            model="claude-sonnet-4-6",
            max_tokens=8192,
            temperature=0.1,
            response_format="text",
        )
        assert req.max_tokens == 8192
        assert req.temperature == 0.1


class TestExtractionResponse:
    """Test ExtractionResponse dataclass."""

    def test_fields(self):
        resp = ExtractionResponse(
            text='{"title": "Test"}',
            input_tokens=100,
            output_tokens=50,
            model="claude-haiku-4-5-20251001",
            provider="anthropic",
            cost_estimate=0.00035,
        )
        assert resp.provider == "anthropic"
        assert resp.cost_estimate == 0.00035


# ── Validator tests ──────────────────────────────────────────────


class TestValidator:
    """Test post-extraction validation."""

    def test_accepts_good_json(self):
        """Valid structured JSON passes validation."""
        from src.providers.validator import _validate_structure, _parse_json

        good_json = '{"title": "Test Doc", "summary": "A test", "key_points": ["fact1"]}'
        parsed = _parse_json(good_json)
        assert _validate_structure(parsed) is True

    def test_rejects_no_title(self):
        """Missing title fails validation."""
        from src.providers.validator import _validate_structure

        assert _validate_structure({"summary": "text"}) is False

    def test_rejects_no_content(self):
        """Missing both summary and key_points fails validation."""
        from src.providers.validator import _validate_structure

        assert _validate_structure({"title": "Test"}) is False

    def test_parse_json_strips_fences(self):
        """JSON parsing handles markdown code fences."""
        from src.providers.validator import _parse_json

        fenced = '```json\n{"title": "Test"}\n```'
        parsed = _parse_json(fenced)
        assert parsed["title"] == "Test"

    def test_parse_json_plain(self):
        """Plain JSON parses correctly."""
        from src.providers.validator import _parse_json

        plain = '{"title": "Test", "summary": "ok"}'
        parsed = _parse_json(plain)
        assert parsed["title"] == "Test"


# ── Cost tracker tests ───────────────────────────────────────────


class TestCostTracker:
    """Test cost logging and monthly spend calculation."""

    def test_log_and_read_cost(self):
        """Costs logged to JSONL and monthly total computed."""
        from src.providers import cost_tracker

        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False, mode="w") as f:
            tmp_path = Path(f.name)

        # Patch COST_LOG to temp file
        original = cost_tracker.COST_LOG
        cost_tracker.COST_LOG = tmp_path

        try:
            # Clean start
            if tmp_path.exists():
                tmp_path.unlink()

            assert cost_tracker.get_monthly_spend() == 0.0

            cost_tracker.log_cost(
                model="claude-haiku-4-5-20251001",
                provider="anthropic",
                input_tokens=1000,
                output_tokens=500,
                cost=0.0035,
            )
            cost_tracker.log_cost(
                model="gemini-3-flash-preview",
                provider="google",
                input_tokens=2000,
                output_tokens=1000,
                cost=0.004,
            )

            total = cost_tracker.get_monthly_spend()
            assert abs(total - 0.0075) < 0.0001

            # Verify JSONL format
            with open(tmp_path) as f:
                lines = f.readlines()
            assert len(lines) == 2
            entry = json.loads(lines[0])
            assert entry["model"] == "claude-haiku-4-5-20251001"
            assert entry["provider"] == "anthropic"
        finally:
            cost_tracker.COST_LOG = original
            if tmp_path.exists():
                tmp_path.unlink()

    def test_budget_check(self):
        """Budget check returns False when exceeded."""
        from src.providers import cost_tracker

        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False, mode="w") as f:
            tmp_path = Path(f.name)

        original = cost_tracker.COST_LOG
        cost_tracker.COST_LOG = tmp_path

        try:
            if tmp_path.exists():
                tmp_path.unlink()

            # Within budget
            assert cost_tracker.check_budget(20.0) is True

            # Log a huge cost
            cost_tracker.log_cost("test", "test", 0, 0, 25.0)

            # Over budget
            assert cost_tracker.check_budget(20.0) is False
        finally:
            cost_tracker.COST_LOG = original
            if tmp_path.exists():
                tmp_path.unlink()


# ── Model set tests ──────────────────────────────────────────────


class TestModelSets:
    """Verify model set definitions."""

    def test_anthropic_models_defined(self):
        assert "claude-haiku-4-5-20251001" in ANTHROPIC_MODELS
        assert "claude-sonnet-4-6" in ANTHROPIC_MODELS

    def test_gemini_models_defined(self):
        assert "gemini-3-flash-preview" in GEMINI_MODELS

    def test_no_overlap(self):
        """Anthropic and Gemini model sets don't overlap."""
        assert ANTHROPIC_MODELS & GEMINI_MODELS == set()
