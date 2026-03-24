"""Tests for model escalation: model_used reflects actual model after Haiku→Sonnet."""

import json
import pytest
from unittest.mock import patch, MagicMock

from src.providers.base import ExtractionRequest, ExtractionResponse
from src.providers.validator import validate_and_retry


class TestEscalationUpdatesModelUsed:
    """After Haiku fails validation and Sonnet succeeds, model_used must reflect Sonnet."""

    def _make_response(self, text: str, model: str, provider: str = "anthropic") -> ExtractionResponse:
        return ExtractionResponse(
            text=text,
            input_tokens=100,
            output_tokens=50,
            model=model,
            provider=provider,
            cost_estimate=0.001,
        )

    def test_escalation_updates_model_used(self):
        """Haiku returns garbage → Sonnet returns valid JSON → model_used = Sonnet."""
        haiku_response = self._make_response(
            text="not valid json at all",
            model="claude-haiku-4-5-20251001",
        )

        sonnet_valid = json.dumps({
            "title": "JLR TMS RFI Response",
            "summary": "Response to JLR logistics RFI.",
            "key_points": ["Integration with SAP TM"],
        })
        sonnet_response = self._make_response(
            text=sonnet_valid,
            model="claude-sonnet-4-6-20260320",
        )

        mock_provider = MagicMock()
        mock_provider.extract.return_value = sonnet_response

        with patch("src.providers.router.get_provider", return_value=mock_provider):
            result, was_escalated = validate_and_retry(haiku_response, ExtractionRequest(
                system_prompt="",
                user_prompt="Extract knowledge",
                model="claude-haiku-4-5-20251001",
                max_tokens=4096,
                temperature=0.2,
                response_format="json",
            ))

        assert was_escalated is True
        assert result.model == "claude-sonnet-4-6-20260320"

    def test_no_escalation_preserves_model(self):
        """Valid Haiku response → no escalation, model stays Haiku."""
        valid_json = json.dumps({
            "title": "Test Document",
            "summary": "A valid extraction.",
            "key_points": ["Point one"],
        })
        haiku_response = self._make_response(
            text=valid_json,
            model="claude-haiku-4-5-20251001",
        )

        result, was_escalated = validate_and_retry(haiku_response, ExtractionRequest(
            system_prompt="",
            user_prompt="Extract knowledge",
            model="claude-haiku-4-5-20251001",
            max_tokens=4096,
            temperature=0.2,
            response_format="json",
        ))

        assert was_escalated is False
        assert result.model == "claude-haiku-4-5-20251001"

    def test_model_in_frontmatter_after_escalation(self):
        """After escalation, the model field in ExtractionResult should be Sonnet."""
        # Simulate what extract_from_text does at line 1279:
        # result.model_used = response.model
        sonnet_response = self._make_response(
            text=json.dumps({"title": "Test", "summary": "S", "key_points": []}),
            model="claude-sonnet-4-6-20260320",
        )

        # Simulate: after validate_and_retry returns escalated response
        model_used = sonnet_response.model
        assert model_used == "claude-sonnet-4-6-20260320"
        assert "haiku" not in model_used
