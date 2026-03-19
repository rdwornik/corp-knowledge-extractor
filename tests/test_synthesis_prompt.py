"""Tests for synthesis prompt enrichment — key_facts, action_items, entities."""

import json
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from src.extract import ExtractionResult
from src.inventory import SourceFile, FileType
from src.synthesize import _run_synthesis


def _make_result(
    name="test.pptx",
    title="Test Doc",
    summary="A test document",
    facts=None,
    overlay=None,
    raw_json=None,
):
    sf = SourceFile(path=Path(name), name=name, type=FileType.SLIDES, size_bytes=100)
    r = ExtractionResult(source_file=sf, title=title, summary=summary)
    r.facts = facts or []
    r.overlay = overlay or {}
    r.raw_json = raw_json or {}
    return r


class TestSynthesisPrompt:
    def test_synthesis_prompt_includes_facts(self):
        """Key facts appear in the synthesis prompt sent to Gemini."""
        result = _make_result(
            facts=[
                {"fact": "Revenue grew 15% to $2.3M in Q4 2025"},
                {"fact": "1706 customers onboarded globally"},
                {"fact": "Short"},
            ],
        )
        extracts = {"test.pptx": result}
        config = {
            "gemini": {"api_key_env": "GEMINI_API_KEY", "model": "gemini-3-flash-preview"},
            "prompts": {"synthesize": "Synthesize the following:"},
        }

        captured_prompt = {}

        def mock_generate(**kwargs):
            captured_prompt["contents"] = kwargs.get("contents", [])
            mock_resp = MagicMock()
            mock_resp.text = '{"synthesis": "test"}'
            return mock_resp

        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = mock_generate

        with (
            patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}),
            patch("google.genai.Client", return_value=mock_client),
        ):
            _run_synthesis(extracts, config)

        prompt_text = str(captured_prompt["contents"][0].text)
        assert "Revenue grew 15%" in prompt_text
        assert "1706 customers" in prompt_text
        assert "Do not generalize" in prompt_text

    def test_synthesis_prompt_includes_actions(self):
        """Action items from overlays appear in the synthesis prompt."""
        result = _make_result(
            overlay={
                "action_items": [
                    {"action": "Schedule architecture review", "owner": "Alice"},
                ],
                "decisions_made": ["Approved cloud migration plan"],
            },
        )
        extracts = {"test.pptx": result}
        config = {
            "gemini": {"api_key_env": "GEMINI_API_KEY", "model": "gemini-3-flash-preview"},
            "prompts": {"synthesize": "Synthesize the following:"},
        }

        captured_prompt = {}

        def mock_generate(**kwargs):
            captured_prompt["contents"] = kwargs.get("contents", [])
            mock_resp = MagicMock()
            mock_resp.text = '{"synthesis": "test"}'
            return mock_resp

        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = mock_generate

        with (
            patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}),
            patch("google.genai.Client", return_value=mock_client),
        ):
            _run_synthesis(extracts, config)

        prompt_text = str(captured_prompt["contents"][0].text)
        assert "Schedule architecture review" in prompt_text
        assert "Approved cloud migration plan" in prompt_text
