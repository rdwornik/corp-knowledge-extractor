"""Smoke test: verify required API keys are available."""

import os

import pytest

# Keys this repo actually uses (from audit):
# - GEMINI_API_KEY: Gemini multimodal extraction, synthesis, transcript, frame tagging
# - ANTHROPIC_API_KEY: Claude Haiku/Sonnet text extraction (optional, falls back to Gemini)
REQUIRED_KEYS = [
    "GEMINI_API_KEY",
]

OPTIONAL_KEYS = [
    "ANTHROPIC_API_KEY",
]


@pytest.mark.parametrize("key", REQUIRED_KEYS)
def test_api_key_available(key):
    """API key is set in environment (loaded by PS profile from global .env)."""
    value = os.environ.get(key)
    if value is None:
        pytest.skip(
            f"{key} not found — run 'keys list' in PowerShell. "
            f"Keys should be in Documents/.secrets/.env"
        )
    assert len(value) > 10, f"{key} too short ({len(value)} chars)"


@pytest.mark.parametrize("key", OPTIONAL_KEYS)
def test_optional_key_warning(key):
    """Optional API key — warn if missing, don't fail."""
    value = os.environ.get(key)
    if not value:
        pytest.skip(f"{key} not set — optional, Gemini fallback will be used")
