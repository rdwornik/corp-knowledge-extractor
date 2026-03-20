"""Gemini Flash / Pro provider for multimodal and large-context extraction."""

import logging
import os

from src.providers.base import ExtractionProvider, ExtractionRequest, ExtractionResponse

log = logging.getLogger(__name__)

# Per-million-token pricing (USD)
GEMINI_PRICING = {
    "gemini-3-flash-preview": {"input": 0.50, "output": 3.00},
    "gemini-3.1-flash-lite": {"input": 0.25, "output": 1.50},
    "gemini-3.1-pro-preview": {"input": 2.00, "output": 12.00},
}


class GeminiProvider(ExtractionProvider):
    """Gemini-based extraction provider for text-only requests.

    This is used when Gemini is selected for text extraction (large context,
    batch mode, or explicit override). Multimodal extraction (Tier 3) still
    uses the direct Gemini calls in extract.py since it needs file uploads
    and inline image parts that don't fit the simple text request model.
    """

    def __init__(self):
        from google import genai

        api_key_env = os.environ.get("GEMINI_API_KEY_ENV", "GEMINI_API_KEY")
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise ValueError(f"{api_key_env} not set. Check global env (keys list).")
        self.client = genai.Client(api_key=api_key)

    def extract(self, request: ExtractionRequest) -> ExtractionResponse:
        """Send text extraction request to Gemini."""
        from google.genai import types

        log.info("Calling Gemini %s (%d max_tokens)...", request.model, request.max_tokens)

        # Gemini uses system_instruction + user content
        config_kwargs = {
            "temperature": request.temperature,
            "max_output_tokens": request.max_tokens,
        }
        if request.system_prompt:
            config_kwargs["system_instruction"] = request.system_prompt
        if request.response_format == "json":
            config_kwargs["response_mime_type"] = "application/json"

        response = self.client.models.generate_content(
            model=request.model,
            contents=[types.Part.from_text(text=request.user_prompt)],
            config=types.GenerateContentConfig(**config_kwargs),
        )

        text = response.text or ""
        input_tokens = 0
        output_tokens = 0
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            input_tokens = getattr(response.usage_metadata, "prompt_token_count", 0) or 0
            output_tokens = getattr(response.usage_metadata, "candidates_token_count", 0) or 0

        pricing = GEMINI_PRICING.get(request.model, {"input": 0.50, "output": 3.00})
        cost = (input_tokens * pricing["input"] + output_tokens * pricing["output"]) / 1_000_000

        log.info(
            "Gemini response: %d input + %d output tokens, cost=$%.6f",
            input_tokens,
            output_tokens,
            cost,
        )

        return ExtractionResponse(
            text=text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model=request.model,
            provider="google",
            cost_estimate=cost,
        )
