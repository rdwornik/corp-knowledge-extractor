"""Claude Haiku / Sonnet provider for text extraction."""

import logging
import os

from src.providers.base import ExtractionProvider, ExtractionRequest, ExtractionResponse

log = logging.getLogger(__name__)

# Per-million-token pricing (USD)
ANTHROPIC_PRICING = {
    "claude-haiku-4-5-20251001": {"input": 1.00, "output": 5.00},
    "claude-sonnet-4-6": {"input": 3.00, "output": 15.00},
}


class AnthropicProvider(ExtractionProvider):
    """Claude-based extraction provider."""

    def __init__(self):
        import anthropic

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not set. Set it in .env or environment.")
        self.client = anthropic.Anthropic(api_key=api_key)

    def extract(self, request: ExtractionRequest) -> ExtractionResponse:
        """Send text extraction request to Claude."""
        log.info("Calling Claude %s (%d max_tokens)...", request.model, request.max_tokens)

        response = self.client.messages.create(
            model=request.model,
            max_tokens=request.max_tokens,
            system=request.system_prompt,
            messages=[{"role": "user", "content": request.user_prompt}],
            temperature=request.temperature,
        )

        text = response.content[0].text
        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens

        pricing = ANTHROPIC_PRICING.get(request.model, {"input": 1.0, "output": 5.0})
        cost = (input_tokens * pricing["input"] + output_tokens * pricing["output"]) / 1_000_000

        log.info(
            "Claude response: %d input + %d output tokens, cost=$%.6f",
            input_tokens,
            output_tokens,
            cost,
        )

        return ExtractionResponse(
            text=text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model=request.model,
            provider="anthropic",
            cost_estimate=cost,
        )
