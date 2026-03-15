"""Post-extraction validation with Sonnet escalation on failure."""

import json
import logging

from src.providers.base import ExtractionRequest, ExtractionResponse

logger = logging.getLogger(__name__)


def validate_and_retry(
    response: ExtractionResponse,
    original_request: ExtractionRequest,
) -> tuple[ExtractionResponse, bool]:
    """Validate extraction JSON. If invalid, retry once with Sonnet.

    Args:
        response: The extraction response to validate
        original_request: The original request (for retry)

    Returns:
        (valid_response, was_escalated) tuple
    """
    try:
        parsed = _parse_json(response.text)
        if _validate_structure(parsed):
            return response, False
    except (json.JSONDecodeError, ValueError):
        pass

    logger.warning(
        "Validation failed for %s (provider=%s), escalating to Sonnet",
        response.model,
        response.provider,
    )

    from src.providers.router import ESCALATION_MODEL, get_provider

    escalation_request = ExtractionRequest(
        system_prompt=original_request.system_prompt,
        user_prompt=original_request.user_prompt,
        model=ESCALATION_MODEL,
        max_tokens=original_request.max_tokens,
        temperature=0.1,
        response_format=original_request.response_format,
    )

    provider = get_provider(ESCALATION_MODEL)
    retry_response = provider.extract(escalation_request)

    return retry_response, True


def _parse_json(text: str) -> dict:
    """Parse JSON, stripping markdown fences if present."""
    cleaned = text.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    elif cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    return json.loads(cleaned.strip())


def _validate_structure(data: dict) -> bool:
    """Check that extracted data has minimum required structure."""
    # Must have a title
    if not data.get("title"):
        return False
    # Must have some content indicator
    if not data.get("summary") and not data.get("key_points"):
        return False
    return True
