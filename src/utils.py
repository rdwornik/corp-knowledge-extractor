"""
Shared utilities for the knowledge extraction pipeline.
"""

import json
import logging
import re

log = logging.getLogger(__name__)


def parse_llm_json(text: str) -> dict:
    """
    Parse JSON from an LLM response, handling common issues:
    - Markdown code fences (```json ... ```)
    - Trailing commas before } or ]
    - JSON object buried in surrounding prose

    Args:
        text: Raw text from Gemini response

    Returns:
        Parsed dict

    Raises:
        ValueError: If all parsing strategies fail
    """
    # Strip markdown fences
    text = re.sub(r"^```(?:json)?\s*", "", text.strip())
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()

    # Strategy 1: direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Strategy 2: remove trailing commas before } or ]
    cleaned = re.sub(r",\s*([}\]])", r"\1", text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Strategy 3: extract first {...} block from surrounding prose
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            # Try trailing-comma fix on the extracted block too
            try:
                return json.loads(re.sub(r",\s*([}\]])", r"\1", match.group()))
            except json.JSONDecodeError:
                pass

    log.error("Cannot parse LLM JSON. First 500 chars: %s", text[:500])
    raise ValueError(f"Failed to parse LLM JSON response")
