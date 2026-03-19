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

    # Strategy 4: json-repair (handles unescaped inner quotes, trailing commas, etc.)
    try:
        from json_repair import repair_json

        repaired = repair_json(text, return_objects=True)
        if isinstance(repaired, dict) and repaired:
            log.debug("json-repair recovered malformed JSON")
            return repaired
    except Exception:
        pass

    log.error("Cannot parse LLM JSON. Full response (%d chars):\n%s", len(text), text[:5000])
    raise ValueError("Failed to parse LLM JSON response")


def normalize_string_list(items: list) -> list[str]:
    """Ensure all items in a list are strings. LLMs sometimes return dicts.

    Handles:
    - str → pass through
    - dict with "name" or "title" key → "name (role)" or "title"
    - anything else → str(item)
    """
    result = []
    for item in items:
        if isinstance(item, str):
            result.append(item)
        elif isinstance(item, dict):
            name = item.get("name", item.get("title", ""))
            role = item.get("role", "")
            if name and role:
                result.append(f"{name} ({role})")
            elif name:
                result.append(str(name))
            else:
                result.append(str(item))
        else:
            result.append(str(item))
    return result
