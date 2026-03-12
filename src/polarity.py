"""
Deterministic polarity detection for extracted facts.

Tags each fact as positive/negative/unknown using keyword pattern matching.
No LLM — purely regex-based. Conservative: defaults to unknown.

Usage:
    from src.polarity import detect_polarity

    detect_polarity("WMS supports REST API")        # "positive"
    detect_polarity("WMS does NOT use Snowflake")    # "negative"
    detect_polarity("Data goes through validation")  # "unknown"
"""

import re

NEGATIVE_PATTERNS = [
    r"\bnot\s+support",
    r"\bnot\s+supported\b",
    r"\bnot\s+available\b",
    r"\bnot\s+included\b",
    r"\bnot\s+recommended\b",
    r"\bnot\s+applicable\b",
    r"\bdoes\s+not\b",
    r"\bdo\s+not\b",
    r"\bcannot\b",
    r"\bcan't\b",
    r"\bdoesn't\b",
    r"\bdon't\b",
    r"\bno\s+support\b",
    r"\bunavailable\b",
    r"\bexcluded\b",
    r"\blacks?\b",
    r"\bwithout\b",
    r"\bn/a\b",
    # Single "not" last — less specific, checked after multi-word patterns
    r"\bnot\b",
]

POSITIVE_PATTERNS = [
    r"\bsupports?\b",
    r"\bprovides?\b",
    r"\bincludes?\b",
    r"\boffers?\b",
    r"\benables?\b",
    r"\bdelivers?\b",
    r"\bavailable\b",
    r"\bbuilt[\s-]in\b",
    r"\bnative\b",
    r"\bcertified\b",
    r"\bcompliant\b",
    r"\bintegrated\b",
]


def detect_polarity(fact_text: str) -> str:
    """Classify a fact as positive, negative, or unknown.

    Conservative — defaults to unknown. If both positive and negative
    patterns match (ambiguous), returns unknown.

    Args:
        fact_text: The fact string to classify

    Returns:
        "positive", "negative", or "unknown"
    """
    if not fact_text:
        return "unknown"

    text_lower = fact_text.lower()

    has_negative = any(re.search(p, text_lower) for p in NEGATIVE_PATTERNS)
    has_positive = any(re.search(p, text_lower) for p in POSITIVE_PATTERNS)

    if has_negative and has_positive:
        return "unknown"
    if has_negative:
        return "negative"
    if has_positive:
        return "positive"
    return "unknown"
