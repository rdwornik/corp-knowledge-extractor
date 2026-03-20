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


# Extended keyword lists for richer classification
_POSITIVE_KEYWORDS = [
    "improvement", "increase", "growth", "reduction in cost", "faster",
    "better", "success", "achieved", "optimized", "streamlined", "automated",
    "enhanced", "enabled", "efficient", "savings", "benefit",
]
_NEGATIVE_KEYWORDS = [
    "risk", "concern", "fear", "barrier", "limitation", "does not",
    "cannot", "failure", "degradation", "challenge", "delay", "expensive",
    "complex", "issue", "problem", "downtime", "outage",
]


def classify_fact_polarity(fact: str) -> str:
    """Classify a single fact as positive/negative/mixed/neutral using keywords.

    Broader than detect_polarity() which uses regex patterns — this uses
    substring keyword matching for sentiment-like classification.
    """
    if not fact:
        return "neutral"
    fact_lower = fact.lower()
    has_positive = any(kw in fact_lower for kw in _POSITIVE_KEYWORDS)
    has_negative = any(kw in fact_lower for kw in _NEGATIVE_KEYWORDS)

    if has_positive and has_negative:
        return "mixed"
    elif has_positive:
        return "positive"
    elif has_negative:
        return "negative"
    else:
        return "neutral"


def classify_note_polarity(facts: list[dict]) -> str:
    """Compute dominant_polarity for a note from its facts."""
    polarities = [f.get("polarity", "neutral") for f in facts if isinstance(f, dict)]
    if not polarities:
        return "neutral"
    pos = sum(1 for p in polarities if p == "positive")
    neg = sum(1 for p in polarities if p == "negative")
    if pos > neg * 2:
        return "positive"
    if neg > pos * 2:
        return "negative"
    if pos > 0 and neg > 0:
        return "mixed"
    return "neutral"
