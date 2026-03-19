"""Fact validation layer — cross-reference extracted facts against source text.

Detects numeric mismatches, magnitude errors, and common hallucination patterns
by comparing numbers in AI-generated facts against numbers in source material.
"""

import re
import logging
from datetime import date

logger = logging.getLogger(__name__)

# Currency symbols to strip
_CURRENCY_RE = re.compile(r"^[\$€£¥]")

# Suffix multipliers
_SUFFIX_MAP = {"k": 1_000, "m": 1_000_000, "b": 1_000_000_000}

# Word multipliers
_WORD_MAP = {"thousand": 1_000, "million": 1_000_000, "billion": 1_000_000_000}

# Pattern: optional currency, digits with commas/dots, optional suffix or word
_NUMBER_RE = re.compile(
    r"[\$€£¥]?\s*(\d[\d,]*\.?\d*)\s*(%|[KkMmBb]\b|million|billion|thousand)?",
    re.IGNORECASE,
)


def normalize_number(text: str) -> float | None:
    """Normalize numeric expressions to float.

    Handles: 1,706 | $2M | $950K | 10.8% | 5-8% | 1.7 million | 15 weeks
    Returns None if no number found.
    """
    text = text.strip()
    if not text:
        return None

    # Handle ranges like "5-8%" — take first value
    range_match = re.match(
        r"[\$€£¥]?\s*(\d[\d,]*\.?\d*)\s*[-–—]\s*(\d[\d,]*\.?\d*)\s*(%|[KkMmBb]\b|million|billion|thousand)?",
        text,
        re.IGNORECASE,
    )
    if range_match:
        num_str = range_match.group(1).replace(",", "")
        suffix = (range_match.group(3) or "").strip().lower()
        try:
            value = float(num_str)
        except ValueError:
            return None
        return _apply_suffix(value, suffix)

    m = _NUMBER_RE.search(text)
    if not m:
        return None

    num_str = m.group(1).replace(",", "")
    suffix = (m.group(2) or "").strip().lower()

    try:
        value = float(num_str)
    except ValueError:
        return None

    return _apply_suffix(value, suffix)


def _apply_suffix(value: float, suffix: str) -> float:
    """Apply K/M/B/word suffix multiplier."""
    if suffix == "%":
        return value  # Keep as percentage value
    if suffix in _SUFFIX_MAP:
        return value * _SUFFIX_MAP[suffix]
    if suffix in _WORD_MAP:
        return value * _WORD_MAP[suffix]
    return value


def extract_numbers_from_text(text: str) -> set[float]:
    """Extract all numeric values from a text string."""
    results = set()
    for m in _NUMBER_RE.finditer(text):
        num_str = m.group(1).replace(",", "")
        suffix = (m.group(2) or "").strip().lower()
        try:
            value = float(num_str)
        except ValueError:
            continue
        results.add(_apply_suffix(value, suffix))
    return results


def _numbers_match(a: float, b: float, tolerance: float = 0.01) -> bool:
    """Check if two numbers match within tolerance (±1% for rounding)."""
    if a == b:
        return True
    if a == 0 or b == 0:
        return False
    ratio = abs(a - b) / max(abs(a), abs(b))
    return ratio <= tolerance


def _magnitude_ratio(a: float, b: float) -> float:
    """Return ratio between two numbers (always >= 1)."""
    if a == 0 or b == 0:
        return float("inf")
    ratio = a / b if a > b else b / a
    return ratio


def validate_fact_against_source(fact: str, source_text: str) -> dict:
    """Cross-reference a fact's numbers against source text.

    Returns dict with status, matched/missing numbers, and anomalies.
    """
    fact_numbers = sorted(extract_numbers_from_text(fact))
    source_numbers = extract_numbers_from_text(source_text)
    anomalies = check_anomalies(fact)

    # Non-numeric fact — nothing to verify
    if not fact_numbers:
        return {
            "fact": fact,
            "status": "verified",
            "fact_numbers": [],
            "source_numbers": [],
            "missing_numbers": [],
            "anomalies": anomalies,
        }

    matched = []
    missing = []

    for fn in fact_numbers:
        found = any(_numbers_match(fn, sn) for sn in source_numbers)
        if found:
            matched.append(fn)
        else:
            # Check for magnitude errors (only when source has numbers to compare)
            best_ratio = min(
                (_magnitude_ratio(fn, sn) for sn in source_numbers),
                default=None,
            )
            if best_ratio is not None and best_ratio > 100:
                missing.append(fn)
                anomalies.append(
                    f"Magnitude error: {fn} not in source (closest is {best_ratio:.0f}x off)"
                )
            else:
                missing.append(fn)

    if not missing:
        status = "verified"
    elif anomalies:
        status = "flagged_mismatch"
    else:
        status = "unverified"

    return {
        "fact": fact,
        "status": status,
        "fact_numbers": fact_numbers,
        "source_numbers": sorted(matched),
        "missing_numbers": missing,
        "anomalies": anomalies,
    }


def check_anomalies(fact: str) -> list[str]:
    """Regex-based anomaly detection for common hallucination patterns."""
    anomalies = []

    # Check percentages > 100 that aren't valid growth/increase patterns
    for m in re.finditer(r"(\d[\d,]*\.?\d*)\s*%", fact):
        val = float(m.group(1).replace(",", ""))
        if val > 100:
            # Valid if context mentions growth/increase/improvement/return
            context = fact.lower()
            valid_contexts = (
                "growth", "increase", "improvement", "return", "roi",
                "rise", "gain", "surge", "jump", "boost", "up",
                "over", "exceed", "above", "more than",
            )
            if not any(ctx in context for ctx in valid_contexts):
                anomalies.append(f"Percentage over 100%: {val}%")

    # Check for magnitude confusion within the same fact (K vs M)
    # Only compare numbers that are both "large" (>= 1000) to avoid
    # false positives from small counts mixed with monetary values
    numbers = sorted(n for n in extract_numbers_from_text(fact) if n >= 1000)
    if len(numbers) >= 2:
        for i in range(len(numbers)):
            for j in range(i + 1, len(numbers)):
                ratio = _magnitude_ratio(numbers[i], numbers[j])
                if ratio > 400:
                    anomalies.append(
                        f"Suspicious magnitude spread: {numbers[i]} vs {numbers[j]} ({ratio:.0f}x)"
                    )

    # Check for future dates (> current year + 1)
    current_year = date.today().year
    for m in re.finditer(r"\b(20\d{2})\b", fact):
        year = int(m.group(1))
        if year > current_year + 1:
            anomalies.append(f"Future date: {year}")

    return anomalies
