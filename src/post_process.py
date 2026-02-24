"""
Post-processor for extraction results.
- Term normalization (find-replace from config)
- Taxonomy normalization (match to canonical names, log unknowns)
- Cardinality enforcement (max topics/products/people)
- Deterministic Links line generation
"""
import logging
import yaml
from pathlib import Path
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Hard caps applied in code — prompt also asks for these limits
_CARDINALITY_CAPS = {
    "topics": 8,
    "products": 4,
    "people": 3,
}


@dataclass
class PostProcessResult:
    """Result of post-processing with metadata about what changed."""
    data: dict
    normalized_terms: list[str]   # terms that were normalized (e.g. "DR -> Disaster Recovery")
    unknown_terms: list[str]      # terms not in taxonomy, logged for review
    truncated_fields: list[str]   # fields that hit cardinality cap
    links_line: str               # deterministic Links line for markdown


def post_process_extraction(result: dict, config: dict, taxonomy_path: Path) -> PostProcessResult:
    """Apply post-processing to raw Gemini extraction result.

    Args:
        result: Raw dict from parse_llm_json()
        config: Unified config dict from load_config()
        taxonomy_path: Path to config/taxonomy.yaml

    Returns:
        PostProcessResult with cleaned data and metadata
    """
    taxonomy = _load_taxonomy(taxonomy_path)
    pp_config = config.get("post_processing", {})
    normalized_terms: list[str] = []
    unknown_terms: list[str] = []
    truncated_fields: list[str] = []

    # 1. Term normalization (simple find-replace across all string values)
    normalizations = pp_config.get("term_normalization", {})
    if normalizations:
        result = _normalize_terms(result, normalizations)

    # 2. Taxonomy normalization for topics
    if "topics" in result and isinstance(result["topics"], list):
        result["topics"], norm, unknown = _normalize_to_taxonomy(
            result["topics"], taxonomy.get("topics", [])
        )
        normalized_terms.extend(norm)
        unknown_terms.extend(unknown)

    # 3. Taxonomy normalization for products
    if "products" in result and isinstance(result["products"], list):
        result["products"], norm, unknown = _normalize_to_taxonomy(
            result["products"], taxonomy.get("products", [])
        )
        normalized_terms.extend(norm)
        unknown_terms.extend(unknown)

    # 4. Cardinality caps
    for field_name, cap in _CARDINALITY_CAPS.items():
        if field_name in result and isinstance(result[field_name], list):
            if len(result[field_name]) > cap:
                logger.warning("Truncating %s: %d → %d", field_name, len(result[field_name]), cap)
                result[field_name] = result[field_name][:cap]
                truncated_fields.append(field_name)

    # 5. Generate deterministic Links line
    links_line = _build_links_line(result)

    # 6. Log unknown terms for weekly review
    if unknown_terms:
        _append_to_taxonomy_review(unknown_terms, taxonomy_path.parent / "taxonomy_review.yaml")
        logger.info("Unknown terms logged for review: %s", unknown_terms)

    return PostProcessResult(
        data=result,
        normalized_terms=normalized_terms,
        unknown_terms=unknown_terms,
        truncated_fields=truncated_fields,
        links_line=links_line,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_taxonomy(path: Path) -> dict:
    """Load taxonomy.yaml. Return empty dict if not found."""
    if not path.exists():
        logger.warning("Taxonomy file not found: %s", path)
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _normalize_to_taxonomy(
    values: list[str],
    taxonomy_entries: list[dict],
) -> tuple[list[str], list[str], list[str]]:
    """Match extracted values to canonical names via aliases.

    Returns:
        (normalized_values, which_were_normalized, which_were_unknown)
    """
    # Build alias → canonical lookup (case-insensitive)
    alias_map: dict[str, str] = {}
    for entry in taxonomy_entries:
        canonical = entry["name"]
        alias_map[canonical.lower()] = canonical
        for alias in entry.get("aliases", []):
            alias_map[alias.lower()] = canonical

    normalized: list[str] = []
    norm_log: list[str] = []
    unknown: list[str] = []
    seen: set[str] = set()

    for val in values:
        canonical = alias_map.get(val.lower())
        if canonical:
            if canonical not in seen:
                normalized.append(canonical)
                seen.add(canonical)
            if val != canonical:
                norm_log.append(f"{val} -> {canonical}")
        else:
            if val not in seen:
                normalized.append(val)  # keep as-is but flag
                seen.add(val)
                unknown.append(val)

    return normalized, norm_log, unknown


def _normalize_terms(data, normalizations: dict):
    """Recursively find-replace terms in all string values."""
    if isinstance(data, str):
        for old, new in normalizations.items():
            data = data.replace(old, new)
        return data
    elif isinstance(data, dict):
        return {k: _normalize_terms(v, normalizations) for k, v in data.items()}
    elif isinstance(data, list):
        return [_normalize_terms(item, normalizations) for item in data]
    return data


def _build_links_line(result: dict) -> str:
    """Build deterministic **Links:** line from topics, products, people."""
    parts: list[str] = []

    for topic in result.get("topics", []):
        parts.append(f"[[{topic}]]")

    for product in result.get("products", []):
        parts.append(f"[[{product}]]")

    for person in result.get("people", []):
        # Strip role annotation: "Mike Geller (Presenter)" -> "Mike Geller"
        name = person.split("(")[0].strip()
        if name:
            parts.append(f"[[{name}]]")

    return "**Links:** " + " · ".join(parts) if parts else ""


def _append_to_taxonomy_review(terms: list[str], review_path: Path) -> None:
    """Append unknown terms to taxonomy_review.yaml for batch approval."""
    data: dict = {"pending": []}
    if review_path.exists():
        with open(review_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {"pending": []}

    existing = set(data.get("pending", []))
    added = False
    for term in terms:
        if term not in existing:
            data["pending"].append(term)
            added = True

    if added:
        with open(review_path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
