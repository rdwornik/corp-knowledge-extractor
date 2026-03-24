"""
Post-processor for extraction results.
Delegates to corp_os_meta for normalization, validation, and link generation.
Adds CKE-specific logic: unknown term logging to local file.
"""

import logging
import re
import yaml
from pathlib import Path
from dataclasses import dataclass, field

from corp_os_meta import (
    normalize_frontmatter,
    validate_frontmatter,
    generate_links_line,
    ValidationResult,
)
from corp_os_meta.models import NoteFrontmatter
from corp_os_meta.normalize import load_taxonomy
from src.utils import normalize_string_list

logger = logging.getLogger(__name__)


def enforce_type_from_extension(result: dict, source_path: str) -> dict:
    """Override content type based on file extension.

    File extension is the ground truth for document type — LLM classification
    can misidentify DOCX/XLSX as "presentation" etc.
    """
    ext = Path(source_path).suffix.lower()
    TYPE_MAP = {
        ".docx": "document",
        ".xlsx": "spreadsheet",
        ".csv": "spreadsheet",
        ".pdf": "document",
        ".pptx": "presentation",
        ".mp4": "presentation",
        ".mkv": "presentation",
        ".avi": "presentation",
    }
    if ext in TYPE_MAP:
        result["content_type"] = TYPE_MAP[ext]
        if "type" in result:
            result["type"] = TYPE_MAP[ext]
    return result


def normalize_company_names(text: str) -> str:
    """Fix known LLM company name duplications."""
    if not text:
        return text
    text = re.sub(r'(?i)\b(Blue\s+){2,}Yonder\b', 'Blue Yonder', text)
    return text


# Short product names → canonical Blue Yonder forms
PRODUCT_ALIASES = {
    "Demand Planning": "Blue Yonder Demand Planning",
    "Supply Planning": "Blue Yonder Supply Planning",
    "Control Tower": "Blue Yonder Control Tower",
    "WMS": "Blue Yonder WMS",
    "TMS": "Blue Yonder TMS",
    "OMS": "Blue Yonder OMS",
    "Platform": "Blue Yonder Platform",
}


def normalize_product_names(products: list[str]) -> list[str]:
    """Normalize short product names to canonical Blue Yonder forms.

    Applied after corp-os-meta normalization to catch remaining short forms.
    Deduplicates: ["Demand Planning", "Blue Yonder Demand Planning"] → ["Blue Yonder Demand Planning"]
    """
    result = []
    for p in products:
        canonical = PRODUCT_ALIASES.get(p.strip(), p)
        if canonical not in result:
            result.append(canonical)
    return result


@dataclass
class PostProcessResult:
    """Result of post-processing with metadata about what changed."""

    data: dict
    links_line: str
    validation_result: ValidationResult
    validated_note: NoteFrontmatter | None
    changes: list[str] = field(default_factory=list)
    unknown_terms: list[str] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)


def post_process_extraction(
    raw_result: dict,
    source_tool: str = "knowledge-extractor",
    source_file: str = "",
    client: str | None = None,
    project: str | None = None,
) -> PostProcessResult:
    """Apply corp-os-meta normalization and validation to raw extraction result.

    Args:
        raw_result: Raw dict from Gemini extraction (parsed JSON)
        source_tool: Tool identifier for frontmatter
        source_file: Original file path/name
        client: Override client from manifest (takes precedence over Gemini)
        project: Override project from manifest (takes precedence over Gemini)

    Returns:
        PostProcessResult with normalized data, links line, and validation status
    """
    # Ensure required fields for corp-os-meta
    raw_result.setdefault("source_tool", source_tool)
    raw_result.setdefault("source_file", source_file)
    raw_result.setdefault("schema_version", 2)

    # Manifest-provided client/project override Gemini's guess
    if client:
        raw_result["client"] = client
    if project:
        raw_result["project"] = project

    # Map CKE field names to corp-os-meta field names if needed
    if "content_type" in raw_result and "type" not in raw_result:
        raw_result["type"] = raw_result.pop("content_type")

    # Map Gemini quality values to schema enum (full|partial|fragment)
    _quality_map = {"high": "full", "medium": "partial", "low": "fragment"}
    if "quality" in raw_result:
        raw_result["quality"] = _quality_map.get(raw_result["quality"], raw_result["quality"])

    # Schema v2 defaults — safe values if LLM didn't produce them
    raw_result.setdefault("confidentiality", "internal")
    raw_result.setdefault("authority", "tribal")
    raw_result.setdefault("layer", "learning")
    raw_result.setdefault("source_type", "documentation")
    raw_result.setdefault("domains", [])

    # Normalize list fields: LLMs sometimes return dicts instead of strings
    for list_field in ("topics", "products", "people", "domains"):
        if list_field in raw_result and isinstance(raw_result[list_field], list):
            raw_result[list_field] = normalize_string_list(raw_result[list_field])

    # Fix duplicated company names (Gemini sometimes doubles "Blue Yonder")
    for str_field in ("title", "summary"):
        val = raw_result.get(str_field, "")
        if val:
            raw_result[str_field] = normalize_company_names(val)

    # Normalize using corp-os-meta taxonomy
    taxonomy = load_taxonomy()
    normalized_data, changes, unknown = normalize_frontmatter(raw_result, taxonomy)

    # Apply company name normalization after corp-os-meta (it may copy raw strings)
    for str_field in ("title", "summary"):
        val = normalized_data.get(str_field, "")
        if val:
            normalized_data[str_field] = normalize_company_names(val)

    # Normalize short product names to canonical Blue Yonder forms
    if "products" in normalized_data and isinstance(normalized_data["products"], list):
        normalized_data["products"] = normalize_product_names(normalized_data["products"])

    if changes:
        logger.info("Normalized: %s", ", ".join(changes))
    if unknown:
        logger.info("Unknown terms: %s", unknown)
        _log_unknown_terms(unknown)

    # Enforce type from file extension (overrides LLM classification)
    if source_file:
        normalized_data = enforce_type_from_extension(normalized_data, source_file)

    # Validate using corp-os-meta
    validation_result, validated_note, issues = validate_frontmatter(normalized_data)

    if issues:
        logger.warning("Validation issues: %s", issues)

    # Generate deterministic links line
    links_line = ""
    if validated_note:
        links_line = generate_links_line(validated_note)
    else:
        # Quarantined — still generate links from raw data for the note
        links_parts = []
        for topic in normalize_string_list(normalized_data.get("topics") or []):
            links_parts.append(f"[[{topic}]]")
        for product in normalize_string_list(normalized_data.get("products") or []):
            links_parts.append(f"[[{product}]]")
        for person in normalize_string_list(normalized_data.get("people") or []):
            name = person.split("(")[0].strip()
            links_parts.append(f"[[{name}]]")
        links_line = "**Links:** " + " . ".join(links_parts) if links_parts else ""

    return PostProcessResult(
        data=normalized_data,
        links_line=links_line,
        validation_result=validation_result,
        validated_note=validated_note,
        changes=changes,
        unknown_terms=unknown,
        issues=issues,
    )


def _normalize_tag(value: str) -> str:
    """Normalize tag value: lowercase, hyphens, strip special chars."""
    tag = value.lower()
    tag = tag.replace("&", "").replace("_", "-").replace(" ", "-")
    tag = re.sub(r'[^a-z0-9\-/]', '', tag)
    tag = re.sub(r'-+', '-', tag).strip('-')
    return tag


def generate_tags(frontmatter: dict) -> list[str]:
    """Generate hierarchical tags from frontmatter fields."""
    tags = []

    for product in (frontmatter.get("products") or []):
        tags.append(f"product/{_normalize_tag(product)}")

    for topic in (frontmatter.get("topics") or []):
        tags.append(f"topic/{_normalize_tag(topic)}")

    for domain in (frontmatter.get("domains") or []):
        tags.append(f"domain/{_normalize_tag(domain)}")

    client = frontmatter.get("client")
    if client:
        tags.append(f"client/{_normalize_tag(client)}")

    doc_type = frontmatter.get("doc_type")
    if doc_type:
        tags.append(f"type/{_normalize_tag(doc_type)}")

    source_type = frontmatter.get("source_type")
    if source_type:
        tags.append(f"source/{_normalize_tag(source_type)}")

    # Deduplicate preserving order
    seen = set()
    return [t for t in tags if not (t in seen or seen.add(t))]


def validate_tags(tags: list[str]) -> list[dict]:
    """Validate tags against corp-os-meta taxonomy.

    Returns list of {"tag": str, "valid": bool, "reason": str}.
    All tags are kept regardless of validity — this is informational only.
    """
    try:
        taxonomy = load_taxonomy()
    except Exception:
        logger.warning("Could not load taxonomy, skipping tag validation")
        return [{"tag": t, "valid": True, "reason": "unvalidated"} for t in tags]

    VALID_PREFIXES = {"product/", "topic/", "domain/", "client/", "type/", "source/"}

    results = []
    for tag in tags:
        # Check prefix
        prefix_valid = any(tag.startswith(p) for p in VALID_PREFIXES)
        if not prefix_valid:
            logger.warning("Tag with unknown prefix: %s", tag)
            results.append({"tag": tag, "valid": False, "reason": "unknown_prefix"})
            continue

        # Extract prefix and value
        prefix, _, value = tag.partition("/")

        # Check value against taxonomy known values
        known_values = _get_known_values(taxonomy, prefix)
        if known_values and value not in known_values:
            logger.warning("Tag value not in taxonomy: %s", tag)
            results.append({"tag": tag, "valid": True, "reason": "unvalidated"})
        else:
            results.append({"tag": tag, "valid": True, "reason": "validated"})

    return results


def _get_known_values(taxonomy: dict, prefix: str) -> set:
    """Get known normalized values for a tag prefix from taxonomy."""
    TAXONOMY_MAP = {
        "product": "products",
        "topic": "topics",
        "domain": "domains",
        "client": "clients",
        "type": "document_types",
        "source": "source_types",
    }
    key = TAXONOMY_MAP.get(prefix, "")
    values = taxonomy.get(key, [])
    if isinstance(values, list):
        result = set()
        for v in values:
            if isinstance(v, dict):
                result.add(_normalize_tag(v.get("name", "")))
            elif isinstance(v, str):
                result.add(_normalize_tag(v))
        return result
    return set()


def _log_unknown_terms(terms: list[str]):
    """Append unknown terms to local review file for batch approval."""
    review_path = Path(__file__).parent.parent / "config" / "taxonomy_review.yaml"
    data = {"pending": []}
    if review_path.exists():
        with open(review_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {"pending": []}

    existing = set(data.get("pending", []))
    added = []
    for term in terms:
        if term not in existing:
            data["pending"].append(term)
            added.append(term)

    if added:
        with open(review_path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
        logger.info("Added to taxonomy_review.yaml: %s", added)
