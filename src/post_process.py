"""
Post-processor for extraction results.
Delegates to corp_os_meta for normalization, validation, and link generation.
Adds CKE-specific logic: unknown term logging to local file.
"""
import logging
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

logger = logging.getLogger(__name__)


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
) -> PostProcessResult:
    """Apply corp-os-meta normalization and validation to raw extraction result.

    Args:
        raw_result: Raw dict from Gemini extraction (parsed JSON)
        source_tool: Tool identifier for frontmatter
        source_file: Original file path/name

    Returns:
        PostProcessResult with normalized data, links line, and validation status
    """
    # Ensure required fields for corp-os-meta
    raw_result.setdefault("source_tool", source_tool)
    raw_result.setdefault("source_file", source_file)
    raw_result.setdefault("schema_version", 1)

    # Map CKE field names to corp-os-meta field names if needed
    if "content_type" in raw_result and "type" not in raw_result:
        raw_result["type"] = raw_result.pop("content_type")

    # Normalize using corp-os-meta taxonomy
    taxonomy = load_taxonomy()
    normalized_data, changes, unknown = normalize_frontmatter(raw_result, taxonomy)

    if changes:
        logger.info("Normalized: %s", ", ".join(changes))
    if unknown:
        logger.info("Unknown terms: %s", unknown)
        _log_unknown_terms(unknown)

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
        for topic in normalized_data.get("topics", []):
            links_parts.append(f"[[{topic}]]")
        for product in normalized_data.get("products", []):
            links_parts.append(f"[[{product}]]")
        for person in normalized_data.get("people", []):
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


def _log_unknown_terms(terms: list[str]):
    """Append unknown terms to local review file for batch approval."""
    review_path = Path("config/taxonomy_review.yaml")
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
