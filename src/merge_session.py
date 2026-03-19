"""Merge correlated PPTX + MP4 extractions into a single training_session note."""

import hashlib
import logging
from datetime import datetime
from pathlib import Path

from src.fact_validation import extract_numbers_from_text, _numbers_match
from src.utils import normalize_string_list

logger = logging.getLogger(__name__)


def merge_correlated(
    pptx_extraction: dict,
    video_extraction: dict,
    pptx_path: Path,
    video_path: Path,
    pptx_hash: str,
    video_hash: str,
    correlation_confidence: int,
    correlation_method: str,
) -> dict:
    """Merge PPTX structured facts + MP4 visual/speaker content into session artifact.

    Returns dict with frontmatter + markdown content ready for output.
    """
    sorted_hashes = sorted([pptx_hash, video_hash])
    session_hash = hashlib.sha256("|".join(sorted_hashes).encode()).hexdigest()

    session_id = _generate_session_id(pptx_path, video_path)

    topics = _dedupe_list(
        (pptx_extraction.get("topics") or []) + (video_extraction.get("topics") or [])
    )
    products = _dedupe_list(
        (pptx_extraction.get("products") or []) + (video_extraction.get("products") or [])
    )
    entities = _dedupe_list(
        (pptx_extraction.get("entities_mentioned") or [])
        + (video_extraction.get("entities_mentioned") or [])
    )
    people = _dedupe_list(
        (pptx_extraction.get("people") or []) + (video_extraction.get("people") or [])
    )

    # Merge key facts from both sources with conservative dedup
    pptx_facts = _tag_facts(pptx_extraction.get("key_facts") or [], "pptx")
    mp4_facts = _tag_facts(video_extraction.get("key_facts") or [], "mp4")
    key_facts = deduplicate_facts(pptx_facts, mp4_facts)

    # Overlay from both sources — merge PPTX + MP4
    pptx_overlay = None
    mp4_overlay = None
    overlay_type = None
    for key in pptx_extraction:
        if key.endswith("_overlay"):
            pptx_overlay = pptx_extraction[key]
            overlay_type = key
            break
    for key in video_extraction:
        if key.endswith("_overlay"):
            mp4_overlay = video_extraction[key]
            if not overlay_type:
                overlay_type = key
            break
    overlay = merge_training_overlays(pptx_overlay or {}, mp4_overlay or {})

    title = pptx_extraction.get("title") or video_extraction.get("title") or "Untitled Session"

    frontmatter = {
        "title": title,
        "type": "training_session",
        "source_type": "correlated_session",
        "session_id": session_id,
        "session_hash": session_hash,
        "date": datetime.now().strftime("%Y-%m-%d"),
        "sources": [
            {
                "modality": "mp4",
                "path": str(video_path).replace("\\", "/"),
                "hash": video_hash,
                "extracted_at": video_extraction.get("extracted_at", ""),
                "contribution": [
                    "slide_frames",
                    "speaker_commentary",
                    "visual_context",
                ],
            },
            {
                "modality": "pptx",
                "path": str(pptx_path).replace("\\", "/"),
                "hash": pptx_hash,
                "extracted_at": pptx_extraction.get("extracted_at", ""),
                "contribution": [
                    "key_facts",
                    "overlay",
                    "entities",
                    "topics",
                    "products",
                ],
            },
        ],
        "correlation": {
            "confidence": correlation_confidence,
            "method": correlation_method,
        },
        "topics": topics,
        "products": products,
        "entities_mentioned": entities,
        "people": people,
        "key_facts": key_facts,
        "domains": _dedupe_list(
            (pptx_extraction.get("domains") or [])
            + (video_extraction.get("domains") or [])
        ),
        "confidentiality": "internal",
        "authority": "approved",
        "language": "en",
        "extraction_version": 2,
        "depth": "deep",
        "schema_version": 2,
        "source_tool": "knowledge-extractor",
    }

    if overlay and overlay_type:
        frontmatter[overlay_type] = overlay

    markdown = _build_markdown(
        title, key_facts, overlay, overlay_type, video_extraction, pptx_path, video_path
    )

    return {
        "frontmatter": frontmatter,
        "markdown": markdown,
        "session_id": session_id,
    }


def merge_training_overlays(pptx_overlay: dict, mp4_overlay: dict) -> dict:
    """Merge overlay fields from PPTX and MP4 sources.

    MP4 typically has richer attendees (actual names) and action_items (with deadlines).
    PPTX typically has structured concerns and questions from the deck.
    """
    if not pptx_overlay and not mp4_overlay:
        return {}
    if not mp4_overlay:
        return dict(pptx_overlay)
    if not pptx_overlay:
        return dict(mp4_overlay)

    merged = {}

    # List fields: deduplicate by string content
    list_fields = [
        "attendees", "decisions_made", "action_items",
        "questions_raised", "concerns_expressed", "next_steps",
    ]
    for field in list_fields:
        pptx_items = pptx_overlay.get(field) or []
        mp4_items = mp4_overlay.get(field) or []
        merged[field] = _dedupe_overlay_items(pptx_items + mp4_items)

    # Copy any non-list fields from either source (PPTX takes precedence)
    for key in set(list(pptx_overlay) + list(mp4_overlay)):
        if key not in list_fields and key not in merged:
            merged[key] = pptx_overlay.get(key) or mp4_overlay.get(key)

    # Remove empty fields
    return {k: v for k, v in merged.items() if v}


def _dedupe_overlay_items(items: list) -> list:
    """Deduplicate overlay items (strings or dicts) by content."""
    seen: set[str] = set()
    result = []
    for item in items:
        if isinstance(item, dict):
            # Dedup by primary text field (action, decision, question, etc.)
            key_text = (
                item.get("action") or item.get("decision") or
                item.get("question") or item.get("concern") or
                item.get("name") or str(item)
            ).lower().strip()
        else:
            key_text = str(item).lower().strip()
        if key_text not in seen:
            seen.add(key_text)
            result.append(item)
    return result


def _tag_facts(raw_facts: list, modality: str) -> list[dict]:
    """Convert raw facts (str or dict) to tagged dicts."""
    tagged = []
    for fact in raw_facts:
        if isinstance(fact, str):
            tagged.append({"fact": fact, "source_modality": modality})
        elif isinstance(fact, dict):
            entry = dict(fact)
            entry["source_modality"] = modality
            tagged.append(entry)
    return tagged


def _is_year(n: float) -> bool:
    """Check if a number looks like a calendar year (2000-2099)."""
    return n == int(n) and 2000 <= n <= 2099


def _meaningful_numbers(nums: set[float]) -> set[float]:
    """Filter out year-like numbers — they don't identify a fact's claim."""
    return {n for n in nums if not _is_year(n)}


def _fact_words(text: str) -> set[str]:
    """Extract meaningful words for overlap comparison."""
    stop_words = {"the", "a", "an", "is", "are", "was", "were", "in", "on", "at",
                  "to", "for", "of", "with", "and", "or", "by", "from", "has", "had"}
    return {w.lower().strip(".,;:!?()") for w in text.split()
            if w.lower() not in stop_words and len(w) > 2 and not w.replace(",", "").isdigit()}


def _facts_match(a: dict, b: dict) -> bool:
    """Check if two facts refer to the same claim via shared numbers and text overlap.

    Match criteria: at least one non-year number in common (±1% tolerance)
    AND meaningful word overlap (>20% shared words).
    """
    a_text = a.get("fact", "")
    b_text = b.get("fact", "")

    a_nums = _meaningful_numbers(extract_numbers_from_text(a_text))
    b_nums = _meaningful_numbers(extract_numbers_from_text(b_text))

    # If both have meaningful numbers, require at least one match
    if a_nums and b_nums:
        has_number_match = any(
            _numbers_match(an, bn)
            for an in a_nums
            for bn in b_nums
        )
        if not has_number_match:
            return False
        # Also require some text overlap (entity similarity)
        a_words = _fact_words(a_text)
        b_words = _fact_words(b_text)
        if not a_words or not b_words:
            return has_number_match
        overlap = len(a_words & b_words) / min(len(a_words), len(b_words))
        return overlap > 0.2

    return False


def _facts_conflict(a: dict, b: dict) -> bool:
    """Check if two facts mention the same entity but different numbers."""
    a_text = a.get("fact", "")
    b_text = b.get("fact", "")

    a_nums = _meaningful_numbers(extract_numbers_from_text(a_text))
    b_nums = _meaningful_numbers(extract_numbers_from_text(b_text))

    if not a_nums or not b_nums:
        return False

    # Need shared words (entity overlap) but NO shared numbers
    a_words = _fact_words(a_text)
    b_words = _fact_words(b_text)

    if not a_words or not b_words:
        return False
    overlap = len(a_words & b_words) / min(len(a_words), len(b_words))
    if overlap < 0.3:
        return False

    # Has entity overlap but no number match → conflict
    has_number_match = any(
        _numbers_match(an, bn) for an in a_nums for bn in b_nums
    )
    return not has_number_match


def deduplicate_facts(pptx_facts: list[dict], mp4_facts: list[dict]) -> list[dict]:
    """Merge facts from PPTX and MP4 with conservative deduplication.

    - Matching facts: keep PPTX version (canonical), mark modality="both"
    - Conflicting facts: keep both, add conflict_detected=True
    - Supplementary MP4 facts: add with modality="mp4"
    """
    result = list(pptx_facts)
    matched_mp4: set[int] = set()

    for mi, mp4_fact in enumerate(mp4_facts):
        for pi, pptx_fact in enumerate(result):
            if _facts_match(pptx_fact, mp4_fact):
                result[pi]["source_modality"] = "both"
                matched_mp4.add(mi)
                break
            elif _facts_conflict(pptx_fact, mp4_fact):
                mp4_fact["conflict_detected"] = True
                result[pi]["conflict_detected"] = True
                result.append(mp4_fact)
                matched_mp4.add(mi)
                break

    # Add supplementary MP4 facts not matched
    for mi, mp4_fact in enumerate(mp4_facts):
        if mi not in matched_mp4:
            result.append(mp4_fact)

    return result


def _generate_session_id(pptx_path: Path, video_path: Path) -> str:
    """Generate a short session ID from filenames."""
    stem = pptx_path.stem.lower().replace(" ", "-").replace("_", "-")
    return stem[:50]


def _dedupe_list(items: list) -> list:
    """Deduplicate list preserving order, case-insensitive. Handles dict items."""
    normalized = normalize_string_list(items)
    seen: set[str] = set()
    result = []
    for item in normalized:
        key = item.lower().strip()
        if key not in seen:
            seen.add(key)
            result.append(item)
    return result


def _build_markdown(
    title: str,
    key_facts: list,
    overlay: dict | None,
    overlay_type: str | None,
    video_extraction: dict,
    pptx_path: Path,
    video_path: Path,
) -> str:
    """Build the markdown body for the merged session note."""
    sections = []

    video_summary = video_extraction.get("summary", "")
    if video_summary:
        sections.append(f"## Executive Summary\n\n{video_summary}")

    if key_facts:
        facts_md = "\n".join(
            f"- {f['fact'] if isinstance(f, dict) else f}" for f in key_facts
        )
        sections.append(f"## Key Facts\n\n{facts_md}")

    if overlay and overlay_type:
        overlay_name = overlay_type.replace("_overlay", "").replace("_", " ").title()
        overlay_md = _format_overlay(overlay)
        sections.append(f"## {overlay_name} Details\n\n{overlay_md}")

    slides = video_extraction.get("slides", [])
    if slides:
        slide_sections = []
        for slide in slides:
            slide_num = slide.get("slide_number", slide.get("frame_index", "?"))
            slide_title = slide.get("title", f"Slide {slide_num}")

            slide_md = f"### Slide {slide_num}: {slide_title}\n"

            frame_path = slide.get("frame_path")
            if frame_path:
                slide_md += f"\n![Slide {slide_num}]({frame_path})\n"

            commentary = slide.get("speaker_explanation", "")
            if commentary:
                slide_md += f"\n**Speaker Commentary:** {commentary}\n"

            so_what = slide.get("so_what", "")
            if so_what:
                slide_md += f"\n**So what:** {so_what}\n"

            slide_facts = [
                f
                for f in key_facts
                if isinstance(f, dict)
                and str(f.get("slide_ref", "")) == str(slide_num)
            ]
            if slide_facts:
                facts_list = "\n".join(f"- {f['fact']}" for f in slide_facts)
                slide_md += f"\n**Key Facts (from deck):**\n{facts_list}\n"

            warning = slide.get("warning", "")
            if warning:
                slide_md += f"\n> WARNING: {warning}\n"

            slide_sections.append(slide_md)

        sections.append(
            "## Slide-by-Slide Walkthrough\n\n" + "\n---\n".join(slide_sections)
        )

    quotes = video_extraction.get("notable_quotes", "")
    if quotes:
        sections.append(f"## Notable Quotes\n\n{quotes}")

    sections.append(
        f"## Source Provenance\n\n"
        f"- **Video:** `{video_path.name}`\n"
        f"- **Deck:** `{pptx_path.name}`\n"
        f"- **Merge method:** Correlated session (PPTX structured facts + MP4 visual/speaker context)"
    )

    return "\n\n".join(sections)


def _format_overlay(overlay: dict) -> str:
    """Format overlay dict as readable markdown."""
    lines = []
    for key, value in overlay.items():
        label = key.replace("_", " ").title()
        if isinstance(value, list) and value:
            if isinstance(value[0], dict):
                for item in value:
                    item_str = ", ".join(f"{k}: {v}" for k, v in item.items() if v)
                    lines.append(f"- **{label}:** {item_str}")
            else:
                lines.append(f"- **{label}:** {', '.join(str(v) for v in value)}")
        elif isinstance(value, str) and value:
            lines.append(f"- **{label}:** {value}")
    return "\n".join(lines) if lines else "No details available."
