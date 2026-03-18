"""Merge correlated PPTX + MP4 extractions into a single training_session note."""

import hashlib
import logging
from datetime import datetime
from pathlib import Path

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

    # Key facts from PPTX (structured), tag with source modality
    key_facts = []
    for fact in pptx_extraction.get("key_facts") or []:
        if isinstance(fact, str):
            key_facts.append({"fact": fact, "source_modality": "pptx"})
        elif isinstance(fact, dict):
            fact["source_modality"] = "pptx"
            key_facts.append(fact)

    # Overlay from PPTX
    overlay = None
    overlay_type = None
    for key in pptx_extraction:
        if key.endswith("_overlay"):
            overlay = pptx_extraction[key]
            overlay_type = key
            break

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


def _generate_session_id(pptx_path: Path, video_path: Path) -> str:
    """Generate a short session ID from filenames."""
    stem = pptx_path.stem.lower().replace(" ", "-").replace("_", "-")
    return stem[:50]


def _dedupe_list(items: list) -> list:
    """Deduplicate list preserving order, case-insensitive."""
    seen: set[str] = set()
    result = []
    for item in items:
        key = str(item).lower().strip()
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
