"""
Build the output knowledge package from groups and extractions.

Output structure:
    {package_name}/
    ├── index.md
    ├── source/
    │   ├── video/
    │   ├── docs/
    │   └── notes/
    ├── extract/
    │   ├── _meta.yaml
    │   ├── {filename}.md
    │   └── synthesis.md
    └── .history/       (re-extract only)

Usage:
    from src.synthesize import build_package

    pkg_path = build_package(groups, extracts, output_dir, "My Meeting", config)
"""

import hashlib
import json
import logging
import os
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from src.compress import compress_video, needs_compression
from src.correlate import FileGroup
from src.extract import ExtractionResult
from src.inventory import FileType, SourceFile
from src.post_process import generate_tags
from src.utils import parse_llm_json

from src.transcript import TranscriptResult
from src.utils import normalize_string_list

log = logging.getLogger(__name__)

PIPELINE_VERSION = "2.0.0"

# Map FileType → source subdirectory
_SOURCE_SUBDIR = {
    FileType.VIDEO: "video",
    FileType.AUDIO: "video",  # audio alongside video
    FileType.DOCUMENT: "docs",
    FileType.SLIDES: "docs",
    FileType.SPREADSHEET: "docs",
    FileType.NOTE: "notes",
    FileType.TRANSCRIPT: "notes",
    FileType.UNKNOWN: "notes",
}


def _tojson_raw(value):
    """JSON serialize without HTML entity escaping."""
    return json.dumps(value, ensure_ascii=False)


def normalize_output_filename(source_filename: str, extracted_at: str, source_hash: str) -> str:
    """Generate normalized output filename.

    Format: {date}_{normalized_name}_{hash4}
    Example: 2026-03-22_sgdbf_architectural_requirements_a7b2
    """
    date = extracted_at[:10] if extracted_at else datetime.now().strftime("%Y-%m-%d")

    stem = Path(source_filename).stem.lower()
    stem = re.sub(r'[^a-z0-9\s]', '', stem)
    stem = re.sub(r'\s+', '_', stem).strip('_')
    stem = stem[:64]

    hash4 = source_hash[:4] if source_hash else "0000"

    return f"{date}_{stem}_{hash4}"


def compute_quality_score(
    key_facts: list,
    facts_with_status: list[dict],
    overlay_fields_populated: int,
    content_chars: int,
    entities_count: int,
) -> int:
    """Compute 0-100 quality score for frontmatter."""
    score = 0

    # Facts: max 30
    specific_facts = [f for f in key_facts if len(str(f)) >= 30]
    score += min(30, len(specific_facts) * 2)

    # Verification: max 20
    if facts_with_status:
        verified = sum(1 for f in facts_with_status if f.get("verification_status") == "verified")
        ratio = verified / len(facts_with_status) if facts_with_status else 0
        score += int(min(20, ratio * 20))

    # Overlay: max 20
    score += min(20, overlay_fields_populated * 4)

    # Content depth: max 15
    score += min(15, int(content_chars / 1000 * 3))

    # Entities: max 15
    score += min(15, entities_count * 3)

    return min(100, score)


def _get_jinja_env() -> Environment:
    templates_dir = Path(__file__).parent.parent / "templates"
    env = Environment(
        loader=FileSystemLoader(str(templates_dir)),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    env.filters["tojson_raw"] = _tojson_raw
    return env


def _copy_source_file(
    source_file: SourceFile,
    source_dir: Path,
    config: dict,
) -> Path:
    """Copy (or compress) a source file into the package source/ directory."""
    subdir = _SOURCE_SUBDIR.get(source_file.type, "notes")
    dest_dir = source_dir / subdir
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / source_file.path.name

    if source_file.type == FileType.VIDEO and config.get("compression", {}).get("enabled", True):
        if needs_compression(source_file.path, config):
            log.info("Compressing %s...", source_file.path.name)
            return compress_video(source_file.path, dest, config)

    log.info("Copying %s → %s", source_file.path.name, dest)
    shutil.copy2(source_file.path, dest)
    return dest


def _run_synthesis(
    extracts: dict[str, ExtractionResult],
    config: dict,
) -> dict:
    """Make a second Gemini call to synthesize across all files."""
    from google import genai
    from google.genai import types

    api_key_env = config.get("gemini", {}).get("api_key_env", "GEMINI_API_KEY")
    api_key = os.environ.get(api_key_env)
    if not api_key:
        log.warning("No Gemini API key — skipping synthesis")
        return {}

    synth_prompt = (config.get("prompts") or {}).get("synthesize", "")
    if not synth_prompt:
        log.warning("No 'synthesize' prompt in config — skipping synthesis")
        return {}

    # Build context from all extracts — include facts, overlays, entities
    extracts_data = []
    for e in extracts.values():
        entry = {
            "file": e.source_file.name,
            "title": e.title,
            "summary": e.summary,
            "key_points": e.key_points,
            "topics": e.topics,
            "people": e.people,
            "products": e.products,
        }
        # Top 10 key_facts by length (longer = more specific)
        if e.facts:
            sorted_facts = sorted(e.facts, key=lambda f: len(f.get("fact", "")), reverse=True)
            entry["key_facts"] = [f["fact"] for f in sorted_facts[:10]]
        # Action items and decisions from overlay
        if e.overlay:
            if e.overlay.get("action_items"):
                entry["action_items"] = e.overlay["action_items"]
            if e.overlay.get("decisions_made"):
                entry["decisions_made"] = e.overlay["decisions_made"]
        # Entities
        if e.raw_json.get("entities_mentioned"):
            entry["entities_mentioned"] = e.raw_json["entities_mentioned"]
        extracts_data.append(entry)

    extracts_summary = json.dumps(extracts_data, indent=2)

    combined_prompt = (
        f"{synth_prompt}\n\n"
        f"--- EXTRACTED KNOWLEDGE FROM {len(extracts)} FILES ---\n{extracts_summary}\n\n"
        "IMPORTANT: The synthesis should reference specific metrics, customer names, "
        "and action items from the extracted data. Do not generalize — be specific."
    )

    model = config.get("gemini", {}).get("model", "gemini-3-flash-preview")
    client = genai.Client(api_key=api_key)

    try:
        response = client.models.generate_content(
            model=model,
            contents=[types.Part.from_text(text=combined_prompt)],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
            ),
        )
        response_text = response.text or "{}"
        log.debug("Raw Gemini synthesis response: %s", response_text[:1000])
        return parse_llm_json(response_text)
    except Exception as exc:
        log.error("Synthesis call failed: %s", exc)
        return {}


def _prompt_hash(config: dict) -> str:
    """Hash the extract prompt for change tracking."""
    prompt = (config.get("prompts") or {}).get("extract", "")
    return hashlib.sha256(prompt.encode()).hexdigest()[:12]


def write_transcript_note(
    transcript: TranscriptResult,
    title: str,
    extraction_note_filename: str,
    extract_dir: Path,
) -> Path | None:
    """Write a transcript markdown note if transcript succeeded.

    Args:
        transcript: TranscriptResult from generate_transcript()
        title: Title from the extraction result
        extraction_note_filename: Filename of the linked extraction note
        extract_dir: Directory to write the transcript note

    Returns:
        Path to the written file, or None if transcript failed.
    """
    if transcript.status != "complete" or not transcript.text:
        return None

    extract_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(transcript.source_path).stem
    note_path = extract_dir / f"{stem}_transcript.md"

    source_path = transcript.source_path.replace("\\", "/")
    frontmatter = (
        f"---\n"
        f"type: transcript\n"
        f"source: {source_path}\n"
        f"title: \"{title} — Full Transcript\"\n"
        f"duration_min: {transcript.duration_min}\n"
        f"word_count: {transcript.word_count}\n"
        f"linked_extraction: {extraction_note_filename}\n"
        f"---\n\n"
    )

    content = frontmatter + transcript.text
    note_path.write_text(content, encoding="utf-8")
    log.info("Wrote transcript note: %s (%d words)", note_path.name, transcript.word_count)
    return note_path


def build_package(
    groups: list[FileGroup],
    extracts: dict[str, ExtractionResult],
    output_dir: Path,
    package_name: str,
    config: dict,
) -> Path:
    """
    Build the complete output package directory.

    Args:
        groups: FileGroup list from correlate_files()
        extracts: Map of filename stem → ExtractionResult
        output_dir: Parent directory for the package
        package_name: Name for the package folder
        config: Unified config dict from load_config()

    Returns:
        Path to the created package directory
    """
    pkg_dir = output_dir / package_name
    pkg_dir.mkdir(parents=True, exist_ok=True)

    source_dir = pkg_dir / "source"
    extract_dir = pkg_dir / "extract"
    extract_dir.mkdir(parents=True, exist_ok=True)

    env = _get_jinja_env()
    now = datetime.now(timezone.utc)
    model = config.get("gemini", {}).get("model", "gemini-3-flash-preview")

    # --- Copy source files ---
    all_files: list[SourceFile] = []
    for group in groups:
        all_files.append(group.primary)
        all_files.extend(group.related)

    for source_file in all_files:
        try:
            _copy_source_file(source_file, source_dir, config)
        except Exception as exc:
            log.error("Failed to copy %s: %s", source_file.path.name, exc)

    # --- Write per-file extract markdown ---
    for stem, result in extracts.items():
        tmpl = env.get_template("extract.md.j2")

        # Provenance metadata from extraction result
        routing_reason = result.routing_reason or "unknown"
        prompt_version = result.prompt_version
        file_model = result.model_used or model

        # Build tags from frontmatter fields
        tag_input = {
            "products": result.products,
            "topics": result.topics,
            "domains": result.domains,
            "client": result.client,
            "doc_type": result.doc_type,
            "source_type": result.source_type,
        }
        tags = generate_tags(tag_input)

        # Compute quality score from assembled data
        _key_facts = result.raw_json.get("key_facts") or []
        _entities = result.raw_json.get("entities_mentioned") or []
        _overlay_fields = len(result.overlay) if result.overlay else 0
        _content_chars = len(result.summary or "") + sum(len(p) for p in result.key_points)
        quality_score = compute_quality_score(
            key_facts=_key_facts,
            facts_with_status=result.facts,
            overlay_fields_populated=_overlay_fields,
            content_chars=_content_chars,
            entities_count=len(_entities),
        )

        content = tmpl.render(
            source_file=str(result.source_file.path).replace("\\", "/"),
            content_type=result.content_type,
            title=result.title,
            date=now.strftime("%Y-%m-%d"),
            summary=result.summary,
            key_points=result.key_points,
            topics=result.topics,
            people=result.people,
            products=result.products,
            slides=result.slides,
            language=result.language,
            quality=result.quality,
            duration_min=result.duration_min,
            transcript_excerpt=result.transcript_excerpt,
            model=file_model,
            tokens_used=result.tokens_used,
            links_line=result.links_line,
            source_tool=result.source_tool,
            source_type=result.source_type,
            layer=result.layer,
            domains=result.domains,
            confidentiality=result.confidentiality,
            authority=result.authority,
            client=result.client,
            project=result.project,
            valid_to=result.raw_json.get("valid_to"),
            slides_subdir="slides" if result.source_file.path.suffix.lower() == ".pptx" else "frames",
            # Deep extraction fields
            doc_type=result.doc_type,
            extraction_version=result.extraction_version,
            depth=result.depth,
            key_facts=result.raw_json.get("key_facts") or [],
            entities_mentioned=result.raw_json.get("entities_mentioned") or [],
            overlay=result.overlay,
            slide_summaries=result.raw_json.get("slide_summaries") or [],
            # Freshness tracking
            freshness=result.freshness,
            # Fact validation — only show flagged/mismatched facts
            flagged_facts=[
                f for f in result.facts
                if f.get("verification_status") in ("flagged_mismatch", "unverified")
                and f.get("anomalies")
            ],
            # Tags
            tags=tags,
            # Quality + Provenance
            quality_score=quality_score,
            routing_reason=routing_reason,
            prompt_version=prompt_version,
        )
        # Normalized output filename: {date}_{name}_{hash4}.md
        extracted_at = (result.freshness or {}).get("extracted_at", now.isoformat())
        source_hash = (result.freshness or {}).get("source_hash", "")
        out_stem = normalize_output_filename(
            result.source_file.path.name, extracted_at, source_hash,
        )
        result.output_stem = out_stem
        out_path = extract_dir / f"{out_stem}.md"
        out_path.write_text(content, encoding="utf-8")
        log.info("Wrote %s", out_path.name)

    # --- Run cross-file synthesis ---
    synthesis_data = {}
    if len(extracts) > 0:
        log.info("Running synthesis across %d files...", len(extracts))
        synthesis_data = _run_synthesis(extracts, config)

    # --- Write synthesis.md ---
    synthesis_md_path = extract_dir / "synthesis.md"
    if synthesis_data:
        synthesis_lines = [
            "# Synthesis\n",
            f"## Executive Summary\n\n{synthesis_data.get('executive_summary', '')}\n",
        ]
        takeaways = synthesis_data.get("key_takeaways") or []
        if takeaways:
            synthesis_lines.append("## Key Takeaways\n")
            synthesis_lines.extend(f"- {t}" for t in takeaways)
            synthesis_lines.append("")

        relationships = synthesis_data.get("relationships", "")
        if relationships:
            synthesis_lines.append(f"## Relationships\n\n{relationships}\n")

        action_items = synthesis_data.get("action_items") or []
        if action_items:
            synthesis_lines.append("## Action Items\n")
            synthesis_lines.extend(f"- {a}" for a in action_items)
            synthesis_lines.append("")

        open_questions = synthesis_data.get("open_questions") or []
        if open_questions:
            synthesis_lines.append("## Open Questions\n")
            synthesis_lines.extend(f"- {q}" for q in open_questions)

        synthesis_md_path.write_text("\n".join(synthesis_lines), encoding="utf-8")
    else:
        synthesis_md_path.write_text("# Synthesis\n\n_Synthesis not available._\n", encoding="utf-8")

    # --- Write _meta.yaml ---
    meta_tmpl = env.get_template("meta.yaml.j2")
    # Determine cover image
    slides_dir = pkg_dir / "source" / "slides"
    frames_dir = pkg_dir / "source" / "frames"
    cover_slide = None
    cover_frame = None
    if slides_dir.exists() and list(slides_dir.glob("slide_001.png")):
        cover_slide = "slides/slide_001.png"
    elif frames_dir.exists() and sorted(frames_dir.glob("*.png")):
        first_frame = sorted(frames_dir.glob("*.png"))[0]
        cover_frame = f"frames/{first_frame.name}"

    meta_content = meta_tmpl.render(
        extracted_at=now.isoformat(),
        model=model,
        pipeline_version=PIPELINE_VERSION,
        prompt_hash=_prompt_hash(config),
        cover_slide=cover_slide,
        cover_frame=cover_frame,
        source_files=[
            {
                "path": str(f.path),
                "type": f.type.value,
                "size_bytes": f.size_bytes,
            }
            for f in all_files
        ],
    )
    (extract_dir / "_meta.yaml").write_text(meta_content, encoding="utf-8")

    # --- Aggregate index data ---
    all_topics: list[str] = []
    all_people: list[str] = []
    all_products: list[str] = []
    all_types: set[str] = set()

    for r in extracts.values():
        all_topics.extend(r.topics)
        all_people.extend(r.people)
        all_products.extend(r.products)
        all_types.add(r.source_file.type.value)

    # Deduplicate preserving order (normalizes dicts from LLM lists first)
    def dedup(lst: list) -> list:
        normalized = normalize_string_list(lst)
        seen = set()
        result = []
        for x in normalized:
            key = x.lower()
            if key not in seen:
                seen.add(key)
                result.append(x)
        return result

    # Prefer synthesis data for index if available
    exec_summary = synthesis_data.get("executive_summary") or (
        next(iter(extracts.values())).summary if extracts else ""
    )
    key_takeaways = synthesis_data.get("key_takeaways") or (
        next(iter(extracts.values())).key_points[:5] if extracts else []
    )
    index_title = synthesis_data.get("title") or package_name

    # --- Write index.md ---
    index_tmpl = env.get_template("index.md.j2")
    index_content = index_tmpl.render(
        id=package_name.lower().replace(" ", "-"),
        title=index_title,
        date=now.strftime("%Y-%m-%d"),
        types=sorted(all_types),
        topics=dedup(all_topics)[:20],
        people=dedup(all_people)[:20],
        products=dedup(all_products)[:20],
        file_count=len(all_files),
        model=model,
        executive_summary=exec_summary,
        groups=groups,
        key_takeaways=key_takeaways[:10],
        all_people=dedup(synthesis_data.get("all_people") or all_people)[:20],
        all_products=dedup(synthesis_data.get("all_products") or all_products)[:20],
        action_items=synthesis_data.get("action_items") or [],
        open_questions=synthesis_data.get("open_questions") or [],
    )
    (pkg_dir / "index.md").write_text(index_content, encoding="utf-8")

    log.info("Package written to: %s", pkg_dir)
    return pkg_dir
