"""
Gemini API-based knowledge extraction.

For plain files (audio, docs, notes): single Gemini call with text/binary content.
For videos WITH sampled frames: uploads video + sends sampled frame images inline
so Gemini can identify unique slides and produce per-slide analysis.

Usage:
    from src.extract import extract_knowledge, ExtractionResult, ExtractionError
    from src.inventory import SourceFile
    from src.frames.sampler import SampledFrame

    # Plain extraction
    result = extract_knowledge(source_file, config)

    # Video with pre-sampled frames (AI picks unique slides)
    result = extract_knowledge(source_file, config, sampled_frames=frames)
"""

import json
import logging
import mimetypes
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

from src.inventory import SourceFile, FileType
from src.frames.sampler import SampledFrame
from src.text_extract import TextExtractionResult
from src.utils import parse_llm_json
from src.post_process import post_process_extraction, PostProcessResult
from src.taxonomy_prompt import get_taxonomy_for_prompt

log = logging.getLogger(__name__)

# Max frames to send to Gemini in a single request (avoid token overload)
MAX_FRAMES_PER_REQUEST = 200


@dataclass
class SlideAnalysis:
    slide_number: int
    frame_index: int = 0          # sample_NNNN index from Gemini's response
    slide_title: str = ""
    timestamp_approx: str = ""
    speaker_insight: str = ""     # What the speaker SAID — the main content
    so_what: str = ""             # Practical implication paragraph
    critical_notes: str | None = None  # Red flags / things to verify
    key_facts: list[str] = field(default_factory=list)


@dataclass
class ExtractionResult:
    source_file: SourceFile
    title: str
    summary: str
    key_points: list[str] = field(default_factory=list)
    topics: list[str] = field(default_factory=list)
    people: list[str] = field(default_factory=list)
    products: list[str] = field(default_factory=list)
    content_type: str = "unknown"
    language: str = "en"
    quality: str = "medium"
    duration_min: int | None = None
    transcript_excerpt: str = ""
    slides: list[SlideAnalysis] = field(default_factory=list)
    raw_json: dict = field(default_factory=dict)
    tokens_used: int = 0
    links_line: str = ""          # Deterministic Links line from post-processing
    source_tool: str = "knowledge-extractor"
    validation_result: str = "valid"  # "valid", "warnings", "quarantine"
    # Schema v2 knowledge dimensions
    source_type: str = "documentation"
    layer: str = "learning"
    domains: list[str] = field(default_factory=list)
    confidentiality: str = "internal"
    authority: str = "tribal"
    client: str | None = None


class ExtractionError(Exception):
    pass


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_client(config: dict):
    from google import genai

    api_key_env = config.get("gemini", {}).get("api_key_env", "GEMINI_API_KEY")
    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise ExtractionError(f"API key not found. Set {api_key_env} in your .env file.")
    return genai.Client(api_key=api_key)


def _get_model(config: dict) -> str:
    return config.get("gemini", {}).get("model", "gemini-2.5-flash")


def _get_prompt(config: dict, prompt_name: str) -> str:
    prompts = config.get("prompts") or {}
    prompt = prompts.get(prompt_name, "")
    if not prompt:
        raise ExtractionError(f"No '{prompt_name}' prompt found in config['prompts']")

    anon_terms = (config.get("anonymization") or {}).get("custom_terms", [])
    if anon_terms:
        prompt += f"\n\nRedact these terms from all output: {', '.join(anon_terms)}"

    # Inject canonical taxonomy so LLM uses correct terms from the start
    if prompt_name in ("extract", "extract_with_slides"):
        prompt += f"\n\n{get_taxonomy_for_prompt()}"

    return prompt


def _upload_and_wait(client, path: Path, config: dict):
    """Upload a file via the Gemini File API and poll until ACTIVE."""
    from google.genai import types

    polling_sec = config.get("gemini", {}).get("polling_interval_sec", 5)
    upload_timeout = config.get("gemini", {}).get("upload_timeout_sec", 300)

    log.info("Uploading %s to Gemini File API...", path.name)
    uploaded = client.files.upload(file=path)

    deadline = time.time() + upload_timeout
    while True:
        file_state = uploaded.state
        state_str = file_state.name if hasattr(file_state, "name") else str(file_state)

        if state_str == "ACTIVE":
            log.info("File %s is ACTIVE", path.name)
            return uploaded
        if state_str == "FAILED":
            raise ExtractionError(f"File upload failed for {path.name}")
        if time.time() > deadline:
            raise ExtractionError(f"File upload timed out after {upload_timeout}s for {path.name}")

        log.debug("File state: %s, waiting %ss...", state_str, polling_sec)
        time.sleep(polling_sec)
        uploaded = client.files.get(name=uploaded.name)


def _slides_from_json(slides_data: list) -> list[SlideAnalysis]:
    result = []
    for i, s in enumerate(slides_data or []):
        result.append(SlideAnalysis(
            slide_number=int(s.get("slide_number") or (i + 1)),
            frame_index=int(s.get("frame_index") or 0),
            slide_title=s.get("slide_title") or "",
            timestamp_approx=s.get("timestamp_approx") or "",
            speaker_insight=s.get("speaker_insight") or "",
            so_what=s.get("so_what") or "",
            critical_notes=s.get("critical_notes") or None,
            key_facts=s.get("key_facts") or [],
        ))
    return result


def _result_from_json(data: dict, source_file: SourceFile, tokens: int) -> ExtractionResult:
    return ExtractionResult(
        source_file=source_file,
        title=data.get("title") or source_file.name,
        summary=data.get("summary") or "",
        key_points=data.get("key_points") or [],
        topics=data.get("topics") or [],
        people=data.get("people") or [],
        products=data.get("products") or [],
        content_type=data.get("content_type") or data.get("type") or "unknown",
        language=data.get("language") or "en",
        quality=data.get("quality") or "medium",
        duration_min=data.get("duration_min"),
        transcript_excerpt=data.get("transcript_excerpt") or "",
        slides=_slides_from_json(data.get("slides")),
        raw_json=data,
        tokens_used=tokens,
        source_type=data.get("source_type") or "documentation",
        layer=data.get("layer") or "learning",
        domains=data.get("domains") or [],
        confidentiality=data.get("confidentiality") or "internal",
        authority=data.get("authority") or "tribal",
        client=data.get("client"),
    )


def _parse_response(response_text: str, source_file: SourceFile) -> dict:
    log.debug("Raw Gemini response for %s: %s", source_file.path.name, response_text[:1000])
    try:
        return parse_llm_json(response_text)
    except ValueError as exc:
        raise ExtractionError(f"Failed to parse JSON response for {source_file.path.name}: {exc}")


def _build_sampled_frame_contents(
    client,
    file: SourceFile,
    sampled_frames: list[SampledFrame],
    config: dict,
) -> list:
    """
    Build Gemini content parts for video + sampled frames.

    Structure: [video_file_ref, frame_0_image, frame_1_image, ..., prompt_text]

    Frame images are sent inline as PNG bytes (small enough for inline).
    The prompt includes the frame count so Gemini knows what it received.
    """
    from google.genai import types

    prompt = _get_prompt(config, "extract_with_slides")

    # Upload video via File API
    uploaded = _upload_and_wait(client, file.path, config)
    parts = [types.Part.from_uri(file_uri=uploaded.uri, mime_type=uploaded.mime_type)]

    # Cap frames sent to avoid token overload
    selected = sampled_frames[:MAX_FRAMES_PER_REQUEST]
    if len(sampled_frames) > MAX_FRAMES_PER_REQUEST:
        log.warning(
            "Capping frames sent to Gemini: %d → %d",
            len(sampled_frames), MAX_FRAMES_PER_REQUEST,
        )

    # Add each sampled frame as inline image bytes
    for sf in selected:
        mime = mimetypes.guess_type(str(sf.path))[0] or "image/png"
        data = sf.path.read_bytes()
        parts.append(types.Part.from_bytes(data=data, mime_type=mime))

    # Append prompt with frame count hint
    parts.append(types.Part.from_text(
        text=f"[{len(selected)} sampled frame(s) provided above, sample_0000 through sample_{len(selected)-1:04d}.]\n\n{prompt}"
    ))

    return parts


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_knowledge(
    file: SourceFile,
    config: dict,
    sampled_frames: list[SampledFrame] | None = None,
) -> ExtractionResult:
    """
    Send a file to Gemini and return structured extracted knowledge.

    Strategy:
    - VIDEO + sampled_frames → File API upload + inline frame images + extract_with_slides prompt
      Gemini identifies unique slides, returns slides[] with frame_index references
    - VIDEO / AUDIO (no frames) → File API upload + extract prompt
    - DOCUMENT / SLIDES > 20MB → File API upload + extract prompt
    - DOCUMENT / SLIDES ≤ 20MB → inline bytes + extract prompt
    - NOTE / TRANSCRIPT / SPREADSHEET → text embedded in prompt

    Args:
        file: SourceFile to extract from
        config: Unified config dict from load_config()
        sampled_frames: Time-sampled frames from sampler.py (video only)

    Returns:
        ExtractionResult with slides[] populated when sampled_frames given

    Raises:
        ExtractionError: Caller logs and skips
    """
    from google import genai
    from google.genai import types

    client = _get_client(config)
    model = _get_model(config)

    INLINE_SIZE_LIMIT = 20 * 1024 * 1024  # 20MB

    # --- Build content parts ---
    if file.type == FileType.VIDEO and sampled_frames:
        log.info(
            "Extracting %s with %d sampled frames (AI slide selection)...",
            file.path.name, len(sampled_frames),
        )
        contents = _build_sampled_frame_contents(client, file, sampled_frames, config)

    elif file.type in (FileType.VIDEO, FileType.AUDIO):
        log.info("Extracting %s (no frames)...", file.path.name)
        prompt = _get_prompt(config, "extract")
        uploaded = _upload_and_wait(client, file.path, config)
        contents = [
            types.Part.from_uri(file_uri=uploaded.uri, mime_type=uploaded.mime_type),
            types.Part.from_text(text=prompt),
        ]

    elif file.type in (FileType.DOCUMENT, FileType.SLIDES):
        log.info("Extracting %s (%s)...", file.path.name, file.type.value)
        prompt = _get_prompt(config, "extract")
        if file.size_bytes > INLINE_SIZE_LIMIT:
            uploaded = _upload_and_wait(client, file.path, config)
            contents = [
                types.Part.from_uri(file_uri=uploaded.uri, mime_type=uploaded.mime_type),
                types.Part.from_text(text=prompt),
            ]
        else:
            mime = mimetypes.guess_type(str(file.path))[0] or "application/octet-stream"
            contents = [
                types.Part.from_bytes(data=file.path.read_bytes(), mime_type=mime),
                types.Part.from_text(text=prompt),
            ]

    elif file.type in (FileType.NOTE, FileType.TRANSCRIPT, FileType.SPREADSHEET):
        log.info("Extracting %s (text)...", file.path.name)
        prompt = _get_prompt(config, "extract")
        try:
            text_content = file.path.read_text(encoding="utf-8", errors="replace")
        except Exception as exc:
            raise ExtractionError(f"Could not read {file.path.name}: {exc}")
        contents = [types.Part.from_text(
            text=f"{prompt}\n\n--- FILE CONTENT ---\n{text_content[:50000]}"
        )]

    else:
        raise ExtractionError(f"Unsupported file type {file.type} for {file.path.name}")

    # --- Call Gemini ---
    response = client.models.generate_content(
        model=model,
        contents=contents,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
        ),
    )

    response_text = response.text or ""
    tokens = 0
    if hasattr(response, "usage_metadata") and response.usage_metadata:
        tokens = getattr(response.usage_metadata, "total_token_count", 0) or 0

    data = _parse_response(response_text, file)

    # Post-process: normalize terms, apply taxonomy, cap cardinality, build links line
    pp = post_process_extraction(
        raw_result=data,
        source_tool="knowledge-extractor",
        source_file=str(file.path),
    )
    if pp.changes:
        log.debug("Normalized: %s", pp.changes)

    result = _result_from_json(pp.data, file, tokens)
    result.links_line = pp.links_line
    result.validation_result = pp.validation_result.value

    log.info(
        "Extracted: '%s' | slides=%d | topics=%s | tokens=%d",
        result.title, len(result.slides), result.topics[:3], tokens,
    )
    return result


def extract_from_text(
    file: SourceFile,
    config: dict,
    text_result: TextExtractionResult,
) -> ExtractionResult:
    """
    Tier 2: Send pre-extracted text to Gemini 2.0 Flash (text-only, cheaper).

    Uses the cheaper text model since we already have good text content
    and don't need multimodal processing.

    Args:
        file: SourceFile metadata
        config: Unified config dict
        text_result: Pre-extracted text from local extraction (Tier 1)

    Returns:
        ExtractionResult with AI-structured knowledge
    """
    from google.genai import types

    client = _get_client(config)
    model = "gemini-2.0-flash"  # Always use cheaper model for text-only

    prompt = _get_prompt(config, "extract")

    # Truncate very long text to stay within token limits
    text_content = text_result.text[:80000]

    log.info(
        "Extracting %s via text-only AI (%d chars, model=%s)...",
        file.path.name, len(text_content), model,
    )

    contents = [types.Part.from_text(
        text=f"{prompt}\n\n--- FILE CONTENT ({text_result.extractor}) ---\n{text_content}"
    )]

    response = client.models.generate_content(
        model=model,
        contents=contents,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
        ),
    )

    response_text = response.text or ""
    tokens = 0
    if hasattr(response, "usage_metadata") and response.usage_metadata:
        tokens = getattr(response.usage_metadata, "total_token_count", 0) or 0

    data = _parse_response(response_text, file)

    # Post-process via corp-os-meta
    pp = post_process_extraction(
        raw_result=data,
        source_tool="knowledge-extractor",
        source_file=str(file.path),
    )

    result = _result_from_json(pp.data, file, tokens)
    result.links_line = pp.links_line
    result.validation_result = pp.validation_result.value

    log.info(
        "Tier 2 extracted: '%s' | topics=%s | tokens=%d",
        result.title, result.topics[:3], tokens,
    )
    return result


def extract_local(
    file: SourceFile,
    text_result: TextExtractionResult,
) -> ExtractionResult:
    """
    Tier 1: Build ExtractionResult from locally-extracted text only (FREE).

    No API call. Uses the file name as title and the raw text as summary.
    Still post-processes through corp-os-meta for normalization.

    Args:
        file: SourceFile metadata
        text_result: Pre-extracted text from local extraction

    Returns:
        ExtractionResult with basic structure (no AI insight)
    """
    log.info(
        "Local extraction for %s (%d chars, no API call)...",
        file.path.name, text_result.char_count,
    )

    # Build a minimal raw result for post-processing
    raw_result = {
        "title": file.name,
        "summary": text_result.text[:2000],
        "topics": [],
        "products": [],
        "people": [],
        "type": file.type.value,
        "language": "en",
        "quality": "local",
    }

    pp = post_process_extraction(
        raw_result=raw_result,
        source_tool="knowledge-extractor",
        source_file=str(file.path),
    )

    result = _result_from_json(pp.data, file, tokens=0)
    result.links_line = pp.links_line
    result.validation_result = pp.validation_result.value

    log.info("Tier 1 local extraction: '%s'", result.title)
    return result
