"""
Gemini API-based knowledge extraction.

For plain files (audio, docs, notes): single Gemini call.
For videos WITH extracted frames: uploads video + sends frame images inline
so Gemini can produce per-slide analysis (slide_title, speaker_explanation, etc.)

Usage:
    from src.extract import extract_knowledge, ExtractionResult, ExtractionError
    from src.inventory import SourceFile

    # Plain extraction
    result = extract_knowledge(source_file, config)

    # Video with pre-extracted slide frames
    result = extract_knowledge(source_file, config, frames=[Path("frame_001.png"), ...])
"""

import json
import logging
import mimetypes
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

from src.inventory import SourceFile, FileType

log = logging.getLogger(__name__)

# Max frames to send to Gemini in a single request (to avoid token overload)
MAX_FRAMES_PER_REQUEST = 50


@dataclass
class SlideInfo:
    slide_number: int
    slide_title: str = ""
    slide_content: str = ""
    speaker_explanation: str = ""
    key_points: list[str] = field(default_factory=list)
    timestamp_approx: str = ""


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
    slides: list[SlideInfo] = field(default_factory=list)  # populated for videos with frames
    raw_json: dict = field(default_factory=dict)
    tokens_used: int = 0


class ExtractionError(Exception):
    pass


def _get_client(config: dict):
    """Create and return authenticated Gemini client."""
    from google import genai

    api_key_env = config.get("gemini", {}).get("api_key_env", "GEMINI_API_KEY")
    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise ExtractionError(
            f"API key not found. Set {api_key_env} in your .env file."
        )
    return genai.Client(api_key=api_key)


def _get_model(config: dict) -> str:
    return config.get("gemini", {}).get("model", "gemini-2.0-flash")


def _get_prompt(config: dict, prompt_name: str) -> str:
    """Get a named prompt from config, injecting anonymization terms."""
    prompts = config.get("prompts") or {}
    prompt = prompts.get(prompt_name, "")
    if not prompt:
        raise ExtractionError(f"No '{prompt_name}' prompt found in config['prompts']")

    anon_terms = (config.get("anonymization") or {}).get("custom_terms", [])
    if anon_terms:
        prompt += f"\n\nRedact these terms from all output: {', '.join(anon_terms)}"

    return prompt


def _strip_fences(text: str) -> str:
    """Remove markdown code fences from LLM response."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _parse_response(response_text: str, source_file: SourceFile) -> dict:
    """Parse JSON from LLM response, with error context."""
    cleaned = _strip_fences(response_text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ExtractionError(
            f"Failed to parse JSON response for {source_file.path.name}: {exc}\n"
            f"Response (first 500 chars): {response_text[:500]}"
        )


def _slides_from_json(slides_data: list) -> list[SlideInfo]:
    """Parse slides array from LLM JSON into SlideInfo objects."""
    result = []
    for s in (slides_data or []):
        result.append(SlideInfo(
            slide_number=int(s.get("slide_number", len(result) + 1)),
            slide_title=s.get("slide_title") or "",
            slide_content=s.get("slide_content") or "",
            speaker_explanation=s.get("speaker_explanation") or "",
            key_points=s.get("key_points") or [],
            timestamp_approx=s.get("timestamp_approx") or "",
        ))
    return result


def _result_from_json(data: dict, source_file: SourceFile, tokens: int) -> ExtractionResult:
    """Build ExtractionResult from parsed JSON, filling defaults for missing keys."""
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
    )


def _upload_and_wait(client, path: Path, config: dict):
    """Upload file via Gemini File API and poll until ACTIVE."""
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
            raise ExtractionError(
                f"File upload timed out after {upload_timeout}s for {path.name}"
            )

        log.debug("File state: %s, waiting %ss...", state_str, polling_sec)
        time.sleep(polling_sec)
        uploaded = client.files.get(name=uploaded.name)


def _build_video_with_frames_contents(
    client,
    file: SourceFile,
    frames: list[Path],
    config: dict,
):
    """
    Build Gemini content parts for a video + slide frames request.

    Returns (contents, prompt_used).
    """
    from google.genai import types

    prompt = _get_prompt(config, "extract_with_frames")

    # Upload video
    uploaded = _upload_and_wait(client, file.path, config)
    parts = [types.Part.from_uri(file_uri=uploaded.uri, mime_type=uploaded.mime_type)]

    # Add frame images inline (capped at MAX_FRAMES_PER_REQUEST)
    selected = frames[:MAX_FRAMES_PER_REQUEST]
    if len(frames) > MAX_FRAMES_PER_REQUEST:
        log.warning(
            "Capping frames from %d to %d for Gemini request",
            len(frames), MAX_FRAMES_PER_REQUEST,
        )

    for frame_path in selected:
        mime = mimetypes.guess_type(str(frame_path))[0] or "image/png"
        data = frame_path.read_bytes()
        parts.append(types.Part.from_bytes(data=data, mime_type=mime))

    # Append count hint so Gemini knows how many frames it received
    parts.append(types.Part.from_text(
        text=f"[{len(selected)} slide frame(s) provided above, in order.]\n\n{prompt}"
    ))

    return parts


def extract_knowledge(
    file: SourceFile,
    config: dict,
    frames: list[Path] | None = None,
) -> ExtractionResult:
    """
    Send a file to Gemini and return structured extracted knowledge.

    Strategy:
    - VIDEO + frames provided → upload video + send frame images + extract_with_frames prompt
    - VIDEO / AUDIO (no frames) → File API upload + extract prompt
    - DOCUMENT / SLIDES > 20MB → File API upload + extract prompt
    - DOCUMENT / SLIDES ≤ 20MB → inline bytes + extract prompt
    - NOTE / TRANSCRIPT / SPREADSHEET → text embedded in prompt

    Args:
        file: SourceFile to extract from
        config: Unified config dict from load_config()
        frames: Optional list of pre-extracted slide frame Paths (video only)

    Returns:
        ExtractionResult with structured knowledge (+ slides[] when frames given)

    Raises:
        ExtractionError: If extraction fails (caller logs and skips)
    """
    from google import genai
    from google.genai import types

    client = _get_client(config)
    model = _get_model(config)

    INLINE_SIZE_LIMIT = 20 * 1024 * 1024  # 20MB

    # Build content parts
    if file.type == FileType.VIDEO and frames:
        log.info(
            "Extracting knowledge from %s with %d frames...",
            file.path.name, len(frames),
        )
        contents = _build_video_with_frames_contents(client, file, frames, config)

    elif file.type in (FileType.VIDEO, FileType.AUDIO):
        log.info("Extracting knowledge from %s (no frames)...", file.path.name)
        prompt = _get_prompt(config, "extract")
        uploaded = _upload_and_wait(client, file.path, config)
        contents = [
            types.Part.from_uri(file_uri=uploaded.uri, mime_type=uploaded.mime_type),
            types.Part.from_text(text=prompt),
        ]

    elif file.type in (FileType.DOCUMENT, FileType.SLIDES):
        log.info("Extracting knowledge from %s (%s)...", file.path.name, file.type.value)
        prompt = _get_prompt(config, "extract")
        if file.size_bytes > INLINE_SIZE_LIMIT:
            uploaded = _upload_and_wait(client, file.path, config)
            contents = [
                types.Part.from_uri(file_uri=uploaded.uri, mime_type=uploaded.mime_type),
                types.Part.from_text(text=prompt),
            ]
        else:
            mime = mimetypes.guess_type(str(file.path))[0] or "application/octet-stream"
            data = file.path.read_bytes()
            contents = [
                types.Part.from_bytes(data=data, mime_type=mime),
                types.Part.from_text(text=prompt),
            ]

    elif file.type in (FileType.NOTE, FileType.TRANSCRIPT, FileType.SPREADSHEET):
        log.info("Extracting knowledge from %s (text)...", file.path.name)
        prompt = _get_prompt(config, "extract")
        try:
            text_content = file.path.read_text(encoding="utf-8", errors="replace")
        except Exception as exc:
            raise ExtractionError(f"Could not read {file.path.name}: {exc}")
        combined_prompt = f"{prompt}\n\n--- FILE CONTENT ---\n{text_content[:50000]}"
        contents = [types.Part.from_text(text=combined_prompt)]

    else:
        raise ExtractionError(
            f"Unsupported file type {file.type} for {file.path.name}"
        )

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
    result = _result_from_json(data, file, tokens)

    log.info(
        "Extracted: '%s' | slides=%d | topics=%s | tokens=%d",
        result.title,
        len(result.slides),
        result.topics[:3],
        tokens,
    )
    return result
