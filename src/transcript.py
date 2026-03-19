"""Dedicated Gemini call for full verbatim video transcript.

Generates a timestamped transcript from an already-uploaded video file.
Designed to run as a second call after extraction, reusing the Gemini file URI.
"""

import logging
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path

from google import genai
from google.genai import types

log = logging.getLogger(__name__)

TRANSCRIPT_PROMPT = (
    "Transcribe everything spoken in this video verbatim. "
    "Include timestamps every 30-90 seconds formatted as [MM:SS]. "
    "When you can identify different speakers, label them "
    "(e.g., Speaker 1, Speaker 2, or by name if introduced). "
    "Output as plain text with timestamps, no JSON."
)

MAX_OUTPUT_TOKENS = 32768
MAX_RETRIES = 3
RETRY_BASE_SEC = 2


@dataclass
class TranscriptResult:
    text: str
    word_count: int
    duration_min: int
    status: str  # "complete", "failed"
    source_path: str


def sanitize_transcript(text: str) -> str:
    """Strip markdown control chars and potential HTML injection."""
    # Remove backticks
    text = text.replace("`", "")
    # Strip HTML-like angle brackets
    text = re.sub(r"<[^>]*>", "", text)
    return text


def generate_transcript(
    video_path: Path,
    gemini_file_uri: str,
    config: dict,
) -> TranscriptResult:
    """Dedicated Gemini call for full verbatim transcript.

    Args:
        video_path: Original video path (for metadata)
        gemini_file_uri: URI of already-uploaded video in Gemini File API
        config: Unified config dict

    Returns:
        TranscriptResult with status "complete" or "failed"
    """
    api_key_env = config.get("gemini", {}).get("api_key_env", "GEMINI_API_KEY")
    api_key = os.environ.get(api_key_env)
    if not api_key:
        log.warning("No Gemini API key for transcript generation")
        return TranscriptResult(
            text="", word_count=0, duration_min=0,
            status="failed", source_path=str(video_path),
        )

    client = genai.Client(api_key=api_key)
    model = config.get("model_override") or config.get("gemini", {}).get("model", "gemini-3-flash-preview")

    contents = [
        types.Part.from_uri(file_uri=gemini_file_uri, mime_type="video/mp4"),
        types.Part.from_text(text=TRANSCRIPT_PROMPT),
    ]

    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.models.generate_content(
                model=model,
                contents=contents,
                config=types.GenerateContentConfig(
                    max_output_tokens=MAX_OUTPUT_TOKENS,
                ),
            )
            raw_text = response.text or ""
            clean_text = sanitize_transcript(raw_text)
            word_count = len(clean_text.split())

            log.info(
                "Transcript generated for %s: %d words (attempt %d)",
                video_path.name, word_count, attempt,
            )
            return TranscriptResult(
                text=clean_text,
                word_count=word_count,
                duration_min=0,  # caller sets this
                status="complete",
                source_path=str(video_path),
            )

        except Exception as exc:
            last_error = exc
            wait = RETRY_BASE_SEC ** attempt
            log.warning(
                "Transcript attempt %d/%d failed for %s: %s — retrying in %ds",
                attempt, MAX_RETRIES, video_path.name, exc, wait,
            )
            if attempt < MAX_RETRIES:
                time.sleep(wait)

    log.error("Transcript generation failed after %d attempts for %s: %s",
              MAX_RETRIES, video_path.name, last_error)
    return TranscriptResult(
        text="", word_count=0, duration_min=0,
        status="failed", source_path=str(video_path),
    )
