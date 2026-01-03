import os
import subprocess
import tempfile
import sys
import time
import re
from pathlib import Path
from typing import List, Dict, Optional
from dotenv import load_dotenv
from groq import Groq
from groq import RateLimitError

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.preprocess_audio import preprocess_for_transcription, get_file_size_mb
from src.transcribe.chunker import split_and_get_metadata, merge_transcripts
from config.config_loader import get

load_dotenv()


def parse_wait_time(error_message: str) -> Optional[int]:
    """
    Parse wait time from Groq rate limit error message.

    Example error message:
    "Rate limit reached. Please try again in 19m35.5s."

    Args:
        error_message: Error message string

    Returns:
        Wait time in seconds, or None if not found
    """
    # Pattern: "try again in 19m35.5s" or "try again in 35s" or "try again in 1h5m"
    pattern = r'try again in (?:(\d+)h)?(?:(\d+)m)?(?:([\d.]+)s)?'
    match = re.search(pattern, error_message, re.IGNORECASE)

    if match:
        hours = int(match.group(1)) if match.group(1) else 0
        minutes = int(match.group(2)) if match.group(2) else 0
        seconds = float(match.group(3)) if match.group(3) else 0

        total_seconds = hours * 3600 + minutes * 60 + seconds
        return int(total_seconds) + 5  # Add 5s buffer

    return None


def wait_with_progress(seconds: int, reason: str = "Rate limit"):
    """
    Wait for specified seconds with progress updates.

    Args:
        seconds: Number of seconds to wait
        reason: Reason for waiting (for display)
    """
    minutes = seconds // 60
    remaining_secs = seconds % 60

    if minutes > 0:
        print(f"  {reason}. Waiting {minutes}m {remaining_secs}s...")
    else:
        print(f"  {reason}. Waiting {remaining_secs}s...")

    # Wait in chunks with progress
    interval = 30  # Update every 30s
    elapsed = 0

    while elapsed < seconds:
        chunk = min(interval, seconds - elapsed)
        time.sleep(chunk)
        elapsed += chunk

        if elapsed < seconds:
            remaining = seconds - elapsed
            remaining_min = remaining // 60
            remaining_sec = remaining % 60
            print(f"    ... {remaining_min}m {remaining_sec}s remaining")

    print(f"  ✓ Wait complete, retrying...")


def extract_audio(video_path: str, output_path: str) -> str:
    """Extract and compress audio from video."""
    subprocess.run([
        "ffmpeg", "-y", "-i", video_path,
        "-vn", "-ac", "1", "-ar", "16000", "-b:a", "32k",
        output_path
    ], check=True, capture_output=True)
    return output_path


def _transcribe_chunk(
    audio_path: str,
    model: str = "whisper-large-v3",
    client: Optional[Groq] = None,
    retry_enabled: bool = True,
    max_retries: int = 3
) -> List[Dict]:
    """
    Transcribe a single audio chunk using Groq API with retry logic.

    Automatically retries on rate limit errors with exponential backoff.

    Args:
        audio_path: Path to audio file (must be < 25MB)
        model: Whisper model name
        client: Groq client (will create if None)
        retry_enabled: Enable automatic retry on rate limit
        max_retries: Maximum number of retries

    Returns:
        List of segments with timestamps

    Raises:
        RateLimitError: If max retries exceeded
        ValueError: If file too large or API key missing
    """
    if client is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in .env")
        client = Groq(api_key=api_key)

    file_size_mb = get_file_size_mb(audio_path)

    if file_size_mb > 25:
        raise ValueError(
            f"Chunk too large ({file_size_mb:.1f}MB). "
            f"Maximum size is 25MB. Use split_audio() first."
        )

    # Load retry config
    try:
        retry_config = get("settings", "transcription.retry", {})
        if isinstance(retry_config, dict):
            retry_enabled = retry_config.get("enabled", retry_enabled)
            max_retries = retry_config.get("max_retries", max_retries)
            initial_delay = retry_config.get("initial_delay", 60)
            backoff_multiplier = retry_config.get("backoff_multiplier", 1.5)
        else:
            initial_delay = 60
            backoff_multiplier = 1.5
    except:
        initial_delay = 60
        backoff_multiplier = 1.5

    print(f"  Transcribing {os.path.basename(audio_path)} ({file_size_mb:.1f}MB)...")

    retry_count = 0
    last_error = None

    while retry_count <= max_retries:
        try:
            with open(audio_path, "rb") as file:
                transcription = client.audio.transcriptions.create(
                    file=(os.path.basename(audio_path), file.read()),
                    model=model,
                    response_format="verbose_json"
                )

            segments = [
                {
                    "start": seg["start"],
                    "end": seg["end"],
                    "text": seg["text"].strip()
                }
                for seg in transcription.segments
            ]

            print(f"  ✓ Transcribed {len(segments)} segments")

            return segments

        except RateLimitError as e:
            last_error = e
            retry_count += 1

            if not retry_enabled or retry_count > max_retries:
                print(f"\n  ❌ Rate limit exceeded after {retry_count - 1} retries")
                print(f"  Suggestion: Use --provider openai for faster processing")
                raise

            # Parse wait time from error message
            error_msg = str(e)
            wait_time = parse_wait_time(error_msg)

            if wait_time is None:
                # Fallback to exponential backoff if can't parse
                wait_time = int(initial_delay * (backoff_multiplier ** (retry_count - 1)))

            print(f"\n  ⚠️  Rate limit reached (retry {retry_count}/{max_retries})")
            wait_with_progress(wait_time, "Rate limit")

    # Should not reach here, but just in case
    if last_error:
        raise last_error
    else:
        raise ValueError("Transcription failed for unknown reason")


def _transcribe_chunk_simple(
    audio_path: str,
    model: str = "whisper-large-v3",
    client: Optional[Groq] = None
) -> List[Dict]:
    """
    Simple transcription without retry (for backwards compatibility).

    Args:
        audio_path: Path to audio file
        model: Whisper model name
        client: Groq client

    Returns:
        List of segments
    """
    return _transcribe_chunk(
        audio_path,
        model=model,
        client=client,
        retry_enabled=False,
        max_retries=0
    )


def transcribe_groq(
    file_path: str,
    model: str = "whisper-large-v3",
    enable_preprocessing: bool = True,
    enable_silence_removal: bool = True,
    enable_chunking: bool = True,
    max_chunk_size_mb: int = 24,
    overlap_seconds: float = 5.0
) -> List[Dict]:
    """
    Transcribe audio/video using Groq API with automatic preprocessing.

    Handles large files by:
    1. Extracting audio from video
    2. Removing silence to reduce size
    3. Splitting into chunks if needed
    4. Transcribing each chunk
    5. Merging transcripts with corrected timestamps

    Args:
        file_path: Path to audio/video file
        model: Whisper model (whisper-large-v3)
        enable_preprocessing: Enable audio optimization and silence removal
        enable_silence_removal: Remove silence from audio
        enable_chunking: Enable chunking if file is too large
        max_chunk_size_mb: Maximum chunk size in MB
        overlap_seconds: Overlap between chunks for context

    Returns:
        List of {"start": float, "end": float, "text": str}

    Example:
        >>> # Transcribe 73MB file (5 hours)
        >>> segments = transcribe_groq("long_meeting.mp4")
        >>> # Automatically: removes silence (73MB → 45MB), splits into 2 chunks, merges
    """
    print(f"\n=== Transcription: {os.path.basename(file_path)} ===")

    # Load config
    try:
        silence_threshold = get("settings", "transcription.silence_removal.threshold_db", -40)
        min_silence_duration = get("settings", "transcription.silence_removal.min_silence_duration", 2.0)
    except:
        # Fallback to defaults if config not available
        silence_threshold = -40
        min_silence_duration = 2.0

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not found in .env")

    client = Groq(api_key=api_key)

    # Step 1: Extract audio from video if needed
    temp_audio = None
    if file_path.lower().endswith(('.mp4', '.mkv', '.avi', '.mov', '.webm')):
        temp_audio = os.path.join(tempfile.gettempdir(), "groq_temp_audio.mp3")
        print("  Converting video to audio...")
        extract_audio(file_path, temp_audio)
        audio_path = temp_audio
    else:
        audio_path = file_path

    original_size_mb = get_file_size_mb(audio_path)
    print(f"  Audio size: {original_size_mb:.1f}MB")

    # Step 2: Preprocess if enabled and file is large
    if enable_preprocessing and original_size_mb > max_chunk_size_mb:
        print(f"  File exceeds {max_chunk_size_mb}MB, preprocessing...")

        processed_audio, stats = preprocess_for_transcription(
            audio_path,
            remove_silence_enabled=enable_silence_removal,
            threshold_db=silence_threshold,
            min_silence_duration=min_silence_duration,
            verbose=True
        )

        audio_path = processed_audio
        processed_size_mb = get_file_size_mb(audio_path)

        print(f"  Preprocessed: {original_size_mb:.1f}MB → {processed_size_mb:.1f}MB "
              f"({stats['total_reduction_percent']:.1f}% reduction)")
    else:
        processed_size_mb = original_size_mb

    # Step 3: Check if chunking is needed
    if processed_size_mb > max_chunk_size_mb:
        if not enable_chunking:
            raise ValueError(
                f"File too large ({processed_size_mb:.1f}MB) and chunking is disabled. "
                f"Enable chunking or reduce file size below {max_chunk_size_mb}MB."
            )

        print(f"\n  File still exceeds {max_chunk_size_mb}MB after preprocessing")
        print(f"  Splitting into chunks...")

        # Split audio into chunks
        chunk_paths, chunk_durations = split_and_get_metadata(
            audio_path,
            max_size_mb=max_chunk_size_mb,
            overlap_seconds=overlap_seconds,
            verbose=True
        )

        print(f"\n  Transcribing {len(chunk_paths)} chunks...")

        # Transcribe each chunk
        chunk_transcripts = []
        for i, chunk_path in enumerate(chunk_paths):
            print(f"\n  === Chunk {i+1}/{len(chunk_paths)} ===")
            transcript = _transcribe_chunk(chunk_path, model=model, client=client)
            chunk_transcripts.append(transcript)

        # Merge transcripts
        print(f"\n  Merging transcripts...")
        segments = merge_transcripts(
            chunk_transcripts,
            chunk_durations=chunk_durations,
            overlap_seconds=overlap_seconds,
            verbose=True
        )

        # Cleanup chunk files
        for chunk_path in chunk_paths:
            if os.path.exists(chunk_path):
                os.remove(chunk_path)

    else:
        # File is small enough, transcribe directly
        print(f"  File size OK ({processed_size_mb:.1f}MB), transcribing...")
        segments = _transcribe_chunk(audio_path, model=model, client=client)

    # Cleanup temp files
    if temp_audio and os.path.exists(temp_audio):
        os.remove(temp_audio)

    if enable_preprocessing and audio_path != file_path and os.path.exists(audio_path):
        os.remove(audio_path)

    total_duration = segments[-1]["end"] if segments else 0
    print(f"\n✓ Transcription complete: {len(segments)} segments, {total_duration/60:.1f} min")

    return segments