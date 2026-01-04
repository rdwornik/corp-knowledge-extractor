"""
OpenAI Whisper API transcription backend.

Provides transcription using OpenAI's Whisper API with:
- No file size limits (large files supported)
- Fast processing (~10x faster than realtime)
- Paid service ($0.006/minute)
- Consistent output format compatible with Groq backend
"""

import os
import subprocess
import tempfile
import sys
from pathlib import Path
from typing import List, Dict, Optional
from dotenv import load_dotenv
from openai import OpenAI

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.transcribe.chunker import split_and_get_metadata, merge_transcripts

load_dotenv()


def get_file_size_mb(file_path: str) -> float:
    """Get file size in megabytes."""
    return os.path.getsize(file_path) / (1024 * 1024)


def extract_audio(video_path: str, output_path: str) -> str:
    """
    Extract and compress audio from video.

    Args:
        video_path: Path to video file
        output_path: Path for output audio

    Returns:
        Path to extracted audio file
    """
    subprocess.run([
        "ffmpeg", "-y", "-i", video_path,
        "-vn", "-ac", "1", "-ar", "16000", "-b:a", "32k",
        output_path
    ], check=True, capture_output=True)
    return output_path


def _transcribe_chunk(
    audio_path: str,
    model: str = "whisper-1",
    client: Optional[OpenAI] = None
) -> List[Dict]:
    """
    Transcribe a single audio chunk using OpenAI Whisper API.

    Args:
        audio_path: Path to audio file (must be < 25MB)
        model: Whisper model name
        client: OpenAI client (will create if None)

    Returns:
        List of segments with timestamps
    """
    if client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in .env")
        client = OpenAI(api_key=api_key)

    file_size_mb = get_file_size_mb(audio_path)

    if file_size_mb > 25:
        raise ValueError(
            f"Chunk too large ({file_size_mb:.1f}MB). "
            f"Maximum size is 25MB. Use split_audio() first."
        )

    print(f"  Transcribing {os.path.basename(audio_path)} ({file_size_mb:.1f}MB)...")

    with open(audio_path, "rb") as audio_file:
        transcription = client.audio.transcriptions.create(
            model=model,
            file=audio_file,
            response_format="verbose_json",
            timestamp_granularities=["segment"]
        )

    # Parse segments
    segments = []
    if hasattr(transcription, 'segments') and transcription.segments:
        for seg in transcription.segments:
            segments.append({
                "start": seg.get("start", 0),
                "end": seg.get("end", 0),
                "text": seg.get("text", "").strip()
            })
    else:
        # Fallback: if no segments, create one segment with full text
        if hasattr(transcription, 'text'):
            segments.append({
                "start": 0,
                "end": 0,
                "text": transcription.text.strip()
            })

    print(f"  ✓ Transcribed {len(segments)} segments")

    return segments


def transcribe_openai(
    file_path: str,
    model: str = "whisper-1",
    enable_chunking: bool = True,
    max_chunk_size_mb: int = 24,
    overlap_seconds: float = 5.0,
    verbose: bool = True
) -> List[Dict]:
    """
    Transcribe audio/video using OpenAI Whisper API with automatic chunking.

    Handles large files by:
    1. Extracting audio from video
    2. Splitting into chunks if needed (OpenAI has 25MB limit)
    3. Transcribing each chunk
    4. Merging transcripts with corrected timestamps

    Args:
        file_path: Path to audio/video file
        model: Whisper model name (default: "whisper-1")
        enable_chunking: Enable chunking if file is too large
        max_chunk_size_mb: Maximum chunk size in MB
        overlap_seconds: Overlap between chunks for context
        verbose: Print progress information

    Returns:
        List of {"start": float, "end": float, "text": str}

    Example:
        >>> segments = transcribe_openai("5_hour_meeting.mp4")
        >>> # Automatically chunks if > 25MB, merges transcripts
        >>> print(f"Transcribed {len(segments)} segments")
    """
    if verbose:
        print(f"\n=== OpenAI Whisper Transcription ===")
        print(f"File: {os.path.basename(file_path)}")

    # Get API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY not found in .env file. "
            "Please add your OpenAI API key to use this provider."
        )

    client = OpenAI(api_key=api_key)

    # Step 1: Extract audio from video if needed
    temp_audio = None
    if file_path.lower().endswith(('.mp4', '.mkv', '.avi', '.mov', '.webm')):
        temp_audio = os.path.join(tempfile.gettempdir(), "openai_temp_audio.mp3")
        if verbose:
            print("  Converting video to audio...")
        extract_audio(file_path, temp_audio)
        audio_path = temp_audio
    else:
        audio_path = file_path

    # Check file size
    file_size_mb = get_file_size_mb(audio_path)
    if verbose:
        print(f"  Audio size: {file_size_mb:.1f}MB")

    # Step 2: Check if chunking is needed
    if file_size_mb > max_chunk_size_mb:
        if not enable_chunking:
            raise ValueError(
                f"File too large ({file_size_mb:.1f}MB) and chunking is disabled. "
                f"Enable chunking or reduce file size below {max_chunk_size_mb}MB."
            )

        print(f"\n  File exceeds {max_chunk_size_mb}MB")
        print(f"  Splitting into chunks...")

        # Split audio into chunks
        chunk_paths, chunk_durations = split_and_get_metadata(
            audio_path,
            max_size_mb=max_chunk_size_mb,
            overlap_seconds=overlap_seconds,
            verbose=verbose
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
            verbose=verbose
        )

        # Cleanup chunk files
        for chunk_path in chunk_paths:
            if os.path.exists(chunk_path):
                os.remove(chunk_path)

    else:
        # File is small enough, transcribe directly
        if verbose:
            print(f"  File size OK ({file_size_mb:.1f}MB), transcribing...")
        segments = _transcribe_chunk(audio_path, model=model, client=client)

    # Cleanup temp file
    if temp_audio and os.path.exists(temp_audio):
        os.remove(temp_audio)

    total_duration = segments[-1]["end"] if segments else 0
    if verbose:
        print(f"\n✓ Transcription complete: {len(segments)} segments, {total_duration/60:.1f} min")
        print(f"✓ Cost: ~${(total_duration/60) * 0.006:.2f}")

    return segments


def transcribe_openai_with_preprocessing(
    file_path: str,
    enable_preprocessing: bool = True,
    enable_silence_removal: bool = True,
    model: str = "whisper-1",
    verbose: bool = True
) -> List[Dict]:
    """
    Transcribe using OpenAI with optional preprocessing to reduce cost.

    Preprocessing with silence removal can significantly reduce costs
    for files with pauses, even though OpenAI handles large files well.

    Args:
        file_path: Path to audio/video file
        enable_preprocessing: Enable audio optimization
        enable_silence_removal: Remove silence to reduce file size (and cost)
        model: Whisper model name
        verbose: Print progress information

    Returns:
        List of transcript segments

    Example:
        >>> # 73MB file with 40% silence
        >>> segments = transcribe_openai_with_preprocessing(
        ...     "meeting.mp4",
        ...     enable_silence_removal=True
        ... )
        >>> # Saves ~$0.75 by removing 40% of audio
    """
    from scripts.preprocess_audio import preprocess_for_transcription
    from config.config_loader import get

    if verbose:
        print(f"\n=== OpenAI Transcription with Preprocessing ===")

    # Get original file size
    original_size_mb = get_file_size_mb(file_path)

    # Preprocess if enabled and beneficial
    processed_path = file_path
    if enable_preprocessing and original_size_mb > 10:  # Only preprocess if > 10MB
        if verbose:
            print(f"  Original file: {original_size_mb:.1f}MB")

        # Load config
        try:
            threshold_db = get("settings", "transcription.silence_removal.threshold_db", -40)
            min_silence_duration = get("settings", "transcription.silence_removal.min_silence_duration", 2.0)
        except:
            threshold_db = -40
            min_silence_duration = 2.0

        processed_path, stats = preprocess_for_transcription(
            file_path,
            remove_silence_enabled=enable_silence_removal,
            threshold_db=threshold_db,
            min_silence_duration=min_silence_duration,
            verbose=verbose
        )

        if verbose:
            original_cost = (stats['original_size_mb'] / 60) * 0.006
            new_cost = (stats['final_size_mb'] / 60) * 0.006
            savings = original_cost - new_cost

            print(f"\n  Cost savings from preprocessing:")
            print(f"    Original: ${original_cost:.2f}")
            print(f"    Optimized: ${new_cost:.2f}")
            print(f"    Savings: ${savings:.2f} ({stats['total_reduction_percent']:.1f}%)")

    # Transcribe
    segments = transcribe_openai(processed_path, model=model, verbose=verbose)

    # Cleanup preprocessed file if created
    if processed_path != file_path and os.path.exists(processed_path):
        os.remove(processed_path)

    return segments


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python openai_backend.py <audio_file>")
        print("\nExample:")
        print("  python openai_backend.py meeting.mp4")
        print("  python openai_backend.py --preprocess meeting.mp4")
        sys.exit(1)

    # Check for preprocessing flag
    preprocess = "--preprocess" in sys.argv
    file_path = [arg for arg in sys.argv[1:] if not arg.startswith("--")][0]

    if not os.path.exists(file_path):
        print(f"Error: File not found: {file_path}")
        sys.exit(1)

    # Transcribe
    if preprocess:
        segments = transcribe_openai_with_preprocessing(file_path)
    else:
        segments = transcribe_openai(file_path)

    print(f"\n✓ Transcription complete!")
    print(f"  Total segments: {len(segments)}")
    if segments:
        print(f"  Duration: {segments[-1]['end']/60:.1f} minutes")
        print(f"  First segment: \"{segments[0]['text'][:50]}...\"")
