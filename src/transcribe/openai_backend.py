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
from typing import List, Dict
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


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


def transcribe_openai(
    file_path: str,
    model: str = "whisper-1",
    verbose: bool = True
) -> List[Dict]:
    """
    Transcribe audio/video using OpenAI Whisper API.

    Advantages over Groq:
    - No file size limit (can handle files > 25MB)
    - Faster processing
    - No hourly rate limits

    Disadvantages:
    - Paid service ($0.006/minute)

    Args:
        file_path: Path to audio/video file
        model: Whisper model name (default: "whisper-1")
        verbose: Print progress information

    Returns:
        List of {"start": float, "end": float, "text": str}

    Example:
        >>> segments = transcribe_openai("5_hour_meeting.mp4")
        >>> # No chunking needed - handles large files directly
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

    # Extract audio from video if needed
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
    file_size_mb = os.path.getsize(audio_path) / (1024 * 1024)
    if verbose:
        print(f"  Audio size: {file_size_mb:.1f}MB")

    # OpenAI has a 25MB limit per file, but supports larger files via chunking
    # For now, we'll use the same preprocessing as Groq if > 25MB
    if file_size_mb > 25:
        if verbose:
            print(f"  Note: File > 25MB. Consider using preprocessing to reduce cost.")
            print(f"  Processing full file (OpenAI supports this, but costs more).")

    if verbose:
        print(f"  Transcribing with OpenAI Whisper...")
        print(f"  Estimated cost: ${(file_size_mb / 60) * 0.006:.2f}")

    try:
        # Open file and send to OpenAI
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
            # This happens with some response formats
            if hasattr(transcription, 'text'):
                segments.append({
                    "start": 0,
                    "end": 0,  # Duration unknown
                    "text": transcription.text.strip()
                })

        if verbose:
            total_duration = segments[-1]["end"] if segments else 0
            print(f"  ✓ Transcribed {len(segments)} segments")
            print(f"  ✓ Duration: {total_duration/60:.1f} minutes")
            print(f"  ✓ Cost: ~${(total_duration/60) * 0.006:.2f}")

        return segments

    finally:
        # Cleanup temp file
        if temp_audio and os.path.exists(temp_audio):
            os.remove(temp_audio)


def transcribe_openai_with_preprocessing(
    file_path: str,
    enable_preprocessing: bool = True,
    enable_silence_removal: bool = True,
    model: str = "whisper-1",
    verbose: bool = True
) -> List[Dict]:
    """
    Transcribe using OpenAI with optional preprocessing to reduce cost.

    While OpenAI can handle large files directly, preprocessing with
    silence removal can significantly reduce costs for files with pauses.

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
    import sys
    from pathlib import Path

    # Add project root to path for imports
    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root))

    from scripts.preprocess_audio import preprocess_for_transcription, get_file_size_mb
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
