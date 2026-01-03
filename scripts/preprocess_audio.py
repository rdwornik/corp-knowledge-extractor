"""
Audio preprocessing utilities for large transcription files.

Provides silence removal and audio optimization to reduce file sizes
while preserving speech content for transcription.
"""

import os
import subprocess
import tempfile
from pathlib import Path
from typing import Optional


def get_audio_duration(audio_path: str) -> float:
    """
    Get duration of audio file in seconds using FFprobe.

    Args:
        audio_path: Path to audio file

    Returns:
        Duration in seconds
    """
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        audio_path
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return float(result.stdout.strip())


def get_file_size_mb(file_path: str) -> float:
    """Get file size in megabytes."""
    return os.path.getsize(file_path) / (1024 * 1024)


def remove_silence(
    input_path: str,
    output_path: Optional[str] = None,
    threshold_db: int = -40,
    min_silence_duration: float = 2.0,
    verbose: bool = True
) -> str:
    """
    Remove silence from audio using FFmpeg silenceremove filter.

    This significantly reduces file size for recordings with pauses,
    while preserving all speech content with natural timing.

    Args:
        input_path: Path to input audio file
        output_path: Path for output (default: temp file)
        threshold_db: Silence threshold in dB (default: -40)
            - Higher values (e.g., -30) = more aggressive removal
            - Lower values (e.g., -50) = keep more quiet parts
        min_silence_duration: Minimum silence duration to remove in seconds
        verbose: Print progress information

    Returns:
        Path to output file with silence removed

    Example:
        >>> # Remove silence from 73MB file
        >>> output = remove_silence("meeting.mp3", threshold_db=-40)
        >>> # Result: 45MB file with same content
    """
    if output_path is None:
        output_path = os.path.join(
            tempfile.gettempdir(),
            f"silence_removed_{os.path.basename(input_path)}"
        )

    original_size = get_file_size_mb(input_path)
    original_duration = get_audio_duration(input_path)

    if verbose:
        print(f"  Original audio: {original_size:.1f}MB, {original_duration/60:.1f} min")
        print(f"  Removing silence (threshold: {threshold_db}dB, min duration: {min_silence_duration}s)...")

    # FFmpeg silenceremove filter
    # Format: silenceremove=start_periods=1:start_duration=1:start_threshold=-40dB:
    #         detection=peak:stop_periods=-1:stop_duration=2:stop_threshold=-40dB
    cmd = [
        "ffmpeg", "-y", "-i", input_path,
        "-af", (
            f"silenceremove="
            f"start_periods=1:"
            f"start_duration=0.5:"
            f"start_threshold={threshold_db}dB:"
            f"detection=peak:"
            f"stop_periods=-1:"
            f"stop_duration={min_silence_duration}:"
            f"stop_threshold={threshold_db}dB"
        ),
        "-ac", "1",  # Mono
        "-ar", "16000",  # 16kHz sample rate (Whisper optimized)
        "-b:a", "32k",  # 32kbps bitrate
        output_path
    ]

    subprocess.run(cmd, check=True, capture_output=True)

    new_size = get_file_size_mb(output_path)
    new_duration = get_audio_duration(output_path)

    reduction_percent = ((original_size - new_size) / original_size) * 100
    silence_removed = original_duration - new_duration

    if verbose:
        print(f"  ✓ After silence removal: {new_size:.1f}MB ({reduction_percent:.1f}% reduction)")
        print(f"  ✓ Duration: {new_duration/60:.1f} min ({silence_removed:.0f}s silence removed)")

    return output_path


def optimize_audio(
    input_path: str,
    output_path: Optional[str] = None,
    target_bitrate: str = "32k",
    sample_rate: int = 16000,
    channels: int = 1,
    verbose: bool = True
) -> str:
    """
    Optimize audio for transcription (mono, low bitrate, Whisper-optimized sample rate).

    Args:
        input_path: Path to input audio/video file
        output_path: Path for output (default: temp file)
        target_bitrate: Audio bitrate (default: 32k for speech)
        sample_rate: Sample rate in Hz (default: 16000 for Whisper)
        channels: Number of channels (default: 1 for mono)
        verbose: Print progress information

    Returns:
        Path to optimized audio file
    """
    if output_path is None:
        output_path = os.path.join(
            tempfile.gettempdir(),
            f"optimized_{Path(input_path).stem}.mp3"
        )

    original_size = get_file_size_mb(input_path)

    if verbose:
        print(f"  Optimizing audio: {original_size:.1f}MB")
        print(f"  Target: {channels} channel(s), {sample_rate}Hz, {target_bitrate}")

    cmd = [
        "ffmpeg", "-y", "-i", input_path,
        "-vn",  # No video
        "-ac", str(channels),
        "-ar", str(sample_rate),
        "-b:a", target_bitrate,
        output_path
    ]

    subprocess.run(cmd, check=True, capture_output=True)

    new_size = get_file_size_mb(output_path)
    reduction_percent = ((original_size - new_size) / original_size) * 100

    if verbose:
        print(f"  ✓ Optimized: {new_size:.1f}MB ({reduction_percent:.1f}% reduction)")

    return output_path


def preprocess_for_transcription(
    input_path: str,
    remove_silence_enabled: bool = True,
    threshold_db: int = -40,
    min_silence_duration: float = 2.0,
    verbose: bool = True
) -> tuple[str, dict]:
    """
    Complete preprocessing pipeline for transcription.

    Combines optimization and silence removal to minimize file size
    while preserving all speech content.

    Args:
        input_path: Path to input audio/video file
        remove_silence_enabled: Enable silence removal (default: True)
        threshold_db: Silence threshold
        min_silence_duration: Minimum silence duration to remove
        verbose: Print progress information

    Returns:
        Tuple of (output_path, stats_dict)

    Example:
        >>> output, stats = preprocess_for_transcription("meeting.mp4")
        >>> print(f"Reduced from {stats['original_size_mb']:.1f}MB to {stats['final_size_mb']:.1f}MB")
    """
    stats = {
        "original_size_mb": get_file_size_mb(input_path),
        "original_path": input_path,
        "steps": []
    }

    if verbose:
        print(f"\n=== Audio Preprocessing ===")
        print(f"Input: {input_path}")
        print(f"Size: {stats['original_size_mb']:.1f}MB\n")

    # Step 1: Extract and optimize audio
    temp_optimized = os.path.join(
        tempfile.gettempdir(),
        f"optimized_{Path(input_path).stem}.mp3"
    )

    optimized_path = optimize_audio(input_path, temp_optimized, verbose=verbose)
    optimized_size = get_file_size_mb(optimized_path)

    stats["steps"].append({
        "name": "optimization",
        "size_mb": optimized_size
    })

    current_path = optimized_path

    # Step 2: Remove silence if enabled
    if remove_silence_enabled:
        temp_silence_removed = os.path.join(
            tempfile.gettempdir(),
            f"silence_removed_{Path(input_path).stem}.mp3"
        )

        silence_removed_path = remove_silence(
            current_path,
            temp_silence_removed,
            threshold_db=threshold_db,
            min_silence_duration=min_silence_duration,
            verbose=verbose
        )

        silence_removed_size = get_file_size_mb(silence_removed_path)

        stats["steps"].append({
            "name": "silence_removal",
            "size_mb": silence_removed_size
        })

        current_path = silence_removed_path

    stats["final_size_mb"] = get_file_size_mb(current_path)
    stats["total_reduction_percent"] = (
        (stats["original_size_mb"] - stats["final_size_mb"]) / stats["original_size_mb"]
    ) * 100

    if verbose:
        print(f"\n=== Preprocessing Complete ===")
        print(f"Original: {stats['original_size_mb']:.1f}MB")
        print(f"Final: {stats['final_size_mb']:.1f}MB")
        print(f"Reduction: {stats['total_reduction_percent']:.1f}%\n")

    return current_path, stats


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python preprocess_audio.py <input_file> [--no-silence-removal]")
        print("\nExample:")
        print("  python preprocess_audio.py meeting.mp4")
        print("  python preprocess_audio.py meeting.mp3 --no-silence-removal")
        sys.exit(1)

    input_file = sys.argv[1]
    remove_silence_flag = "--no-silence-removal" not in sys.argv

    if not os.path.exists(input_file):
        print(f"Error: File not found: {input_file}")
        sys.exit(1)

    output_file, stats = preprocess_for_transcription(
        input_file,
        remove_silence_enabled=remove_silence_flag
    )

    print(f"Output saved to: {output_file}")
    print(f"\nStatistics:")
    print(f"  Original size: {stats['original_size_mb']:.1f}MB")
    print(f"  Final size: {stats['final_size_mb']:.1f}MB")
    print(f"  Total reduction: {stats['total_reduction_percent']:.1f}%")
