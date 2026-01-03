"""
Audio chunking and transcript merging for large files.

Splits audio files into smaller chunks for API processing,
then merges transcripts with corrected timestamps.
"""

import os
import subprocess
import tempfile
from pathlib import Path
from typing import List, Dict
from pydub import AudioSegment
from pydub.silence import detect_nonsilent


def get_audio_duration(audio_path: str) -> float:
    """Get duration of audio file in seconds using FFprobe."""
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


def find_silence_boundaries(
    audio_path: str,
    min_silence_len: int = 1000,
    silence_thresh: int = -40
) -> List[int]:
    """
    Find silence boundaries in audio for intelligent chunk splitting.

    Args:
        audio_path: Path to audio file
        min_silence_len: Minimum silence length in ms (default: 1000ms)
        silence_thresh: Silence threshold in dB (default: -40dB)

    Returns:
        List of timestamps (in ms) where silence occurs
    """
    audio = AudioSegment.from_file(audio_path)

    # Detect non-silent chunks
    nonsilent_ranges = detect_nonsilent(
        audio,
        min_silence_len=min_silence_len,
        silence_thresh=silence_thresh
    )

    # Extract silence boundaries (end of each non-silent chunk)
    silence_boundaries = []

    for i, (start, end) in enumerate(nonsilent_ranges):
        if i < len(nonsilent_ranges) - 1:
            # Add boundary at the end of this chunk
            silence_boundaries.append(end)

    return silence_boundaries


def split_audio(
    input_path: str,
    max_size_mb: int = 24,
    overlap_seconds: float = 5.0,
    output_dir: str = None,
    verbose: bool = True
) -> List[str]:
    """
    Split audio into chunks smaller than max_size_mb.

    Splits at silence boundaries to avoid cutting words. Adds overlap
    between chunks to preserve context at boundaries.

    Args:
        input_path: Path to input audio file
        max_size_mb: Maximum chunk size in MB (default: 24)
        overlap_seconds: Overlap between chunks in seconds (default: 5.0)
        output_dir: Output directory for chunks (default: temp dir)
        verbose: Print progress information

    Returns:
        List of paths to audio chunks

    Example:
        >>> chunks = split_audio("large_meeting.mp3", max_size_mb=24)
        >>> print(f"Split into {len(chunks)} chunks")
        Split into 3 chunks
    """
    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="audio_chunks_")
    else:
        os.makedirs(output_dir, exist_ok=True)

    file_size_mb = get_file_size_mb(input_path)
    duration_seconds = get_audio_duration(input_path)

    if verbose:
        print(f"  Splitting audio: {file_size_mb:.1f}MB, {duration_seconds/60:.1f} min")

    # Calculate approximate number of chunks needed
    num_chunks = int(file_size_mb / max_size_mb) + 1

    if num_chunks == 1:
        # File is small enough, no splitting needed
        if verbose:
            print(f"  No splitting needed (file < {max_size_mb}MB)")
        return [input_path]

    # Load audio with pydub
    audio = AudioSegment.from_file(input_path)
    total_duration_ms = len(audio)
    overlap_ms = int(overlap_seconds * 1000)

    # Find silence boundaries for intelligent splitting
    silence_boundaries = find_silence_boundaries(input_path)

    if verbose:
        print(f"  Target: {num_chunks} chunks of ~{max_size_mb}MB each")
        print(f"  Found {len(silence_boundaries)} silence boundaries")

    # Calculate target chunk duration
    target_chunk_duration_ms = total_duration_ms / num_chunks

    # Split at silence boundaries closest to target durations
    chunk_paths = []
    current_start = 0

    for chunk_idx in range(num_chunks):
        # Calculate target split point
        target_split = (chunk_idx + 1) * target_chunk_duration_ms

        if chunk_idx == num_chunks - 1:
            # Last chunk: take everything remaining
            chunk_end = total_duration_ms
        else:
            # Find silence boundary closest to target split
            if silence_boundaries:
                chunk_end = min(
                    silence_boundaries,
                    key=lambda x: abs(x - target_split)
                )
                # Remove used boundary
                silence_boundaries = [b for b in silence_boundaries if b != chunk_end]
            else:
                # No silence boundaries, split at target
                chunk_end = int(target_split)

        # Add overlap to start (except first chunk)
        actual_start = max(0, current_start - overlap_ms) if chunk_idx > 0 else 0

        # Extract chunk
        chunk = audio[actual_start:chunk_end]

        # Save chunk
        chunk_filename = f"chunk_{chunk_idx:03d}.mp3"
        chunk_path = os.path.join(output_dir, chunk_filename)

        chunk.export(
            chunk_path,
            format="mp3",
            parameters=["-ac", "1", "-ar", "16000", "-b:a", "32k"]
        )

        chunk_size = get_file_size_mb(chunk_path)
        chunk_duration = len(chunk) / 1000

        if verbose:
            print(f"  Chunk {chunk_idx + 1}/{num_chunks}: {chunk_size:.1f}MB, "
                  f"{chunk_duration/60:.1f} min (offset: {current_start/1000:.1f}s)")

        chunk_paths.append(chunk_path)

        # Move to next chunk start
        current_start = chunk_end

        # Stop if we've reached the end
        if chunk_end >= total_duration_ms:
            break

    if verbose:
        print(f"  ✓ Split into {len(chunk_paths)} chunks")

    return chunk_paths


def merge_transcripts(
    transcripts: List[List[Dict]],
    chunk_durations: List[float] = None,
    overlap_seconds: float = 5.0,
    verbose: bool = True
) -> List[Dict]:
    """
    Merge transcripts from multiple chunks with corrected timestamps.

    Handles overlapping segments and adjusts timestamps to match
    original audio timeline.

    Args:
        transcripts: List of transcript lists from each chunk
        chunk_durations: Duration of each chunk in seconds (for timestamp offset)
        overlap_seconds: Overlap between chunks in seconds
        verbose: Print progress information

    Returns:
        Merged transcript with corrected timestamps

    Example:
        >>> chunk1 = [{"start": 0, "end": 5, "text": "Hello"}]
        >>> chunk2 = [{"start": 0, "end": 5, "text": "world"}]
        >>> merged = merge_transcripts([chunk1, chunk2], chunk_durations=[10, 10])
        >>> # Result: [{"start": 0, "end": 5, "text": "Hello"},
        >>> #          {"start": 10, "end": 15, "text": "world"}]
    """
    if not transcripts:
        return []

    if len(transcripts) == 1:
        # Single transcript, no merging needed
        return transcripts[0]

    if verbose:
        print(f"  Merging {len(transcripts)} transcript chunks...")

    merged = []
    cumulative_offset = 0.0

    for chunk_idx, transcript in enumerate(transcripts):
        if verbose:
            print(f"  Processing chunk {chunk_idx + 1}: {len(transcript)} segments, "
                  f"offset: {cumulative_offset:.1f}s")

        for segment in transcript:
            # Adjust timestamps with cumulative offset
            adjusted_segment = {
                "start": segment["start"] + cumulative_offset,
                "end": segment["end"] + cumulative_offset,
                "text": segment["text"]
            }

            # Skip overlapping segments (first N seconds of each chunk after the first)
            if chunk_idx > 0:
                # This is not the first chunk
                if segment["start"] < overlap_seconds:
                    # Skip this segment (it's in the overlap region)
                    continue

            merged.append(adjusted_segment)

        # Update cumulative offset for next chunk
        # Subtract overlap because chunks overlap
        if chunk_idx < len(transcripts) - 1 and chunk_durations:
            # Use actual chunk duration minus overlap
            cumulative_offset += chunk_durations[chunk_idx] - overlap_seconds
        elif transcript:
            # Use last segment end time as duration estimate
            cumulative_offset = transcript[-1]["end"] + cumulative_offset

    if verbose:
        print(f"  ✓ Merged into {len(merged)} segments (total duration: {merged[-1]['end']/60:.1f} min)")

    return merged


def split_and_get_metadata(
    input_path: str,
    max_size_mb: int = 24,
    overlap_seconds: float = 5.0,
    output_dir: str = None,
    verbose: bool = True
) -> tuple[List[str], List[float]]:
    """
    Split audio and return chunk paths with their durations.

    Convenience function that combines splitting with metadata extraction.

    Args:
        input_path: Path to input audio file
        max_size_mb: Maximum chunk size in MB
        overlap_seconds: Overlap between chunks
        output_dir: Output directory for chunks
        verbose: Print progress information

    Returns:
        Tuple of (chunk_paths, chunk_durations)
    """
    chunk_paths = split_audio(
        input_path,
        max_size_mb=max_size_mb,
        overlap_seconds=overlap_seconds,
        output_dir=output_dir,
        verbose=verbose
    )

    # Get duration of each chunk
    chunk_durations = []
    for chunk_path in chunk_paths:
        duration = get_audio_duration(chunk_path)
        chunk_durations.append(duration)

    return chunk_paths, chunk_durations


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python chunker.py <input_audio> [max_size_mb]")
        print("\nExample:")
        print("  python chunker.py large_meeting.mp3 24")
        sys.exit(1)

    input_file = sys.argv[1]
    max_size = int(sys.argv[2]) if len(sys.argv) > 2 else 24

    if not os.path.exists(input_file):
        print(f"Error: File not found: {input_file}")
        sys.exit(1)

    print(f"\n=== Audio Chunking ===")
    chunk_paths, chunk_durations = split_and_get_metadata(
        input_file,
        max_size_mb=max_size,
        verbose=True
    )

    print(f"\n=== Results ===")
    print(f"Created {len(chunk_paths)} chunks:")
    for i, (path, duration) in enumerate(zip(chunk_paths, chunk_durations)):
        size = get_file_size_mb(path)
        print(f"  {i+1}. {os.path.basename(path)}: {size:.1f}MB, {duration/60:.1f} min")
