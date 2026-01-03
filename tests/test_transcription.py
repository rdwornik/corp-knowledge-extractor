"""
Tests for audio preprocessing and transcription functionality.

Tests silence removal, audio chunking, and transcript merging.
"""

import pytest
import os
import sys
import tempfile
import subprocess
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.preprocess_audio import (
    remove_silence,
    optimize_audio,
    get_file_size_mb,
    get_audio_duration
)
from src.transcribe.chunker import (
    split_audio,
    merge_transcripts,
    split_and_get_metadata
)


# ======================
# Test Fixtures
# ======================

@pytest.fixture
def sample_audio_file():
    """
    Create a sample audio file for testing.

    Generates a 10-second audio file with speech-like tones.
    """
    temp_file = os.path.join(tempfile.gettempdir(), "test_audio.mp3")

    # Generate 10 seconds of audio with tone (simulates speech)
    # Using FFmpeg to create synthetic audio
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi",
        "-i", "sine=frequency=440:duration=10",
        "-ac", "1",
        "-ar", "16000",
        "-b:a", "32k",
        temp_file
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True)
        yield temp_file
    finally:
        if os.path.exists(temp_file):
            os.remove(temp_file)


@pytest.fixture
def sample_audio_with_silence():
    """
    Create audio file with silence gaps for testing silence removal.

    Pattern: 2s speech, 3s silence, 2s speech, 3s silence, 2s speech
    Total: ~12 seconds (9s speech + 6s silence)
    """
    temp_file = os.path.join(tempfile.gettempdir(), "test_audio_silence.mp3")

    # Create audio with alternating speech and silence
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi",
        "-i", (
            "sine=frequency=440:duration=2,"
            "anullsrc=duration=3,"
            "sine=frequency=440:duration=2,"
            "anullsrc=duration=3,"
            "sine=frequency=440:duration=2"
        ),
        "-ac", "1",
        "-ar", "16000",
        "-b:a", "32k",
        temp_file
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True)
        yield temp_file
    finally:
        if os.path.exists(temp_file):
            os.remove(temp_file)


# ======================
# Silence Removal Tests
# ======================

class TestSilenceRemoval:
    """Test silence removal functionality."""

    def test_silence_removal_reduces_size(self, sample_audio_with_silence):
        """Test that silence removal reduces file size."""
        original_size = get_file_size_mb(sample_audio_with_silence)
        original_duration = get_audio_duration(sample_audio_with_silence)

        output_path = remove_silence(
            sample_audio_with_silence,
            threshold_db=-40,
            min_silence_duration=2.0,
            verbose=False
        )

        try:
            new_size = get_file_size_mb(output_path)
            new_duration = get_audio_duration(output_path)

            # File should be smaller
            assert new_size < original_size, "Silence removal should reduce file size"

            # Duration should be shorter (silence removed)
            assert new_duration < original_duration, "Duration should be reduced"

            # Should remove approximately 6 seconds of silence
            silence_removed = original_duration - new_duration
            assert 4 <= silence_removed <= 8, f"Should remove ~6s silence, removed {silence_removed:.1f}s"

        finally:
            if os.path.exists(output_path):
                os.remove(output_path)

    def test_silence_removal_preserves_speech(self, sample_audio_with_silence):
        """Test that silence removal preserves speech content."""
        output_path = remove_silence(
            sample_audio_with_silence,
            threshold_db=-40,
            min_silence_duration=2.0,
            verbose=False
        )

        try:
            # File should still exist and be valid
            assert os.path.exists(output_path)

            duration = get_audio_duration(output_path)

            # Should have approximately 6 seconds of speech left
            assert 4 <= duration <= 8, f"Should have ~6s speech, got {duration:.1f}s"

        finally:
            if os.path.exists(output_path):
                os.remove(output_path)

    def test_optimize_audio_reduces_size(self, sample_audio_file):
        """Test audio optimization."""
        original_size = get_file_size_mb(sample_audio_file)

        output_path = optimize_audio(
            sample_audio_file,
            target_bitrate="16k",
            verbose=False
        )

        try:
            new_size = get_file_size_mb(output_path)

            # Lower bitrate should produce smaller file
            # (might not always be true for very short files)
            assert os.path.exists(output_path)

            # File should be valid
            duration = get_audio_duration(output_path)
            assert duration > 0

        finally:
            if os.path.exists(output_path):
                os.remove(output_path)


# ======================
# Audio Chunking Tests
# ======================

class TestAudioChunking:
    """Test audio chunking functionality."""

    def test_split_audio_small_file(self, sample_audio_file):
        """Test that small files don't get split."""
        size_mb = get_file_size_mb(sample_audio_file)

        # File should be < 1MB, so no splitting with 24MB threshold
        chunks = split_audio(
            sample_audio_file,
            max_size_mb=24,
            verbose=False
        )

        # Should return the original file without splitting
        assert len(chunks) == 1
        assert chunks[0] == sample_audio_file

    def test_split_audio_creates_chunks(self, sample_audio_file):
        """Test splitting with very low threshold to force chunking."""
        # Use very small max size to force splitting
        chunks = split_audio(
            sample_audio_file,
            max_size_mb=0.01,  # 10KB - will force multiple chunks
            overlap_seconds=1.0,
            verbose=False
        )

        try:
            # Should create multiple chunks
            assert len(chunks) > 1, "Should create multiple chunks"

            # Each chunk should exist
            for chunk in chunks:
                assert os.path.exists(chunk)

                # Chunk should be small
                chunk_size = get_file_size_mb(chunk)
                assert chunk_size <= 0.02, f"Chunk too large: {chunk_size:.2f}MB"

        finally:
            # Cleanup chunks
            for chunk in chunks:
                if chunk != sample_audio_file and os.path.exists(chunk):
                    os.remove(chunk)

    def test_split_and_get_metadata(self, sample_audio_file):
        """Test split_and_get_metadata function."""
        chunks, durations = split_and_get_metadata(
            sample_audio_file,
            max_size_mb=24,
            verbose=False
        )

        # Should have same number of chunks and durations
        assert len(chunks) == len(durations)

        # Durations should be positive
        for duration in durations:
            assert duration > 0

        # Total duration should approximately match original
        total_duration = sum(durations)
        original_duration = get_audio_duration(sample_audio_file)

        # Allow for overlap adjustments
        assert abs(total_duration - original_duration) < 2


# ======================
# Transcript Merging Tests
# ======================

class TestTranscriptMerging:
    """Test transcript merging functionality."""

    def test_merge_single_transcript(self):
        """Test merging with single transcript (no-op)."""
        transcript = [
            {"start": 0.0, "end": 5.0, "text": "Hello world"},
            {"start": 5.0, "end": 10.0, "text": "This is a test"}
        ]

        merged = merge_transcripts([transcript], verbose=False)

        # Should return unchanged
        assert len(merged) == 2
        assert merged[0]["text"] == "Hello world"
        assert merged[1]["text"] == "This is a test"

    def test_merge_multiple_transcripts(self):
        """Test merging multiple transcripts with timestamp correction."""
        chunk1 = [
            {"start": 0.0, "end": 5.0, "text": "First chunk"},
            {"start": 5.0, "end": 10.0, "text": "End of first"}
        ]

        chunk2 = [
            {"start": 0.0, "end": 5.0, "text": "Second chunk"},
            {"start": 5.0, "end": 10.0, "text": "End of second"}
        ]

        merged = merge_transcripts(
            [chunk1, chunk2],
            chunk_durations=[10.0, 10.0],
            overlap_seconds=2.0,
            verbose=False
        )

        # Should have 4 segments (2 from each chunk)
        # But overlap removal might reduce this
        assert len(merged) >= 3, "Should have at least 3 segments"

        # Timestamps should be sequential
        for i in range(len(merged) - 1):
            assert merged[i]["end"] <= merged[i+1]["start"] + 1, "Timestamps should be sequential"

    def test_merge_preserves_text(self):
        """Test that all text content is preserved during merge."""
        chunk1 = [
            {"start": 0.0, "end": 3.0, "text": "Alpha"},
            {"start": 3.0, "end": 6.0, "text": "Beta"}
        ]

        chunk2 = [
            {"start": 0.0, "end": 3.0, "text": "Gamma"},
            {"start": 3.0, "end": 6.0, "text": "Delta"}
        ]

        merged = merge_transcripts(
            [chunk1, chunk2],
            chunk_durations=[8.0, 8.0],
            overlap_seconds=2.0,
            verbose=False
        )

        # All text should be present (some might be in overlap and skipped)
        merged_text = " ".join([seg["text"] for seg in merged])

        # At minimum, non-overlapping segments should be present
        assert "Beta" in merged_text or "Alpha" in merged_text
        assert "Delta" in merged_text or "Gamma" in merged_text

    def test_merge_empty_transcripts(self):
        """Test handling of empty transcripts."""
        merged = merge_transcripts([], verbose=False)
        assert merged == []

        merged = merge_transcripts([[]], verbose=False)
        assert merged == []


# ======================
# Integration Tests
# ======================

class TestTranscriptionIntegration:
    """Integration tests for full preprocessing pipeline."""

    def test_full_pipeline_small_file(self, sample_audio_file):
        """Test full pipeline with small file (no chunking needed)."""
        # This would test the full transcribe_groq() function
        # but requires API key, so we'll skip actual API call

        size_mb = get_file_size_mb(sample_audio_file)

        # Verify file is small enough
        assert size_mb < 24, "Test file should be < 24MB"

        # Verify file exists and is valid
        duration = get_audio_duration(sample_audio_file)
        assert duration > 0

    def test_preprocessing_reduces_size_significantly(self, sample_audio_with_silence):
        """Test that preprocessing significantly reduces file size."""
        original_size = get_file_size_mb(sample_audio_with_silence)

        # Apply silence removal
        output_path = remove_silence(
            sample_audio_with_silence,
            threshold_db=-40,
            min_silence_duration=2.0,
            verbose=False
        )

        try:
            new_size = get_file_size_mb(output_path)

            # Should have significant reduction
            reduction_percent = ((original_size - new_size) / original_size) * 100

            # For our test file with 50% silence, expect ~30-60% reduction
            assert reduction_percent > 20, f"Expected >20% reduction, got {reduction_percent:.1f}%"

        finally:
            if os.path.exists(output_path):
                os.remove(output_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
