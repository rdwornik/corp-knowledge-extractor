"""Tests for transcript generation and note writing."""

from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from src.transcript import (
    TranscriptResult,
    generate_transcript,
    sanitize_transcript,
    MAX_RETRIES,
)
from src.synthesize import write_transcript_note


@pytest.fixture
def config():
    return {
        "gemini": {
            "model": "gemini-3-flash-preview",
            "api_key_env": "GEMINI_API_KEY",
        }
    }


class TestGenerateTranscript:
    def test_generate_transcript_success(self, tmp_path, config):
        """Mock Gemini response → TranscriptResult with text."""
        video = tmp_path / "talk.mp4"
        video.touch()

        mock_response = MagicMock()
        mock_response.text = "[00:00] Speaker 1: Hello everyone, welcome to the session."

        with (
            patch.dict("os.environ", {"GEMINI_API_KEY": "fake-key"}),
            patch("src.transcript.genai") as mock_genai,
        ):
            mock_client = MagicMock()
            mock_client.models.generate_content.return_value = mock_response
            mock_genai.Client.return_value = mock_client

            result = generate_transcript(video, "gs://fake/uri", config)

        assert result.status == "complete"
        assert result.word_count > 0
        assert "Hello everyone" in result.text

    def test_generate_transcript_retry(self, tmp_path, config):
        """First call fails, second succeeds → status complete."""
        video = tmp_path / "talk.mp4"
        video.touch()

        mock_response = MagicMock()
        mock_response.text = "[00:00] Transcript content here."

        with (
            patch.dict("os.environ", {"GEMINI_API_KEY": "fake-key"}),
            patch("src.transcript.genai") as mock_genai,
            patch("src.transcript.time.sleep"),  # don't wait in tests
        ):
            mock_client = MagicMock()
            mock_client.models.generate_content.side_effect = [
                RuntimeError("Server error"),
                mock_response,
            ]
            mock_genai.Client.return_value = mock_client

            result = generate_transcript(video, "gs://fake/uri", config)

        assert result.status == "complete"
        assert "Transcript content" in result.text

    def test_generate_transcript_all_fail(self, tmp_path, config):
        """3 failures → status failed, empty text."""
        video = tmp_path / "talk.mp4"
        video.touch()

        with (
            patch.dict("os.environ", {"GEMINI_API_KEY": "fake-key"}),
            patch("src.transcript.genai") as mock_genai,
            patch("src.transcript.time.sleep"),
        ):
            mock_client = MagicMock()
            mock_client.models.generate_content.side_effect = RuntimeError("Persistent error")
            mock_genai.Client.return_value = mock_client

            result = generate_transcript(video, "gs://fake/uri", config)

        assert result.status == "failed"
        assert result.text == ""
        assert result.word_count == 0
        assert mock_client.models.generate_content.call_count == MAX_RETRIES


class TestTranscriptNoteWriting:
    def test_transcript_file_written(self, tmp_path):
        """Synthesize writes _transcript.md with frontmatter."""
        tr = TranscriptResult(
            text="[00:00] Welcome to the session.\n[00:30] Let's begin.",
            word_count=10,
            duration_min=45,
            status="complete",
            source_path=str(tmp_path / "talk.mp4"),
        )
        extract_dir = tmp_path / "extract"

        result = write_transcript_note(
            tr, "My Talk Title", "talk.md", extract_dir,
        )

        assert result is not None
        assert result.exists()
        assert result.name == "talk_transcript.md"

        content = result.read_text(encoding="utf-8")
        assert "type: transcript" in content
        assert 'title: "My Talk Title — Full Transcript"' in content
        assert "duration_min: 45" in content
        assert "word_count: 10" in content
        assert "linked_extraction: talk.md" in content
        assert "[00:00] Welcome to the session." in content

    def test_transcript_not_written_on_failure(self, tmp_path):
        """status=failed → no file created."""
        tr = TranscriptResult(
            text="", word_count=0, duration_min=0,
            status="failed", source_path=str(tmp_path / "talk.mp4"),
        )
        extract_dir = tmp_path / "extract"

        result = write_transcript_note(
            tr, "Title", "talk.md", extract_dir,
        )

        assert result is None
        assert not (extract_dir / "talk_transcript.md").exists()


class TestTranscriptSanitization:
    def test_transcript_sanitization(self):
        """Backticks and HTML tags stripped."""
        raw = "```python\ncode\n``` and <script>alert('xss')</script> and normal text"
        clean = sanitize_transcript(raw)
        assert "`" not in clean
        assert "<script>" not in clean
        assert "</script>" not in clean
        assert "normal text" in clean

    def test_sanitize_angle_brackets(self):
        assert "<b>bold</b>" != sanitize_transcript("<b>bold</b>")
        assert "bold" in sanitize_transcript("<b>bold</b>")

    def test_sanitize_preserves_timestamps(self):
        text = "[00:30] Speaker said something."
        assert sanitize_transcript(text) == text
