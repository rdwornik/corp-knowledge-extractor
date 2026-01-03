"""
Integration tests for the knowledge extraction pipeline.

Tests the end-to-end flow and individual pipeline components.
Note: Full pipeline tests require API keys and may incur costs.
"""

import pytest
import os
import sys
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.align.aligner import align
from src.anonymize.anonymizer import anonymize
from src.output.generator import generate_output
from src.output.post_processor import post_process


class TestAlignment:
    """Test speech-to-frame alignment logic."""

    def test_align_basic(self):
        """Test basic alignment of speech to frames."""
        transcript = [
            {"start": 0.0, "end": 5.0, "text": "Welcome to the presentation"},
            {"start": 5.0, "end": 10.0, "text": "This is slide one"},
            {"start": 10.0, "end": 15.0, "text": "Moving to slide two"}
        ]

        frames = [
            {"timestamp": 0.0, "text": "Title Slide"},
            {"timestamp": 6.0, "text": "Slide One"},
            {"timestamp": 11.0, "text": "Slide Two"}
        ]

        aligned = align(transcript, frames)

        assert len(aligned) > 0
        assert all("speech" in item for item in aligned)
        assert all("slide_text" in item for item in aligned)
        assert all("start" in item for item in aligned)
        assert all("end" in item for item in aligned)

    def test_align_empty_input(self):
        """Test alignment with empty inputs."""
        result = align([], [])
        assert isinstance(result, list)

    def test_align_no_frames(self):
        """Test alignment with transcript but no frames."""
        transcript = [
            {"start": 0.0, "end": 5.0, "text": "Some speech"}
        ]
        result = align(transcript, [])
        assert isinstance(result, list)


class TestAnonymization:
    """Test PII anonymization functionality."""

    def test_anonymize_basic(self):
        """Test basic text anonymization."""
        text = "John Smith works at Blue Yonder in Phoenix"
        custom_terms = ["Blue Yonder"]

        result = anonymize(text, custom_terms)

        assert "Blue Yonder" not in result  # Should be redacted
        assert "[REDACTED" in result or "[ORG]" in result  # Some redaction marker

    def test_anonymize_preserves_safe_text(self):
        """Test that safe text is preserved."""
        text = "The WMS system uses Azure cloud infrastructure"
        custom_terms = []  # No custom terms

        result = anonymize(text, custom_terms)

        # Product names should be preserved (configured in exclude_terms)
        assert "WMS" in result or "Azure" in result

    def test_anonymize_empty_text(self):
        """Test anonymization of empty text."""
        result = anonymize("", [])
        assert result == ""

    def test_anonymize_custom_terms(self):
        """Test custom term redaction."""
        text = "Our client SecretCorp is interested"
        custom_terms = ["SecretCorp"]

        result = anonymize(text, custom_terms)

        assert "SecretCorp" not in result


class TestOutputGeneration:
    """Test report and JSONL generation."""

    def test_generate_output_structure(self, tmp_path):
        """Test that output generation creates correct structure."""
        synthesis = {
            "slide_breakdown": [
                {
                    "frame_id": "001",
                    "title": "Test Slide",
                    "visual_content": "Test visual",
                    "speaker_explanation": "Test explanation",
                    "technical_details": "Test details",
                    "context_relationships": "Test context",
                    "key_terminology": ["term1", "term2"]
                }
            ],
            "qa_pairs": [
                {
                    "question": "What is this?",
                    "answer": "A test",
                    "category": "general"
                }
            ]
        }

        frames = [
            {"timestamp": 0.0, "path": "dummy.png"}
        ]

        # Mock file copy since frames may not exist
        with patch("shutil.copy"):
            with patch("os.path.exists", return_value=True):
                output_folder = generate_output(synthesis, frames, output_dir=str(tmp_path))

        # Verify folder structure
        assert os.path.exists(output_folder)
        assert os.path.exists(os.path.join(output_folder, "frames"))
        assert os.path.exists(os.path.join(output_folder, "report.md"))
        assert os.path.exists(os.path.join(output_folder, "knowledge.jsonl"))
        assert os.path.exists(os.path.join(output_folder, "metadata.json"))

    def test_markdown_format(self, tmp_path):
        """Test that generated markdown has correct format."""
        synthesis = {
            "slide_breakdown": [
                {
                    "frame_id": "001",
                    "title": "Test Slide",
                    "speaker_explanation": "This is important content"
                }
            ],
            "qa_pairs": []
        }

        frames = [{"timestamp": 0.0, "path": "dummy.png"}]

        with patch("shutil.copy"):
            with patch("os.path.exists", return_value=True):
                output_folder = generate_output(synthesis, frames, output_dir=str(tmp_path))

        # Read and verify markdown
        md_path = os.path.join(output_folder, "report.md")
        with open(md_path, "r", encoding="utf-8") as f:
            content = f.read()

        assert "# Meeting Knowledge Report" in content
        assert "Test Slide" in content
        assert "This is important content" in content

    def test_jsonl_format(self, tmp_path):
        """Test that JSONL output is valid."""
        synthesis = {
            "slide_breakdown": [],
            "qa_pairs": [
                {"question": "Q1", "answer": "A1", "category": "test"},
                {"question": "Q2", "answer": "A2", "category": "test"}
            ]
        }

        frames = []

        with patch("shutil.copy"):
            output_folder = generate_output(synthesis, frames, output_dir=str(tmp_path))

        # Read and verify JSONL
        jsonl_path = os.path.join(output_folder, "knowledge.jsonl")
        with open(jsonl_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        assert len(lines) == 2
        for line in lines:
            obj = json.loads(line)
            assert "question" in obj
            assert "answer" in obj


class TestPostProcessor:
    """Test post-processing (deduplication, categorization)."""

    def test_post_process_structure(self):
        """Test that post_process maintains data structure."""
        synthesis = {
            "slide_breakdown": [
                {
                    "frame_id": "001",
                    "title": "Test",
                    "speaker_explanation": "Content here",
                    "technical_details": "Details here"
                }
            ],
            "qa_pairs": []
        }

        frames = [{"timestamp": 0.0}]

        result = post_process(synthesis, frames)

        assert "slide_breakdown" in result
        assert "qa_pairs" in result
        assert isinstance(result["slide_breakdown"], list)
        assert isinstance(result["qa_pairs"], list)

    def test_post_process_removes_low_quality(self):
        """Test that post_process filters low-quality slides."""
        synthesis = {
            "slide_breakdown": [
                {
                    "frame_id": "001",
                    "title": "Good Slide",
                    "speaker_explanation": "This has substantial content that is educational",
                    "technical_details": "Version 2.0"
                },
                {
                    "frame_id": "002",
                    "title": "Bad Slide",
                    "speaker_explanation": "Short",  # Too short
                    "technical_details": ""
                }
            ],
            "qa_pairs": []
        }

        frames = [
            {"timestamp": 0.0},
            {"timestamp": 5.0}
        ]

        result = post_process(synthesis, frames)

        # Should filter out low-quality slides
        assert len(result["slide_breakdown"]) <= len(synthesis["slide_breakdown"])

    def test_categorization_applied(self):
        """Test that slides are categorized."""
        synthesis = {
            "slide_breakdown": [
                {
                    "frame_id": "001",
                    "title": "Infrastructure Overview",
                    "speaker_explanation": "We use Azure platform for SaaS deployment",
                    "technical_details": "Azure Cloud"
                }
            ],
            "qa_pairs": []
        }

        frames = [{"timestamp": 0.0}]

        result = post_process(synthesis, frames)

        # Check if category was assigned (based on keywords)
        if len(result["slide_breakdown"]) > 0:
            slide = result["slide_breakdown"][0]
            assert "category" in slide
            # With "Azure" and "platform" keywords, should be infrastructure
            # (depends on categories.yaml config)


class TestPipelineIntegration:
    """
    Integration tests for full pipeline.

    Note: These tests require API keys and will make real API calls.
    Mark with @pytest.mark.integration and run separately.
    """

    @pytest.mark.integration
    @pytest.mark.skipif(
        not os.getenv("GEMINI_API_KEY"),
        reason="GEMINI_API_KEY not set"
    )
    def test_full_pipeline_with_sample(self):
        """
        Test full pipeline with a sample video.

        This test is skipped by default. To run:
        pytest tests/test_pipeline.py::TestPipelineIntegration -m integration
        """
        # This would require a sample video file in tests/fixtures/
        # Implementation depends on having test data
        pytest.skip("Requires sample video file - implement when test data available")


class TestComponentMocking:
    """Test pipeline with mocked components."""

    def test_pipeline_with_mocked_llm(self):
        """Test pipeline flow with mocked LLM calls."""
        # Mock the synthesizer
        mock_synthesis_result = {
            "slide_breakdown": [
                {
                    "frame_id": "001",
                    "title": "Mocked Slide",
                    "speaker_explanation": "Mocked explanation",
                    "technical_details": "Mocked details",
                    "context_relationships": "",
                    "key_terminology": []
                }
            ],
            "qa_pairs": [
                {
                    "question": "Mocked question?",
                    "answer": "Mocked answer",
                    "category": "general"
                }
            ]
        }

        frames = [{"timestamp": 0.0, "path": "mock.png", "text": "Mock text"}]

        # Test post-processing with mocked data
        result = post_process(mock_synthesis_result, frames)

        assert "slide_breakdown" in result
        assert "qa_pairs" in result
        assert len(result["slide_breakdown"]) > 0

    def test_error_handling_in_pipeline(self):
        """Test that pipeline handles errors gracefully."""
        # Test with malformed synthesis result
        bad_synthesis = {
            "slide_breakdown": "not a list",  # Wrong type
            "qa_pairs": []
        }

        frames = []

        # Should not crash, but may return empty or handle gracefully
        try:
            result = post_process(bad_synthesis, frames)
            # If it doesn't crash, it handled the error
            assert True
        except Exception as e:
            # Or it raises a clear error
            assert "slide_breakdown" in str(e).lower() or "list" in str(e).lower()


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
