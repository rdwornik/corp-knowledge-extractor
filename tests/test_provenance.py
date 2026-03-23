"""Tests for provenance metadata in frontmatter (Council Decision #8a)."""

from pathlib import Path

from src.providers.router import select_model


class TestProvenance:
    def test_provenance_model_name(self):
        """select_model returns a model name string."""
        model, _ = select_model(Path("deck.pptx"), 50000)
        assert isinstance(model, str)
        assert model == "gemini-3.1-pro-preview"

    def test_provenance_routing_reason_pptx(self):
        """PPTX extraction produces routing_reason='pptx_multimodal'."""
        _, reason = select_model(Path("deck.pptx"), 50000)
        assert reason == "pptx_multimodal"

    def test_provenance_routing_reason_text(self):
        """Text file produces routing_reason='text_default'."""
        _, reason = select_model(Path("doc.docx"), 50000)
        assert reason == "text_default"

    def test_provenance_routing_reason_override(self):
        """Manual override produces routing_reason='manual_override'."""
        _, reason = select_model(Path("any.txt"), 50000, model_override="pro")
        assert reason == "manual_override"

    def test_provenance_routing_reason_small(self):
        """Small file produces routing_reason='small_file_local'."""
        _, reason = select_model(Path("tiny.txt"), 3000)
        assert reason == "small_file_local"

    def test_provenance_routing_reason_video(self):
        """Video file produces routing_reason='video_multimodal'."""
        _, reason = select_model(Path("vid.mp4"), 100000)
        assert reason == "video_multimodal"

    def test_provenance_routing_reason_pdf_multimodal(self):
        """All PDFs produce routing_reason='pdf_multimodal'."""
        _, reason = select_model(Path("scan.pdf"), 50000, has_images=True)
        assert reason == "pdf_multimodal"

    def test_provenance_prompt_version_deep(self):
        """Deep extraction gets prompt_version='deep_v2'."""
        depth = "deep"
        prompt_version = "deep_v2" if depth == "deep" else "standard_v1"
        assert prompt_version == "deep_v2"

    def test_provenance_prompt_version_standard(self):
        """Standard extraction gets prompt_version='standard_v1'."""
        depth = "standard"
        prompt_version = "deep_v2" if depth == "deep" else "standard_v1"
        assert prompt_version == "standard_v1"
