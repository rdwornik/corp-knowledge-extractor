"""Tests for policy-based auto model routing (Council Decision #8a)."""

from pathlib import Path

from src.providers.router import select_model


class TestSelectModel:
    def test_select_model_pptx(self):
        model, reason = select_model(Path("deck.pptx"), 50000)
        assert model == "gemini-3.1-pro-preview"
        assert reason == "pptx_multimodal"

    def test_select_model_mp4(self):
        model, reason = select_model(Path("video.mp4"), 100000)
        assert model == "gemini-3.1-pro-preview"
        assert reason == "video_multimodal"

    def test_select_model_mkv(self):
        model, reason = select_model(Path("video.mkv"), 100000)
        assert model == "gemini-3.1-pro-preview"
        assert reason == "video_multimodal"

    def test_select_model_pdf_text(self):
        model, reason = select_model(Path("report.pdf"), 50000, has_images=False)
        assert model == "gemini-3-flash-preview"
        assert reason == "text_default"

    def test_select_model_pdf_images(self):
        model, reason = select_model(Path("report.pdf"), 50000, has_images=True)
        assert model == "gemini-3.1-pro-preview"
        assert reason == "pdf_with_images"

    def test_select_model_docx(self):
        model, reason = select_model(Path("document.docx"), 50000)
        assert model == "gemini-3-flash-preview"
        assert reason == "text_default"

    def test_select_model_xlsx(self):
        model, reason = select_model(Path("data.xlsx"), 50000)
        assert model == "gemini-3-flash-preview"
        assert reason == "text_default"

    def test_select_model_tiny(self):
        model, reason = select_model(Path("small.txt"), 3000)
        assert model == "free"
        assert reason == "small_file_local"

    def test_select_model_override_pro(self):
        model, reason = select_model(Path("any.txt"), 50000, model_override="pro")
        assert model == "gemini-3.1-pro-preview"
        assert reason == "manual_override"

    def test_select_model_override_flash(self):
        model, reason = select_model(Path("any.txt"), 50000, model_override="flash")
        assert model == "gemini-3-flash-preview"
        assert reason == "manual_override"

    def test_select_model_override_full_name(self):
        model, reason = select_model(
            Path("any.txt"), 50000, model_override="gemini-3.1-pro-preview"
        )
        assert model == "gemini-3.1-pro-preview"
        assert reason == "manual_override"

    def test_select_model_wav(self):
        model, reason = select_model(Path("audio.wav"), 100000)
        assert model == "gemini-3.1-pro-preview"
        assert reason == "video_multimodal"

    def test_select_model_mov(self):
        model, reason = select_model(Path("clip.mov"), 100000)
        assert model == "gemini-3.1-pro-preview"
        assert reason == "video_multimodal"
