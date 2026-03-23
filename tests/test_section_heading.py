"""Tests for section heading style based on source file type."""

import json
from pathlib import Path
from jinja2 import Environment, FileSystemLoader


def _get_env():
    templates_dir = Path(__file__).parent.parent / "templates"
    env = Environment(
        loader=FileSystemLoader(str(templates_dir)),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    env.filters["tojson_raw"] = lambda v: json.dumps(v, ensure_ascii=False)
    return env


def _render(source_file: str) -> str:
    """Render extract.md.j2 with a single slide and return the output."""
    env = _get_env()
    tmpl = env.get_template("extract.md.j2")

    class FakeSlide:
        slide_number = 1
        slide_title = "Introduction"
        timestamp_approx = "00:00"
        speaker_insight = "Speaker talked about intro."
        so_what = ""
        critical_notes = ""
        key_facts = []

    return tmpl.render(
        source_file=source_file,
        content_type="presentation",
        title="Test Title",
        date="2026-03-23",
        summary="Test summary.",
        key_points=[],
        topics=["Testing"],
        people=[],
        products=[],
        slides=[FakeSlide()],
        language="en",
        quality="full",
        duration_min=None,
        transcript_excerpt="",
        model="gemini-3-flash-preview",
        tokens_used=100,
        links_line="",
        source_tool="knowledge-extractor",
    )


def test_section_heading_pptx():
    output = _render("C:/Users/test/docs/presentation.pptx")
    assert "## Slide 1: Introduction" in output
    assert "## Section" not in output


def test_section_heading_docx():
    output = _render("C:/Users/test/docs/report.docx")
    assert "## Section 1: Introduction" in output
    assert "## Slide" not in output


def test_section_heading_pdf():
    output = _render("C:/Users/test/docs/document.pdf")
    assert "## Section 1: Introduction" in output
    assert "## Slide" not in output


def test_section_heading_mp4():
    output = _render("C:/Users/test/videos/recording.mp4")
    assert "## Slide 1: Introduction" in output
    assert "## Section" not in output


def test_section_heading_image_alt_text_pptx():
    output = _render("C:/Users/test/docs/presentation.pptx")
    assert "![Slide 1]" in output


def test_section_heading_image_alt_text_pdf():
    output = _render("C:/Users/test/docs/document.pdf")
    assert "![Section 1]" in output
