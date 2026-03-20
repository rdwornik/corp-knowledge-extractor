"""Tests for cover_slide/cover_frame in _meta.yaml."""

import yaml
from pathlib import Path

from jinja2 import Environment, FileSystemLoader


class TestMetaCover:
    def test_meta_cover_slide(self):
        """Package with slides → cover_slide in yaml."""
        env = Environment(loader=FileSystemLoader("templates"))
        tmpl = env.get_template("meta.yaml.j2")
        content = tmpl.render(
            extracted_at="2026-03-20T12:00:00",
            model="gemini-3-flash-preview",
            pipeline_version="0.4.0",
            prompt_hash="abc123",
            cover_slide="slides/slide_001.png",
            cover_frame=None,
            source_files=[{"path": "test.pptx", "type": "slides", "size_bytes": 1000}],
        )
        meta = yaml.safe_load(content)
        assert meta["cover_slide"] == "slides/slide_001.png"
        assert "cover_frame" not in meta

    def test_meta_cover_frame(self):
        """Package with frames → cover_frame in yaml."""
        env = Environment(loader=FileSystemLoader("templates"))
        tmpl = env.get_template("meta.yaml.j2")
        content = tmpl.render(
            extracted_at="2026-03-20T12:00:00",
            model="gemini-3-flash-preview",
            pipeline_version="0.4.0",
            prompt_hash="abc123",
            cover_slide=None,
            cover_frame="frames/slide_001.png",
            source_files=[{"path": "test.mp4", "type": "video", "size_bytes": 5000}],
        )
        meta = yaml.safe_load(content)
        assert meta["cover_frame"] == "frames/slide_001.png"
        assert "cover_slide" not in meta

    def test_meta_no_cover(self):
        """No slides or frames → no cover field."""
        env = Environment(loader=FileSystemLoader("templates"))
        tmpl = env.get_template("meta.yaml.j2")
        content = tmpl.render(
            extracted_at="2026-03-20T12:00:00",
            model="gemini-3-flash-preview",
            pipeline_version="0.4.0",
            prompt_hash="abc123",
            cover_slide=None,
            cover_frame=None,
            source_files=[{"path": "test.pdf", "type": "document", "size_bytes": 2000}],
        )
        meta = yaml.safe_load(content)
        assert "cover_slide" not in meta
        assert "cover_frame" not in meta
