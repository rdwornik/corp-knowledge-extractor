"""Tests for file:/// link in note body."""

from jinja2 import Environment, FileSystemLoader


class TestFileLink:
    def _render(self, source_file):
        env = Environment(loader=FileSystemLoader("templates"))
        env.filters["tojson_raw"] = lambda v: str(v)
        tmpl = env.get_template("extract.md.j2")
        return tmpl.render(
            source_file=source_file, content_type="document",
            title="Test", date="2026-01-01", topics=[], people=[],
            products=[], language="en", quality="full", tokens_used=0,
            summary="Summary.", links_line="", slides=[],
            key_points=[], transcript_excerpt="", model="test",
            flagged_facts=[],
        )

    def test_file_link_present(self):
        output = self._render("C:/Users/test/file.pdf")
        assert "file:///" in output
        assert "[Open original]" in output

    def test_file_link_forward_slashes(self):
        output = self._render("C:\\Users\\test\\file.pptx")
        link_line = [l for l in output.split("\n") if "file:///" in l][0]
        assert "\\" not in link_line
        assert "C:/Users/test/file.pptx" in link_line
