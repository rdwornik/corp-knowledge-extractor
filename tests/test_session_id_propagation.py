"""Tests for session_id propagation to individual notes."""

import json
import yaml
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from run import _propagate_session_id


class TestSessionIdPropagation:
    def test_individual_gets_session_id(self, tmp_path):
        """After merge, individual JSON has session_id."""
        json_path = tmp_path / "deck.json"
        json_path.write_text(json.dumps({"title": "Test", "key_facts": []}), encoding="utf-8")

        md_path = tmp_path / "deck.md"
        md_path.write_text("---\ntitle: Test\n---\n\n# Test\n", encoding="utf-8")

        _propagate_session_id(tmp_path, "deck", "cognitive-friday-s4e1")

        # Check JSON
        data = json.loads(json_path.read_text(encoding="utf-8"))
        assert data["session_id"] == "cognitive-friday-s4e1"

        # Check MD frontmatter
        text = md_path.read_text(encoding="utf-8")
        assert "session_id:" in text
        assert "cognitive-friday-s4e1" in text

    def test_standalone_no_session_id(self, tmp_path):
        """File without session merge → no session_id."""
        json_path = tmp_path / "standalone.json"
        json_path.write_text(json.dumps({"title": "Solo"}), encoding="utf-8")

        # Don't call _propagate_session_id
        data = json.loads(json_path.read_text(encoding="utf-8"))
        assert "session_id" not in data

    def test_idempotent(self, tmp_path):
        """Calling twice doesn't duplicate session_id in frontmatter."""
        md_path = tmp_path / "deck.md"
        md_path.write_text("---\ntitle: Test\n---\n\n# Test\n", encoding="utf-8")
        json_path = tmp_path / "deck.json"
        json_path.write_text(json.dumps({"title": "Test"}), encoding="utf-8")

        _propagate_session_id(tmp_path, "deck", "session-1")
        _propagate_session_id(tmp_path, "deck", "session-1")

        text = md_path.read_text(encoding="utf-8")
        assert text.count("session_id:") == 1
