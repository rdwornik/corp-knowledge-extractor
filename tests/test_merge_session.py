"""Tests for session merge logic."""

from pathlib import Path

from src.merge_session import merge_correlated, _dedupe_list


class TestMergeCorrelated:
    def test_produces_session_type(self):
        result = merge_correlated(
            pptx_extraction={"title": "Test", "topics": ["A"], "key_facts": ["Fact 1"]},
            video_extraction={"title": "Test", "topics": ["B"], "slides": []},
            pptx_path=Path("test.pptx"),
            video_path=Path("test.mp4"),
            pptx_hash="abc",
            video_hash="def",
            correlation_confidence=90,
            correlation_method="test",
        )
        assert result["frontmatter"]["type"] == "training_session"
        assert result["frontmatter"]["source_type"] == "correlated_session"
        assert len(result["frontmatter"]["sources"]) == 2

    def test_deduplicates_topics(self):
        result = merge_correlated(
            pptx_extraction={"title": "T", "topics": ["SaaS", "ML"], "key_facts": []},
            video_extraction={"title": "T", "topics": ["ML", "Cloud"], "slides": []},
            pptx_path=Path("t.pptx"),
            video_path=Path("t.mp4"),
            pptx_hash="a",
            video_hash="b",
            correlation_confidence=90,
            correlation_method="test",
        )
        topics = result["frontmatter"]["topics"]
        assert len(topics) == 3  # SaaS, ML, Cloud — no dupes

    def test_session_hash_deterministic(self):
        kwargs = dict(
            pptx_extraction={"title": "T"},
            video_extraction={"title": "T"},
            pptx_path=Path("t.pptx"),
            video_path=Path("t.mp4"),
            pptx_hash="abc",
            video_hash="def",
            correlation_confidence=90,
            correlation_method="test",
        )
        r1 = merge_correlated(**kwargs)
        r2 = merge_correlated(**kwargs)
        assert r1["frontmatter"]["session_hash"] == r2["frontmatter"]["session_hash"]

    def test_session_hash_independent_of_order(self):
        """Hash should be the same regardless of which file is pptx vs video."""
        r1 = merge_correlated(
            pptx_extraction={"title": "T"},
            video_extraction={"title": "T"},
            pptx_path=Path("t.pptx"),
            video_path=Path("t.mp4"),
            pptx_hash="abc",
            video_hash="def",
            correlation_confidence=90,
            correlation_method="test",
        )
        r2 = merge_correlated(
            pptx_extraction={"title": "T"},
            video_extraction={"title": "T"},
            pptx_path=Path("t.pptx"),
            video_path=Path("t.mp4"),
            pptx_hash="def",
            video_hash="abc",
            correlation_confidence=90,
            correlation_method="test",
        )
        assert r1["frontmatter"]["session_hash"] == r2["frontmatter"]["session_hash"]

    def test_key_facts_tagged_with_modality(self):
        result = merge_correlated(
            pptx_extraction={"title": "T", "key_facts": ["Fact A", {"fact": "Fact B", "page": 3}]},
            video_extraction={"title": "T", "slides": []},
            pptx_path=Path("t.pptx"),
            video_path=Path("t.mp4"),
            pptx_hash="a",
            video_hash="b",
            correlation_confidence=90,
            correlation_method="test",
        )
        facts = result["frontmatter"]["key_facts"]
        assert len(facts) == 2
        assert facts[0]["source_modality"] == "pptx"
        assert facts[1]["source_modality"] == "pptx"

    def test_overlay_preserved(self):
        result = merge_correlated(
            pptx_extraction={
                "title": "T",
                "training_overlay": {"attendees": ["Alice"], "decisions_made": ["Go live"]},
            },
            video_extraction={"title": "T", "slides": []},
            pptx_path=Path("t.pptx"),
            video_path=Path("t.mp4"),
            pptx_hash="a",
            video_hash="b",
            correlation_confidence=90,
            correlation_method="test",
        )
        assert "training_overlay" in result["frontmatter"]
        assert result["frontmatter"]["training_overlay"]["attendees"] == ["Alice"]

    def test_prefers_pptx_title(self):
        result = merge_correlated(
            pptx_extraction={"title": "Clean Deck Title"},
            video_extraction={"title": "Recording 2026-03-15 14.30", "slides": []},
            pptx_path=Path("t.pptx"),
            video_path=Path("t.mp4"),
            pptx_hash="a",
            video_hash="b",
            correlation_confidence=90,
            correlation_method="test",
        )
        assert result["frontmatter"]["title"] == "Clean Deck Title"

    def test_markdown_has_provenance(self):
        result = merge_correlated(
            pptx_extraction={"title": "T"},
            video_extraction={"title": "T", "slides": []},
            pptx_path=Path("deck.pptx"),
            video_path=Path("recording.mp4"),
            pptx_hash="a",
            video_hash="b",
            correlation_confidence=90,
            correlation_method="test",
        )
        assert "deck.pptx" in result["markdown"]
        assert "recording.mp4" in result["markdown"]
        assert "Source Provenance" in result["markdown"]


class TestDedupeList:
    def test_basic(self):
        assert _dedupe_list(["A", "b", "a", "C"]) == ["A", "b", "C"]

    def test_preserves_order(self):
        assert _dedupe_list(["Z", "A", "Z", "B"]) == ["Z", "A", "B"]

    def test_empty(self):
        assert _dedupe_list([]) == []
