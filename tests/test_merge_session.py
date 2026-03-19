"""Tests for session merge logic."""

from pathlib import Path

from src.merge_session import merge_correlated, _dedupe_list, deduplicate_facts, merge_training_overlays


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


class TestDeduplicateFacts:
    """Conservative fact deduplication between PPTX and MP4 sources."""

    def test_merge_dedup_exact_match(self):
        """Same fact in both → kept once, modality='both'."""
        pptx = [{"fact": "1706 customers onboarded", "source_modality": "pptx"}]
        mp4 = [{"fact": "1706 customers onboarded across regions", "source_modality": "mp4"}]

        result = deduplicate_facts(pptx, mp4)

        both_facts = [f for f in result if f["source_modality"] == "both"]
        assert len(both_facts) == 1
        # PPTX wording kept
        assert both_facts[0]["fact"] == "1706 customers onboarded"

    def test_merge_dedup_supplementary(self):
        """MP4 fact not in PPTX → added with modality='mp4'."""
        pptx = [{"fact": "1706 customers onboarded", "source_modality": "pptx"}]
        mp4 = [{"fact": "Speaker mentioned 42 integrations available", "source_modality": "mp4"}]

        result = deduplicate_facts(pptx, mp4)

        assert len(result) == 2
        mp4_facts = [f for f in result if f["source_modality"] == "mp4"]
        assert len(mp4_facts) == 1
        assert "42 integrations" in mp4_facts[0]["fact"]

    def test_merge_dedup_conflict(self):
        """Same entity different number → both kept, conflict flagged."""
        pptx = [{"fact": "Revenue was $2M in 2025", "source_modality": "pptx"}]
        mp4 = [{"fact": "Revenue reached $3M in 2025", "source_modality": "mp4"}]

        result = deduplicate_facts(pptx, mp4)

        conflicted = [f for f in result if f.get("conflict_detected")]
        assert len(conflicted) >= 1

    def test_merge_dedup_number_matching(self):
        """'1,706 customers' vs '1706 total customers' → matched as same."""
        pptx = [{"fact": "Serving 1,706 customers globally", "source_modality": "pptx"}]
        mp4 = [{"fact": "1706 total customers served", "source_modality": "mp4"}]

        result = deduplicate_facts(pptx, mp4)

        both_facts = [f for f in result if f["source_modality"] == "both"]
        assert len(both_facts) == 1

    def test_merge_preserves_pptx_canonical(self):
        """When matched, PPTX wording is kept (canonical from deck)."""
        pptx = [{"fact": "Platform handles 1706 transactions per second", "source_modality": "pptx"}]
        mp4 = [{"fact": "About 1706 transactions every second on the platform", "source_modality": "mp4"}]

        result = deduplicate_facts(pptx, mp4)

        both_facts = [f for f in result if f["source_modality"] == "both"]
        assert len(both_facts) == 1
        # PPTX version preserved
        assert both_facts[0]["fact"] == "Platform handles 1706 transactions per second"

    def test_no_conflict_same_numbers(self):
        """'30-50% reduction' from both → matched, no conflict."""
        pptx = [{"fact": "30-50% reduction in stockholding", "source_modality": "pptx"}]
        mp4 = [{"fact": "30-50% reduction in stockholding", "source_modality": "mp4"}]

        result = deduplicate_facts(pptx, mp4)

        assert len(result) == 1
        assert result[0]["source_modality"] == "both"
        assert not result[0].get("conflict_detected")

    def test_real_conflict_different_numbers(self):
        """'7-9% reduction' vs '7.5% reduction' → conflict detected."""
        pptx = [{"fact": "7-9% stockholding reduction", "source_modality": "pptx"}]
        mp4 = [{"fact": "7.5% stockholding reduction", "source_modality": "mp4"}]

        result = deduplicate_facts(pptx, mp4)

        conflicted = [f for f in result if f.get("conflict_detected")]
        assert len(conflicted) >= 1

    def test_no_conflict_rephrased(self):
        """'1706 customers' vs '1,706 potential customers' → matched, no conflict."""
        pptx = [{"fact": "1706 customers", "source_modality": "pptx"}]
        mp4 = [{"fact": "1,706 potential customers", "source_modality": "mp4"}]

        result = deduplicate_facts(pptx, mp4)

        both_facts = [f for f in result if f["source_modality"] == "both"]
        assert len(both_facts) == 1
        assert not both_facts[0].get("conflict_detected")


class TestMergeTrainingOverlays:
    def test_merge_overlays_combines_attendees(self):
        """PPTX has 3 attendees, MP4 has 5 with 2 overlap → 6 unique."""
        pptx_overlay = {
            "attendees": ["Alice", "Bob", "Charlie"],
        }
        mp4_overlay = {
            "attendees": ["Bob", "Charlie", "Diana", "Eve", "Frank"],
        }

        result = merge_training_overlays(pptx_overlay, mp4_overlay)

        assert len(result["attendees"]) == 6
        names = [a.lower() if isinstance(a, str) else "" for a in result["attendees"]]
        assert "alice" in names
        assert "frank" in names

    def test_merge_overlays_combines_action_items(self):
        """Different action items from both → all preserved."""
        pptx_overlay = {
            "action_items": [
                {"action": "Review architecture", "owner": "Alice"},
            ],
        }
        mp4_overlay = {
            "action_items": [
                {"action": "Schedule follow-up meeting", "owner": "Bob"},
                {"action": "Update documentation", "owner": "Charlie"},
            ],
        }

        result = merge_training_overlays(pptx_overlay, mp4_overlay)

        assert len(result["action_items"]) == 3

    def test_merge_overlays_one_empty(self):
        """MP4 has no overlay → PPTX overlay returned as-is."""
        pptx_overlay = {
            "attendees": ["Alice"],
            "decisions_made": ["Approved migration plan"],
        }

        result = merge_training_overlays(pptx_overlay, {})

        assert result["attendees"] == ["Alice"]
        assert result["decisions_made"] == ["Approved migration plan"]

    def test_merge_overlays_both_empty(self):
        """Both empty → empty overlay."""
        result = merge_training_overlays({}, {})
        assert result == {}
