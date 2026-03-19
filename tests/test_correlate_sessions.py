"""Tests for session correlation detection (Stage 1 + Stage 2)."""

import pytest
from pathlib import Path

from src.correlate_sessions import (
    CorrelationCandidate,
    normalize_stem,
    filename_similarity,
    detect_stage1,
    confirm_stage2,
    word_jaccard,
)


class TestNormalizeStem:
    def test_basic(self):
        assert normalize_stem("Cognitive Friday S4E1 J2CC.pptx") == "cognitive friday s4e1 j2cc"

    def test_strips_noise(self):
        assert normalize_stem("Report_final_v2.pptx") == "report"

    def test_normalizes_separators(self):
        assert normalize_stem("my-file_name.mp4") == "my file name"

    def test_collapses_spaces(self):
        assert normalize_stem("file  _  name.txt") == "file name"


class TestFilenameSimilarity:
    def test_identical(self):
        sim = filename_similarity("Session1.pptx", "Session1.mp4")
        assert sim == 1.0

    def test_high_similarity(self):
        sim = filename_similarity(
            "Cognitive Friday S4E1 J2CC.pptx",
            "Cognitive Friday S4E1 - Journey to the Cloud.mp4",
        )
        assert sim > 0.5

    def test_low_similarity(self):
        sim = filename_similarity("Budget_2026.xlsx", "Platform_Architecture.mp4")
        assert sim < 0.3


class TestDetectStage1:
    def test_finds_pair(self, tmp_path):
        pptx = tmp_path / "Session1.pptx"
        mp4 = tmp_path / "Session1.mp4"
        pptx.touch()
        mp4.touch()
        group = detect_stage1([pptx, mp4])
        assert len(group.candidates) == 1
        assert group.candidates[0].pptx_path == pptx
        assert group.candidates[0].video_path == mp4

    def test_different_folders_no_match(self, tmp_path):
        dir1 = tmp_path / "a"
        dir2 = tmp_path / "b"
        dir1.mkdir()
        dir2.mkdir()
        pptx = dir1 / "Session1.pptx"
        mp4 = dir2 / "Session1.mp4"
        pptx.touch()
        mp4.touch()
        group = detect_stage1([pptx, mp4])
        assert len(group.candidates) == 0

    def test_no_video(self, tmp_path):
        pptx = tmp_path / "Deck.pptx"
        pptx.touch()
        group = detect_stage1([pptx])
        assert len(group.candidates) == 0
        assert len(group.standalone) == 1

    def test_low_similarity_excluded(self, tmp_path):
        pptx = tmp_path / "Budget2026.pptx"
        mp4 = tmp_path / "Architecture_Deep_Dive.mp4"
        pptx.touch()
        mp4.touch()
        group = detect_stage1([pptx, mp4])
        assert len(group.candidates) == 0
        assert len(group.standalone) == 2

    def test_mkv_extension(self, tmp_path):
        pptx = tmp_path / "Session.pptx"
        mkv = tmp_path / "Session.mkv"
        pptx.touch()
        mkv.touch()
        group = detect_stage1([pptx, mkv])
        assert len(group.candidates) == 1

    def test_standalone_files_tracked(self, tmp_path):
        pptx = tmp_path / "Deck.pptx"
        mp4 = tmp_path / "Deck.mp4"
        txt = tmp_path / "Notes.txt"
        pptx.touch()
        mp4.touch()
        txt.touch()
        group = detect_stage1([pptx, mp4, txt])
        assert len(group.candidates) == 1
        assert len(group.standalone) == 1
        assert group.standalone[0] == txt


class TestWordJaccard:
    def test_word_jaccard_identical(self):
        assert word_jaccard("hello world", "hello world") == 1.0

    def test_word_jaccard_partial(self):
        result = word_jaccard("journey cognitive cloud", "cognitive cloud training")
        assert result == pytest.approx(0.5, abs=0.01)

    def test_word_jaccard_disjoint(self):
        assert word_jaccard("budget review", "platform architecture") == 0.0

    def test_word_jaccard_empty_both(self):
        assert word_jaccard("", "") == 1.0

    def test_word_jaccard_one_empty(self):
        assert word_jaccard("hello", "") == 0.0


class TestConfirmStage2:
    def test_title_match_confirms(self):
        """filename_sim=0.8 (pass) + title_jaccard high (pass) → merge."""
        candidate = CorrelationCandidate(
            pptx_path=Path("a.pptx"),
            video_path=Path("a.mp4"),
            filename_similarity=0.8,
            stage1_confidence=80,
        )
        candidate = confirm_stage2(
            candidate,
            {"title": "Journey to Cognitive Cloud", "slide_count": 15},
            {"title": "Journey to Cognitive Cloud Session", "slides": [{}] * 14},
        )
        assert candidate.merge_decision == "merge"
        assert candidate.stage2_confirmed is True

    def test_slide_count_match_confirms(self):
        """filename_sim=0.7 (pass) + slide_count ±1 (pass) → merge."""
        candidate = CorrelationCandidate(
            pptx_path=Path("a.pptx"),
            video_path=Path("a.mp4"),
            filename_similarity=0.7,
            stage1_confidence=70,
        )
        candidate = confirm_stage2(
            candidate,
            {"title": "Deck A", "slide_count": 20},
            {"title": "Video B", "slides": [{}] * 19},
        )
        assert candidate.merge_decision == "merge"

    def test_no_match_becomes_crosslink(self):
        """Only filename_sim barely passes (0.65 not > 0.65) — 0 signals → crosslink."""
        candidate = CorrelationCandidate(
            pptx_path=Path("a.pptx"),
            video_path=Path("b.mp4"),
            filename_similarity=0.65,
            stage1_confidence=65,
        )
        candidate = confirm_stage2(
            candidate,
            {"title": "Budget Review 2026", "slide_count": 5},
            {"title": "Platform Architecture Deep Dive", "slides": [{}] * 30},
        )
        assert candidate.merge_decision == "crosslink"
        assert candidate.stage2_confirmed is False

    def test_stage2_two_signals_merge(self):
        """filename_sim=0.71 + title_jaccard=0.55 → 2 signals → merge."""
        candidate = CorrelationCandidate(
            pptx_path=Path("a.pptx"),
            video_path=Path("a.mp4"),
            filename_similarity=0.71,
            stage1_confidence=71,
        )
        # Titles with >0.50 Jaccard: "cloud platform demo" vs "cloud platform overview"
        # Jaccard = {cloud, platform} / {cloud, platform, demo, overview} = 2/4 = 0.50
        # Need > 0.50, so use closer titles
        candidate = confirm_stage2(
            candidate,
            {"title": "Cloud Platform Demo Session", "slide_count": 5},
            {"title": "Cloud Platform Demo Recording", "slides": [{}] * 30},
        )
        assert candidate.merge_decision == "merge"
        assert candidate.stage2_confirmed is True

    def test_stage2_one_signal_crosslink(self):
        """Only filename_sim passes → 1 signal → crosslink."""
        candidate = CorrelationCandidate(
            pptx_path=Path("x.pptx"),
            video_path=Path("x.mp4"),
            filename_similarity=0.75,
            stage1_confidence=75,
        )
        candidate = confirm_stage2(
            candidate,
            {"title": "Budget Review 2026", "slide_count": 5},
            {"title": "Platform Architecture Deep Dive", "slides": [{}] * 30},
        )
        assert candidate.merge_decision == "crosslink"

    def test_stage2_slide_count_and_topics(self):
        """slide_count ±2 + topic_overlap 0.7 → 2 signals → merge."""
        candidate = CorrelationCandidate(
            pptx_path=Path("a.pptx"),
            video_path=Path("a.mp4"),
            filename_similarity=0.50,  # below threshold
            stage1_confidence=50,
        )
        candidate = confirm_stage2(
            candidate,
            {
                "title": "Totally Different Title A",
                "slide_count": 20,
                "topics": ["SLA", "Disaster Recovery", "Cloud Migration"],
            },
            {
                "title": "Completely Unrelated Title B",
                "slides": [{}] * 18,
                "topics": ["SLA", "Disaster Recovery", "Cloud Migration", "Pricing"],
            },
        )
        assert candidate.merge_decision == "merge"

    def test_stage2_real_cognitive_friday(self):
        """Actual S4E1 Cognitive Friday titles → should merge."""
        candidate = CorrelationCandidate(
            pptx_path=Path("Cognitive Friday S4E1 J2CC.pptx"),
            video_path=Path("Cognitive Friday S4E1 - Journey to the Cloud.mp4"),
            filename_similarity=0.72,
            stage1_confidence=72,
        )
        candidate = confirm_stage2(
            candidate,
            {
                "title": "Cognitive Friday S4E1 - Journey to the Cognitive Cloud",
                "slide_count": 25,
                "topics": ["Cloud Migration", "SLA"],
            },
            {
                "title": "Cognitive Friday Season 4 Episode 1 - Journey to the Cognitive Cloud",
                "slides": [{}] * 23,
                "topics": ["Cloud Migration", "SLA", "Platform"],
            },
        )
        assert candidate.merge_decision == "merge"
        assert candidate.stage2_confirmed is True
