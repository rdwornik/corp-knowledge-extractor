"""Unit tests for src/correlate.py — file grouping logic."""

import pytest
from pathlib import Path

from src.inventory import SourceFile, FileType
from src.correlate import correlate_files, FileGroup, _stem_prefix


def _sf(name: str, ft: FileType, size: int = 1000) -> SourceFile:
    """Helper to create a SourceFile without touching the filesystem."""
    return SourceFile(
        path=Path(f"/fake/{name}"),
        type=ft,
        size_bytes=size,
        name=Path(name).stem,
    )


# ---------------------------------------------------------------------------
# stem_prefix helper
# ---------------------------------------------------------------------------


def test_stem_prefix_keeps_identifier_digits():
    # Digits that are part of the group ID must be preserved
    assert _stem_prefix("session1") == "session1"


def test_stem_prefix_strips_slides_suffix():
    # Strips "_slides" but keeps identifier digit intact
    assert _stem_prefix("session1_slides") == "session1"


def test_stem_prefix_strips_notes_suffix():
    assert _stem_prefix("meeting_notes") == "meeting"


def test_stem_prefix_no_change():
    assert _stem_prefix("training") == "training"


# ---------------------------------------------------------------------------
# Single group for small inputs
# ---------------------------------------------------------------------------


def test_single_group_for_one_file():
    files = [_sf("video.mp4", FileType.VIDEO)]
    groups = correlate_files(files, {})
    assert len(groups) == 1
    assert groups[0].primary.name == "video"


def test_single_group_for_two_files():
    files = [
        _sf("video.mp4", FileType.VIDEO),
        _sf("slides.pptx", FileType.SLIDES),
    ]
    groups = correlate_files(files, {})
    assert len(groups) == 1


def test_single_group_for_three_files():
    files = [
        _sf("video.mp4", FileType.VIDEO),
        _sf("slides.pptx", FileType.SLIDES),
        _sf("notes.md", FileType.NOTE),
    ]
    groups = correlate_files(files, {})
    assert len(groups) == 1


# ---------------------------------------------------------------------------
# Filename matching (> 3 files)
# ---------------------------------------------------------------------------


def test_filename_matching_groups_session_files():
    files = [
        _sf("session1.mp4", FileType.VIDEO),
        _sf("session1_slides.pptx", FileType.SLIDES),
        _sf("session2.mp4", FileType.VIDEO),
        _sf("session2_notes.md", FileType.NOTE),
    ]
    groups = correlate_files(files, {})
    # session1 and session2 should be separate groups
    assert len(groups) == 2


def test_unrelated_files_become_separate_groups():
    files = [
        _sf("alpha.mp4", FileType.VIDEO),
        _sf("beta.mp4", FileType.VIDEO),
        _sf("gamma.mp4", FileType.VIDEO),
        _sf("delta.mp4", FileType.VIDEO),
    ]
    groups = correlate_files(files, {})
    assert len(groups) == 4


# ---------------------------------------------------------------------------
# Primary selection
# ---------------------------------------------------------------------------


def test_video_is_primary():
    files = [
        _sf("session1.mp4", FileType.VIDEO),
        _sf("session1_slides.pptx", FileType.SLIDES),
        _sf("session1_notes.md", FileType.NOTE),
        _sf("session1_report.pdf", FileType.DOCUMENT),
    ]
    groups = correlate_files(files, {})
    assert groups[0].primary.type == FileType.VIDEO


def test_audio_is_primary_when_no_video():
    files = [
        _sf("meeting1.mp3", FileType.AUDIO),
        _sf("meeting1_notes.md", FileType.NOTE),
        _sf("meeting1_slides.pptx", FileType.SLIDES),
        _sf("meeting1_report.pdf", FileType.DOCUMENT),
    ]
    groups = correlate_files(files, {})
    assert groups[0].primary.type == FileType.AUDIO


def test_related_files_not_primary():
    files = [
        _sf("talk.mp4", FileType.VIDEO),
        _sf("talk_slides.pptx", FileType.SLIDES),
    ]
    groups = correlate_files(files, {})
    assert groups[0].primary.type == FileType.VIDEO
    assert len(groups[0].related) == 1
    assert groups[0].related[0].type == FileType.SLIDES


# ---------------------------------------------------------------------------
# Empty input
# ---------------------------------------------------------------------------


def test_empty_input_returns_empty():
    groups = correlate_files([], {})
    assert groups == []
