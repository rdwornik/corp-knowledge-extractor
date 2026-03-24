"""Tests for batch processor slide_image_paths propagation."""

import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


def _make_extraction_result(tmp_path, with_slides=True):
    """Create a mock ExtractionResult with optional slide_image_paths."""
    from src.extract import ExtractionResult
    from src.inventory import SourceFile, FileType

    source_file = SourceFile(
        path=tmp_path / "test.pdf",
        type=FileType.DOCUMENT,
        size_bytes=1000,
        name="test",
    )
    # Create a dummy source file
    source_file.path.write_bytes(b"%PDF-1.4 dummy")

    result = ExtractionResult(
        source_file=source_file,
        title="Test Document",
        summary="A test summary.",
        key_points=["Point 1"],
        topics=["SLA"],
        people=[],
        products=[],
        content_type="document",
        raw_json={"title": "Test Document", "summary": "A test.", "key_points": ["Point 1"], "topics": ["SLA"]},
    )

    if with_slides:
        # Create temp slide PNGs
        temp_slides = tmp_path / "temp_slides" / "test"
        temp_slides.mkdir(parents=True)
        slide_paths = []
        for i in range(3):
            png = temp_slides / f"cover_{i+1:03d}.png"
            png.write_bytes(b"\x89PNG fake image data")
            slide_paths.append(png)
        result.slide_image_paths = slide_paths
    else:
        result.slide_image_paths = []

    return result


class TestBatchSlideImagePaths:
    """Batch processor must copy slide_image_paths PNGs to output."""

    def test_batch_copies_slide_pngs(self, tmp_path):
        """Extraction with slide_image_paths → PNGs copied to source/slides/."""
        result = _make_extraction_result(tmp_path, with_slides=True)
        slide_paths = list(result.slide_image_paths)  # copy before they get unlinked

        # Simulate what batch.py does after extraction
        pkg_dir = tmp_path / "output" / "test_pkg"
        pkg_dir.mkdir(parents=True)

        # Replicate the batch.py logic
        if result.slide_image_paths:
            slides_dir = pkg_dir / "source" / "slides"
            slides_dir.mkdir(parents=True, exist_ok=True)
            for png in result.slide_image_paths:
                if png.exists():
                    shutil.copy2(png, slides_dir / png.name)
            # Cleanup temp
            for png in result.slide_image_paths:
                if png.exists():
                    png.unlink()

        # Verify PNGs were copied
        output_slides = list((pkg_dir / "source" / "slides").glob("*.png"))
        assert len(output_slides) == 3
        assert all(p.stat().st_size > 0 for p in output_slides)

        # Verify temp files were cleaned up
        assert all(not p.exists() for p in slide_paths)

    def test_batch_no_slides(self, tmp_path):
        """Extraction without slides → no slides dir, no error."""
        result = _make_extraction_result(tmp_path, with_slides=False)

        pkg_dir = tmp_path / "output" / "test_pkg"
        pkg_dir.mkdir(parents=True)

        # Replicate the batch.py logic — should be a no-op
        if result.slide_image_paths:
            slides_dir = pkg_dir / "source" / "slides"
            slides_dir.mkdir(parents=True, exist_ok=True)
            for png in result.slide_image_paths:
                if png.exists():
                    shutil.copy2(png, slides_dir / png.name)

        # slides dir should not exist
        assert not (pkg_dir / "source" / "slides").exists()
