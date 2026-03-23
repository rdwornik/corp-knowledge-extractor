"""Tests for --force flag propagation in batch processor and manifest CLI."""

import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.manifest import Manifest, ManifestEntry, FileStatus, save_status
from src.batch import BatchProcessor


class TestBatchProcessorForce:
    def _make_manifest(self, tmp_path):
        test_file = tmp_path / "test.txt"
        test_file.write_text("hello world", encoding="utf-8")
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        return Manifest(
            schema_version=1, project="test", output_dir=output_dir,
            files=[ManifestEntry(id="file-1", path=test_file, doc_type="document", name="test.txt")],
        )

    def _mark_done(self, output_dir: Path, entry_id: str):
        save_status(output_dir, {
            entry_id: {"status": FileStatus.DONE.value, "processed_at": "2026-03-23T00:00:00"},
        })

    def test_force_true_reprocesses_done(self, tmp_path):
        """force=True + existing DONE → re-extracts anyway."""
        manifest = self._make_manifest(tmp_path)
        self._mark_done(manifest.output_dir, "file-1")

        processor = BatchProcessor(manifest, {}, resume=True, force=True)
        with patch.object(processor, "_process_single", return_value=2) as mock_proc:
            summary = processor.process_all()

        assert summary["skipped"] == 0
        assert summary["done"] == 1
        mock_proc.assert_called_once()

    def test_no_force_skips_done(self, tmp_path):
        """force=False + resume=True + DONE → skipped."""
        manifest = self._make_manifest(tmp_path)
        self._mark_done(manifest.output_dir, "file-1")

        processor = BatchProcessor(manifest, {}, resume=True, force=False)
        with patch.object(processor, "_process_single", return_value=2) as mock_proc:
            summary = processor.process_all()

        assert summary["skipped"] == 1
        assert summary["done"] == 0
        mock_proc.assert_not_called()


class TestBatchJobRunnerForce:
    def test_runner_accepts_force(self):
        from src.batch_api import BatchJobRunner
        manifest = MagicMock()
        manifest.files = []
        manifest.output_dir = Path("/tmp/test")
        runner = BatchJobRunner(manifest, {}, force=True)
        assert runner.force is True

    def test_runner_default_no_force(self):
        from src.batch_api import BatchJobRunner
        manifest = MagicMock()
        manifest.files = []
        manifest.output_dir = Path("/tmp/test")
        runner = BatchJobRunner(manifest, {})
        assert runner.force is False


class TestProcessManifestCLI:
    def test_force_flag_exists(self):
        from scripts.run import process_manifest
        param_names = [p.name for p in process_manifest.params]
        assert "force" in param_names

    def test_force_is_flag(self):
        from scripts.run import process_manifest
        force_param = next(p for p in process_manifest.params if p.name == "force")
        assert force_param.is_flag is True
