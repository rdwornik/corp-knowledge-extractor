"""Tests for normalize_string_list and MemoryError guard."""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.utils import normalize_string_list


class TestNormalizeStringList:
    def test_normalize_string_list_strings(self):
        assert normalize_string_list(["a", "b"]) == ["a", "b"]

    def test_normalize_string_list_dicts(self):
        result = normalize_string_list([{"name": "Rob", "role": "PM"}])
        assert result == ["Rob (PM)"]

    def test_normalize_string_list_mixed(self):
        result = normalize_string_list(["Alice", {"name": "Bob", "role": "Dev"}])
        assert result == ["Alice", "Bob (Dev)"]

    def test_normalize_string_list_empty_dict(self):
        result = normalize_string_list([{}])
        assert result == ["{}"]

    def test_normalize_string_list_nested(self):
        """Dict with 'title' key (no 'name') uses title."""
        result = normalize_string_list([{"title": "Azure"}])
        assert result == ["Azure"]

    def test_normalize_string_list_dict_name_only(self):
        result = normalize_string_list([{"name": "SAP"}])
        assert result == ["SAP"]

    def test_normalize_string_list_int(self):
        result = normalize_string_list([42])
        assert result == ["42"]

    def test_normalize_string_list_empty(self):
        assert normalize_string_list([]) == []

    def test_normalize_string_list_dict_with_both_name_and_title(self):
        """'name' takes precedence over 'title'."""
        result = normalize_string_list([{"name": "WMS", "title": "Warehouse"}])
        assert result == ["WMS"]


class TestDedupWithDicts:
    def test_dedup_with_dicts_no_crash(self):
        """Synthesize dedup should not crash on dict items."""
        from src.synthesize import normalize_string_list as nsl
        items = [{"name": "A"}, "B", {"name": "A"}]
        normalized = nsl(items)
        seen = set()
        result = []
        for x in normalized:
            key = x.lower()
            if key not in seen:
                seen.add(key)
                result.append(x)
        assert "A" in result
        assert "B" in result
        assert len(result) == 2


class TestUploadMemoryError:
    def test_upload_memory_error(self, tmp_path):
        """MemoryError during upload → graceful ExtractionError."""
        from src.extract import _upload_and_wait, ExtractionError

        # Create a small file (the size check is based on stat, but MemoryError
        # comes from the SDK reading the file)
        video = tmp_path / "huge.mp4"
        video.write_bytes(b"x" * 1024)

        mock_client = MagicMock()
        mock_client.files.upload.side_effect = MemoryError("out of memory")

        config = {"gemini": {"polling_interval_sec": 1, "upload_timeout_sec": 10}}

        with pytest.raises(ExtractionError, match="File too large for upload"):
            _upload_and_wait(mock_client, video, config)

    def test_upload_large_file_warns(self, tmp_path, caplog):
        """Files > 300MB log a warning about compression."""
        from src.extract import _upload_and_wait

        # We can't create a 300MB file in tests, so mock stat
        video = tmp_path / "big.mp4"
        video.write_bytes(b"x" * 100)

        mock_uploaded = MagicMock()
        mock_uploaded.state.name = "ACTIVE"
        mock_uploaded.uri = "gs://fake"

        mock_client = MagicMock()
        mock_client.files.upload.return_value = mock_uploaded

        config = {"gemini": {"polling_interval_sec": 1, "upload_timeout_sec": 10}}

        with patch.object(Path, "stat") as mock_stat:
            mock_stat.return_value = MagicMock(st_size=400 * 1024 * 1024)
            import logging
            with caplog.at_level(logging.WARNING):
                result = _upload_and_wait(mock_client, video, config)

        assert result == mock_uploaded
        assert any("compression is recommended" in r.message for r in caplog.records)
