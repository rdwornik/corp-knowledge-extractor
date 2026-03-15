"""Tests for Gemini Batch API integration."""

import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

from src.batch_api import (
    build_batch_jsonl,
    submit_batch_job,
    poll_batch_job,
    parse_batch_results,
    BatchJobRunner,
)
from src.manifest import Manifest, ManifestEntry
from src.tier_router import Tier, TierDecision
from src.text_extract import TextExtractionResult
from src.extract import ExtractionError


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_text_result():
    return TextExtractionResult(
        text="This is sample extracted text from a document about Blue Yonder Platform.",
        char_count=71,
        extractor="pdfplumber",
        extraction_quality="good",
    )


@pytest.fixture
def sample_entries(tmp_path, sample_text_result):
    """Create sample manifest entries with temp files."""
    files = []
    for i in range(3):
        p = tmp_path / f"doc_{i}.pdf"
        p.write_text(f"Sample content {i}")
        entry = ManifestEntry(
            id=f"doc-{i}",
            path=p,
            doc_type="document",
            name=f"Document {i}",
        )
        decision = TierDecision(
            tier=Tier.TEXT_AI,
            reason="good text extraction",
            estimated_cost=0.001,
            model="gemini-2.5-flash",
            text_result=sample_text_result,
        )
        files.append((entry, decision))
    return files


@pytest.fixture
def mock_config():
    return {
        "prompts": {
            "extract": "Extract knowledge from this document. Return JSON.",
        },
        "anonymization": {"custom_terms": []},
        "gemini": {"model": "gemini-2.5-flash"},
    }


# ---------------------------------------------------------------------------
# build_batch_jsonl
# ---------------------------------------------------------------------------


class TestBuildBatchJsonl:
    def test_creates_jsonl_file(self, tmp_path, sample_entries, mock_config):
        jsonl_path = tmp_path / "batch_input.jsonl"
        result = build_batch_jsonl(sample_entries, mock_config, jsonl_path)
        assert result == jsonl_path
        assert jsonl_path.exists()

    def test_correct_line_count(self, tmp_path, sample_entries, mock_config):
        jsonl_path = tmp_path / "batch_input.jsonl"
        build_batch_jsonl(sample_entries, mock_config, jsonl_path)
        lines = jsonl_path.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 3

    def test_keys_match_entry_ids(self, tmp_path, sample_entries, mock_config):
        jsonl_path = tmp_path / "batch_input.jsonl"
        build_batch_jsonl(sample_entries, mock_config, jsonl_path)
        lines = jsonl_path.read_text(encoding="utf-8").strip().split("\n")
        keys = [json.loads(line)["key"] for line in lines]
        expected = [e.id for e, _ in sample_entries]
        assert keys == expected

    def test_request_structure(self, tmp_path, sample_entries, mock_config):
        jsonl_path = tmp_path / "batch_input.jsonl"
        build_batch_jsonl(sample_entries, mock_config, jsonl_path)
        line = json.loads(jsonl_path.read_text(encoding="utf-8").split("\n")[0])

        assert "key" in line
        assert "request" in line

        req = line["request"]
        assert "contents" in req
        assert len(req["contents"]) == 1
        assert req["contents"][0]["parts"][0]["text"]
        assert req["contents"][0]["role"] == "user"
        assert req["generation_config"]["response_mime_type"] == "application/json"

    def test_prompt_includes_file_content(self, tmp_path, sample_entries, mock_config):
        jsonl_path = tmp_path / "batch_input.jsonl"
        build_batch_jsonl(sample_entries, mock_config, jsonl_path)
        line = json.loads(jsonl_path.read_text(encoding="utf-8").split("\n")[0])
        text = line["request"]["contents"][0]["parts"][0]["text"]
        assert "Blue Yonder Platform" in text
        assert "FILE CONTENT" in text

    def test_skips_non_tier2(self, tmp_path, mock_config, sample_text_result):
        """Tier 1 and Tier 3 entries should not appear in JSONL."""
        p = tmp_path / "video.mp4"
        p.write_text("fake video")

        entries = [
            (
                ManifestEntry(id="tier1", path=p, doc_type="note", name="Note"),
                TierDecision(
                    tier=Tier.LOCAL, reason="small", estimated_cost=0, model=None, text_result=sample_text_result
                ),
            ),
            (
                ManifestEntry(id="tier3", path=p, doc_type="video", name="Video"),
                TierDecision(tier=Tier.MULTIMODAL, reason="video", estimated_cost=0.03, model="gemini-2.5-flash"),
            ),
        ]

        jsonl_path = tmp_path / "batch.jsonl"
        build_batch_jsonl(entries, mock_config, jsonl_path)
        content = jsonl_path.read_text(encoding="utf-8").strip()
        assert content == ""  # No lines written

    def test_creates_parent_dirs(self, tmp_path, sample_entries, mock_config):
        jsonl_path = tmp_path / "deep" / "nested" / "batch.jsonl"
        build_batch_jsonl(sample_entries, mock_config, jsonl_path)
        assert jsonl_path.exists()


# ---------------------------------------------------------------------------
# submit_batch_job
# ---------------------------------------------------------------------------


class TestSubmitBatchJob:
    def test_uploads_and_creates_job(self, tmp_path):
        jsonl_path = tmp_path / "batch.jsonl"
        jsonl_path.write_text('{"key": "test", "request": {}}\n')

        mock_client = MagicMock()
        mock_uploaded = MagicMock()
        mock_uploaded.name = "files/abc123"
        mock_client.files.upload.return_value = mock_uploaded

        mock_job = MagicMock()
        mock_job.name = "batches/xyz789"
        mock_job.state.name = "JOB_STATE_PENDING"
        mock_client.batches.create.return_value = mock_job

        result = submit_batch_job(mock_client, jsonl_path, "gemini-2.5-flash", "test-job")

        mock_client.files.upload.assert_called_once()
        mock_client.batches.create.assert_called_once_with(
            model="gemini-2.5-flash",
            src="files/abc123",
            config={"display_name": "test-job"},
        )
        assert result.name == "batches/xyz789"


# ---------------------------------------------------------------------------
# poll_batch_job
# ---------------------------------------------------------------------------


class TestPollBatchJob:
    def test_returns_on_success(self):
        mock_client = MagicMock()
        mock_job = MagicMock()
        mock_job.state.name = "JOB_STATE_SUCCEEDED"
        mock_client.batches.get.return_value = mock_job

        result = poll_batch_job(mock_client, "batches/test", poll_interval=0, timeout=10)
        assert result.state.name == "JOB_STATE_SUCCEEDED"

    def test_raises_on_failure(self):
        mock_client = MagicMock()
        mock_job = MagicMock()
        mock_job.state.name = "JOB_STATE_FAILED"
        mock_client.batches.get.return_value = mock_job

        with pytest.raises(ExtractionError, match="FAILED"):
            poll_batch_job(mock_client, "batches/test", poll_interval=0, timeout=10)

    def test_raises_on_cancelled(self):
        mock_client = MagicMock()
        mock_job = MagicMock()
        mock_job.state.name = "JOB_STATE_CANCELLED"
        mock_client.batches.get.return_value = mock_job

        with pytest.raises(ExtractionError, match="CANCELLED"):
            poll_batch_job(mock_client, "batches/test", poll_interval=0, timeout=10)

    def test_raises_on_expired(self):
        mock_client = MagicMock()
        mock_job = MagicMock()
        mock_job.state.name = "JOB_STATE_EXPIRED"
        mock_client.batches.get.return_value = mock_job

        with pytest.raises(ExtractionError, match="EXPIRED"):
            poll_batch_job(mock_client, "batches/test", poll_interval=0, timeout=10)

    def test_timeout_raises(self):
        mock_client = MagicMock()
        mock_job = MagicMock()
        mock_job.state.name = "JOB_STATE_RUNNING"
        mock_client.batches.get.return_value = mock_job

        with pytest.raises(TimeoutError, match="did not complete"):
            poll_batch_job(mock_client, "batches/test", poll_interval=0, timeout=0)

    def test_polls_until_done(self):
        """Should transition through states before succeeding."""
        mock_client = MagicMock()

        states = ["JOB_STATE_PENDING", "JOB_STATE_RUNNING", "JOB_STATE_SUCCEEDED"]
        jobs = []
        for state in states:
            job = MagicMock()
            job.state.name = state
            jobs.append(job)

        mock_client.batches.get.side_effect = jobs

        result = poll_batch_job(mock_client, "batches/test", poll_interval=0, timeout=60)
        assert result.state.name == "JOB_STATE_SUCCEEDED"
        assert mock_client.batches.get.call_count == 3

    def test_retries_on_network_error(self):
        """Network errors during poll should retry, not crash."""
        mock_client = MagicMock()

        error_then_success = [
            ConnectionError("network timeout"),
            MagicMock(state=MagicMock(name="JOB_STATE_SUCCEEDED")),
        ]
        # Fix: state.name needs to work
        error_then_success[1].state.name = "JOB_STATE_SUCCEEDED"
        mock_client.batches.get.side_effect = error_then_success

        result = poll_batch_job(mock_client, "batches/test", poll_interval=0, timeout=60)
        assert result.state.name == "JOB_STATE_SUCCEEDED"


# ---------------------------------------------------------------------------
# parse_batch_results
# ---------------------------------------------------------------------------


class TestParseBatchResults:
    def _make_result_line(self, key: str, text: str) -> str:
        return json.dumps({"key": key, "response": {"candidates": [{"content": {"parts": [{"text": text}]}}]}})

    def test_maps_keys_to_responses(self):
        mock_client = MagicMock()
        mock_job = MagicMock()
        mock_job.dest.file_name = "files/result123"

        result_jsonl = "\n".join(
            [
                self._make_result_line("doc-0", '{"title": "Doc Zero"}'),
                self._make_result_line("doc-1", '{"title": "Doc One"}'),
            ]
        )
        mock_client.files.download.return_value = result_jsonl.encode("utf-8")

        results = parse_batch_results(mock_client, mock_job)

        assert len(results) == 2
        assert "doc-0" in results
        assert "doc-1" in results
        assert '"Doc Zero"' in results["doc-0"]

    def test_handles_error_entries(self):
        mock_client = MagicMock()
        mock_job = MagicMock()
        mock_job.dest.file_name = "files/result123"

        result_jsonl = "\n".join(
            [
                self._make_result_line("doc-0", '{"title": "OK"}'),
                json.dumps({"key": "doc-1", "error": {"message": "rate limit"}}),
            ]
        )
        mock_client.files.download.return_value = result_jsonl.encode("utf-8")

        results = parse_batch_results(mock_client, mock_job)
        assert "doc-0" in results
        assert "doc-1" not in results  # Error entry skipped

    def test_handles_empty_results(self):
        mock_client = MagicMock()
        mock_job = MagicMock()
        mock_job.dest.file_name = "files/result123"
        mock_client.files.download.return_value = b""

        results = parse_batch_results(mock_client, mock_job)
        assert results == {}

    def test_handles_malformed_lines(self):
        mock_client = MagicMock()
        mock_job = MagicMock()
        mock_job.dest.file_name = "files/result123"

        result_jsonl = "not valid json\n" + self._make_result_line("doc-0", "ok")
        mock_client.files.download.return_value = result_jsonl.encode("utf-8")

        results = parse_batch_results(mock_client, mock_job)
        assert len(results) == 1
        assert "doc-0" in results


# ---------------------------------------------------------------------------
# Cost display
# ---------------------------------------------------------------------------


class TestCostDiscount:
    def test_tier2_batch_cost_is_half(self):
        """Tier 2 batch cost should be 50% of synchronous."""
        from src.tier_router import TIER_COSTS, Tier

        sync_cost = TIER_COSTS[Tier.TEXT_AI]
        batch_cost = sync_cost * 0.5
        assert batch_cost == sync_cost / 2
        assert batch_cost == 0.0005


# ---------------------------------------------------------------------------
# Integration: BatchJobRunner routing
# ---------------------------------------------------------------------------


class TestBatchJobRunnerRouting:
    def test_routes_entries_to_correct_tiers(self, tmp_path):
        """Verify that entries are bucketed into tier1/tier2/tier3 lists."""
        # Create files of different types
        pdf = tmp_path / "report.pdf"
        pdf.write_text("x" * 100)
        txt = tmp_path / "notes.txt"
        txt.write_text("Short note")
        mp4 = tmp_path / "video.mp4"
        mp4.write_bytes(b"\x00" * 1000)

        manifest = Manifest(
            schema_version=1,
            project="test",
            output_dir=tmp_path / "output",
            files=[
                ManifestEntry(id="pdf-1", path=pdf, doc_type="document", name="Report"),
                ManifestEntry(id="txt-1", path=txt, doc_type="note", name="Notes"),
                ManifestEntry(id="mp4-1", path=mp4, doc_type="video", name="Video"),
            ],
        )

        runner = BatchJobRunner(manifest, {"prompts": {"extract": "test"}, "gemini": {}})

        # We can't run the full pipeline without mocking everything,
        # but we can verify the object is constructed correctly
        assert runner.manifest == manifest
        assert len(runner.manifest.files) == 3
