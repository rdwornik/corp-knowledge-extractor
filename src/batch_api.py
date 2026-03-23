"""
Gemini Batch API integration for CKE.

Submits multiple extraction requests as a single JSONL batch job
at 50% of the synchronous API cost. Turnaround: typically minutes
to hours (up to 24h).

Only Tier 2 (text-only) requests are batched. Tier 1 (local) runs
locally with no API call. Tier 3 (multimodal) falls back to
synchronous since it requires file uploads that can't be embedded
in JSONL.

Usage:
    from src.batch_api import BatchJobRunner

    runner = BatchJobRunner(manifest, config)
    summary = runner.run(poll_interval=60, timeout=86400)
"""

import json
import logging
import time
from datetime import datetime
from pathlib import Path

from src.manifest import Manifest, ManifestEntry, FileStatus, save_status
from src.extract import (
    _get_client,
    _get_model,
    _get_prompt,
    _prepend_user_context,
    _result_from_json,
    ExtractionResult,
    ExtractionError,
)
from src.inventory import SourceFile, FileType
from src.tier_router import route_tier, Tier, TierDecision, TIER_COSTS
from src.post_process import post_process_extraction
from src.utils import parse_llm_json

logger = logging.getLogger(__name__)

# Map manifest doc_type strings to FileType enum values
_DOC_TYPE_MAP = {
    "video": "VIDEO",
    "audio": "AUDIO",
    "document": "DOCUMENT",
    "presentation": "SLIDES",
    "slides": "SLIDES",
    "spreadsheet": "SPREADSHEET",
    "note": "NOTE",
    "transcript": "TRANSCRIPT",
}


def _source_file_from_entry(entry: ManifestEntry) -> SourceFile:
    """Build a SourceFile from a ManifestEntry."""
    ft_name = _DOC_TYPE_MAP.get(entry.doc_type.lower(), "DOCUMENT")
    file_type = FileType[ft_name]
    return SourceFile(
        path=entry.path,
        type=file_type,
        size_bytes=entry.path.stat().st_size,
        name=entry.path.stem,
    )


def build_batch_jsonl(
    entries: list[tuple[ManifestEntry, TierDecision]],
    config: dict,
    output_path: Path,
) -> Path:
    """
    Build JSONL file for Gemini Batch API.

    Each line: {"key": "<entry_id>", "request": {"contents": [...], "generation_config": {...}}}

    Only includes Tier 2 entries (text-only).

    Args:
        entries: List of (ManifestEntry, TierDecision) tuples for Tier 2 files
        config: Unified config dict
        output_path: Where to write the JSONL file

    Returns:
        Path to the written JSONL file
    """
    prompt = _get_prompt(config, "extract")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for entry, decision in entries:
            if decision.tier != Tier.TEXT_AI or not decision.text_result:
                continue

            text_content = decision.text_result.text[:80000]
            entry_prompt = _prepend_user_context(prompt, entry.user_context)
            full_prompt = f"{entry_prompt}\n\n--- FILE CONTENT ({decision.text_result.extractor}) ---\n{text_content}"

            line = {
                "key": entry.id,
                "request": {
                    "contents": [
                        {
                            "parts": [{"text": full_prompt}],
                            "role": "user",
                        }
                    ],
                    "generation_config": {
                        "response_mime_type": "application/json",
                    },
                },
            }
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
            count += 1

    logger.info("Built batch JSONL: %d requests -> %s", count, output_path)
    return output_path


def submit_batch_job(
    client,
    jsonl_path: Path,
    model: str,
    display_name: str | None = None,
) -> object:
    """
    Upload JSONL and submit batch job to Gemini.

    Args:
        client: google.genai.Client instance
        jsonl_path: Path to JSONL input file
        model: Gemini model name (e.g. "gemini-3-flash-preview")
        display_name: Human-readable job name

    Returns:
        Batch job object (has .name, .state, .dest)
    """
    from google.genai import types

    logger.info("Uploading batch JSONL: %s", jsonl_path.name)
    uploaded = client.files.upload(
        file=jsonl_path,
        config=types.UploadFileConfig(
            display_name=display_name or jsonl_path.stem,
            mime_type="jsonl",
        ),
    )
    logger.info("Uploaded: %s", uploaded.name)

    batch_job = client.batches.create(
        model=model,
        src=uploaded.name,
        config={"display_name": display_name or f"cke-batch-{int(time.time())}"},
    )

    logger.info("Batch job created: %s (state: %s)", batch_job.name, batch_job.state.name)
    return batch_job


def poll_batch_job(
    client,
    job_name: str,
    poll_interval: int = 60,
    timeout: int = 86400,
) -> object:
    """
    Poll batch job until completion.

    Args:
        client: google.genai.Client
        job_name: Batch job name from submit_batch_job()
        poll_interval: Seconds between status checks
        timeout: Max seconds to wait

    Returns:
        Completed batch job object

    Raises:
        TimeoutError: If job doesn't complete within timeout
        ExtractionError: If job fails or is cancelled
    """
    TERMINAL_STATES = {
        "JOB_STATE_SUCCEEDED",
        "JOB_STATE_FAILED",
        "JOB_STATE_CANCELLED",
        "JOB_STATE_EXPIRED",
    }

    start = time.time()
    last_state = None
    retries = 0
    max_retries = 5

    while True:
        elapsed = time.time() - start
        if elapsed > timeout:
            raise TimeoutError(
                f"Batch job {job_name} did not complete within {timeout}s. "
                f"Check manually: client.batches.get(name='{job_name}')"
            )

        try:
            job = client.batches.get(name=job_name)
            retries = 0  # reset on success
        except Exception as e:
            retries += 1
            if retries > max_retries:
                raise ExtractionError(f"Failed to poll batch job after {max_retries} retries: {e}") from e
            wait = min(poll_interval * retries, 300)
            logger.warning("Poll error (retry %d/%d): %s. Waiting %ds...", retries, max_retries, e, wait)
            time.sleep(wait)
            continue

        state = job.state.name if hasattr(job.state, "name") else str(job.state)

        if state != last_state:
            logger.info(
                "Batch %s: %s (%.0fs elapsed)",
                job_name,
                state,
                elapsed,
            )
            last_state = state

        if state in TERMINAL_STATES:
            if state == "JOB_STATE_FAILED":
                raise ExtractionError(f"Batch job {job_name} FAILED")
            if state == "JOB_STATE_CANCELLED":
                raise ExtractionError(f"Batch job {job_name} was CANCELLED")
            if state == "JOB_STATE_EXPIRED":
                raise ExtractionError(f"Batch job {job_name} EXPIRED (ran >48h without completing)")
            return job  # SUCCEEDED

        time.sleep(poll_interval)


def parse_batch_results(
    client,
    job,
) -> dict[str, str]:
    """
    Download batch results and parse into {key: response_text} mapping.

    Args:
        client: google.genai.Client
        job: Completed batch job object

    Returns:
        Dict mapping entry key -> raw Gemini response text
    """
    result_file = job.dest.file_name
    logger.info("Downloading batch results: %s", result_file)

    result_bytes = client.files.download(file=result_file)
    result_text = result_bytes.decode("utf-8")

    results = {}
    for line_num, line in enumerate(result_text.strip().split("\n"), 1):
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
            key = entry.get("key", f"unknown-{line_num}")

            # Extract response text from the batch result structure
            response = entry.get("response", {})

            # The response contains candidates[0].content.parts[0].text
            candidates = response.get("candidates", [])
            if candidates:
                parts = candidates[0].get("content", {}).get("parts", [])
                if parts:
                    results[key] = parts[0].get("text", "")
                    continue

            # Fallback: check for error
            error = entry.get("error")
            if error:
                logger.error("Batch entry %s failed: %s", key, error)
                continue

            logger.warning("Batch entry %s: no response text found", key)

        except json.JSONDecodeError as e:
            logger.error("Failed to parse batch result line %d: %s", line_num, e)

    logger.info("Parsed %d batch results", len(results))
    return results


class BatchJobRunner:
    """
    Orchestrates the full batch extraction flow:
    1. Route all manifest entries to tiers
    2. Process Tier 1 locally (free, immediate)
    3. Build JSONL for Tier 2 entries
    4. Submit batch job, poll, parse results
    5. Fall back to synchronous for Tier 3
    6. Post-process all results and write output
    """

    def __init__(
        self,
        manifest: Manifest,
        config: dict,
        force_tier: int | None = None,
        resume: bool = False,
    ):
        self.manifest = manifest
        self.config = config
        self.force_tier = force_tier
        self.resume = resume
        self.statuses: dict[str, dict] = {}

    def run(
        self,
        poll_interval: int = 60,
        timeout: int = 86400,
    ) -> dict:
        """
        Execute batch processing. Returns summary dict compatible with
        the synchronous BatchProcessor.process_all() output.
        """
        from src.extract import extract_local, extract_knowledge
        from src.manifest import load_status
        from src.correlate import correlate_files
        from src.synthesize import build_package
        from src.frames.sampler import sample_frames
        from src.compress import needs_compression, compress_video
        from scripts.run import keep_slide_frames

        output_dir = self.manifest.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        existing_status = {}
        if self.resume:
            existing_status = load_status(output_dir)

        summary = {
            "total": len(self.manifest.files),
            "done": 0,
            "error": 0,
            "skipped": 0,
            "cost": 0.0,
            "tiers": {1: 0, 2: 0, 3: 0},
            "batch_mode": True,
        }

        # --- Phase 1: Route all entries ---
        tier2_entries: list[tuple[ManifestEntry, TierDecision, SourceFile]] = []
        tier1_entries: list[tuple[ManifestEntry, TierDecision, SourceFile]] = []
        tier3_entries: list[tuple[ManifestEntry, TierDecision, SourceFile]] = []

        for entry in self.manifest.files:
            if self.resume and existing_status.get(entry.id) == FileStatus.DONE:
                logger.info("Skipping (already done): %s", entry.name)
                self.statuses[entry.id] = {
                    "status": FileStatus.DONE.value,
                    "skipped_reason": "resume",
                }
                summary["skipped"] += 1
                continue

            if not entry.path.exists():
                logger.warning("File not found: %s", entry.path)
                self.statuses[entry.id] = {
                    "status": FileStatus.ERROR.value,
                    "error": f"File not found: {entry.path}",
                }
                summary["error"] += 1
                continue

            source_file = _source_file_from_entry(entry)
            decision = route_tier(source_file, force_tier=self.force_tier)
            logger.info("Tier %d: %s (%s)", decision.tier.value, entry.name, decision.reason)

            if decision.tier == Tier.LOCAL:
                tier1_entries.append((entry, decision, source_file))
            elif decision.tier == Tier.TEXT_AI:
                tier2_entries.append((entry, decision, source_file))
            else:
                tier3_entries.append((entry, decision, source_file))

        logger.info(
            "Routing: %d local, %d batch (text-AI), %d synchronous (multimodal)",
            len(tier1_entries),
            len(tier2_entries),
            len(tier3_entries),
        )

        # --- Phase 2: Process Tier 1 locally (free, instant) ---
        for entry, decision, source_file in tier1_entries:
            try:
                result = extract_local(source_file, decision.text_result)
                self._write_output(entry, result, source_file, output_dir, correlate_files, build_package)
                self.statuses[entry.id] = {
                    "status": FileStatus.DONE.value,
                    "processed_at": datetime.now().isoformat(),
                    "tier": 1,
                }
                summary["done"] += 1
                summary["tiers"][1] += 1
                logger.info("  [Tier 1] Done: %s", entry.name)
            except Exception as e:
                logger.error("  [Tier 1] Error %s: %s", entry.name, e)
                self.statuses[entry.id] = {
                    "status": FileStatus.ERROR.value,
                    "error": str(e),
                }
                summary["error"] += 1
            save_status(output_dir, self.statuses)

        # --- Phase 3: Batch Tier 2 via Gemini Batch API ---
        if tier2_entries:
            batch_results = self._run_batch_tier2(
                tier2_entries,
                poll_interval,
                timeout,
            )
            for entry, decision, source_file in tier2_entries:
                response_text = batch_results.get(entry.id)
                if response_text is None:
                    logger.error("  [Batch] No result for %s", entry.name)
                    self.statuses[entry.id] = {
                        "status": FileStatus.ERROR.value,
                        "error": "No response in batch results",
                    }
                    summary["error"] += 1
                    save_status(output_dir, self.statuses)
                    continue

                try:
                    data = parse_llm_json(response_text)
                    pp = post_process_extraction(
                        raw_result=data,
                        source_tool="knowledge-extractor",
                        source_file=str(entry.path),
                        client=entry.client,
                        project=entry.project,
                    )
                    result = _result_from_json(pp.data, source_file, tokens=0)
                    result.links_line = pp.links_line
                    result.validation_result = pp.validation_result.value

                    if entry.client:
                        result.client = entry.client
                    if entry.project:
                        result.project = entry.project

                    self._write_output(entry, result, source_file, output_dir, correlate_files, build_package)
                    self.statuses[entry.id] = {
                        "status": FileStatus.DONE.value,
                        "processed_at": datetime.now().isoformat(),
                        "tier": 2,
                        "batch": True,
                    }
                    summary["done"] += 1
                    summary["tiers"][2] += 1
                    summary["cost"] += TIER_COSTS[Tier.TEXT_AI] * 0.5  # 50% discount
                    logger.info("  [Batch] Done: %s", entry.name)

                except Exception as e:
                    logger.error("  [Batch] Error processing %s: %s", entry.name, e, exc_info=True)
                    self.statuses[entry.id] = {
                        "status": FileStatus.ERROR.value,
                        "error": str(e),
                    }
                    summary["error"] += 1

                save_status(output_dir, self.statuses)

        # --- Phase 4: Synchronous Tier 3 fallback ---
        for entry, decision, source_file in tier3_entries:
            try:
                logger.info("  [Tier 3] Synchronous: %s", entry.name)

                # Compress if needed
                pkg_dir = output_dir / entry.id
                if source_file.type == FileType.VIDEO:
                    if needs_compression(source_file.path, self.config):
                        compressed_out = pkg_dir / "source" / "video" / source_file.path.name
                        compress_video(source_file.path, compressed_out, self.config)

                # Sample frames for video
                sampled_frames = None
                if source_file.type == FileType.VIDEO:
                    temp_dir = pkg_dir / "temp_frames" / source_file.name
                    sampled_frames = sample_frames(source_file.path, temp_dir, self.config)

                result = extract_knowledge(source_file, self.config, sampled_frames=sampled_frames, user_context=entry.user_context)

                # Keep slide frames
                if result.slides and sampled_frames:
                    output_frames_dir = pkg_dir / "source" / "frames"
                    keep_slide_frames(sampled_frames, result.slides, output_frames_dir, self.config)
                elif sampled_frames:
                    if self.config.get("frame_sampling", {}).get("cleanup_non_slides", True):
                        for sf in sampled_frames:
                            if sf.path.exists():
                                sf.path.unlink()

                if entry.client:
                    result.client = entry.client
                if entry.project:
                    result.project = entry.project

                self._write_output(entry, result, source_file, output_dir, correlate_files, build_package)
                self.statuses[entry.id] = {
                    "status": FileStatus.DONE.value,
                    "processed_at": datetime.now().isoformat(),
                    "tier": 3,
                }
                summary["done"] += 1
                summary["tiers"][3] += 1
                summary["cost"] += TIER_COSTS[Tier.MULTIMODAL]
                logger.info("  [Tier 3] Done: %s", entry.name)

            except Exception as e:
                logger.error("  [Tier 3] Error %s: %s", entry.name, e, exc_info=True)
                self.statuses[entry.id] = {
                    "status": FileStatus.ERROR.value,
                    "error": str(e),
                }
                summary["error"] += 1
            save_status(output_dir, self.statuses)

        logger.info(
            "Batch complete: %d done, %d errors, %d skipped | cost: $%.4f | tiers: %s",
            summary["done"],
            summary["error"],
            summary["skipped"],
            summary["cost"],
            summary["tiers"],
        )
        return summary

    def _run_batch_tier2(
        self,
        entries: list[tuple[ManifestEntry, TierDecision, SourceFile]],
        poll_interval: int,
        timeout: int,
    ) -> dict[str, str]:
        """Submit Tier 2 entries as batch, return {entry_id: response_text}."""
        client = _get_client(self.config)
        model = _get_model(self.config)

        # Build JSONL
        output_dir = self.manifest.output_dir
        jsonl_path = output_dir / ".batch" / "batch_input.jsonl"
        build_batch_jsonl(
            [(e, d) for e, d, _ in entries],
            self.config,
            jsonl_path,
        )

        # Submit
        display_name = f"cke-{self.manifest.project}-{int(time.time())}"
        job = submit_batch_job(client, jsonl_path, model, display_name)

        logger.info(
            "Batch job submitted: %s (%d requests). Polling every %ds (timeout: %ds)...",
            job.name,
            len(entries),
            poll_interval,
            timeout,
        )

        # Poll
        completed_job = poll_batch_job(client, job.name, poll_interval, timeout)

        # Parse results
        return parse_batch_results(client, completed_job)

    def _write_output(
        self,
        entry: ManifestEntry,
        result: ExtractionResult,
        source_file: SourceFile,
        output_dir: Path,
        correlate_files,
        build_package,
    ):
        """Write package output for a single entry (same as synchronous path)."""
        extracts = {source_file.name: result}
        groups = correlate_files([source_file], extracts)
        build_package(groups, extracts, output_dir, entry.id, self.config)
        self._write_extract_json(entry, result, output_dir / entry.id)

    def _write_extract_json(self, entry: ManifestEntry, result: ExtractionResult, pkg_dir: Path):
        """Write machine-readable extract.json."""
        pp = post_process_extraction(
            raw_result=dict(result.raw_json),
            source_tool="knowledge-extractor",
            source_file=str(entry.path).replace("\\", "/"),
            client=entry.client,
            project=entry.project,
        )

        extract_dir = pkg_dir / "extract"
        extract_dir.mkdir(parents=True, exist_ok=True)
        extract_json_path = extract_dir / "extract.json"

        with open(extract_json_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "schema_version": 2,
                    "id": entry.id,
                    "source_file": str(entry.path).replace("\\", "/"),
                    "doc_type": entry.doc_type,
                    "project": entry.project or self.manifest.project,
                    "client": pp.data.get("client"),
                    "title": result.title,
                    "summary": result.summary,
                    "topics": pp.data.get("topics", []),
                    "products": pp.data.get("products", []),
                    "people": pp.data.get("people", []),
                    "domains": pp.data.get("domains", []),
                    "key_points": result.key_points,
                    "slides_count": len(result.slides),
                    "links_line": pp.links_line,
                    "validation_result": pp.validation_result.value,
                    "unknown_terms": pp.unknown_terms,
                    "source_date": result.source_date,
                    "facts": result.facts,
                    "batch_api": True,
                    "processed_at": datetime.now().isoformat(),
                },
                f,
                indent=2,
                ensure_ascii=False,
                default=str,
            )
