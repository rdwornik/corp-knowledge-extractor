"""
Batch processor for manifest-driven extraction.
Initializes Gemini client once, processes all files sequentially,
tracks status per file, supports resume.
"""
import json
import shutil
import time
import logging
from datetime import datetime
from pathlib import Path

from src.manifest import Manifest, ManifestEntry, FileStatus, load_status, save_status

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


class BatchProcessor:
    """Process a manifest of files through the extraction pipeline."""

    def __init__(self, manifest: Manifest, config: dict, max_rpm: int = 100,
                 resume: bool = False, force_tier: int | None = None):
        self.manifest = manifest
        self.config = config
        self.max_rpm = max_rpm
        self.resume = resume
        self.force_tier = force_tier
        self.statuses: dict[str, dict] = {}
        self._last_request_time = 0.0
        self._min_interval = 60.0 / max_rpm
        self._cost_total = 0.0

    def process_all(self) -> dict:
        """Process all files in manifest. Returns summary dict."""
        from src.inventory import SourceFile, FileType, scan_input
        from src.extract import extract_knowledge, extract_from_text, extract_local, ExtractionError
        from src.correlate import correlate_files
        from src.synthesize import build_package
        from src.frames.sampler import sample_frames
        from src.compress import needs_compression, compress_video
        from src.tier_router import route_tier, Tier

        output_dir = self.manifest.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        # Load existing status for resume
        existing_status = {}
        if self.resume:
            existing_status = load_status(output_dir)

        summary = {"total": len(self.manifest.files), "done": 0, "error": 0, "skipped": 0,
                   "cost": 0.0, "tiers": {1: 0, 2: 0, 3: 0}}

        for i, entry in enumerate(self.manifest.files, 1):
            logger.info("[%d/%d] Processing: %s", i, summary["total"], entry.name)

            # Skip if already done (resume mode)
            if self.resume and existing_status.get(entry.id) == FileStatus.DONE:
                logger.info("  Skipping (already done): %s", entry.name)
                self.statuses[entry.id] = {
                    "status": FileStatus.DONE.value,
                    "skipped_reason": "resume",
                }
                summary["skipped"] += 1
                continue

            # Skip if file doesn't exist
            if not entry.path.exists():
                logger.warning("  File not found: %s", entry.path)
                self.statuses[entry.id] = {
                    "status": FileStatus.ERROR.value,
                    "error": f"File not found: {entry.path}",
                }
                summary["error"] += 1
                save_status(output_dir, self.statuses)
                continue

            # Rate limiting
            self._rate_limit()

            try:
                tier_used = self._process_single(
                    entry, output_dir, FileType, SourceFile,
                    extract_knowledge, extract_from_text, extract_local,
                    correlate_files, build_package,
                    sample_frames, needs_compression, compress_video,
                    route_tier, Tier,
                )
                self.statuses[entry.id] = {
                    "status": FileStatus.DONE.value,
                    "processed_at": datetime.now().isoformat(),
                    "tier": tier_used,
                }
                summary["done"] += 1
                summary["tiers"][tier_used] = summary["tiers"].get(tier_used, 0) + 1
                from src.tier_router import TIER_COSTS, Tier as _T
                summary["cost"] += TIER_COSTS[_T(tier_used)]
                logger.info("  Done (Tier %d): %s", tier_used, entry.name)

            except Exception as e:
                logger.error("  Error processing %s: %s", entry.name, e, exc_info=True)
                self.statuses[entry.id] = {
                    "status": FileStatus.ERROR.value,
                    "error": str(e),
                }
                summary["error"] += 1

            # Save status after each file for resume support
            save_status(output_dir, self.statuses)

        logger.info(
            "Batch complete: %d done, %d errors, %d skipped | cost: $%.4f | tiers: %s",
            summary["done"], summary["error"], summary["skipped"],
            summary["cost"], summary["tiers"],
        )
        return summary

    def _process_single(
        self,
        entry: ManifestEntry,
        output_dir: Path,
        FileType,
        SourceFile,
        extract_knowledge,
        extract_from_text,
        extract_local,
        correlate_files,
        build_package,
        sample_frames,
        needs_compression,
        compress_video,
        route_tier,
        Tier,
    ) -> int:
        """Process a single manifest entry through the full pipeline. Returns tier used."""
        from scripts.run import keep_slide_frames

        # Build SourceFile from manifest entry
        ft_name = _DOC_TYPE_MAP.get(entry.doc_type.lower(), "DOCUMENT")
        file_type = FileType[ft_name]
        source_file = SourceFile(
            path=entry.path,
            type=file_type,
            size_bytes=entry.path.stat().st_size,
            name=entry.path.stem,
        )

        pkg_dir = output_dir / entry.id
        output_frames_dir = pkg_dir / "source" / "frames"

        # Route to appropriate tier
        decision = route_tier(source_file, force_tier=self.force_tier)
        logger.info("  Tier %d: %s (%s)", decision.tier.value, entry.name, decision.reason)

        # Compress video if needed (Tier 3 only)
        if file_type == FileType.VIDEO and decision.tier == Tier.MULTIMODAL:
            if needs_compression(source_file.path, self.config):
                compressed_out = pkg_dir / "source" / "video" / source_file.path.name
                compress_video(source_file.path, compressed_out, self.config)

        # Sample frames for video (Tier 3 only)
        sampled_frames = None
        if file_type == FileType.VIDEO and decision.tier == Tier.MULTIMODAL:
            temp_dir = pkg_dir / "temp_frames" / source_file.name
            sampled_frames = sample_frames(source_file.path, temp_dir, self.config)
            logger.info("  Sampled %d frames", len(sampled_frames))

        # Extract knowledge via appropriate tier
        if decision.tier == Tier.LOCAL and decision.text_result:
            result = extract_local(source_file, decision.text_result)
        elif decision.tier == Tier.TEXT_AI and decision.text_result:
            result = extract_from_text(source_file, self.config, decision.text_result)
        else:
            result = extract_knowledge(source_file, self.config, sampled_frames=sampled_frames)

        # Keep slide frames, cleanup temp
        if result.slides and sampled_frames:
            keep_slide_frames(sampled_frames, result.slides, output_frames_dir, self.config)
            logger.info("  Kept %d slide frames", len(result.slides))
        elif sampled_frames:
            if self.config.get("frame_sampling", {}).get("cleanup_non_slides", True):
                for sf in sampled_frames:
                    if sf.path.exists():
                        sf.path.unlink()

        # Inject manifest client/project into result so they flow to templates
        if entry.client:
            result.client = entry.client
        if entry.project:
            result.project = entry.project

        # Correlate (single file → single group)
        extracts = {source_file.name: result}
        groups = correlate_files([source_file], extracts)

        # Build package (markdown output)
        build_package(groups, extracts, output_dir, entry.id, self.config)

        # Write machine-readable extract.json
        self._write_extract_json(entry, result, pkg_dir)

        return decision.tier.value

    def _write_extract_json(self, entry: ManifestEntry, result, pkg_dir: Path):
        """Write machine-readable extract.json for CPE consumption."""
        from src.post_process import post_process_extraction

        pp = post_process_extraction(
            raw_result=dict(result.raw_json),
            source_tool="knowledge-extractor",
            source_file=str(entry.path).replace('\\', '/'),
            client=entry.client,
            project=entry.project,
        )

        extract_dir = pkg_dir / "extract"
        extract_dir.mkdir(parents=True, exist_ok=True)
        extract_json_path = extract_dir / "extract.json"

        with open(extract_json_path, "w", encoding="utf-8") as f:
            json.dump({
                "schema_version": 2,
                "id": entry.id,
                "source_file": str(entry.path).replace('\\', '/'),
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
                "processed_at": datetime.now().isoformat(),
            }, f, indent=2, ensure_ascii=False, default=str)

    def _rate_limit(self):
        """Ensure minimum interval between requests."""
        now = time.time()
        elapsed = now - self._last_request_time
        if elapsed < self._min_interval:
            wait = self._min_interval - elapsed
            logger.debug("Rate limiting: waiting %.1fs", wait)
            time.sleep(wait)
        self._last_request_time = time.time()
