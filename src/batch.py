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

    def __init__(self, manifest: Manifest, config: dict, max_rpm: int = 100, resume: bool = False):
        self.manifest = manifest
        self.config = config
        self.max_rpm = max_rpm
        self.resume = resume
        self.statuses: dict[str, dict] = {}
        self._last_request_time = 0.0
        self._min_interval = 60.0 / max_rpm

    def process_all(self) -> dict:
        """Process all files in manifest. Returns summary dict."""
        from src.inventory import SourceFile, FileType, scan_input
        from src.extract import extract_knowledge, ExtractionError
        from src.correlate import correlate_files
        from src.synthesize import build_package
        from src.frames.sampler import sample_frames
        from src.compress import needs_compression, compress_video

        output_dir = self.manifest.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        # Load existing status for resume
        existing_status = {}
        if self.resume:
            existing_status = load_status(output_dir)

        summary = {"total": len(self.manifest.files), "done": 0, "error": 0, "skipped": 0}

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
                result = self._process_single(
                    entry, output_dir, FileType, SourceFile,
                    extract_knowledge, correlate_files, build_package,
                    sample_frames, needs_compression, compress_video,
                )
                self.statuses[entry.id] = {
                    "status": FileStatus.DONE.value,
                    "processed_at": datetime.now().isoformat(),
                }
                summary["done"] += 1
                logger.info("  Done: %s", entry.name)

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
            "Batch complete: %d done, %d errors, %d skipped",
            summary["done"], summary["error"], summary["skipped"],
        )
        return summary

    def _process_single(
        self,
        entry: ManifestEntry,
        output_dir: Path,
        FileType,
        SourceFile,
        extract_knowledge,
        correlate_files,
        build_package,
        sample_frames,
        needs_compression,
        compress_video,
    ) -> None:
        """Process a single manifest entry through the full pipeline."""
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

        # Compress video if needed
        if file_type == FileType.VIDEO and needs_compression(source_file.path, self.config):
            compressed_out = pkg_dir / "source" / "video" / source_file.path.name
            compress_video(source_file.path, compressed_out, self.config)

        # Sample frames for video
        sampled_frames = None
        if file_type == FileType.VIDEO:
            temp_dir = pkg_dir / "temp_frames" / source_file.name
            sampled_frames = sample_frames(source_file.path, temp_dir, self.config)
            logger.info("  Sampled %d frames", len(sampled_frames))

        # Extract knowledge via Gemini
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

        # Correlate (single file → single group)
        extracts = {source_file.name: result}
        groups = correlate_files([source_file], extracts)

        # Build package (markdown output)
        build_package(groups, extracts, output_dir, entry.id, self.config)

        # Write machine-readable extract.json
        self._write_extract_json(entry, result, pkg_dir)

    def _write_extract_json(self, entry: ManifestEntry, result, pkg_dir: Path):
        """Write machine-readable extract.json for CPE consumption."""
        from src.post_process import post_process_extraction

        pp = post_process_extraction(
            raw_result=dict(result.raw_json),
            source_tool="knowledge-extractor",
            source_file=str(entry.path),
        )

        extract_dir = pkg_dir / "extract"
        extract_dir.mkdir(parents=True, exist_ok=True)
        extract_json_path = extract_dir / "extract.json"

        with open(extract_json_path, "w", encoding="utf-8") as f:
            json.dump({
                "schema_version": 1,
                "id": entry.id,
                "source_file": str(entry.path),
                "doc_type": entry.doc_type,
                "project": self.manifest.project,
                "title": result.title,
                "summary": result.summary,
                "topics": pp.data.get("topics", []),
                "products": pp.data.get("products", []),
                "people": pp.data.get("people", []),
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
