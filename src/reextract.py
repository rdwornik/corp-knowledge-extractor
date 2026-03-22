"""
Re-run extraction on an existing package without touching source files.

The source/ directory is immutable — only extract/ is regenerated.
Previous extracts are archived in .history/{date}_{model}/.

Usage:
    from src.reextract import reextract_package
    from pathlib import Path

    reextract_package(Path("output/My Meeting"), config)
"""

import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger(__name__)


def reextract_package(package_path: Path, config: dict) -> None:
    """
    Re-run extraction on an existing package.

    Steps:
    1. Validate package (must have source/ and extract/)
    2. Read existing extract/_meta.yaml for provenance
    3. Move extract/ to .history/{date}_{model}/
    4. Scan source/ files
    5. Re-run extract_knowledge() on each
    6. Re-run synthesis
    7. Write new _meta.yaml + index.md

    NOTE: Never modifies source/ — it is immutable.

    Args:
        package_path: Path to existing package directory
        config: Unified config dict from load_config()
    """
    # Inline imports to avoid circular imports at module level
    from src.inventory import scan_input
    from src.extract import extract_knowledge, ExtractionError
    from src.correlate import correlate_files
    from src.synthesize import build_package

    source_dir = package_path / "source"
    extract_dir = package_path / "extract"

    # --- Validate ---
    if not source_dir.exists():
        raise ValueError(f"Package is missing source/ directory: {package_path}")
    if not extract_dir.exists():
        raise ValueError(f"Package is missing extract/ directory: {package_path}")

    # --- Archive existing extract ---
    now = datetime.now(timezone.utc)
    model = config.get("gemini", {}).get("model", "gemini-3-flash-preview")
    archive_name = f"{now.strftime('%Y%m%d_%H%M%S')}_{model.replace('-', '_')}"
    history_dir = package_path / ".history" / archive_name
    history_dir.mkdir(parents=True, exist_ok=True)

    log.info("Archiving extract/ → .history/%s/", archive_name)
    shutil.copytree(str(extract_dir), str(history_dir / "extract"))
    shutil.rmtree(str(extract_dir))

    # --- Scan source files ---
    log.info("Scanning source files in %s...", source_dir)
    files = scan_input(source_dir, config)
    if not files:
        raise ValueError(f"No supported files found in source/: {source_dir}")
    log.info("Found %d source files", len(files))

    # --- Re-extract ---
    extracts = {}
    for f in files:
        try:
            result = extract_knowledge(f, config)
            extracts[f.name] = result
            log.info("Re-extracted: %s", f.name)
        except ExtractionError as exc:
            log.warning("Skipping %s: %s", f.name, exc)

    if not extracts:
        log.error("No files were successfully re-extracted")
        # Restore archive
        shutil.copytree(str(history_dir / "extract"), str(extract_dir))
        raise RuntimeError("Re-extraction produced no results; restored previous extract/")

    # --- Correlate and rebuild package ---
    groups = correlate_files(files, extracts)

    # Use parent as output_dir, package name from directory
    output_dir = package_path.parent
    package_name = package_path.name

    build_package(groups, extracts, output_dir, package_name, config)
    log.info("Re-extraction complete: %s", package_path)
