"""Freshness tracking fields for extraction output.

Computes source_path, source_hash, source_mtime, extracted_at for each
extracted document to enable change detection and re-extraction decisions.
"""

import hashlib
from datetime import datetime
from pathlib import Path


def compute_source_hash(filepath: Path) -> str:
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_freshness_fields(filepath: Path) -> dict:
    """Compute all freshness tracking fields for a source file.

    Returns dict with source_path, source_hash, source_mtime, extracted_at.
    """
    return {
        "source_path": str(filepath).replace("\\", "/"),
        "source_hash": compute_source_hash(filepath),
        "source_mtime": datetime.fromtimestamp(filepath.stat().st_mtime).isoformat(),
        "extracted_at": datetime.now().isoformat(),
    }
