"""
Input file scanning and classification.

Usage:
    from src.inventory import scan_input, FileType, SourceFile

    config = load_config()
    files = scan_input(Path("data/input"), config)
    for f in files:
        print(f.type, f.path)
"""

import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

log = logging.getLogger(__name__)


class FileType(Enum):
    VIDEO = "video"
    AUDIO = "audio"
    DOCUMENT = "document"
    SLIDES = "slides"
    SPREADSHEET = "spreadsheet"
    NOTE = "note"
    TRANSCRIPT = "transcript"
    UNKNOWN = "unknown"


@dataclass
class SourceFile:
    path: Path
    type: FileType
    size_bytes: int
    name: str  # stem without extension


def _build_ext_map(config: dict) -> dict[str, FileType]:
    """Build reverse map: extension → FileType from config['file_types']."""
    file_types = config.get("file_types") or {}
    ext_map: dict[str, FileType] = {}
    for type_name, extensions in file_types.items():
        try:
            ft = FileType(type_name)
        except ValueError:
            log.warning("Unknown file type in config: %s", type_name)
            continue
        for ext in extensions:
            ext_map[ext.lower()] = ft
    return ext_map


def _classify(path: Path, ext_map: dict[str, FileType]) -> FileType:
    """Classify a file by extension."""
    ext = path.suffix.lower()
    return ext_map.get(ext, FileType.UNKNOWN)


def scan_input(input_path: Path, config: dict) -> list[SourceFile]:
    """
    Classify a file, or recursively scan a folder.

    Extension→FileType mapping comes from config['file_types'].
    Skips hidden files/dirs (names starting with '.').
    A single file input returns a list with one item.

    Args:
        input_path: Path to file or directory
        config: Unified config dict from load_config()

    Returns:
        List of SourceFile objects, sorted by path
    """
    ext_map = _build_ext_map(config)
    results: list[SourceFile] = []

    if input_path.is_file():
        ft = _classify(input_path, ext_map)
        if ft == FileType.UNKNOWN:
            log.warning("Unknown file type: %s", input_path)
        results.append(
            SourceFile(
                path=input_path,
                type=ft,
                size_bytes=input_path.stat().st_size,
                name=input_path.stem,
            )
        )
    elif input_path.is_dir():
        for p in sorted(input_path.rglob("*")):
            # Skip hidden files and directories
            if any(part.startswith(".") for part in p.parts):
                continue
            if not p.is_file():
                continue
            ft = _classify(p, ext_map)
            if ft == FileType.UNKNOWN:
                log.warning("Unknown file type (skipping): %s", p)
                continue
            results.append(
                SourceFile(
                    path=p,
                    type=ft,
                    size_bytes=p.stat().st_size,
                    name=p.stem,
                )
            )
    else:
        raise ValueError(f"Input path does not exist: {input_path}")

    return results
