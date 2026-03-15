"""
Manifest schema for batch processing.
Reads JSON manifest, validates entries, tracks status.
"""

import json
import logging
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class FileStatus(str, Enum):
    PENDING = "pending"
    DONE = "done"
    ERROR = "error"
    SKIPPED = "skipped"


@dataclass
class ManifestEntry:
    id: str
    path: Path
    doc_type: str
    name: str
    client: str | None = None
    project: str | None = None
    status: FileStatus = FileStatus.PENDING
    error: str | None = None


@dataclass
class Manifest:
    schema_version: int
    project: str
    output_dir: Path
    files: list[ManifestEntry]
    config: dict = field(default_factory=dict)

    @classmethod
    def from_file(cls, path: Path) -> "Manifest":
        """Load manifest from JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if data.get("schema_version", 0) != 1:
            raise ValueError(f"Unsupported manifest schema version: {data.get('schema_version')}")

        files = []
        for entry in data.get("files", []):
            files.append(
                ManifestEntry(
                    id=entry["id"],
                    path=Path(entry["path"]),
                    doc_type=entry.get("doc_type", "document"),
                    name=entry.get("name", entry["id"]),
                    client=entry.get("client"),
                    project=entry.get("project"),
                )
            )

        return cls(
            schema_version=data["schema_version"],
            project=data.get("project", "unknown"),
            output_dir=Path(data["output_dir"]),
            files=files,
            config=data.get("config", {}),
        )


def load_status(output_dir: Path) -> dict[str, FileStatus]:
    """Load processing status from status.json in output directory."""
    status_path = output_dir / "status.json"
    if not status_path.exists():
        return {}
    with open(status_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {k: FileStatus(v["status"]) for k, v in data.items()}


def save_status(output_dir: Path, statuses: dict[str, dict]):
    """Save processing status to status.json."""
    status_path = output_dir / "status.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(status_path, "w", encoding="utf-8") as f:
        json.dump(statuses, f, indent=2, ensure_ascii=False)
