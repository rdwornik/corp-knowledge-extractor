"""
Group related input files into logical FileGroups.

Heuristics (v1):
1. If ≤3 files total → single group
2. Filename similarity: group by common stem prefix (strip trailing digits)
3. Video files are primary; other types are related
4. Ungrouped files each become solo groups

Usage:
    from src.correlate import correlate_files, FileGroup

    groups = correlate_files(files, extracts)
"""

import re
import logging
from dataclasses import dataclass, field

from src.inventory import SourceFile, FileType
from src.extract import ExtractionResult

log = logging.getLogger(__name__)


@dataclass
class FileGroup:
    primary: SourceFile
    related: list[SourceFile] = field(default_factory=list)
    group_name: str = ""


def _stem_prefix(name: str) -> str:
    """
    Reduce a filename stem to its shared prefix by stripping descriptive
    suffixes like '_slides', '_notes', '_transcript'.

    Trailing digits that are part of the identifier are preserved so that
    session1 and session2 remain distinct groups.

    Examples:
        "session1" → "session1"
        "session1_slides" → "session1"
        "session2_notes" → "session2"
        "meeting_notes" → "meeting"
        "training_2024" → "training_2024"
    """
    # Remove common descriptive suffixes (but not digits)
    name = re.sub(
        r"[_\-\s]*(slides|notes|transcript|audio|video|recording|deck|doc|docs)$",
        "",
        name,
        flags=re.IGNORECASE,
    )
    return name.lower().strip("_- ")


def _pick_primary(group_files: list[SourceFile]) -> SourceFile:
    """
    Choose the primary file for a group.
    Priority: VIDEO > AUDIO > SLIDES > DOCUMENT > others.
    Falls back to the first file in the list.
    """
    priority = [FileType.VIDEO, FileType.AUDIO, FileType.SLIDES, FileType.DOCUMENT]
    for ft in priority:
        for f in group_files:
            if f.type == ft:
                return f
    return group_files[0]


def correlate_files(
    files: list[SourceFile],
    extracts: dict[str, ExtractionResult],
) -> list[FileGroup]:
    """
    Group related files together.

    Args:
        files: All SourceFile objects from scan_input()
        extracts: Map of filename stem → ExtractionResult

    Returns:
        List of FileGroup objects
    """
    if not files:
        return []

    # Rule 1: Small input → single group
    if len(files) <= 3:
        primary = _pick_primary(files)
        related = [f for f in files if f is not primary]
        # Use title from primary file's extraction if available
        group_name = (
            extracts.get(primary.name, {})
            and getattr(extracts.get(primary.name), "title", primary.name)
            or primary.name
        )
        if hasattr(group_name, "title") and callable(group_name.title):
            group_name = group_name.title()
        log.info("Small input (%d files) → single group: %s", len(files), group_name)
        return [FileGroup(primary=primary, related=related, group_name=str(group_name))]

    # Rule 2: Group by stem prefix
    prefix_map: dict[str, list[SourceFile]] = {}
    for f in files:
        prefix = _stem_prefix(f.name)
        prefix_map.setdefault(prefix, []).append(f)

    groups: list[FileGroup] = []
    for prefix, group_files in sorted(prefix_map.items()):
        primary = _pick_primary(group_files)
        related = [f for f in group_files if f is not primary]

        # Use extraction title if available, else prefix
        extract = extracts.get(primary.name)
        group_name = extract.title if extract else prefix or primary.name

        groups.append(
            FileGroup(
                primary=primary,
                related=related,
                group_name=group_name,
            )
        )
        log.info(
            "Group '%s': primary=%s, related=%s",
            group_name,
            primary.name,
            [r.name for r in related],
        )

    return groups
