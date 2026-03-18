"""Detect correlated PPTX+MP4 file pairs for session merging.

Stage 1: Pre-extraction filename heuristics (fast, no API).
Stage 2: Post-extraction metadata confirmation (title similarity, slide count alignment).
"""

import logging
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path

logger = logging.getLogger(__name__)

VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov"}


@dataclass
class CorrelationCandidate:
    pptx_path: Path
    video_path: Path
    filename_similarity: float
    stage1_confidence: float  # 0-100
    stage2_confirmed: bool = False
    stage2_confidence: float = 0.0
    merge_decision: str = "pending"  # pending | merge | crosslink | standalone


@dataclass
class SessionGroup:
    candidates: list[CorrelationCandidate] = field(default_factory=list)
    standalone: list[Path] = field(default_factory=list)


def normalize_stem(filename: str) -> str:
    """Normalize filename for comparison: lowercase, strip common noise."""
    stem = Path(filename).stem.lower()
    for noise in ["_final", "_v2", "_v3", "_draft", "_copy", " - copy"]:
        stem = stem.replace(noise, "")
    stem = stem.replace("_", " ").replace("-", " ")
    # Collapse multiple spaces
    while "  " in stem:
        stem = stem.replace("  ", " ")
    return stem.strip()


def filename_similarity(a: str, b: str) -> float:
    """Compute normalized filename similarity (0.0 - 1.0)."""
    norm_a = normalize_stem(a)
    norm_b = normalize_stem(b)
    return SequenceMatcher(None, norm_a, norm_b).ratio()


def detect_stage1(file_paths: list[Path], threshold: float = 0.6) -> SessionGroup:
    """Stage 1: Detect correlated pairs using filename heuristics.

    Rules:
    - Must be in same folder
    - Must be extension pair (.pptx + video)
    - Normalized filename similarity > threshold

    Returns SessionGroup with candidates and standalone files.
    """
    by_folder: dict[Path, list[Path]] = {}
    for p in file_paths:
        by_folder.setdefault(p.parent, []).append(p)

    result = SessionGroup()
    matched_files: set[Path] = set()

    for folder, files in by_folder.items():
        pptx_files = [f for f in files if f.suffix.lower() == ".pptx"]
        video_files = [f for f in files if f.suffix.lower() in VIDEO_EXTENSIONS]

        # Build all (pptx, video, similarity) triples, then greedily assign 1:1
        all_pairs = []
        for pptx in pptx_files:
            for video in video_files:
                sim = filename_similarity(pptx.name, video.name)
                if sim >= threshold:
                    all_pairs.append((sim, pptx, video))

        # Sort by similarity descending — best matches first
        all_pairs.sort(key=lambda t: t[0], reverse=True)

        used_pptx: set[Path] = set()
        used_video: set[Path] = set()

        for sim, pptx, video in all_pairs:
            if pptx in used_pptx or video in used_video:
                continue

            confidence = min(100, int(sim * 100))
            candidate = CorrelationCandidate(
                pptx_path=pptx,
                video_path=video,
                filename_similarity=sim,
                stage1_confidence=confidence,
            )
            result.candidates.append(candidate)
            matched_files.add(pptx)
            matched_files.add(video)
            used_pptx.add(pptx)
            used_video.add(video)
            logger.info(
                "Stage 1 correlation: %s <-> %s (similarity=%.2f, confidence=%d)",
                pptx.name,
                video.name,
                sim,
                confidence,
            )

    for p in file_paths:
        if p not in matched_files:
            result.standalone.append(p)

    return result


def confirm_stage2(
    candidate: CorrelationCandidate,
    pptx_extraction: dict,
    video_extraction: dict,
    title_threshold: float = 0.8,
    slide_tolerance: int = 2,
) -> CorrelationCandidate:
    """Stage 2: Confirm correlation using extracted metadata.

    Confirms if:
    - Extracted titles are similar (>threshold) OR
    - Slide counts align (within tolerance)

    If neither matches: downgrade to crosslink (no merge).
    """
    pptx_title = (pptx_extraction.get("title") or "").lower()
    video_title = (video_extraction.get("title") or "").lower()
    title_sim = SequenceMatcher(None, pptx_title, video_title).ratio()

    pptx_slides = pptx_extraction.get("slide_count", 0)
    video_slides = len(video_extraction.get("slides", []))
    slide_match = (
        abs(pptx_slides - video_slides) <= slide_tolerance
        if (pptx_slides and video_slides)
        else False
    )

    confirmed = title_sim >= title_threshold or slide_match

    if confirmed:
        combined_confidence = min(
            100, int((candidate.stage1_confidence + title_sim * 100) / 2)
        )
        candidate.stage2_confirmed = True
        candidate.stage2_confidence = combined_confidence
        candidate.merge_decision = "merge"
        logger.info(
            "Stage 2 CONFIRMED: %s <-> %s (title_sim=%.2f, slide_match=%s, confidence=%d)",
            candidate.pptx_path.name,
            candidate.video_path.name,
            title_sim,
            slide_match,
            combined_confidence,
        )
    else:
        candidate.stage2_confirmed = False
        candidate.stage2_confidence = candidate.stage1_confidence * 0.5
        candidate.merge_decision = "crosslink"
        logger.warning(
            "Stage 2 REJECTED merge: %s <-> %s (title_sim=%.2f, slide_match=%s). Emitting as crosslink.",
            candidate.pptx_path.name,
            candidate.video_path.name,
            title_sim,
            slide_match,
        )

    return candidate
