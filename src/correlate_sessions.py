"""Detect correlated PPTX+MP4 file pairs for session merging.

Stage 1: Pre-extraction filename heuristics (fast, no API).
Stage 2: Post-extraction metadata confirmation (composite 2-of-N scoring).
"""

import logging
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path

from src.utils import normalize_string_list

logger = logging.getLogger(__name__)


def word_jaccard(a: str, b: str) -> float:
    """Compute word-level Jaccard similarity: |intersection| / |union|."""
    words_a = set(a.lower().split())
    words_b = set(b.lower().split())
    if not words_a and not words_b:
        return 1.0
    if not words_a or not words_b:
        return 0.0
    intersection = words_a & words_b
    union = words_a | words_b
    return len(intersection) / len(union)

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
    filename_sim_threshold: float = 0.65,
    title_jaccard_threshold: float = 0.50,
    slide_tolerance: int = 3,
    slide_pct_tolerance: float = 0.25,
    topic_overlap_threshold: float = 0.60,
) -> CorrelationCandidate:
    """Stage 2: Confirm correlation using composite 2-of-N scoring.

    Signals evaluated:
    1. filename_similarity > 0.65 (from Stage 1)
    2. title_jaccard > 0.50 (word-level Jaccard of extracted titles)
    3. slide_count within ±3 or ±25%
    4. topic_overlap > 0.60 (Jaccard of topics lists)

    Merge if 2+ signals pass; otherwise crosslink.
    """
    signals_passed: list[str] = []

    # Signal 1: filename similarity (already computed in Stage 1)
    if candidate.filename_similarity > filename_sim_threshold:
        signals_passed.append(f"filename_sim={candidate.filename_similarity:.2f}")

    # Signal 2: title Jaccard
    pptx_title = pptx_extraction.get("title") or ""
    video_title = video_extraction.get("title") or ""
    title_jac = word_jaccard(pptx_title, video_title)
    if title_jac > title_jaccard_threshold:
        signals_passed.append(f"title_jaccard={title_jac:.2f}")

    # Signal 3: slide count alignment
    pptx_slides = pptx_extraction.get("slide_count", 0)
    video_slides = len(video_extraction.get("slides", []))
    if pptx_slides and video_slides:
        abs_diff = abs(pptx_slides - video_slides)
        max_slides = max(pptx_slides, video_slides)
        pct_diff = abs_diff / max_slides if max_slides else 1.0
        if abs_diff <= slide_tolerance or pct_diff <= slide_pct_tolerance:
            signals_passed.append(f"slide_count={pptx_slides}vs{video_slides}")

    # Signal 4: topic overlap
    pptx_topics = set(t.lower() for t in normalize_string_list(pptx_extraction.get("topics") or []))
    video_topics = set(t.lower() for t in normalize_string_list(video_extraction.get("topics") or []))
    if pptx_topics and video_topics:
        topic_union = pptx_topics | video_topics
        topic_inter = pptx_topics & video_topics
        topic_jac = len(topic_inter) / len(topic_union)
        if topic_jac > topic_overlap_threshold:
            signals_passed.append(f"topic_overlap={topic_jac:.2f}")

    confirmed = len(signals_passed) >= 2

    if confirmed:
        combined_confidence = min(
            100, int((candidate.stage1_confidence + len(signals_passed) * 25) / 2)
        )
        candidate.stage2_confirmed = True
        candidate.stage2_confidence = combined_confidence
        candidate.merge_decision = "merge"
        logger.info(
            "Stage 2 CONFIRMED: %s <-> %s (%d signals: %s, confidence=%d)",
            candidate.pptx_path.name,
            candidate.video_path.name,
            len(signals_passed),
            ", ".join(signals_passed),
            combined_confidence,
        )
    else:
        candidate.stage2_confirmed = False
        candidate.stage2_confidence = candidate.stage1_confidence * 0.5
        candidate.merge_decision = "crosslink"
        logger.warning(
            "Stage 2 REJECTED merge: %s <-> %s (%d signal(s): %s). Emitting as crosslink.",
            candidate.pptx_path.name,
            candidate.video_path.name,
            len(signals_passed),
            ", ".join(signals_passed) if signals_passed else "none",
        )

    return candidate
