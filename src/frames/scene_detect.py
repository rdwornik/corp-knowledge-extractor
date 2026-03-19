"""FFmpeg-based scene change detection for video frame sampling.

Replaces fixed-interval sampling with content-aware scene detection.
Falls back to time-based sampler if ffmpeg is unavailable.
"""

import logging
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np

from src.frames.sampler import SampledFrame

log = logging.getLogger(__name__)

# Thresholds
SCENE_THRESHOLD = 0.35
MIN_FRAMES_FLOOR = 8
FLOOR_VIDEO_MIN_DURATION = 600  # 10 minutes in seconds
FLOOR_INTERVAL_SEC = 45
CIRCUIT_BREAKER_MAX = 60
CIRCUIT_BREAKER_TARGET = 35
DEDUP_CORRELATION_THRESHOLD = 0.95
DYNAMIC_CAP_EXTRA = 5
DYNAMIC_CAP_MAX = 35


def _run_ffmpeg_scene_detect(video_path: Path, threshold: float = SCENE_THRESHOLD) -> list[float]:
    """Run ffmpeg scene detection and return list of timestamps (seconds)."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise FileNotFoundError("ffmpeg not found on PATH")

    cmd = [
        ffmpeg,
        "-i", str(video_path),
        "-filter:v", f"select=gt(scene\\,{threshold}),showinfo",
        "-vsync", "vfr",
        "-f", "null",
        "-",
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=300,
    )

    # Parse timestamps from showinfo filter output (stderr)
    timestamps = []
    for line in result.stderr.splitlines():
        # showinfo outputs lines like: [Parsed_showinfo_1 ...] n:   0 pts:  12345 pts_time:1.234
        match = re.search(r"pts_time:\s*([\d.]+)", line)
        if match:
            timestamps.append(float(match.group(1)))

    return timestamps


def _get_video_duration(video_path: Path) -> float:
    """Get video duration in seconds using OpenCV."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 0.0
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return total / fps if fps > 0 else 0.0


def _extract_frame_at(video_path: Path, timestamp_sec: float, output_path: Path) -> bool:
    """Extract a single frame at the given timestamp using OpenCV."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return False

    cap.set(cv2.CAP_PROP_POS_MSEC, timestamp_sec * 1000)
    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        return False

    # Downscale if needed
    h, w = frame.shape[:2]
    if w > 1920 or h > 1080:
        scale = min(1920 / w, 1080 / h)
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)))

    cv2.imwrite(str(output_path), frame)
    return True


def _histogram_correlation(img_a: Path, img_b: Path) -> float:
    """Compare two images via histogram correlation. Returns 0.0-1.0."""
    a = cv2.imread(str(img_a))
    b = cv2.imread(str(img_b))
    if a is None or b is None:
        return 0.0

    hist_a = cv2.calcHist([a], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist_b = cv2.calcHist([b], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist_a, hist_a)
    cv2.normalize(hist_b, hist_b)

    return float(cv2.compareHist(hist_a, hist_b, cv2.HISTCMP_CORREL))


def scene_detect(
    video_path: Path,
    output_dir: Path,
    config: dict,
) -> list[SampledFrame]:
    """Detect scene changes and extract frames at those points.

    Algorithm:
    1. Run ffmpeg scene detection → candidate timestamps
    2. If candidates < 8 and video > 10 min: supplement with time-based floor
    3. If candidates > 60: circuit breaker, evenly sample 35
    4. Extract frames, dedup consecutive via histogram correlation (>0.95)
    5. Dynamic cap: min(unique + 5, 35)

    Falls back to time-based sampler if ffmpeg fails.
    """
    from src.frames.sampler import sample_frames

    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        timestamps = _run_ffmpeg_scene_detect(video_path)
    except (FileNotFoundError, subprocess.TimeoutExpired, subprocess.SubprocessError) as exc:
        log.warning("FFmpeg scene detection failed: %s — falling back to time-based sampler", exc)
        return sample_frames(video_path, output_dir, config)

    duration = _get_video_duration(video_path)
    log.info("Scene detection: %d candidates in %.0fs video", len(timestamps), duration)

    # Floor: supplement if too few scenes in long videos
    if len(timestamps) < MIN_FRAMES_FLOOR and duration > FLOOR_VIDEO_MIN_DURATION:
        t = 0.0
        while t < duration:
            if t not in timestamps:
                timestamps.append(t)
            t += FLOOR_INTERVAL_SEC
        timestamps = sorted(set(timestamps))
        log.info("Supplemented to %d frames (floor for long video)", len(timestamps))

    # Circuit breaker: too many scenes
    if len(timestamps) > CIRCUIT_BREAKER_MAX:
        step = len(timestamps) / CIRCUIT_BREAKER_TARGET
        timestamps = [timestamps[int(i * step)] for i in range(CIRCUIT_BREAKER_TARGET)]
        log.info("Circuit breaker: capped to %d frames", len(timestamps))

    # Extract frames
    extracted: list[SampledFrame] = []
    for i, ts in enumerate(timestamps):
        frame_path = output_dir / f"sample_{i:04d}.png"
        if _extract_frame_at(video_path, ts, frame_path):
            extracted.append(SampledFrame(path=frame_path, index=i, timestamp_sec=ts))

    if not extracted:
        log.warning("No frames extracted via scene detection — falling back to sampler")
        return sample_frames(video_path, output_dir, config)

    # Lightweight dedup: remove consecutive frames with >0.95 histogram correlation
    unique: list[SampledFrame] = [extracted[0]]
    for sf in extracted[1:]:
        corr = _histogram_correlation(unique[-1].path, sf.path)
        if corr <= DEDUP_CORRELATION_THRESHOLD:
            unique.append(sf)
        else:
            # Remove the duplicate file
            if sf.path.exists():
                sf.path.unlink()
            log.debug("Deduped frame %d (corr=%.3f with previous)", sf.index, corr)

    # Dynamic cap
    cap = min(len(unique) + DYNAMIC_CAP_EXTRA, DYNAMIC_CAP_MAX)
    if len(unique) > cap:
        # Remove excess frames beyond cap
        for sf in unique[cap:]:
            if sf.path.exists():
                sf.path.unlink()
        unique = unique[:cap]

    # Re-index
    for i, sf in enumerate(unique):
        sf.index = i

    log.info("Scene detection: %d unique frames (from %d candidates)", len(unique), len(extracted))
    return unique
