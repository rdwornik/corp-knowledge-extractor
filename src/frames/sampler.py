"""
Frame sampler — extracts frames at fixed intervals using OpenCV.

No pixel comparison. No change detection. Just dumb time-based sampling.
AI (Gemini) decides which frames are unique slides, not pixel math.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2

log = logging.getLogger(__name__)


@dataclass
class SampledFrame:
    path: Path
    index: int  # Sequential sample index (0-based), matches sample_NNNN filename
    timestamp_sec: float


def sample_frames(
    video_path: Path,
    output_dir: Path,
    config: dict,
) -> list[SampledFrame]:
    """
    Extract one frame every N seconds from a video.

    Args:
        video_path: Path to video file
        output_dir: Directory to write frame images (created if missing)
        config: Must contain config['frame_sampling']['interval_sec']

    Returns:
        List of SampledFrame objects with path and timestamp
    """
    cfg = config.get("frame_sampling", {})
    interval_sec = cfg.get("interval_sec", 10)
    max_frames = cfg.get("max_frames", 500)
    output_format = cfg.get("format", "png")

    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        log.error("Cannot open video: %s", video_path)
        return []

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames_in_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames_in_video / fps if fps > 0 else 0
    frame_interval = max(1, int(fps * interval_sec))
    expected = int(duration_sec / interval_sec) if interval_sec > 0 else 0

    log.info(
        "Sampling %s: %.0fs duration, 1 frame every %gs, ~%d frames expected",
        video_path.name,
        duration_sec,
        interval_sec,
        expected,
    )

    frames: list[SampledFrame] = []
    frame_number = 0
    sample_index = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_number % frame_interval == 0 and sample_index < max_frames:
            timestamp = frame_number / fps
            filename = f"sample_{sample_index:04d}.{output_format}"
            filepath = output_dir / filename
            cv2.imwrite(str(filepath), frame)

            frames.append(
                SampledFrame(
                    path=filepath,
                    index=sample_index,
                    timestamp_sec=timestamp,
                )
            )
            sample_index += 1

        frame_number += 1

    cap.release()
    log.info("Sampled %d frames from %s", len(frames), video_path.name)
    return frames
