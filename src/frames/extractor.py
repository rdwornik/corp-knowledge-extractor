"""
Frame extraction using OpenCV pixel-based change detection.

Extracts frames when a significant pixel change is detected between
sampled video frames (i.e., slide transitions). Returns sequential
frame paths (frame_001.png, frame_002.png, ...) sorted by timestamp.

Usage:
    from src.frames.extractor import extract_frames
    from pathlib import Path

    frames = extract_frames(video_path, output_dir, config)
    # Returns: [Path("output_dir/frame_001.png"), ...]
"""

import logging
from pathlib import Path

import cv2

log = logging.getLogger(__name__)


def extract_frames(
    video_path: Path,
    output_dir: Path,
    config: dict,
) -> list[Path]:
    """
    Extract frames when slide changes are detected using pixel diff.

    Reads settings from config['processing']['frames'] and
    config['processing']['deduplication'].

    Args:
        video_path: Path to input video file
        output_dir: Directory to write frame PNGs to (created if missing)
        config: Unified config dict from load_config()

    Returns:
        Sorted list of Path objects (frame_001.png, frame_002.png, ...)
    """
    frames_cfg = (config.get("processing") or {}).get("frames", {})
    dedup_cfg = (config.get("processing") or {}).get("deduplication", {})

    sample_rate = frames_cfg.get("sample_rate", 1)  # seconds
    threshold = frames_cfg.get("pixel_threshold", 0.05)  # 0.0–1.0
    pixel_diff_thresh = frames_cfg.get("pixel_diff_threshold", 25)  # 0–255
    max_per_minute = frames_cfg.get("max_per_minute", 999)
    max_total = frames_cfg.get("max_total", 500)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_duration = total_frames_count / fps
    frame_interval = max(1, int(fps * sample_rate))

    log.info(
        "Extracting frames: %s (%.1fs, %.1f fps, interval=%d frames)",
        video_path.name,
        total_duration,
        fps,
        frame_interval,
    )

    # Temporary storage: list of (timestamp, grayscale_frame_data, bgr_frame_data)
    raw_frames: list[tuple[float, object, object]] = []

    prev_gray = None
    frame_count = 0
    frames_in_current_minute = 0
    current_minute = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        timestamp = frame_count / fps

        # Check max_total limit
        if len(raw_frames) >= max_total:
            log.info("Reached max_total=%d, stopping extraction at %.1fs", max_total, timestamp)
            break

        # Reset per-minute counter when minute changes
        minute_now = int(timestamp / 60)
        if minute_now > current_minute:
            current_minute = minute_now
            frames_in_current_minute = 0

        if frame_count % frame_interval == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if prev_gray is None:
                save = True
            else:
                diff = cv2.absdiff(gray, prev_gray)
                changed_ratio = (diff > pixel_diff_thresh).sum() / diff.size
                save = changed_ratio > threshold

            if save and frames_in_current_minute >= max_per_minute:
                save = False

            if save:
                raw_frames.append((timestamp, gray.copy(), frame.copy()))
                prev_gray = gray
                frames_in_current_minute += 1

        frame_count += 1

    cap.release()
    log.info("Raw frames captured: %d", len(raw_frames))

    # Deduplicate (pixel-only, no OCR)
    raw_frames = _deduplicate_frames(raw_frames, dedup_cfg)
    log.info("After deduplication: %d frames", len(raw_frames))

    # Sort by timestamp and write with sequential names
    raw_frames.sort(key=lambda x: x[0])
    result_paths: list[Path] = []

    for i, (ts, _gray, bgr) in enumerate(raw_frames):
        filename = f"frame_{i + 1:03d}.png"
        out_path = output_dir / filename
        cv2.imwrite(str(out_path), bgr)
        result_paths.append(out_path)

    log.info("Wrote %d frames to %s", len(result_paths), output_dir)
    return result_paths


def _deduplicate_frames(
    raw_frames: list[tuple[float, object, object]],
    dedup_cfg: dict,
) -> list[tuple[float, object, object]]:
    """
    Remove consecutive frames that are too visually similar (pixel-based only).

    No OCR — comparison uses downscaled grayscale pixel difference.
    """
    if len(raw_frames) <= 1:
        return raw_frames

    pixel_sim_threshold = dedup_cfg.get("pixel_similarity", 0.85)
    comparison_size = dedup_cfg.get("comparison_size", [100, 100])
    cw, ch = comparison_size[0], comparison_size[1]

    unique = [raw_frames[0]]
    prev_small = cv2.resize(raw_frames[0][1], (cw, ch))

    for ts, gray, bgr in raw_frames[1:]:
        curr_small = cv2.resize(gray, (cw, ch))
        diff = cv2.absdiff(prev_small, curr_small)
        pixel_similarity = 1.0 - (diff.sum() / (255.0 * cw * ch))

        if pixel_similarity < pixel_sim_threshold:
            unique.append((ts, gray, bgr))
            prev_small = curr_small

    return unique
