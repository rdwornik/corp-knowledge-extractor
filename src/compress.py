"""
Video compression utilities using FFmpeg.

Reads compression settings from config['compression'].

Usage:
    from src.compress import needs_compression, compress_video
    from pathlib import Path

    config = load_config()
    src = Path("video.mp4")
    dst = Path("video_compressed.mp4")

    if needs_compression(src, config):
        result = compress_video(src, dst, config)
"""

import logging
import shutil
import subprocess
from pathlib import Path

log = logging.getLogger(__name__)


def needs_compression(source: Path, config: dict) -> bool:
    """
    Return True if the file is large enough to warrant compression.

    Threshold comes from config['compression']['skip_if_under_mb'].
    Logs the reason when compression is skipped.
    """
    skip_mb = config.get("compression", {}).get("skip_if_under_mb", 50)
    size_mb = source.stat().st_size / (1024 * 1024)
    if size_mb <= skip_mb:
        log.info(
            "Skipping compression: %s is %.0fMB (threshold: %gMB)",
            source.name,
            size_mb,
            skip_mb,
        )
        return False
    return True


def compress_video(source: Path, target: Path, config: dict) -> Path:
    """
    Compress a video file using FFmpeg with settings from config.

    On FFmpeg failure: logs the error and copies the original file to target.
    Returns the path to the output file (compressed or original copy).

    Args:
        source: Input video path
        target: Output video path
        config: Unified config dict (reads config['compression'])

    Returns:
        Path to output file
    """
    cfg = config.get("compression", {})
    resolution = cfg.get("resolution", "1280:720")
    preset = cfg.get("preset", "ultrafast")
    crf = cfg.get("crf", 28)
    audio_bitrate = cfg.get("audio_bitrate", "64k")

    target.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-i",
        str(source),
        "-c:v",
        "libx264",
        "-crf",
        str(crf),
        "-preset",
        preset,
        "-vf",
        f"scale={resolution}",
        "-c:a",
        "aac",
        "-b:a",
        audio_bitrate,
        "-movflags",
        "+faststart",
        "-y",
        str(target),
    ]

    log.info("Compressing %s → %s", source.name, target.name)

    try:
        subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
        size_in = source.stat().st_size / (1024 * 1024)
        size_out = target.stat().st_size / (1024 * 1024)
        log.info(
            "Compressed %.1fMB → %.1fMB (%.0f%% reduction)",
            size_in,
            size_out,
            (1 - size_out / size_in) * 100,
        )
        return target
    except subprocess.CalledProcessError as exc:
        log.error(
            "FFmpeg compression failed for %s: %s",
            source.name,
            exc.stderr.decode(errors="replace"),
        )
        log.warning("Copying original file to %s", target)
        shutil.copy2(source, target)
        return target
    except FileNotFoundError:
        log.error("ffmpeg not found in PATH. Copying original file to %s", target)
        shutil.copy2(source, target)
        return target
