"""
Video compression CLI utility.

Wraps src/compress.py for standalone use.

Usage:
    python scripts/compress_video.py data/input/training.mp4
    python scripts/compress_video.py input.mp4 --output compressed.mp4
    python scripts/compress_video.py input.mp4 --crf 23
"""

import sys
import argparse
import subprocess
import json
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

# Global API keys (Documents/.secrets/.env)
_global_env = Path.home() / "Documents" / ".secrets" / ".env"
if _global_env.exists():
    load_dotenv(_global_env, override=False)
# Local .env (project-specific vars only)
load_dotenv(override=False)

from config.config_loader import load_config
from src.compress import compress_video, needs_compression


def get_video_info(video_path: str) -> dict:
    """Get video metadata using ffprobe."""
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v", "quiet",
                "-print_format", "json",
                "-show_format",
                "-show_streams",
                video_path,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
        data = json.loads(result.stdout)
        fmt = data.get("format", {})
        video_stream = next(
            (s for s in data.get("streams", []) if s.get("codec_type") == "video"),
            {},
        )
        return {
            "duration": float(fmt.get("duration", 0)),
            "size_mb": round(int(fmt.get("size", 0)) / (1024 * 1024), 1),
            "width": video_stream.get("width", 0),
            "height": video_stream.get("height", 0),
        }
    except Exception as e:
        print(f"Warning: Could not get video info: {e}")
        return {}


def main():
    parser = argparse.ArgumentParser(
        description="Compress training videos for knowledge extraction"
    )
    parser.add_argument("input", help="Input video file path")
    parser.add_argument("-o", "--output", help="Output file path")
    parser.add_argument(
        "--crf",
        type=int,
        default=None,
        help="Quality factor (18-28, lower=better). Overrides config.",
    )
    parser.add_argument(
        "--preset",
        default=None,
        choices=["ultrafast", "fast", "medium", "slow", "veryslow"],
        help="Encoding speed preset. Overrides config.",
    )
    parser.add_argument(
        "--resolution",
        default=None,
        help="Target resolution as W:H (e.g. 1280:720). Overrides config.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Compress even if file is below skip_if_under_mb threshold.",
    )
    args = parser.parse_args()

    source = Path(args.input)
    if not source.exists():
        print(f"Error: File not found: {source}")
        return 1

    config = load_config()

    # CLI overrides
    if args.crf is not None:
        config.setdefault("compression", {})["crf"] = args.crf
    if args.preset is not None:
        config.setdefault("compression", {})["preset"] = args.preset
    if args.resolution is not None:
        config.setdefault("compression", {})["resolution"] = args.resolution

    # Check if compression needed
    if not args.force and not needs_compression(source, config):
        skip_mb = config.get("compression", {}).get("skip_if_under_mb", 500)
        size_mb = source.stat().st_size / (1024 * 1024)
        print(f"File is {size_mb:.1f}MB (threshold: {skip_mb}MB). Use --force to compress anyway.")
        return 0

    # Build output path
    if args.output:
        target = Path(args.output)
    else:
        target = source.parent / f"{source.stem}_compressed{source.suffix}"

    # Show input info
    info = get_video_info(str(source))
    if info:
        print(f"Input:  {source.name}")
        print(f"  Size:       {info.get('size_mb', '?')} MB")
        print(f"  Duration:   {info.get('duration', '?'):.0f}s")
        print(f"  Resolution: {info.get('width', '?')}x{info.get('height', '?')}")
        print()

    result = compress_video(source, target, config)

    # Show output info
    out_info = get_video_info(str(result))
    if out_info and info:
        reduction = (1 - out_info["size_mb"] / max(info["size_mb"], 0.1)) * 100
        print(f"\nOutput: {result}")
        print(f"  Size:       {out_info.get('size_mb', '?')} MB")
        print(f"  Reduction:  {reduction:.0f}%")

    print("\n✓ Done!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
