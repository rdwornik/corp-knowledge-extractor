"""
Batch video compression for multiple files.

Processes all videos in a directory, with progress tracking and error handling.

Usage:
    # Compress all videos in data/input/
    python scripts/batch_compress.py data/input/

    # Compress with custom settings
    python scripts/batch_compress.py data/input/ --crf 23 --preset fast

    # Output to different directory
    python scripts/batch_compress.py data/input/ --output-dir data/compressed/

    # Skip files that already have compressed versions
    python scripts/batch_compress.py data/input/ --skip-existing

    # Dry run (show what would be compressed)
    python scripts/batch_compress.py data/input/ --dry-run
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime
import json

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from compress_video import VideoCompressor
from config.config_loader import get


class BatchCompressor:
    """Batch video compression with progress tracking."""

    def __init__(
        self,
        input_dir: str,
        output_dir: str = None,
        crf: int = 28,
        preset: str = "medium",
        resolution: str = "1280x720",
        audio_bitrate: str = "96k",
        skip_existing: bool = False
    ):
        """
        Initialize batch compressor.

        Args:
            input_dir: Directory containing videos to compress
            output_dir: Directory for compressed videos (default: same as input)
            crf: Quality factor (18-28)
            preset: Encoding speed preset
            resolution: Target resolution
            audio_bitrate: Audio bitrate
            skip_existing: Skip files that already have compressed versions
        """
        self.input_dir = input_dir
        self.output_dir = output_dir or input_dir
        self.crf = crf
        self.preset = preset
        self.resolution = resolution
        self.audio_bitrate = audio_bitrate
        self.skip_existing = skip_existing

        self.compressor = VideoCompressor()

        # Create output directory if needed
        os.makedirs(self.output_dir, exist_ok=True)

    def find_videos(self) -> list:
        """
        Find all video files in input directory.

        Returns:
            List of video file paths
        """
        video_extensions = tuple(get(
            "settings",
            "input.video_extensions",
            [".mp4", ".mkv", ".avi", ".mov"]
        ))

        videos = []
        for filename in os.listdir(self.input_dir):
            if filename.lower().endswith(video_extensions):
                videos.append(os.path.join(self.input_dir, filename))

        return sorted(videos)

    def get_output_path(self, input_path: str) -> str:
        """
        Generate output path for compressed video.

        Args:
            input_path: Input video file path

        Returns:
            Output file path
        """
        filename = os.path.basename(input_path)
        base, ext = os.path.splitext(filename)

        # Add _compressed suffix if output dir is same as input dir
        if os.path.abspath(self.output_dir) == os.path.abspath(self.input_dir):
            output_filename = f"{base}_compressed{ext}"
        else:
            output_filename = filename

        return os.path.join(self.output_dir, output_filename)

    def should_skip(self, input_path: str, output_path: str) -> bool:
        """
        Determine if file should be skipped.

        Args:
            input_path: Input video file
            output_path: Output video file

        Returns:
            True if should skip, False otherwise
        """
        if not self.skip_existing:
            return False

        # Skip if output already exists
        if os.path.exists(output_path):
            print(f"  ⊘ Skipping (already exists): {os.path.basename(input_path)}")
            return True

        return False

    def compress_batch(self, dry_run: bool = False) -> dict:
        """
        Compress all videos in batch.

        Args:
            dry_run: If True, only show what would be compressed

        Returns:
            Summary statistics
        """
        videos = self.find_videos()

        if not videos:
            print(f"No videos found in {self.input_dir}")
            return {}

        print(f"{'='*60}")
        print(f"BATCH VIDEO COMPRESSION")
        print(f"{'='*60}")
        print(f"Input directory:  {self.input_dir}")
        print(f"Output directory: {self.output_dir}")
        print(f"Files found:      {len(videos)}")
        print(f"CRF:              {self.crf}")
        print(f"Preset:           {self.preset}")
        print(f"Resolution:       {self.resolution}")
        print(f"Skip existing:    {self.skip_existing}")
        print(f"{'='*60}\n")

        if dry_run:
            print("DRY RUN MODE - No files will be compressed\n")

        stats = {
            "total_files": len(videos),
            "compressed": 0,
            "skipped": 0,
            "failed": 0,
            "total_size_before_mb": 0,
            "total_size_after_mb": 0,
            "start_time": datetime.now(),
            "results": []
        }

        for i, video_path in enumerate(videos, 1):
            filename = os.path.basename(video_path)
            output_path = self.get_output_path(video_path)

            print(f"[{i}/{len(videos)}] {filename}")

            # Check if should skip
            if self.should_skip(video_path, output_path):
                stats["skipped"] += 1
                stats["results"].append({
                    "file": filename,
                    "status": "skipped",
                    "reason": "already exists"
                })
                print()
                continue

            if dry_run:
                print(f"  → Would compress to: {os.path.basename(output_path)}")
                print()
                continue

            try:
                # Get input size
                input_info = self.compressor.get_video_info(video_path)
                input_size = input_info.get("size_mb", 0)
                stats["total_size_before_mb"] += input_size

                # Compress
                self.compressor.compress_video(
                    video_path,
                    output_path,
                    crf=self.crf,
                    preset=self.preset,
                    resolution=self.resolution,
                    audio_bitrate=self.audio_bitrate
                )

                # Get output size
                output_info = self.compressor.get_video_info(output_path)
                output_size = output_info.get("size_mb", 0)
                stats["total_size_after_mb"] += output_size

                stats["compressed"] += 1
                stats["results"].append({
                    "file": filename,
                    "status": "success",
                    "input_size_mb": input_size,
                    "output_size_mb": output_size,
                    "reduction_pct": round((1 - output_size/max(input_size, 0.1)) * 100, 1)
                })

            except Exception as e:
                print(f"  ✗ Error: {e}")
                stats["failed"] += 1
                stats["results"].append({
                    "file": filename,
                    "status": "failed",
                    "error": str(e)
                })

            print()

        stats["end_time"] = datetime.now()
        stats["duration_seconds"] = (stats["end_time"] - stats["start_time"]).total_seconds()

        return stats

    def print_summary(self, stats: dict):
        """
        Print compression summary.

        Args:
            stats: Statistics from compress_batch()
        """
        print(f"\n{'='*60}")
        print("BATCH COMPRESSION SUMMARY")
        print(f"{'='*60}")
        print(f"Total files:       {stats['total_files']}")
        print(f"Compressed:        {stats['compressed']}")
        print(f"Skipped:           {stats['skipped']}")
        print(f"Failed:            {stats['failed']}")
        print()
        print(f"Total size before: {stats['total_size_before_mb']:.1f} MB")
        print(f"Total size after:  {stats['total_size_after_mb']:.1f} MB")

        if stats['total_size_before_mb'] > 0:
            reduction = (
                1 - stats['total_size_after_mb'] / stats['total_size_before_mb']
            ) * 100
            print(f"Total reduction:   {reduction:.1f}%")

        print(f"Duration:          {stats['duration_seconds']:.1f} seconds")
        print(f"{'='*60}\n")

        # Show individual results
        if stats['results']:
            print("Individual Results:")
            print(f"{'File':<40} {'Status':<12} {'Size Reduction':<15}")
            print("-" * 70)

            for result in stats['results']:
                file = result['file'][:38]
                status = result['status']

                if status == "success":
                    reduction = f"{result['reduction_pct']}%"
                    size_info = f"({result['input_size_mb']:.1f} → {result['output_size_mb']:.1f} MB)"
                    print(f"{file:<40} ✓ {status:<10} {reduction:<8} {size_info}")
                elif status == "skipped":
                    print(f"{file:<40} ⊘ {status:<10} {result.get('reason', '')}")
                else:
                    print(f"{file:<40} ✗ {status:<10} {result.get('error', '')[:30]}")

            print()

    def save_report(self, stats: dict, report_path: str = None):
        """
        Save compression report to JSON.

        Args:
            stats: Statistics from compress_batch()
            report_path: Path to save report (default: output_dir/compression_report.json)
        """
        if report_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_path = os.path.join(
                self.output_dir,
                f"compression_report_{timestamp}.json"
            )

        # Convert datetime to string for JSON serialization
        stats_copy = stats.copy()
        stats_copy["start_time"] = stats["start_time"].isoformat()
        stats_copy["end_time"] = stats["end_time"].isoformat()

        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(stats_copy, f, indent=2)

        print(f"Report saved to: {report_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Batch compress videos in directory"
    )
    parser.add_argument(
        "input_dir",
        help="Directory containing videos to compress"
    )
    parser.add_argument(
        "-o", "--output-dir",
        help="Output directory (default: same as input)"
    )
    parser.add_argument(
        "--crf",
        type=int,
        default=28,
        help="Quality factor (18-28, lower=better, default=28)"
    )
    parser.add_argument(
        "--preset",
        default="medium",
        choices=["ultrafast", "fast", "medium", "slow", "veryslow"],
        help="Encoding speed preset (default: medium)"
    )
    parser.add_argument(
        "--resolution",
        default="1280x720",
        help="Target resolution (default: 1280x720)"
    )
    parser.add_argument(
        "--audio-bitrate",
        default="96k",
        help="Audio bitrate (default: 96k)"
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip files that already have compressed versions"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be compressed without actually compressing"
    )
    parser.add_argument(
        "--save-report",
        action="store_true",
        help="Save compression report to JSON"
    )

    args = parser.parse_args()

    try:
        batch = BatchCompressor(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            crf=args.crf,
            preset=args.preset,
            resolution=args.resolution,
            audio_bitrate=args.audio_bitrate,
            skip_existing=args.skip_existing
        )

        stats = batch.compress_batch(dry_run=args.dry_run)

        if not args.dry_run:
            batch.print_summary(stats)

            if args.save_report:
                batch.save_report(stats)

        return 0

    except Exception as e:
        print(f"✗ Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
