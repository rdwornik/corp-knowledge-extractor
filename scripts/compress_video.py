"""
Video compression utility using FFmpeg.

Compresses large training videos (4GB → 500MB-1GB) while preserving
quality sufficient for OCR and transcription.

Usage:
    # Compress single video
    python scripts/compress_video.py data/input/training.mp4

    # Compress with custom output path
    python scripts/compress_video.py input.mp4 --output compressed.mp4

    # Audio-only extraction (for Type B meetings)
    python scripts/compress_video.py input.mp4 --audio-only

    # Custom quality (CRF 18-28, lower = better quality)
    python scripts/compress_video.py input.mp4 --crf 23

    # Verify quality after compression
    python scripts/compress_video.py input.mp4 --verify
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.config_loader import get


class VideoCompressor:
    """FFmpeg-based video compression."""

    def __init__(self):
        """Initialize compressor with FFmpeg path detection."""
        self.ffmpeg_path = self._find_ffmpeg()
        self.ffprobe_path = self._find_ffprobe()

    def _find_ffmpeg(self) -> str:
        """Find FFmpeg executable."""
        # Check common paths
        paths = [
            "ffmpeg",  # In PATH
            "C:\\ffmpeg\\bin\\ffmpeg.exe",
            "/usr/bin/ffmpeg",
            "/usr/local/bin/ffmpeg"
        ]

        for path in paths:
            try:
                subprocess.run(
                    [path, "-version"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=True
                )
                return path
            except (subprocess.CalledProcessError, FileNotFoundError):
                continue

        raise FileNotFoundError(
            "FFmpeg not found. Install from https://ffmpeg.org/download.html"
        )

    def _find_ffprobe(self) -> str:
        """Find FFprobe executable (comes with FFmpeg)."""
        # Try replacing 'ffmpeg' with 'ffprobe' in found path
        if "ffmpeg" in self.ffmpeg_path:
            return self.ffmpeg_path.replace("ffmpeg", "ffprobe")
        return "ffprobe"

    def get_video_info(self, video_path: str) -> dict:
        """
        Get video metadata using ffprobe.

        Returns:
            Dictionary with duration, size, resolution, bitrate
        """
        try:
            result = subprocess.run(
                [
                    self.ffprobe_path,
                    "-v", "quiet",
                    "-print_format", "json",
                    "-show_format",
                    "-show_streams",
                    video_path
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True
            )

            import json
            data = json.loads(result.stdout)

            # Extract relevant info
            format_info = data.get("format", {})
            video_stream = next(
                (s for s in data.get("streams", []) if s.get("codec_type") == "video"),
                {}
            )

            return {
                "duration": float(format_info.get("duration", 0)),
                "size_bytes": int(format_info.get("size", 0)),
                "size_mb": round(int(format_info.get("size", 0)) / (1024*1024), 1),
                "bitrate": int(format_info.get("bit_rate", 0)),
                "width": video_stream.get("width", 0),
                "height": video_stream.get("height", 0),
                "fps": eval(video_stream.get("r_frame_rate", "0/1"))
            }
        except Exception as e:
            print(f"Warning: Could not get video info: {e}")
            return {}

    def compress_video(
        self,
        input_path: str,
        output_path: str = None,
        crf: int = 28,
        preset: str = "medium",
        resolution: str = "1280x720",
        audio_bitrate: str = "96k"
    ) -> str:
        """
        Compress video using H.264 codec.

        Args:
            input_path: Input video file
            output_path: Output file (default: input_compressed.mp4)
            crf: Constant Rate Factor (18-28, lower=better quality)
            preset: Encoding speed preset (ultrafast, fast, medium, slow)
            resolution: Target resolution (e.g., "1280x720" for 720p)
            audio_bitrate: Audio bitrate (e.g., "96k")

        Returns:
            Path to compressed file
        """
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input file not found: {input_path}")

        # Generate output path if not provided
        if output_path is None:
            base, ext = os.path.splitext(input_path)
            output_path = f"{base}_compressed{ext}"

        # Get input info
        print(f"Analyzing input video: {input_path}")
        input_info = self.get_video_info(input_path)
        print(f"  Size: {input_info.get('size_mb', '?')} MB")
        print(f"  Duration: {input_info.get('duration', '?')} seconds")
        print(f"  Resolution: {input_info.get('width', '?')}x{input_info.get('height', '?')}")
        print()

        # Build FFmpeg command
        cmd = [
            self.ffmpeg_path,
            "-i", input_path,
            "-c:v", "libx264",           # H.264 video codec
            "-crf", str(crf),            # Quality (18-28)
            "-preset", preset,           # Encoding speed
            "-vf", f"scale={resolution}",  # Resize to target resolution
            "-c:a", "aac",               # AAC audio codec
            "-b:a", audio_bitrate,       # Audio bitrate
            "-movflags", "+faststart",   # Optimize for streaming
            "-y",                        # Overwrite output
            output_path
        ]

        print(f"Compressing video...")
        print(f"  CRF: {crf} (lower = better quality)")
        print(f"  Preset: {preset}")
        print(f"  Target resolution: {resolution}")
        print(f"  Audio bitrate: {audio_bitrate}")
        print()

        try:
            # Run compression
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True
            )

            # Get output info
            output_info = self.get_video_info(output_path)
            print(f"✓ Compression complete!")
            print(f"  Output: {output_path}")
            print(f"  Size: {output_info.get('size_mb', '?')} MB")
            print(f"  Compression ratio: {input_info.get('size_mb', 0) / max(output_info.get('size_mb', 1), 0.1):.1f}x")
            print()

            return output_path

        except subprocess.CalledProcessError as e:
            print(f"Error during compression:")
            print(e.stderr.decode())
            raise

    def extract_audio(
        self,
        input_path: str,
        output_path: str = None,
        format: str = "mp3",
        bitrate: str = "96k"
    ) -> str:
        """
        Extract audio-only from video.

        Useful for Type B meetings (client calls) where video isn't needed.

        Args:
            input_path: Input video file
            output_path: Output audio file
            format: Audio format (mp3, m4a, wav)
            bitrate: Audio bitrate

        Returns:
            Path to audio file
        """
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input file not found: {input_path}")

        # Generate output path if not provided
        if output_path is None:
            base = os.path.splitext(input_path)[0]
            output_path = f"{base}_audio.{format}"

        print(f"Extracting audio from: {input_path}")

        cmd = [
            self.ffmpeg_path,
            "-i", input_path,
            "-vn",                    # No video
            "-c:a", "libmp3lame" if format == "mp3" else "aac",
            "-b:a", bitrate,
            "-y",
            output_path
        ]

        try:
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            file_size = os.path.getsize(output_path) / (1024*1024)
            print(f"✓ Audio extracted!")
            print(f"  Output: {output_path}")
            print(f"  Size: {file_size:.1f} MB")
            print()
            return output_path

        except subprocess.CalledProcessError as e:
            print(f"Error during audio extraction:")
            print(e.stderr.decode())
            raise

    def verify_quality(self, original_path: str, compressed_path: str) -> dict:
        """
        Verify compression quality by comparing file sizes and metadata.

        Args:
            original_path: Original video file
            compressed_path: Compressed video file

        Returns:
            Dictionary with comparison metrics
        """
        print("Verifying compression quality...")

        original_info = self.get_video_info(original_path)
        compressed_info = self.get_video_info(compressed_path)

        size_reduction = (
            1 - compressed_info.get("size_mb", 0) / max(original_info.get("size_mb", 1), 0.1)
        ) * 100

        metrics = {
            "original_size_mb": original_info.get("size_mb"),
            "compressed_size_mb": compressed_info.get("size_mb"),
            "size_reduction_pct": round(size_reduction, 1),
            "original_resolution": f"{original_info.get('width')}x{original_info.get('height')}",
            "compressed_resolution": f"{compressed_info.get('width')}x{compressed_info.get('height')}",
            "duration_match": abs(
                original_info.get("duration", 0) - compressed_info.get("duration", 0)
            ) < 1.0
        }

        print(f"\n{'='*50}")
        print("COMPRESSION QUALITY REPORT")
        print(f"{'='*50}")
        print(f"Original size:     {metrics['original_size_mb']} MB")
        print(f"Compressed size:   {metrics['compressed_size_mb']} MB")
        print(f"Size reduction:    {metrics['size_reduction_pct']}%")
        print(f"Original res:      {metrics['original_resolution']}")
        print(f"Compressed res:    {metrics['compressed_resolution']}")
        print(f"Duration match:    {'✓' if metrics['duration_match'] else '✗'}")
        print(f"{'='*50}\n")

        return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Compress training videos for knowledge extraction"
    )
    parser.add_argument(
        "input",
        help="Input video file path"
    )
    parser.add_argument(
        "-o", "--output",
        help="Output file path (default: input_compressed.mp4)"
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
        "--audio-only",
        action="store_true",
        help="Extract audio only (for audio-only meetings)"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify compression quality after processing"
    )

    args = parser.parse_args()

    try:
        compressor = VideoCompressor()

        if args.audio_only:
            # Extract audio only
            output_path = compressor.extract_audio(
                args.input,
                args.output
            )
        else:
            # Compress video
            output_path = compressor.compress_video(
                args.input,
                args.output,
                crf=args.crf,
                preset=args.preset,
                resolution=args.resolution,
                audio_bitrate=args.audio_bitrate
            )

            # Verify if requested
            if args.verify:
                compressor.verify_quality(args.input, output_path)

        print("✓ Done!")
        return 0

    except Exception as e:
        print(f"✗ Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
