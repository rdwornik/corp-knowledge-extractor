"""Tests for FFmpeg scene detection frame sampling."""

import subprocess
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from src.frames.sampler import SampledFrame
from src.frames.scene_detect import (
    scene_detect,
    _run_ffmpeg_scene_detect,
    CIRCUIT_BREAKER_MAX,
    CIRCUIT_BREAKER_TARGET,
    DYNAMIC_CAP_MAX,
    DYNAMIC_CAP_EXTRA,
    MIN_FRAMES_FLOOR,
)


def _make_ffmpeg_stderr(timestamps: list[float]) -> str:
    """Build fake ffmpeg showinfo stderr output."""
    lines = []
    for i, ts in enumerate(timestamps):
        lines.append(f"[Parsed_showinfo_1 @ 0x1234] n:{i:4d} pts:{int(ts * 90000):10d} pts_time:{ts:.6f}")
    return "\n".join(lines)


@pytest.fixture
def config():
    return {"frame_sampling": {"interval_sec": 10, "max_frames": 500, "format": "png"}}


class TestSceneDetectCreatesFrames:
    def test_scene_detect_creates_frames(self, tmp_path, config):
        """Mock ffmpeg output, verify frame list."""
        video = tmp_path / "test.mp4"
        video.touch()
        out_dir = tmp_path / "frames"

        timestamps = [1.0, 5.0, 12.0, 20.0]
        stderr = _make_ffmpeg_stderr(timestamps)

        with (
            patch("src.frames.scene_detect._run_ffmpeg_scene_detect", return_value=timestamps),
            patch("src.frames.scene_detect._get_video_duration", return_value=120.0),
            patch("src.frames.scene_detect._extract_frame_at") as mock_extract,
            patch("src.frames.scene_detect._histogram_correlation", return_value=0.5),
        ):
            # Make extract_frame_at create dummy files
            def fake_extract(vp, ts, out):
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_bytes(b"PNG_FAKE")
                return True
            mock_extract.side_effect = fake_extract

            frames = scene_detect(video, out_dir, config)

        assert len(frames) == 4
        assert all(isinstance(f, SampledFrame) for f in frames)
        assert frames[0].timestamp_sec == 1.0
        assert frames[-1].timestamp_sec == 20.0


class TestSceneDetectFloor:
    def test_scene_detect_floor_8(self, tmp_path, config):
        """Video > 10min with 3 detected scenes → supplemented to >= 8."""
        video = tmp_path / "long.mp4"
        video.touch()
        out_dir = tmp_path / "frames"

        timestamps = [10.0, 200.0, 400.0]  # Only 3 scenes

        with (
            patch("src.frames.scene_detect._run_ffmpeg_scene_detect", return_value=timestamps),
            patch("src.frames.scene_detect._get_video_duration", return_value=900.0),  # 15 min
            patch("src.frames.scene_detect._extract_frame_at") as mock_extract,
            patch("src.frames.scene_detect._histogram_correlation", return_value=0.3),
        ):
            def fake_extract(vp, ts, out):
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_bytes(b"PNG_FAKE")
                return True
            mock_extract.side_effect = fake_extract

            frames = scene_detect(video, out_dir, config)

        assert len(frames) >= MIN_FRAMES_FLOOR


class TestSceneDetectCircuitBreaker:
    def test_scene_detect_circuit_breaker_60(self, tmp_path, config):
        """100 detected scenes → capped at 35."""
        video = tmp_path / "busy.mp4"
        video.touch()
        out_dir = tmp_path / "frames"

        timestamps = [float(i) for i in range(100)]

        with (
            patch("src.frames.scene_detect._run_ffmpeg_scene_detect", return_value=timestamps),
            patch("src.frames.scene_detect._get_video_duration", return_value=300.0),
            patch("src.frames.scene_detect._extract_frame_at") as mock_extract,
            patch("src.frames.scene_detect._histogram_correlation", return_value=0.3),
        ):
            def fake_extract(vp, ts, out):
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_bytes(b"PNG_FAKE")
                return True
            mock_extract.side_effect = fake_extract

            frames = scene_detect(video, out_dir, config)

        assert len(frames) <= DYNAMIC_CAP_MAX


class TestSceneDetectDedup:
    def test_scene_detect_dedup(self, tmp_path, config):
        """Consecutive similar frames removed."""
        video = tmp_path / "dup.mp4"
        video.touch()
        out_dir = tmp_path / "frames"

        timestamps = [1.0, 2.0, 3.0, 10.0]

        call_count = [0]

        def fake_corr(a, b):
            call_count[0] += 1
            # Frame at 2.0 is similar to 1.0 (dup), 3.0 is similar to 1.0 (dup), 10.0 is different
            if call_count[0] <= 2:
                return 0.98  # duplicate
            return 0.3  # different

        with (
            patch("src.frames.scene_detect._run_ffmpeg_scene_detect", return_value=timestamps),
            patch("src.frames.scene_detect._get_video_duration", return_value=60.0),
            patch("src.frames.scene_detect._extract_frame_at") as mock_extract,
            patch("src.frames.scene_detect._histogram_correlation", side_effect=fake_corr),
        ):
            def fake_extract(vp, ts, out):
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_bytes(b"PNG_FAKE")
                return True
            mock_extract.side_effect = fake_extract

            frames = scene_detect(video, out_dir, config)

        # 4 candidates, 2 deduped → 2 unique
        assert len(frames) == 2


class TestSceneDetectDynamicCap:
    def test_scene_detect_dynamic_cap(self, tmp_path, config):
        """20 unique → cap at min(20+5, 35) = 25."""
        video = tmp_path / "many.mp4"
        video.touch()
        out_dir = tmp_path / "frames"

        timestamps = [float(i * 10) for i in range(20)]

        with (
            patch("src.frames.scene_detect._run_ffmpeg_scene_detect", return_value=timestamps),
            patch("src.frames.scene_detect._get_video_duration", return_value=300.0),
            patch("src.frames.scene_detect._extract_frame_at") as mock_extract,
            patch("src.frames.scene_detect._histogram_correlation", return_value=0.3),
        ):
            def fake_extract(vp, ts, out):
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_bytes(b"PNG_FAKE")
                return True
            mock_extract.side_effect = fake_extract

            frames = scene_detect(video, out_dir, config)

        # 20 unique, cap = min(20+5, 35) = 25. 20 < 25, so all kept.
        assert len(frames) == 20


class TestSceneDetectFallback:
    def test_scene_detect_ffmpeg_fallback(self, tmp_path, config):
        """ffmpeg fails → falls back to old sampler."""
        video = tmp_path / "test.mp4"
        video.touch()
        out_dir = tmp_path / "frames"

        mock_frames = [SampledFrame(path=tmp_path / "f.png", index=0, timestamp_sec=0.0)]

        with (
            patch("src.frames.scene_detect._run_ffmpeg_scene_detect", side_effect=FileNotFoundError("ffmpeg not found")),
            patch("src.frames.sampler.sample_frames", return_value=mock_frames) as mock_sampler,
        ):
            frames = scene_detect(video, out_dir, config)

        mock_sampler.assert_called_once()
        assert frames == mock_frames
