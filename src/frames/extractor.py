import cv2
import os
import yaml
from pathlib import Path
from config.config_loader import get, get_path


def load_preset(preset_name: str) -> dict:
    """
    Load frame extraction preset from config/presets/.

    Args:
        preset_name: Name of preset (powerpoint, excel, demo, audio_only, hybrid)

    Returns:
        Dictionary with preset configuration
    """
    preset_path = Path(__file__).parent.parent.parent / "config" / "presets" / f"{preset_name}.yaml"

    if not preset_path.exists():
        raise FileNotFoundError(f"Preset '{preset_name}' not found at {preset_path}")

    with open(preset_path, "r", encoding="utf-8") as f:
        preset = yaml.safe_load(f)

    return preset


def extract_frames(
    video_path: str,
    output_dir: str = None,
    threshold: float = None,
    sample_rate: int = None,
    preset: str = None,
    max_per_minute: int = None,
    max_total: int = None
) -> list[dict]:
    """
    Extract frames when slide changes detected.

    Args:
        video_path: Path to video file
        output_dir: Output directory for frames
        threshold: Pixel change threshold (0.0-1.0)
        sample_rate: Seconds between frame checks
        preset: Preset name (powerpoint, excel, demo, audio_only, hybrid)
        max_per_minute: Maximum frames per minute
        max_total: Maximum total frames

    Returns:
        List of frame dictionaries with timestamp and path
    """
    # Load preset if specified
    preset_config = None
    if preset:
        preset_config = load_preset(preset)
        print(f"Using preset: {preset_config.get('name', preset)}")
        print(f"  Description: {preset_config.get('description', 'N/A')}")

        # Check if frames are disabled (audio-only preset)
        if not preset_config.get("frames", {}).get("enabled", True):
            print("  Frames disabled for this preset (audio-only mode)")
            return []

    # Load defaults from preset or config
    if output_dir is None:
        output_dir = get_path("processing", "frames.output_dir")

    if threshold is None:
        if preset_config:
            threshold = preset_config.get("frames", {}).get("pixel_threshold", 0.05)
        else:
            threshold = get("processing", "frames.pixel_threshold", 0.05)

    if sample_rate is None:
        if preset_config:
            # Check for adaptive mode
            mode = preset_config.get("frames", {}).get("mode")
            if mode == "adaptive":
                sample_rate = preset_config.get("frames", {}).get("initial_sample_rate", 5)
                print(f"  Adaptive mode - starting with sample_rate={sample_rate}s")
            else:
                sample_rate = preset_config.get("frames", {}).get("sample_rate", 1)
        else:
            sample_rate = get("processing", "frames.sample_rate", 1)

    if max_per_minute is None:
        if preset_config:
            max_per_minute = preset_config.get("frames", {}).get("max_per_minute", 999)
        else:
            max_per_minute = get("processing", "frames.max_per_minute", 999)

    if max_total is None:
        if preset_config:
            max_total = preset_config.get("frames", {}).get("max_total", 999)
        else:
            max_total = get("processing", "frames.max_total", 999)

    print(f"  Sample rate: {sample_rate}s")
    print(f"  Pixel threshold: {threshold}")
    print(f"  Max per minute: {max_per_minute}")
    print(f"  Max total: {max_total}")

    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps
    frame_interval = int(fps * sample_rate)

    prev_frame = None
    saved_frames = []
    frame_count = 0
    frames_in_current_minute = 0
    current_minute_start = 0

    # Adaptive mode tracking (for hybrid preset)
    if preset_config and preset_config.get("frames", {}).get("mode") == "adaptive":
        adaptive_tracker = AdaptiveFrameTracker(preset_config, fps)
    else:
        adaptive_tracker = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current_time = frame_count / fps

        # Check max_total limit
        if len(saved_frames) >= max_total:
            print(f"  Reached max_total limit ({max_total}), stopping frame extraction")
            break

        # Check if we've moved to a new minute
        if int(current_time / 60) > current_minute_start:
            current_minute_start = int(current_time / 60)
            frames_in_current_minute = 0

        if frame_count % frame_interval == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if prev_frame is None:
                save = True
            else:
                diff = cv2.absdiff(gray, prev_frame)
                pixel_diff_threshold = get("processing", "frames.pixel_diff_threshold", 25)
                changed = (diff > pixel_diff_threshold).sum() / diff.size
                save = changed > threshold

            # Check max_per_minute limit
            if save and frames_in_current_minute >= max_per_minute:
                save = False
                # print(f"  Skipping frame at {current_time:.1f}s (max_per_minute={max_per_minute} reached)")

            if save:
                timestamp = frame_count / fps
                filename = f"frame_{timestamp:.1f}s.png"
                path = os.path.join(output_dir, filename)
                cv2.imwrite(path, frame)
                saved_frames.append({"timestamp": timestamp, "path": path})
                prev_frame = gray
                frames_in_current_minute += 1

                # Adaptive mode: track activity
                if adaptive_tracker:
                    adaptive_tracker.add_frame(timestamp)

        frame_count += 1

        # Adaptive mode: check for mode switch
        if adaptive_tracker and adaptive_tracker.should_check_switch(current_time):
            new_settings = adaptive_tracker.check_and_switch(len(saved_frames), current_time)
            if new_settings:
                threshold = new_settings.get("pixel_threshold", threshold)
                sample_rate = new_settings.get("sample_rate", sample_rate)
                frame_interval = int(fps * sample_rate)
                max_per_minute = new_settings.get("max_per_minute", max_per_minute)

    cap.release()

    print(f"  Extracted {len(saved_frames)} frames from video ({total_duration:.1f}s)")

    # Deduplicate similar frames
    if preset_config:
        dedup_config = preset_config.get("frames", {}).get("deduplication", {})
        if dedup_config.get("enabled", True):
            pixel_sim = dedup_config.get("pixel_similarity", 0.85)
            saved_frames = _deduplicate_frames(saved_frames, similarity_threshold=pixel_sim)
    else:
        saved_frames = _deduplicate_frames(saved_frames)

    # Filter out junk frames (Teams UI, waiting screens)
    saved_frames = _filter_junk_frames(saved_frames)

    # Sort by timestamp
    saved_frames = sorted(saved_frames, key=lambda x: x["timestamp"])

    print(f"  After deduplication and filtering: {len(saved_frames)} frames")

    return saved_frames


class AdaptiveFrameTracker:
    """
    Tracks frame extraction activity and adapts sampling strategy.
    Used for hybrid preset with adaptive mode.
    """

    def __init__(self, preset_config: dict, fps: float):
        """
        Initialize adaptive tracker.

        Args:
            preset_config: Preset configuration dictionary
            fps: Video frames per second
        """
        self.preset_config = preset_config
        self.fps = fps

        # Get adaptive settings
        frames_config = preset_config.get("frames", {})
        self.analysis_window = frames_config.get("analysis_window", 60)  # seconds
        adaptive_rules = frames_config.get("adaptive_rules", {})
        self.high_activity_threshold = adaptive_rules.get("high_activity_threshold", 5)
        self.low_activity_threshold = adaptive_rules.get("low_activity_threshold", 2)

        # Mode configurations
        self.modes = frames_config.get("modes", {})

        # Tracking
        self.current_mode = "powerpoint"  # Start with default
        self.frame_timestamps = []
        self.last_analysis_time = 0
        self.mode_switches = []

    def add_frame(self, timestamp: float):
        """Record that a frame was extracted at this timestamp."""
        self.frame_timestamps.append(timestamp)

    def should_check_switch(self, current_time: float) -> bool:
        """Check if it's time to analyze and potentially switch modes."""
        return current_time - self.last_analysis_time >= self.analysis_window

    def check_and_switch(self, total_frames: int, current_time: float) -> dict:
        """
        Analyze recent activity and switch mode if needed.

        Returns:
            New settings dict if mode switched, None otherwise
        """
        # Get frames in last analysis window
        window_start = current_time - self.analysis_window
        recent_frames = [
            t for t in self.frame_timestamps
            if t >= window_start
        ]

        frames_per_minute = (len(recent_frames) / self.analysis_window) * 60

        # Determine target mode
        target_mode = self.current_mode

        if frames_per_minute > self.high_activity_threshold:
            # High activity = switch to demo mode
            target_mode = "demo"
        elif frames_per_minute < self.low_activity_threshold:
            # Low activity = switch to powerpoint mode
            target_mode = "powerpoint"
        # else: stay in current mode (medium activity)

        # Switch if different
        if target_mode != self.current_mode:
            print(f"\n  [ADAPTIVE MODE] Analysis at {current_time:.1f}s:")
            print(f"    Frames in last {self.analysis_window}s: {len(recent_frames)}")
            print(f"    Activity: {frames_per_minute:.1f} frames/minute")
            print(f"    Switching: {self.current_mode} -> {target_mode}")

            self.current_mode = target_mode
            self.mode_switches.append({
                "time": current_time,
                "from_mode": self.current_mode,
                "to_mode": target_mode,
                "activity": frames_per_minute
            })

            # Return new settings
            mode_config = self.modes.get(target_mode, {})
            self.last_analysis_time = current_time

            return {
                "sample_rate": mode_config.get("sample_rate", 5),
                "pixel_threshold": mode_config.get("pixel_threshold", 0.15),
                "max_per_minute": mode_config.get("max_per_minute", 10)
            }
        else:
            # No switch, just update last analysis time
            self.last_analysis_time = current_time
            return None


def _deduplicate_frames(frames: list[dict], similarity_threshold: float = None) -> list[dict]:
    """Remove frames that are too similar (pixel or OCR text)."""
    import pytesseract
    from PIL import Image

    if similarity_threshold is None:
        similarity_threshold = get("processing", "deduplication.pixel_similarity", 0.85)

    tesseract_path = get("settings", "tools.tesseract_path")
    pytesseract.pytesseract.tesseract_cmd = tesseract_path

    if len(frames) <= 1:
        return frames

    unique = [frames[0]]
    prev_img = cv2.imread(frames[0]["path"], cv2.IMREAD_GRAYSCALE)
    prev_text = ""

    try:
        prev_text = pytesseract.image_to_string(Image.open(frames[0]["path"])).strip()
    except:
        pass

    for frame in frames[1:]:
        curr_img = cv2.imread(frame["path"], cv2.IMREAD_GRAYSCALE)
        curr_text = ""

        try:
            curr_text = pytesseract.image_to_string(Image.open(frame["path"])).strip()
        except:
            pass

        # Check pixel similarity
        comparison_size = get("processing", "deduplication.comparison_size", [100, 100])
        prev_small = cv2.resize(prev_img, tuple(comparison_size))
        curr_small = cv2.resize(curr_img, tuple(comparison_size))
        diff = cv2.absdiff(prev_small, curr_small)
        pixel_similarity = 1 - (diff.sum() / (255 * comparison_size[0] * comparison_size[1]))

        # Check OCR text similarity
        text_similarity = _text_similarity(prev_text, curr_text)
        text_sim_threshold = get("processing", "deduplication.text_similarity", 0.90)

        # Keep if BOTH are different enough
        if pixel_similarity < similarity_threshold or text_similarity < text_sim_threshold:
            unique.append(frame)
            prev_img = curr_img
            prev_text = curr_text
        else:
            os.remove(frame["path"])

    return unique


def _text_similarity(text1: str, text2: str) -> float:
    """Calculate word overlap similarity."""
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())

    if not words1 or not words2:
        return 0.0

    overlap = len(words1 & words2)
    total = len(words1 | words2)

    return overlap / total if total > 0 else 0.0

def _filter_junk_frames(frames: list[dict]) -> list[dict]:
    """Remove frames with Teams UI, waiting screens, etc."""
    import pytesseract
    from PIL import Image

    # Set tesseract path
    tesseract_path = get("settings", "tools.tesseract_path")
    pytesseract.pytesseract.tesseract_cmd = tesseract_path

    # Load junk patterns from config
    junk_patterns = get("filters", "frame_junk_patterns", [])

    filtered = []
    for frame in frames:
        try:
            img = Image.open(frame["path"])
            text = pytesseract.image_to_string(img).lower()

            is_junk = any(pattern in text for pattern in junk_patterns)

            if not is_junk:
                filtered.append(frame)
            else:
                os.remove(frame["path"])
        except:
            filtered.append(frame)  # Keep if can't read

    return filtered
