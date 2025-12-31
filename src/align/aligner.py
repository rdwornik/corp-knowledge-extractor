from config.config_loader import get


def align(transcript: list[dict], frames: list[dict]) -> list[dict]:
    """
    Match transcript segments with visible slides using timestamp + semantic tags.
    """
    aligned = []

    for seg in transcript:
        best_frame = _find_best_frame(seg, frames)

        aligned.append({
            "start": seg["start"],
            "end": seg["end"],
            "speech": seg["text"],
            "slide_text": best_frame
        })

    return aligned


def _find_best_frame(segment: dict, frames: list[dict], window: int = None) -> str:
    """Find best matching frame using timestamp + semantic tags."""
    if not frames:
        return ""

    # Load config values
    if window is None:
        window = get("processing", "alignment.window", 3)
    tolerance_before = get("processing", "alignment.timestamp_tolerance_before", 5)
    tolerance_after = get("processing", "alignment.timestamp_tolerance_after", 10)
    weights = get("processing", "alignment.weights", {"tags": 0.5, "text": 0.3, "timestamp": 0.2})
    timestamp_divisor = get("processing", "alignment.timestamp_score_divisor", 10)

    seg_start = segment["start"]
    seg_end = segment["end"]
    speech = segment["text"].lower()
    speech_words = set(speech.split())

    # Find frame with closest timestamp
    closest_idx = 0
    min_diff = float('inf')

    for i, frame in enumerate(frames):
        if frame["timestamp"] <= seg_start + tolerance_before:
            diff = abs(frame["timestamp"] - seg_start)
            if diff < min_diff:
                min_diff = diff
                closest_idx = i
    
    # Check window around closest frame
    candidates = []
    for i in range(max(0, closest_idx - window), min(len(frames), closest_idx + window + 1)):
        frame = frames[i]

        if frame["timestamp"] <= seg_end + tolerance_after:
            # Tag-based similarity
            tags = frame.get("tags", [])
            tag_score = _tag_similarity(speech_words, tags)

            # OCR text similarity (fallback)
            text_score = _text_similarity(speech, frame.get("text", "").lower())

            # Timestamp proximity
            timestamp_score = 1.0 / (1.0 + abs(frame["timestamp"] - seg_start) / timestamp_divisor)

            # Combined score using weights from config
            combined_score = (
                weights["tags"] * tag_score +
                weights["text"] * text_score +
                weights["timestamp"] * timestamp_score
            )

            candidates.append((frame, combined_score))
    
    if not candidates:
        return frames[closest_idx].get("text", "")
    
    best = max(candidates, key=lambda x: x[1])
    return best[0].get("text", "")


def _tag_similarity(speech_words: set, tags: list[str]) -> float:
    """Check how many tags appear in speech."""
    if not tags:
        return 0.0
    
    matches = 0
    for tag in tags:
        tag_words = set(tag.lower().split())
        if tag_words & speech_words:
            matches += 1
    
    return matches / len(tags)


def _text_similarity(text1: str, text2: str) -> float:
    """Calculate word overlap similarity."""
    stop_words_list = get("filters", "stop_words", [])
    stop_words = set(stop_words_list)

    words1 = set(w for w in text1.split() if len(w) > 2 and w not in stop_words)
    words2 = set(w for w in text2.split() if len(w) > 2 and w not in stop_words)
    
    if not words1 or not words2:
        return 0.0
    
    overlap = len(words1 & words2)
    return overlap / len(words1) if words1 else 0.0