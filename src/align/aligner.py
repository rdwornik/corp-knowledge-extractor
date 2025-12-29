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


def _find_best_frame(segment: dict, frames: list[dict], window: int = 3) -> str:
    """Find best matching frame using timestamp + semantic tags."""
    if not frames:
        return ""
    
    seg_start = segment["start"]
    seg_end = segment["end"]
    speech = segment["text"].lower()
    speech_words = set(speech.split())
    
    # Find frame with closest timestamp
    closest_idx = 0
    min_diff = float('inf')
    
    for i, frame in enumerate(frames):
        if frame["timestamp"] <= seg_start + 5:
            diff = abs(frame["timestamp"] - seg_start)
            if diff < min_diff:
                min_diff = diff
                closest_idx = i
    
    # Check window around closest frame
    candidates = []
    for i in range(max(0, closest_idx - window), min(len(frames), closest_idx + window + 1)):
        frame = frames[i]
        
        if frame["timestamp"] <= seg_end + 10:
            # Tag-based similarity
            tags = frame.get("tags", [])
            tag_score = _tag_similarity(speech_words, tags)
            
            # OCR text similarity (fallback)
            text_score = _text_similarity(speech, frame.get("text", "").lower())
            
            # Timestamp proximity
            timestamp_score = 1.0 / (1.0 + abs(frame["timestamp"] - seg_start) / 10)
            
            # Combined: 50% tags, 30% text, 20% timestamp
            combined_score = 0.5 * tag_score + 0.3 * text_score + 0.2 * timestamp_score
            
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
    stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 
                  'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                  'would', 'could', 'should', 'may', 'might', 'can', 'to', 'of',
                  'in', 'for', 'on', 'with', 'at', 'by', 'from', 'or', 'and',
                  'so', 'but', 'if', 'then', 'that', 'this', 'it', 'its', 'we',
                  'you', 'they', 'i', 'he', 'she', 'my', 'your', 'our', 'their'}
    
    words1 = set(w for w in text1.split() if len(w) > 2 and w not in stop_words)
    words2 = set(w for w in text2.split() if len(w) > 2 and w not in stop_words)
    
    if not words1 or not words2:
        return 0.0
    
    overlap = len(words1 & words2)
    return overlap / len(words1) if words1 else 0.0