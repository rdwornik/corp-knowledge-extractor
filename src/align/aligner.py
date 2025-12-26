def align(transcript: list[dict], frames: list[dict]) -> list[dict]:
    """
    Match transcript segments with visible slides.
    
    Args:
        transcript: List of {"start": float, "end": float, "text": str}
        frames: List of {"timestamp": float, "path": str, "text": str}
    
    Returns:
        List of {"start": float, "end": float, "speech": str, "slide_text": str}
    """
    aligned = []
    
    for seg in transcript:
        # Find most recent slide at segment start time
        slide_text = ""
        for frame in frames:
            if frame["timestamp"] <= seg["start"]:
                slide_text = frame["text"]
            else:
                break
        
        aligned.append({
            "start": seg["start"],
            "end": seg["end"],
            "speech": seg["text"],
            "slide_text": slide_text
        })
    
    return aligned