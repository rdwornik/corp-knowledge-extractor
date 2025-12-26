import whisper


def transcribe(file_path: str, model_name: str = "medium") -> list[dict]:
    """
    Transcribe audio/video to timestamped segments.
    
    Returns:
        List of {"start": float, "end": float, "text": str}
    """
    model = whisper.load_model(model_name)
    result = model.transcribe(file_path)
    
    segments = [
        {
            "start": seg["start"],
            "end": seg["end"],
            "text": seg["text"].strip()
        }
        for seg in result["segments"]
    ]
    
    return segments