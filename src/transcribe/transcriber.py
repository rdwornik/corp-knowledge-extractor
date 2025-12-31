import whisper
from config.config_loader import get


def transcribe(file_path: str, model_name: str = None) -> list[dict]:
    """
    Transcribe audio/video to timestamped segments.

    Returns:
        List of {"start": float, "end": float, "text": str}
    """
    # Load defaults from config
    if model_name is None:
        model_name = get("settings", "llm.whisper_model", "medium")

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