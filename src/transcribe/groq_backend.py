import os
import subprocess
import tempfile
from dotenv import load_dotenv
from groq import Groq

load_dotenv()


def extract_audio(video_path: str, output_path: str) -> str:
    """Extract and compress audio from video."""
    subprocess.run([
        "ffmpeg", "-y", "-i", video_path,
        "-vn", "-ac", "1", "-ar", "16000", "-b:a", "32k",
        output_path
    ], check=True, capture_output=True)
    return output_path


def transcribe_groq(file_path: str, model: str = "whisper-large-v3") -> list[dict]:
    """
    Transcribe audio/video using Groq API (fast cloud Whisper).
    
    Args:
        file_path: Path to audio/video file
        model: Whisper model (whisper-large-v3)
    
    Returns:
        List of {"start": float, "end": float, "text": str}
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not found in .env")
    
    client = Groq(api_key=api_key)
    
    # Convert video to compressed audio
    temp_audio = None
    if file_path.lower().endswith(('.mp4', '.mkv', '.avi', '.mov', '.webm')):
        temp_audio = os.path.join(tempfile.gettempdir(), "groq_temp_audio.mp3")
        print("  Converting video to audio...")
        extract_audio(file_path, temp_audio)
        file_path = temp_audio
    
    # Check file size
    file_size = os.path.getsize(file_path)
    if file_size > 25 * 1024 * 1024:  # 25MB limit
        raise ValueError(f"File too large ({file_size // 1024 // 1024}MB). Groq limit is 25MB.")
    
    with open(file_path, "rb") as file:
        transcription = client.audio.transcriptions.create(
            file=(os.path.basename(file_path), file.read()),
            model=model,
            response_format="verbose_json"
        )
    
    # Cleanup temp file
    if temp_audio and os.path.exists(temp_audio):
        os.remove(temp_audio)
    
    segments = [
        {
            "start": seg["start"],
            "end": seg["end"],
            "text": seg["text"].strip()
        }
        for seg in transcription.segments
    ]
    
    return segments