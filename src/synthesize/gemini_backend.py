import os
import json
from dotenv import load_dotenv
from google import genai
from .base import BaseSynthesizer
from config.config_loader import get

load_dotenv()


class GeminiSynthesizer(BaseSynthesizer):
    """Google Gemini API backend with chunk processing."""

    def __init__(self, model: str = None, chunk_size: int = None):
        super().__init__()

        # Load defaults from config
        if model is None:
            model = get("settings", "llm.model", "gemini-2.0-flash")
        if chunk_size is None:
            chunk_size = get("processing", "synthesis.chunk_size", 10)

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in .env")

        self.client = genai.Client(api_key=api_key)
        self.model_name = model
        self.chunk_size = chunk_size
    
    def synthesize(self, frames: list[dict], aligned_data: list[dict]) -> dict:
        """
        Process frames directly, using aligned_data for speech context.
        
        Args:
            frames: List of {"timestamp": float, "path": str, "text": str, "tags": list}
            aligned_data: List of {"start": float, "end": float, "speech": str, "slide_text": str}
        """
        # Sort frames by timestamp
        frames = sorted(frames, key=lambda x: x.get("timestamp", 0))
        
        # Build speech lookup by timestamp
        speech_by_time = self._build_speech_lookup(aligned_data)
        
        # Process in chunks
        all_breakdowns = []
        all_qa_pairs = []
        
        total_chunks = (len(frames) + self.chunk_size - 1) // self.chunk_size
        
        for i in range(0, len(frames), self.chunk_size):
            chunk = frames[i:i + self.chunk_size]
            chunk_num = (i // self.chunk_size) + 1
            print(f"    Processing chunk {chunk_num}/{total_chunks} ({len(chunk)} frames)...")
            
            result = self._process_chunk(chunk, speech_by_time, start_index=i)
            
            if "slide_breakdown" in result:
                all_breakdowns.extend(result["slide_breakdown"])
            if "qa_pairs" in result:
                all_qa_pairs.extend(result["qa_pairs"])
        
        return {
            "slide_breakdown": all_breakdowns,
            "qa_pairs": all_qa_pairs
        }
    
    def _build_speech_lookup(self, aligned_data: list[dict]) -> dict:
        """Build a lookup of speech segments by timestamp ranges."""
        segments = []
        for seg in aligned_data:
            segments.append({
                "start": seg["start"],
                "end": seg["end"],
                "speech": seg["speech"]
            })
        return segments
    
    def _find_speech_for_frame(self, frame_timestamp: float, speech_segments: list, next_frame_timestamp: float = None) -> str:
        """Find all speech that occurs while this frame is shown."""
        speeches = []

        # Frame is shown from frame_timestamp until next_frame_timestamp (or +default if last)
        default_range = get("settings", "limits.speech_range_last_frame", 60)
        end_time = next_frame_timestamp if next_frame_timestamp else frame_timestamp + default_range
        
        for seg in speech_segments:
            # Speech overlaps with frame display time
            if seg["start"] < end_time and seg["end"] > frame_timestamp:
                speeches.append(seg["speech"])
        
        return " ".join(speeches)
    
    def _process_chunk(self, frames: list[dict], speech_segments: list, start_index: int = 0) -> dict:
        """Process a chunk of frames."""
        ocr_limit = get("settings", "limits.ocr_text_max_chars", 500)
        content = ""

        for i, frame in enumerate(frames):
            frame_id = f"{start_index + i + 1:03d}"
            timestamp = frame.get("timestamp", 0)
            ocr_text = frame.get("text", "")[:ocr_limit]
            tags = frame.get("tags", [])
            
            # Find next frame timestamp for speech range
            next_timestamp = None
            if i + 1 < len(frames):
                next_timestamp = frames[i + 1].get("timestamp")
            
            speech = self._find_speech_for_frame(timestamp, speech_segments, next_timestamp)
            
            content += f"=== FRAME_ID:{frame_id} (timestamp: {timestamp:.1f}s) ===\n"
            content += f"VISUAL (OCR): {ocr_text}\n"
            content += f"TAGS: {', '.join(tags)}\n"
            content += f"SPEECH: {speech}\n\n"
        
        print("=" * 50)
        print(f"SENDING TO LLM (first 2000 chars):")
        print(content[:2000])
        print("=" * 50)
        
        prompt = f"{self.prompt_template}\n\n---\n\nMEETING CONTENT:\n\n{content}"

        
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt
        )
        text = response.text
        
        # Parse JSON from response
        try:
            start = text.find("{")
            end = text.rfind("}") + 1
            if start != -1 and end > start:
                return json.loads(text[start:end])
        except json.JSONDecodeError:
            pass
        
        return {"raw": text}