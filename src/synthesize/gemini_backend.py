import os
import json
from dotenv import load_dotenv
from google import genai
from .base import BaseSynthesizer

load_dotenv()


class GeminiSynthesizer(BaseSynthesizer):
    """Google Gemini API backend with chunk processing."""
    
    def __init__(self, model: str = "gemini-2.0-flash", chunk_size: int = 10):
        super().__init__()
        
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in .env")
        
        self.client = genai.Client(api_key=api_key)
        self.model_name = model
        self.chunk_size = chunk_size
    
    def synthesize(self, aligned_data: list[dict]) -> dict:
        # Group aligned data by slide
        slides = self._group_by_slide(aligned_data)
        
        # Process in chunks
        all_breakdowns = []
        all_qa_pairs = []
        
        total_chunks = (len(slides) + self.chunk_size - 1) // self.chunk_size
        
        for i in range(0, len(slides), self.chunk_size):
            chunk = slides[i:i + self.chunk_size]
            chunk_num = (i // self.chunk_size) + 1
            print(f"    Processing chunk {chunk_num}/{total_chunks} ({len(chunk)} slides)...")
            
            result = self._process_chunk(chunk, start_index=i)
            
            if "slide_breakdown" in result:
                all_breakdowns.extend(result["slide_breakdown"])
            if "qa_pairs" in result:
                all_qa_pairs.extend(result["qa_pairs"])
        
        return {
            "slide_breakdown": all_breakdowns,
            "qa_pairs": all_qa_pairs
        }
    
    def _group_by_slide(self, aligned_data: list[dict]) -> list[dict]:
        """Group transcript segments by slide."""
        slides = []
        current_slide = None
        current_speech = []
        current_timestamp = 0
        
        for seg in aligned_data:
            if seg['slide_text'] != current_slide:
                if current_slide is not None:
                    slides.append({
                        "slide_text": current_slide,
                        "speech": " ".join(current_speech),
                        "timestamp": current_timestamp
                    })
                current_slide = seg['slide_text']
                current_speech = [seg['speech']]
                current_timestamp = seg['start']
            else:
                current_speech.append(seg['speech'])
        
        if current_slide:
            slides.append({
                "slide_text": current_slide,
                "speech": " ".join(current_speech),
                "timestamp": current_timestamp
            })
        
        return slides
    
    def _process_chunk(self, slides: list[dict], start_index: int = 0) -> dict:
        """Process a chunk of slides."""
        # Build content for this chunk
        content = ""
        for i, slide in enumerate(slides):
            slide_num = start_index + i + 1
            content += f"=== SLIDE {slide_num} (timestamp: {slide['timestamp']:.1f}s) ===\n"
            content += f"VISUAL: {slide['slide_text'][:500]}\n"
            content += f"SPEECH: {slide['speech']}\n\n"
        
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