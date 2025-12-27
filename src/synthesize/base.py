from abc import ABC, abstractmethod
import os


class BaseSynthesizer(ABC):
    """Base class for all LLM backends."""
    
    def __init__(self):
        self.prompt_template = self._load_prompt()
    
    def _load_prompt(self) -> str:
        prompt_path = os.path.join(
            os.path.dirname(__file__),
            "..", "..", "config", "prompts", "knowledge_extraction.txt"
        )
        with open(prompt_path, "r", encoding="utf-8") as f:
            return f.read()
    
    @abstractmethod
    def synthesize(self, aligned_data: list[dict]) -> dict:
        pass
    
    def _build_prompt(self, aligned_data: list[dict]) -> str:
        """Build prompt from aligned data."""
        
        # Group by slide
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
        
        # Build content
        content = ""
        for i, slide in enumerate(slides, 1):
            content += f"=== SLIDE {i} (timestamp: {slide['timestamp']:.1f}s) ===\n"
            content += f"VISUAL: {slide['slide_text'][:500]}\n"
            content += f"SPEECH: {slide['speech']}\n\n"
        
        return f"{self.prompt_template}\n\n---\n\nMEETING CONTENT:\n\n{content}"