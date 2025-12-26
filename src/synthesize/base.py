from abc import ABC, abstractmethod


class BaseSynthesizer(ABC):
    """Base class for all LLM backends."""
    
    @abstractmethod
    def synthesize(self, aligned_data: list[dict]) -> dict:
        pass
    
    def _build_prompt(self, aligned_data: list[dict]) -> str:
        """Build prompt from aligned data."""
        
        # Group by slide
        slides = []
        current_slide = None
        current_speech = []
        
        for seg in aligned_data:
            if seg['slide_text'] != current_slide:
                if current_slide is not None:
                    slides.append({
                        "slide_text": current_slide,
                        "speech": " ".join(current_speech)
                    })
                current_slide = seg['slide_text']
                current_speech = [seg['speech']]
            else:
                current_speech.append(seg['speech'])
        
        if current_slide:
            slides.append({
                "slide_text": current_slide,
                "speech": " ".join(current_speech)
            })
        
        # Build content
        content = ""
        for i, slide in enumerate(slides, 1):
            content += f"SLIDE {i}:\n"
            content += f"Visual: {slide['slide_text'][:300]}\n"
            content += f"Speech: {slide['speech']}\n\n"
        
        return f"""Analyze this meeting recording with slides.

{content}

Provide TWO outputs:

1. SLIDE_BREAKDOWN: For each slide:
   - slide_number
   - title (infer from content)
   - visual_content (what's shown)
   - explanation (what speaker explained)
   - key_concepts (list of concepts taught)

2. QA_PAIRS: Generate question-answer pairs for knowledge base:
   - question (specific, factual)
   - answer (concise, accurate)
   - source_slide (number)

Respond in JSON format:
{{
  "slide_breakdown": [...],
  "qa_pairs": [...]
}}"""