from abc import ABC, abstractmethod


class BaseSynthesizer(ABC):
    """Base class for all LLM backends."""
    
    @abstractmethod
    def synthesize(self, aligned_data: list[dict]) -> dict:
        """
        Turn aligned transcript + slides into structured knowledge.
        
        Args:
            aligned_data: List of {"start", "end", "speech", "slide_text"}
        
        Returns:
            {"summary": str, "topics": list, "action_items": list, "key_terms": list}
        """
        pass
    
    def _build_prompt(self, aligned_data: list[dict]) -> str:
        """Build prompt from aligned data."""
        content = ""
        for seg in aligned_data:
            content += f"[{seg['start']:.1f}s] {seg['speech']}\n"
            if seg['slide_text']:
                content += f"  [SLIDE]: {seg['slide_text'][:200]}...\n"
        
        return f"""Analyze this meeting transcript with slide context.

TRANSCRIPT:
{content}

Extract:
1. SUMMARY: 2-3 sentence overview
2. TOPICS: Main topics discussed (list)
3. ACTION_ITEMS: Tasks mentioned (list)
4. KEY_TERMS: Important terminology (list)

Respond in JSON format."""