import json
import requests
from .base import BaseSynthesizer


class OllamaSynthesizer(BaseSynthesizer):
    """Local LLM via Ollama."""
    
    def __init__(self, model: str = "llama3.2"):
        self.model = model
        self.url = "http://localhost:11434/api/generate"
    
    def synthesize(self, aligned_data: list[dict]) -> dict:
        prompt = self._build_prompt(aligned_data)
        
        response = requests.post(
            self.url,
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False
            }
        )
        
        result = response.json()
        text = result.get("response", "")
        
        # Try to parse JSON from response
        try:
            # Find JSON block in response
            start = text.find("{")
            end = text.rfind("}") + 1
            if start != -1 and end > start:
                return json.loads(text[start:end])
        except json.JSONDecodeError:
            pass
        
        # Fallback: return raw text
        return {"raw": text}