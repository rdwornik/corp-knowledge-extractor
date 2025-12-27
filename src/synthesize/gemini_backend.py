import os
import json
from dotenv import load_dotenv
from google import genai
from .base import BaseSynthesizer

load_dotenv()


class GeminiSynthesizer(BaseSynthesizer):
    """Google Gemini API backend."""
    
    def __init__(self, model: str = "gemini-3-flash-preview"):
        super().__init__()
        
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in .env")
        
        self.client = genai.Client(api_key=api_key)
        self.model_name = model
    
    def synthesize(self, aligned_data: list[dict]) -> dict:
        prompt = self._build_prompt(aligned_data)
        
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