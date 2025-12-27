import os
import json
from dotenv import load_dotenv
import google.generativeai as genai
from .base import BaseSynthesizer

load_dotenv()


class GeminiSynthesizer(BaseSynthesizer):
    """Google Gemini API backend."""
    
    def __init__(self, model: str = "gemini-3-flash-preview"):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in .env")
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model)
    
    def synthesize(self, aligned_data: list[dict]) -> dict:
        prompt = self._build_prompt(aligned_data)
        
        response = self.model.generate_content(prompt)
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