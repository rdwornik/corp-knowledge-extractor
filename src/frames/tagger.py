import os
import json
from dotenv import load_dotenv
from google import genai

load_dotenv()


def tag_frames(frames: list[dict], batch_size: int = 10) -> list[dict]:
    """
    Generate semantic tags for each frame using LLM.
    
    Args:
        frames: List of {"timestamp": float, "path": str, "text": str}
        batch_size: How many frames to process at once
    
    Returns:
        Same list with added "tags" field
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in .env")
    
    client = genai.Client(api_key=api_key)
    
    # Process in batches
    for i in range(0, len(frames), batch_size):
        batch = frames[i:i + batch_size]
        batch_num = (i // batch_size) + 1
        total_batches = (len(frames) + batch_size - 1) // batch_size
        print(f"    Tagging frames {batch_num}/{total_batches}...")
        
        # Build prompt
        prompt = _build_tagging_prompt(batch, start_index=i)
        
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt
        )
        
        # Parse response
        tags_list = _parse_tags_response(response.text, len(batch))
        
        # Assign tags to frames
        for j, tags in enumerate(tags_list):
            frames[i + j]["tags"] = tags
    
    return frames


def _build_tagging_prompt(batch: list[dict], start_index: int) -> str:
    content = ""
    for j, frame in enumerate(batch):
        frame_num = start_index + j + 1
        ocr_text = frame.get("text", "")[:500]
        content += f"FRAME {frame_num}:\n{ocr_text}\n\n"
    
    return f"""Analyze these presentation slides and generate semantic tags for each.

{content}

For EACH frame, provide 3-6 topic tags that describe what the slide is about.
Tags should be concepts, features, or topics (e.g., "public APIs", "security", "disaster recovery", "pricing", "architecture").

Respond in JSON format:
{{
  "frames": [
    {{"frame": 1, "tags": ["tag1", "tag2", "tag3"]}},
    {{"frame": 2, "tags": ["tag1", "tag2", "tag3"]}}
  ]
}}

Be specific and accurate. Use lowercase tags."""


def _parse_tags_response(text: str, expected_count: int) -> list[list[str]]:
    """Parse LLM response to extract tags."""
    try:
        start = text.find("{")
        end = text.rfind("}") + 1
        if start != -1 and end > start:
            data = json.loads(text[start:end])
            tags_list = [f.get("tags", []) for f in data.get("frames", [])]
            
            # Pad if needed
            while len(tags_list) < expected_count:
                tags_list.append([])
            
            return tags_list[:expected_count]
    except:
        pass
    
    # Fallback: empty tags
    return [[] for _ in range(expected_count)]