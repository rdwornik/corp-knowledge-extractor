import os
import json
import shutil
from datetime import datetime
from collections import defaultdict
from config.config_loader import get, get_path


def generate_output(
    synthesis: dict,
    frames: list[dict],
    output_dir: str = None
) -> str:
    """Generate human + machine outputs."""

    # Load defaults from config
    if output_dir is None:
        output_dir = get_path("settings", "output.directory")

    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M")
    folder = os.path.join(output_dir, timestamp)
    frames_dir = os.path.join(folder, "frames")
    os.makedirs(frames_dir, exist_ok=True)
    
    # Sort frames by timestamp
    frames = sorted(frames, key=lambda x: x.get("timestamp", 0))
    
    # Copy frames
    frame_id_to_file = {}
    for i, frame in enumerate(frames):
        frame_id = f"{i+1:03d}"
        new_name = f"frame_{frame_id}.png"
        new_path = os.path.join(frames_dir, new_name)
        if os.path.exists(frame["path"]):
            shutil.copy(frame["path"], new_path)
        frame_id_to_file[frame_id] = new_name
    
    # Generate markdown
    md = _generate_markdown(synthesis, frame_id_to_file)
    with open(os.path.join(folder, "report.md"), "w", encoding="utf-8") as f:
        f.write(md)
    
    # Generate JSONL
    with open(os.path.join(folder, "knowledge.jsonl"), "w", encoding="utf-8") as f:
        for qa in synthesis.get("qa_pairs", []):
            f.write(json.dumps(qa, ensure_ascii=False) + "\n")
    
    # Generate metadata
    meta = {
        "created": timestamp,
        "slides_count": len(frames),
        "qa_count": len(synthesis.get("qa_pairs", []))
    }
    with open(os.path.join(folder, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    
    return folder


def _generate_markdown(synthesis: dict, frame_id_to_file: dict) -> str:
    """Build clean, insight-focused markdown report."""
    md = "# Meeting Knowledge Report\n\n"
    
    breakdowns = synthesis.get("slide_breakdown", [])
    
    # Load categories from config
    category_order = get("categories", "order", ["general"])
    category_titles = get("categories", "titles", {})

    # Group by category if present
    by_category = defaultdict(list)
    for slide in breakdowns:
        category = slide.get("category", "general")
        by_category[category].append(slide)
    
    for category in category_order:
        slides = by_category.get(category, [])
        if not slides:
            continue
        
        md += f"# {category_titles.get(category, category.title())}\n\n"
        
        for slide in slides:
            md += _format_slide(slide, frame_id_to_file)
    
    return md

def _format_slide(slide: dict, frame_id_to_file: dict) -> str:
    """Format a single slide."""
    frame_id_raw = slide.get('frame_id', '')
    if isinstance(frame_id_raw, int):
        frame_id = f"{frame_id_raw:03d}"
    else:
        frame_id = str(frame_id_raw).zfill(3)

    title = slide.get('title', 'Untitled')

    md = f"## {title}\n\n"

    # Image
    if frame_id in frame_id_to_file:
        filename = frame_id_to_file[frame_id]
        md += f"![{title}](frames/{filename})\n\n"

    # Visual content
    visual = slide.get('visual_content', '')
    if visual:
        md += f"**What's shown:** {visual}\n\n"

    # Technical details
    tech = slide.get('technical_details', '')
    if tech:
        md += f"**Technical Details:** {tech}\n\n"

    # Speaker explanation - THE MAIN CONTENT!
    explanation = slide.get('speaker_explanation', '')
    if explanation:
        md += f"**Speaker Explanation:** {explanation}\n\n"

    # Context
    context = slide.get('context_relationships', '')
    if context:
        md += f"**Context & Relationships:** {context}\n\n"

    # Terminology
    terms = slide.get('key_terminology', [])
    if terms:
        if isinstance(terms, list):
            md += f"**Key Terminology:** {', '.join(str(t) for t in terms)}\n\n"
        else:
            md += f"**Key Terminology:** {terms}\n\n"

    md += "---\n\n"
    return md

def _is_valuable(text: str) -> bool:
    """Check if text contains valuable content (not filler)."""
    if not text:
        return False

    text_lower = text.lower()

    # Load filler patterns from config
    filler_patterns = get("filters", "filler_patterns", [])

    for pattern in filler_patterns:
        if pattern in text_lower:
            return False

    # Must have some substance
    min_length = get("settings", "limits.min_valuable_text_length", 20)
    return len(text.strip()) > min_length


def _has_specifics(text: str) -> bool:
    """Check if technical text has specific values (numbers, versions, etc.)."""
    import re

    # Look for numbers, percentages, versions, specific terms
    has_numbers = bool(re.search(r'\d+', text))

    # Load specific terms from config
    specific_terms = get("filters", "specific_terms", [])
    has_specifics = any(term in text.lower() for term in specific_terms)

    return has_numbers or has_specifics