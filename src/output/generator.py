import os
import json
import shutil
from datetime import datetime
from collections import defaultdict


def generate_output(
    synthesis: dict,
    frames: list[dict],
    output_dir: str = "output"
) -> str:
    """Generate human + machine outputs."""
    
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
    
    # Group by category if present
    by_category = defaultdict(list)
    for slide in breakdowns:
        category = slide.get("category", "general")
        by_category[category].append(slide)
    
    category_order = [
        "infrastructure", "sla", "api", "architecture", 
        "security", "configuration", "data", "updates", 
        "warehouse", "general"
    ]
    
    category_titles = {
        "infrastructure": "🏗️ Infrastructure & Platform",
        "sla": "📋 Service Level Agreements",
        "api": "🔌 APIs & Integration",
        "architecture": "🏛️ Technical Architecture",
        "security": "🔒 Security",
        "configuration": "⚙️ Configuration & Customization",
        "data": "📊 Data & Reporting",
        "updates": "🔄 Updates & Versioning",
        "warehouse": "📦 Warehouse Management",
        "general": "📝 General",
    }
    
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
    
    # Skip empty/filler content
    filler_patterns = [
        "n/a", "none", "not applicable",
        "identical to", "same as slide", "same as previous",
        "the slide shows", "the presenter mentions",
        "no additional", "nothing new",
    ]
    
    for pattern in filler_patterns:
        if pattern in text_lower:
            return False
    
    # Must have some substance
    return len(text.strip()) > 20


def _has_specifics(text: str) -> bool:
    """Check if technical text has specific values (numbers, versions, etc.)."""
    import re
    # Look for numbers, percentages, versions, specific terms
    has_numbers = bool(re.search(r'\d+', text))
    has_specifics = any(term in text.lower() for term in [
        '%', 'version', 'api', 'hour', 'minute', 'gb', 'mb', 'tb',
        'http', 'rest', 'kafka', 'azure', 'aws', 'kubernetes'
    ])
    return has_numbers or has_specifics