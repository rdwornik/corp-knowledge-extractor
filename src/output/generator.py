import os
import json
import shutil
from datetime import datetime


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
    
    # SORT FRAMES BY TIMESTAMP FIRST (must match synthesizer order!)
    frames = sorted(frames, key=lambda x: x.get("timestamp", 0))
    
    # Copy frames with frame_id naming
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
    """Build markdown report organized by category."""
    md = "# Meeting Knowledge Report\n\n"
    
    breakdowns = synthesis.get("slide_breakdown", [])
    
    # Group by category
    from collections import defaultdict
    by_category = defaultdict(list)
    for slide in breakdowns:
        category = slide.get("category", "general")
        by_category[category].append(slide)
    
    # Category display order
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
            frame_id_raw = slide.get('frame_id', '')
            if isinstance(frame_id_raw, int):
                frame_id = f"{frame_id_raw:03d}"
            else:
                frame_id = str(frame_id_raw).zfill(3)
            
            title = slide.get('title', 'Untitled')
            
            md += f"## {title}\n\n"
            
            # Note if merged
            merged_from = slide.get("merged_from", [])
            if len(merged_from) > 1:
                md += f"*Combined from frames: {', '.join(str(f) for f in merged_from)}*\n\n"
            
            # Match frame by ID
            if frame_id in frame_id_to_file:
                filename = frame_id_to_file[frame_id]
                md += f"![{title}](frames/{filename})\n\n"
            
            visual = slide.get('visual_content', '')
            if visual:
                md += f"**What's shown:** {visual}\n\n"
            
            tech = slide.get('technical_details', '')
            if tech:
                md += f"**Technical Details:** {tech}\n\n"
            
            explanation = slide.get('speaker_explanation', slide.get('explanation', ''))
            if explanation:
                md += f"**Speaker Explanation:** {explanation}\n\n"
            
            context = slide.get('context_relationships', '')
            if context:
                md += f"**Context & Relationships:** {context}\n\n"
            
            terms = slide.get('key_terminology', [])
            if terms:
                if isinstance(terms, list):
                    md += f"**Key Terminology:** {', '.join(str(t) for t in terms)}\n\n"
                else:
                    md += f"**Key Terminology:** {terms}\n\n"
            
            md += "---\n\n"
    
    return md