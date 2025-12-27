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
    
    # Copy frames
    frame_paths = []
    for i, frame in enumerate(frames):
        new_name = f"slide_{i+1}.png"
        new_path = os.path.join(frames_dir, new_name)
        if os.path.exists(frame["path"]):
            shutil.copy(frame["path"], new_path)
        frame_paths.append(new_name)
    
    # Generate markdown
    md = _generate_markdown(synthesis, frame_paths)
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


def _generate_markdown(synthesis: dict, frame_paths: list[str]) -> str:
    """Build markdown report."""
    md = "# Meeting Knowledge Report\n\n"
    
    breakdowns = synthesis.get("slide_breakdown", [])
    
    for slide in breakdowns:
        slide_num = slide.get('slide_number', 0)
        title = slide.get('title', 'Untitled')
        
        md += f"## Slide {slide_num}: {title}\n\n"
        
        # Embed image if available
        if slide_num > 0 and slide_num <= len(frame_paths):
            md += f"![Slide {slide_num}](frames/{frame_paths[slide_num - 1]})\n\n"
        
        # Visual content
        visual = slide.get('visual_content', '')
        if visual:
            md += f"**What's shown:** {visual}\n\n"
        
        # Technical details (NEW)
        tech = slide.get('technical_details', '')
        if tech:
            md += f"**Technical Details:** {tech}\n\n"
        
        # Speaker explanation
        explanation = slide.get('speaker_explanation', slide.get('explanation', ''))
        if explanation:
            md += f"**Speaker Explanation:** {explanation}\n\n"
        
        # Context & relationships (NEW)
        context = slide.get('context_relationships', '')
        if context:
            md += f"**Context & Relationships:** {context}\n\n"
        
        # Key terminology
        terms = slide.get('key_terminology', [])
        if terms:
            if isinstance(terms, list):
                md += f"**Key Terminology:** {', '.join(str(t) for t in terms)}\n\n"
            else:
                md += f"**Key Terminology:** {terms}\n\n"
        
        md += "---\n\n"
    
    return md