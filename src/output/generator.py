import os
import json
import shutil
from datetime import datetime


def generate_output(
    synthesis: dict,
    frames: list[dict],
    output_dir: str = "output"
) -> str:
    """
    Generate human + machine outputs.
    
    Args:
        synthesis: {"slide_breakdown": [...], "qa_pairs": [...]}
        frames: List of {"timestamp": float, "path": str, "text": str}
        output_dir: Base output directory
    
    Returns:
        Path to created output folder
    """
    # Create folder with timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M")
    folder = os.path.join(output_dir, timestamp)
    frames_dir = os.path.join(folder, "frames")
    os.makedirs(frames_dir, exist_ok=True)
    
    # Copy frames
    frame_paths = []
    for i, frame in enumerate(frames):
        new_name = f"slide_{i+1}.png"
        new_path = os.path.join(frames_dir, new_name)
        shutil.copy(frame["path"], new_path)
        frame_paths.append(new_name)
    
    # Generate markdown
    md = _generate_markdown(synthesis, frame_paths)
    with open(os.path.join(folder, "report.md"), "w", encoding="utf-8") as f:
        f.write(md)
    
    # Generate JSONL
    with open(os.path.join(folder, "knowledge.jsonl"), "w", encoding="utf-8") as f:
        for qa in synthesis.get("qa_pairs", []):
            f.write(json.dumps(qa) + "\n")
    
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
    
    for i, slide in enumerate(synthesis.get("slide_breakdown", [])):
        md += f"## Slide {slide.get('slide_number', i+1)}: {slide.get('title', 'Untitled')}\n\n"
        
        # Embed image if available
        if i < len(frame_paths):
            md += f"![Slide {i+1}](frames/{frame_paths[i]})\n\n"
        
        md += f"**What's shown:** {slide.get('visual_content', 'N/A')}\n\n"
        md += f"**Explanation:** {slide.get('explanation', 'N/A')}\n\n"
        md += f"**Key concepts:** {', '.join(slide.get('key_concepts', []))}\n\n"
        md += "---\n\n"
    
    return md