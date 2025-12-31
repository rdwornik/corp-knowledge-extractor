import re
from collections import defaultdict
from config.config_loader import get


def post_process(synthesis: dict, frames: list[dict]) -> dict:
    """
    Post-process synthesis results to remove duplicates and organize by topic.
    
    Args:
        synthesis: Dict with slide_breakdown and qa_pairs
        frames: Original frames list
    
    Returns:
        Cleaned and organized synthesis
    """
    breakdowns = synthesis.get("slide_breakdown", [])
    qa_pairs = synthesis.get("qa_pairs", [])
    
    # Step 1: Filter junk frames
    breakdowns = _filter_junk_frames(breakdowns)
    
    # Step 2: Deduplicate similar frames
    breakdowns = _deduplicate_frames(breakdowns)
    
    # Step 3: Categorize by topic
    breakdowns = _categorize_by_topic(breakdowns)
    
    # Step 4: Clean up QA pairs (remove those referencing removed frames)
    valid_frame_ids = {b.get("frame_id") for b in breakdowns}
    qa_pairs = [qa for qa in qa_pairs if qa.get("frame_id") in valid_frame_ids]
    
    # Step 5: Deduplicate QA pairs
    qa_pairs = _deduplicate_qa_pairs(qa_pairs)
    
    return {
        "slide_breakdown": breakdowns,
        "qa_pairs": qa_pairs
    }

def _filter_junk_frames(breakdowns: list[dict]) -> list[dict]:
    """Remove junk frames."""
    # Load junk patterns from config
    junk_patterns = get("filters", "slide_junk_patterns", [])

    filtered = []
    for b in breakdowns:
        title = b.get("title", "").lower()
        
        is_junk = any(re.search(pattern, title) for pattern in junk_patterns)
        if is_junk:
            continue
        
        # Check speaker_explanation (not key_insight!)
        explanation = b.get("speaker_explanation", "")
        min_exp_length = get("settings", "limits.min_explanation_length", 30)
        has_content = explanation and len(explanation) > min_exp_length

        if has_content:
            filtered.append(b)
        else:
            # Still keep if has technical details
            tech = b.get("technical_details", "")
            min_tech_length = get("settings", "limits.min_technical_details_length", 10)
            if tech and len(tech) > min_tech_length:
                filtered.append(b)

    return filtered
    
def _deduplicate_frames(breakdowns: list[dict]) -> list[dict]:
    """Merge frames with very similar titles/content."""
    if not breakdowns:
        return []
    
    # Group by normalized title
    groups = defaultdict(list)
    
    for b in breakdowns:
        title = b.get("title", "Untitled")
        # Normalize: remove frame numbers, lowercase, strip
        normalized = re.sub(r'\d+', '', title).lower().strip()
        normalized = re.sub(r'[:\-\s]+', ' ', normalized).strip()
        groups[normalized].append(b)
    
    # Merge each group
    merged = []
    for normalized_title, group in groups.items():
        if len(group) == 1:
            merged.append(group[0])
        else:
            # Merge: keep first frame_id, combine unique content
            best = _merge_frame_group(group)
            merged.append(best)
    
    # Sort by original frame_id
    merged.sort(key=lambda x: _parse_frame_id(x.get("frame_id", "999")))
    
    return merged


def _merge_frame_group(group: list[dict]) -> dict:
    """Merge a group of similar frames into one."""
    # Use first frame as base
    base = group[0].copy()
    
    # Collect all unique content
    all_technical = set()
    all_speech = []
    all_terms = set()
    
    for b in group:
        tech = b.get("technical_details", "")
        if tech:
            # Handle both string and list
            if isinstance(tech, list):
                all_technical.update(str(t) for t in tech)
            else:
                all_technical.add(str(tech))
        
        speech = b.get("speaker_explanation", "")
        if speech and speech not in all_speech:
            all_speech.append(speech)
        
        terms = b.get("key_terminology", [])
        if isinstance(terms, list):
            all_terms.update(str(t) for t in terms)
        elif terms:
            all_terms.add(str(terms))
    
    # Pick longest/richest version for each field
    base["technical_details"] = max(all_technical, key=len) if all_technical else ""
    base["speaker_explanation"] = " | ".join(all_speech) if all_speech else ""
    base["key_terminology"] = list(all_terms)
    
    # Note which frames were merged
    frame_ids = [b.get("frame_id", "") for b in group]
    base["merged_from"] = frame_ids
    
    return base


def _categorize_by_topic(breakdowns: list[dict]) -> list[dict]:
    """Add category field based on content analysis."""
    # Load categories from config
    categories = get("categories", "keywords", {})
    
    for b in breakdowns:
        title = b.get("title", "").lower()
        visual = b.get("visual_content", "").lower()
        tags = " ".join(b.get("tags", [])) if isinstance(b.get("tags"), list) else ""
        combined = f"{title} {visual} {tags}"
        
        # Find best matching category
        best_category = "general"
        best_score = 0
        
        for category, keywords in categories.items():
            score = sum(1 for kw in keywords if kw in combined)
            if score > best_score:
                best_score = score
                best_category = category
        
        b["category"] = best_category
    
    return breakdowns


def _deduplicate_qa_pairs(qa_pairs: list[dict]) -> list[dict]:
    """Remove duplicate or very similar QA pairs."""
    seen_questions = set()
    unique = []
    
    for qa in qa_pairs:
        question = qa.get("question", "").lower().strip()
        # Normalize
        normalized = re.sub(r'\s+', ' ', question)
        normalized = re.sub(r'[^\w\s]', '', normalized)
        
        if normalized not in seen_questions:
            seen_questions.add(normalized)
            unique.append(qa)
    
    return unique


def _parse_frame_id(frame_id) -> int:
    """Parse frame_id to int for sorting."""
    if isinstance(frame_id, int):
        return frame_id
    try:
        return int(str(frame_id).lstrip("0") or "0")
    except:
        return 999