"""Extraction quality evaluator — run after CKE extraction to assess quality.

Usage:
    python eval_extraction.py                          # evaluate all in output/
    python eval_extraction.py output/SomePackage       # evaluate one package
    python eval_extraction.py --compare old/ new/      # compare two extractions

Produces a concise scorecard you can paste into chat for feedback.
"""

import json
import sys
import yaml
from pathlib import Path
from datetime import datetime


def load_frontmatter(md_path: Path) -> dict:
    """Extract YAML frontmatter from markdown file."""
    text = md_path.read_text(encoding="utf-8", errors="replace")
    if not text.startswith("---"):
        return {}
    end = text.find("---", 3)
    if end == -1:
        return {}
    try:
        return yaml.safe_load(text[3:end]) or {}
    except Exception:
        return {}


def count_content(md_path: Path) -> dict:
    """Count content metrics in a markdown note."""
    text = md_path.read_text(encoding="utf-8", errors="replace")

    if text.startswith("---"):
        end = text.find("---", 3)
        if end != -1:
            text = text[end + 3:]

    lines = text.strip().split("\n")
    sections = [l for l in lines if l.startswith("## ")]
    images = [l for l in lines if l.startswith("![")]
    quotes = [l for l in lines if l.startswith("> ") and "WARNING" not in l]
    warnings = [l for l in lines if "> WARNING" in l]
    bullet_points = [l for l in lines if l.strip().startswith("- ")]

    return {
        "total_chars": len(text),
        "total_lines": len(lines),
        "sections": len(sections),
        "images": len(images),
        "quotes": len(quotes),
        "warnings": len(warnings),
        "bullet_points": len(bullet_points),
    }


def assess_fact_quality(key_facts: list) -> dict:
    """Assess quality of key_facts beyond just counting them."""
    if not key_facts:
        return {"count": 0, "avg_length": 0, "specific_count": 0, "generic_count": 0, "quality_ratio": 0.0}

    lengths = [len(str(f)) for f in key_facts]
    specific = [f for f in key_facts if len(str(f)) >= 30]
    generic = [f for f in key_facts if len(str(f)) < 30]

    return {
        "count": len(key_facts),
        "avg_length": round(sum(lengths) / len(lengths)),
        "specific_count": len(specific),
        "generic_count": len(generic),
        "quality_ratio": round(len(specific) / len(key_facts), 2) if key_facts else 0.0,
    }


def assess_enrichment(key_facts: list) -> dict:
    """Check RFP enrichment fields (polarity, locator, source_date) on facts."""
    if not key_facts or not isinstance(key_facts[0], dict):
        return {"has_polarity": False, "has_locator": False, "has_source_date": False,
                "polarity_count": 0, "locator_count": 0}

    polarity_count = sum(1 for f in key_facts if isinstance(f, dict) and f.get("polarity") and f["polarity"] != "unknown")
    locator_count = sum(1 for f in key_facts if isinstance(f, dict) and f.get("locator"))

    return {
        "has_polarity": polarity_count > 0,
        "has_locator": locator_count > 0,
        "has_source_date": any(isinstance(f, dict) and f.get("source_date") for f in key_facts),
        "polarity_count": polarity_count,
        "locator_count": locator_count,
    }


def evaluate_package(package_dir: Path) -> dict:
    """Evaluate a single CKE output package."""
    result = {
        "package": package_dir.name,
        "files": {},
        "score": 0,
        "issues": [],
    }

    md_files = []
    for md in package_dir.rglob("*.md"):
        if md.name in ("index.md", "synthesis.md"):
            continue
        md_files.append(md)

    if not md_files:
        result["issues"].append("No extraction markdown files found")
        return result

    for md in md_files:
        fm = load_frontmatter(md)
        content = count_content(md)
        key_facts = fm.get("key_facts", []) or []
        fact_quality = assess_fact_quality(key_facts)
        enrichment = assess_enrichment(key_facts)

        file_eval = {
            "title": fm.get("title", md.stem),
            "source": fm.get("source", ""),
            "model": fm.get("model", "unknown"),
            "extraction_version": fm.get("extraction_version", 1),
            "depth": fm.get("depth", "standard"),
            "doc_type": fm.get("doc_type", "general"),
            "tokens_used": fm.get("tokens_used", 0),
            "content": content,
            "has_freshness": all(fm.get(k) for k in ["source_path", "source_hash", "extracted_at"]),
            "has_overlay": False,
            "overlay_type": None,
            "overlay_fields_populated": 0,
            "overlay_fields_total": 0,
            "key_facts_count": fact_quality["count"],
            "key_facts_quality": fact_quality,
            "enrichment": enrichment,
            "entities_count": len(fm.get("entities_mentioned", []) or []),
            "topics_count": len(fm.get("topics", []) or []),
            "products_count": len(fm.get("products", []) or []),
            "people_count": len(fm.get("people", []) or []),
            "quality": fm.get("quality", "unknown"),
        }

        # Check for overlay
        overlay_keys = [k for k in fm.keys() if k.endswith("_overlay")]
        if overlay_keys:
            file_eval["has_overlay"] = True
            file_eval["overlay_type"] = overlay_keys[0].replace("_overlay", "")
            overlay = fm[overlay_keys[0]]
            if isinstance(overlay, dict):
                file_eval["overlay_fields_total"] = len(overlay)
                file_eval["overlay_fields_populated"] = sum(
                    1 for v in overlay.values() if v is not None and v != [] and v != ""
                )

        # Slides/frames/images
        slides_dir = package_dir / "source" / "slides"
        frames_dir = package_dir / "source" / "frames"
        images_dir = package_dir / "source" / "images"
        file_eval["slides_count"] = len(list(slides_dir.glob("*.png"))) if slides_dir.exists() else 0
        file_eval["frames_count"] = len(list(frames_dir.glob("*.png"))) if frames_dir.exists() else 0
        file_eval["images_count"] = len(list(images_dir.glob("*.png"))) if images_dir.exists() else 0

        # PPTX coverage: sections vs expected slides
        source_path = fm.get("source", "")
        if source_path.endswith(".pptx"):
            expected_slides = file_eval["slides_count"] or content["images"] or 0
            if expected_slides > 0:
                file_eval["slide_coverage"] = round(content["sections"] / expected_slides, 2)
            else:
                file_eval["slide_coverage"] = None
        else:
            file_eval["slide_coverage"] = None

        # Score (0-100) — weighted by value
        score = 0

        # Key facts: max 20, but penalize generic facts
        specific_facts = fact_quality["specific_count"]
        score += min(20, specific_facts * 2)

        # Overlay: max 20 (raised from 10 — structured data is high value)
        if file_eval["has_overlay"]:
            score += 5  # base for having overlay at all
            score += min(15, file_eval["overlay_fields_populated"] * 3)

        # Freshness: 10
        score += 10 if file_eval["has_freshness"] else 0

        # Entities: max 10
        score += min(10, file_eval["entities_count"] * 2)

        # Topics: max 8
        score += min(8, file_eval["topics_count"])

        # Content depth: max 10
        if content["total_chars"] > 2000:
            score += 10
        elif content["total_chars"] > 1000:
            score += 7
        elif content["total_chars"] > 500:
            score += 4

        # Visual content (slides/frames): 10
        score += 10 if file_eval["slides_count"] > 0 or file_eval["frames_count"] > 0 else 0

        # Extraction version: 7
        score += 7 if file_eval["extraction_version"] >= 2 else 0

        # RFP enrichment bonus: 5
        if enrichment["has_polarity"] or enrichment["has_locator"]:
            score += 3 if enrichment["has_polarity"] else 0
            score += 2 if enrichment["has_locator"] else 0

        file_eval["score"] = min(100, score)

        result["files"][md.name] = file_eval

    scores = [f["score"] for f in result["files"].values()]
    result["score"] = round(sum(scores) / len(scores)) if scores else 0

    return result


def print_scorecard(result: dict):
    """Print concise scorecard for pasting."""
    print(f"\n{'='*70}")
    print(f"EXTRACTION QUALITY: {result['package']}")
    print(f"{'='*70}")
    print(f"Package score: {result['score']}/100")

    for fname, feval in result["files"].items():
        fq = feval["key_facts_quality"]
        en = feval["enrichment"]

        print(f"\n--- {feval['title'][:60]} ---")
        print(f"  Model: {feval['model']} | v{feval['extraction_version']} | {feval['depth']} | {feval['doc_type']}")
        print(f"  Score: {feval['score']}/100 | Quality: {feval['quality']}")
        print(f"  Key facts: {fq['count']} ({fq['specific_count']} specific, {fq['generic_count']} generic, avg {fq['avg_length']} chars)")
        print(f"  Entities: {feval['entities_count']} | Topics: {feval['topics_count']} | Products: {feval['products_count']}")
        print(f"  Content: {feval['content']['total_chars']} chars, {feval['content']['sections']} sections, {feval['content']['images']} images")
        print(f"  Overlay: {'YES (' + feval['overlay_type'] + ', ' + str(feval['overlay_fields_populated']) + '/' + str(feval['overlay_fields_total']) + ' fields)' if feval['has_overlay'] else 'NO'}")
        print(f"  Freshness: {'YES' if feval['has_freshness'] else 'NO'}")
        print(f"  Slides: {feval['slides_count']} | Frames: {feval['frames_count']} | Images: {feval['images_count']}")
        if feval["slide_coverage"] is not None:
            print(f"  Slide coverage: {feval['slide_coverage']:.0%}")
        if en["has_polarity"] or en["has_locator"]:
            print(f"  RFP enrichment: polarity={en['polarity_count']}, locator={en['locator_count']}")

        # Issues
        if fq["count"] == 0:
            print(f"  !! No key_facts extracted")
        elif fq["quality_ratio"] < 0.5:
            print(f"  !! {fq['generic_count']}/{fq['count']} facts are generic (<30 chars)")
        if feval["content"]["total_chars"] < 500:
            print(f"  !! Very short content ({feval['content']['total_chars']} chars)")
        if not feval["has_overlay"] and feval["depth"] == "deep":
            print(f"  !! Deep extraction but no overlay")
        if feval["quality"] == "fragment":
            print(f"  !! Fragment quality — incomplete extraction")
        if feval["slide_coverage"] is not None and feval["slide_coverage"] < 0.5:
            print(f"  !! Low slide coverage — {feval['content']['sections']} sections for {feval['slides_count'] or feval['content']['images']} slides")

    if result["issues"]:
        print(f"\nPackage issues: {', '.join(result['issues'])}")


def compare_packages(old_dir: Path, new_dir: Path):
    """Compare old vs new extraction of same document."""
    old = evaluate_package(old_dir)
    new = evaluate_package(new_dir)

    print(f"\n{'='*70}")
    print(f"COMPARISON: {old['package']} -> {new['package']}")
    print(f"{'='*70}")
    delta = new["score"] - old["score"]
    arrow = "+" if delta > 0 else "" if delta < 0 else "="
    print(f"Score: {old['score']}/100 -> {new['score']}/100 ({arrow}{delta} pts)")

    # Match files by source filename (more reliable than title)
    old_by_source = {}
    for fname, feval in old["files"].items():
        src = Path(feval.get("source", fname)).stem
        old_by_source[src] = feval

    for fname, n in new["files"].items():
        src = Path(n.get("source", fname)).stem
        o = old_by_source.get(src)

        if not o:
            # Fallback: match by title prefix
            for oeval in old["files"].values():
                if oeval["title"][:30] == n["title"][:30]:
                    o = oeval
                    break
        if not o:
            o = list(old["files"].values())[0] if old["files"] else None

        print(f"\n  {n['title'][:55]}")
        if o:
            def delta_str(old_val, new_val, fmt="{:>5}"):
                d = new_val - old_val if isinstance(new_val, (int, float)) else 0
                arrow = "+" if d > 0 else "" if d < 0 else " "
                return f"{fmt.format(old_val)} -> {fmt.format(new_val)} ({arrow}{d})"

            print(f"    Score:      {delta_str(o.get('score',0), n['score'])}")
            print(f"    Key facts:  {delta_str(o['key_facts_count'], n['key_facts_count'])}")
            print(f"    Entities:   {delta_str(o['entities_count'], n['entities_count'])}")
            print(f"    Content:    {delta_str(o['content']['total_chars'], n['content']['total_chars'])}")
            print(f"    Overlay:    {'NO' if not o['has_overlay'] else 'YES'} -> {'NO' if not n['has_overlay'] else 'YES'}")
            print(f"    Slides:     {delta_str(o.get('slides_count',0), n.get('slides_count',0))}")
            print(f"    Freshness:  {'NO' if not o['has_freshness'] else 'YES'} -> {'NO' if not n['has_freshness'] else 'YES'}")
            print(f"    Version:    v{o.get('extraction_version',1)} -> v{n.get('extraction_version',1)}")


if __name__ == "__main__":
    args = sys.argv[1:]

    if "--compare" in args:
        idx = args.index("--compare")
        old_dir = Path(args[idx + 1])
        new_dir = Path(args[idx + 2])
        compare_packages(old_dir, new_dir)
    elif args:
        for arg in args:
            p = Path(arg)
            if p.is_dir():
                result = evaluate_package(p)
                print_scorecard(result)
    else:
        output = Path("output")
        if not output.exists():
            print("No output/ directory found. Run from CKE repo.")
            sys.exit(1)

        packages = sorted(p for p in output.iterdir() if p.is_dir())

        print(f"Evaluating {len(packages)} packages...\n")
        total_score = 0
        for pkg in packages:
            result = evaluate_package(pkg)
            print_scorecard(result)
            total_score += result["score"]

        avg = round(total_score / len(packages)) if packages else 0
        print(f"\n{'='*70}")
        print(f"OVERALL: {len(packages)} packages, average score {avg}/100")
        print(f"{'='*70}")
