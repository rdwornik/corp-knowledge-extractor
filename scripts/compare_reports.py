"""
Report comparison tool for quality regression detection.

Compares two knowledge extraction reports and generates:
1. comparison_report.md - Human-readable diff with improvements/regressions
2. comparison_metrics.json - Machine-readable metrics for CI/CD

Usage:
    # Compare two reports
    python scripts/compare_reports.py output/2025-01-01_1200 output/2025-01-02_1400

    # Compare with baseline
    python scripts/compare_reports.py tests/fixtures/baseline output/latest

    # Fail on regression (for CI/CD)
    python scripts/compare_reports.py baseline.txt new.txt --fail-on-regression

    # Output to specific location
    python scripts/compare_reports.py old/ new/ --output comparison/
"""

import os
import sys
import json
import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.test_quality import QualityChecker


def load_report_data(report_dir: str) -> dict:
    """
    Load all data from a report directory.

    Args:
        report_dir: Path to report directory

    Returns:
        Dictionary with report data
    """
    if not os.path.exists(report_dir):
        raise FileNotFoundError(f"Report directory not found: {report_dir}")

    data = {
        "report_dir": report_dir,
        "markdown_path": os.path.join(report_dir, "report.md"),
        "jsonl_path": os.path.join(report_dir, "knowledge.jsonl"),
        "metadata_path": os.path.join(report_dir, "metadata.json"),
        "frames_dir": os.path.join(report_dir, "frames")
    }

    # Load markdown
    if os.path.exists(data["markdown_path"]):
        with open(data["markdown_path"], "r", encoding="utf-8") as f:
            data["markdown_content"] = f.read()
    else:
        data["markdown_content"] = ""

    # Load JSONL Q&A pairs
    data["qa_pairs"] = []
    if os.path.exists(data["jsonl_path"]):
        with open(data["jsonl_path"], "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data["qa_pairs"].append(json.loads(line))

    # Load metadata
    data["metadata"] = {}
    if os.path.exists(data["metadata_path"]):
        with open(data["metadata_path"], "r", encoding="utf-8") as f:
            data["metadata"] = json.load(f)

    # Count frames
    data["frame_count"] = 0
    if os.path.exists(data["frames_dir"]):
        data["frame_count"] = len([
            f for f in os.listdir(data["frames_dir"])
            if f.endswith(".png")
        ])

    # Extract slide data from markdown
    data["slides"] = extract_slides_from_markdown(data["markdown_content"])

    # Run quality checks
    checker = QualityChecker(report_dir)
    data["quality"] = {
        "speaker_explanation": checker.check_speaker_explanation_quality(),
        "junk_frames": checker.check_no_junk_frames(),
        "categories": checker.check_categories_balanced(),
        "qa_pairs": checker.check_qa_pairs_quality()
    }

    return data


def extract_slides_from_markdown(markdown: str) -> List[dict]:
    """
    Extract slide information from markdown report.

    Returns:
        List of slide dictionaries with title and content
    """
    slides = []

    # Split by ## headers (slide titles)
    pattern = r'^## (.+?)$'
    parts = re.split(pattern, markdown, flags=re.MULTILINE)

    # parts[0] is content before first ##, then alternates title/content
    for i in range(1, len(parts), 2):
        if i + 1 < len(parts):
            title = parts[i].strip()
            content = parts[i + 1].strip()

            # Extract speaker explanation
            explanation_match = re.search(
                r'\*\*Speaker Explanation:\*\* (.+?)(?:\n\n|\*\*|$)',
                content,
                re.DOTALL
            )
            explanation = explanation_match.group(1).strip() if explanation_match else ""

            slides.append({
                "title": title,
                "content": content,
                "explanation": explanation,
                "explanation_length": len(explanation)
            })

    return slides


def compare_reports(old_dir: str, new_dir: str) -> dict:
    """
    Compare two reports and generate metrics.

    Args:
        old_dir: Path to old report directory
        new_dir: Path to new report directory

    Returns:
        Dictionary with comparison metrics
    """
    print(f"Loading old report: {old_dir}")
    old = load_report_data(old_dir)

    print(f"Loading new report: {new_dir}")
    new = load_report_data(new_dir)

    comparison = {
        "timestamp": datetime.now().isoformat(),
        "old_report": old_dir,
        "new_report": new_dir,
        "frames": compare_frames(old, new),
        "slides": compare_slides(old, new),
        "qa_pairs": compare_qa_pairs(old, new),
        "quality": compare_quality(old, new),
        "content_changes": compare_content_changes(old, new)
    }

    # Determine overall verdict
    comparison["verdict"] = determine_verdict(comparison)

    return comparison


def compare_frames(old: dict, new: dict) -> dict:
    """Compare frame counts."""
    old_count = old["frame_count"]
    new_count = new["frame_count"]
    change = new_count - old_count
    change_percent = (change / max(old_count, 1)) * 100

    return {
        "old_count": old_count,
        "new_count": new_count,
        "change": change,
        "change_percent": round(change_percent, 1)
    }


def compare_slides(old: dict, new: dict) -> dict:
    """Compare slide counts and content."""
    old_slides = old["slides"]
    new_slides = new["slides"]

    old_titles = set(s["title"] for s in old_slides)
    new_titles = set(s["title"] for s in new_slides)

    removed = list(old_titles - new_titles)
    added = list(new_titles - old_titles)

    return {
        "old_count": len(old_slides),
        "new_count": len(new_slides),
        "change": len(new_slides) - len(old_slides),
        "removed_titles": removed[:10],  # Limit to 10
        "added_titles": added[:10],
        "total_removed": len(removed),
        "total_added": len(added)
    }


def compare_qa_pairs(old: dict, new: dict) -> dict:
    """Compare Q&A pair counts."""
    old_count = len(old["qa_pairs"])
    new_count = len(new["qa_pairs"])
    change = new_count - old_count
    change_percent = (change / max(old_count, 1)) * 100

    return {
        "old_count": old_count,
        "new_count": new_count,
        "change": change,
        "change_percent": round(change_percent, 1)
    }


def compare_quality(old: dict, new: dict) -> dict:
    """Compare quality metrics."""
    def get_metrics(quality_data):
        return {
            "avg_explanation_length": quality_data["speaker_explanation"][2].get("avg_length", 0),
            "empty_explanations": quality_data["speaker_explanation"][2].get("empty", 0),
            "junk_slides": quality_data["junk_frames"][2].get("junk_slides", 0),
            "general_category_count": quality_data["categories"][2].get("general_count", 0),
            "total_slides": quality_data["categories"][2].get("total_slides", 1)
        }

    old_metrics = get_metrics(old["quality"])
    new_metrics = get_metrics(new["quality"])

    # Calculate changes
    old_general_percent = (old_metrics["general_category_count"] / max(old_metrics["total_slides"], 1)) * 100
    new_general_percent = (new_metrics["general_category_count"] / max(new_metrics["total_slides"], 1)) * 100

    improvements = []
    regressions = []

    # Check explanation length
    length_change = new_metrics["avg_explanation_length"] - old_metrics["avg_explanation_length"]
    length_change_percent = (length_change / max(old_metrics["avg_explanation_length"], 1)) * 100

    if length_change_percent > 10:
        improvements.append(f"Longer explanations (+{length_change_percent:.0f}%)")
    elif length_change_percent < -10:
        regressions.append(f"Shorter explanations ({length_change_percent:.0f}%)")

    # Check junk frames
    junk_change = new_metrics["junk_slides"] - old_metrics["junk_slides"]
    if junk_change < 0:
        improvements.append(f"Fewer junk frames ({abs(junk_change)} less)")
    elif junk_change > 0:
        regressions.append(f"More junk frames (+{junk_change})")

    # Check categorization
    cat_change = new_general_percent - old_general_percent
    if cat_change < -10:
        improvements.append(f"Better categorization ({abs(cat_change):.0f}% less in general)")
    elif cat_change > 10:
        regressions.append(f"Worse categorization (+{cat_change:.0f}% in general)")

    return {
        "old": old_metrics,
        "new": new_metrics,
        "improvements": improvements,
        "regressions": regressions
    }


def compare_content_changes(old: dict, new: dict) -> dict:
    """Compare content changes for slides with same title."""
    old_slides = {s["title"]: s for s in old["slides"]}
    new_slides = {s["title"]: s for s in new["slides"]}

    common_titles = set(old_slides.keys()) & set(new_slides.keys())

    changed_explanations = []

    for title in common_titles:
        old_slide = old_slides[title]
        new_slide = new_slides[title]

        if old_slide["explanation"] != new_slide["explanation"]:
            # Determine change type
            old_len = old_slide["explanation_length"]
            new_len = new_slide["explanation_length"]

            if new_len > old_len * 1.2:
                change_type = "improved"
            elif new_len < old_len * 0.8:
                change_type = "degraded"
            else:
                change_type = "rewritten"

            changed_explanations.append({
                "title": title,
                "old_explanation": old_slide["explanation"][:200] + "...",
                "new_explanation": new_slide["explanation"][:200] + "...",
                "old_length": old_len,
                "new_length": new_len,
                "change_type": change_type
            })

    return {
        "total_common_slides": len(common_titles),
        "changed_explanations": len(changed_explanations),
        "examples": changed_explanations[:5]  # First 5 examples
    }


def determine_verdict(comparison: dict) -> dict:
    """
    Determine overall verdict (improved, degraded, mixed).

    Returns:
        Dictionary with verdict and reasons
    """
    improvements = comparison["quality"]["improvements"]
    regressions = comparison["quality"]["regressions"]

    if regressions and not improvements:
        verdict = "degraded"
        summary = "Quality has degraded - regressions detected"
    elif improvements and not regressions:
        verdict = "improved"
        summary = "Quality has improved - no regressions"
    elif improvements and regressions:
        verdict = "mixed"
        summary = "Mixed changes - some improvements and some regressions"
    else:
        verdict = "unchanged"
        summary = "No significant quality changes detected"

    return {
        "verdict": verdict,
        "summary": summary,
        "has_regressions": len(regressions) > 0
    }


def generate_markdown_report(comparison: dict, output_path: str):
    """Generate human-readable markdown comparison report."""
    md = "# Report Comparison\n\n"

    md += f"**Old Report:** `{comparison['old_report']}`  \n"
    md += f"**New Report:** `{comparison['new_report']}`  \n"
    md += f"**Comparison Date:** {comparison['timestamp']}  \n\n"

    # Summary table
    md += "## Summary\n\n"
    md += "| Metric | Old | New | Change |\n"
    md += "|--------|-----|-----|--------|\n"

    frames = comparison["frames"]
    md += f"| Frames | {frames['old_count']} | {frames['new_count']} | "
    md += f"{frames['change']:+d} ({frames['change_percent']:+.1f}%) |\n"

    slides = comparison["slides"]
    md += f"| Slides in report | {slides['old_count']} | {slides['new_count']} | "
    md += f"{slides['change']:+d} |\n"

    qa = comparison["qa_pairs"]
    md += f"| Q&A pairs | {qa['old_count']} | {qa['new_count']} | "
    md += f"{qa['change']:+d} ({qa['change_percent']:+.1f}%) |\n"

    quality = comparison["quality"]
    old_q = quality["old"]
    new_q = quality["new"]
    md += f"| Avg explanation length | {old_q['avg_explanation_length']:.0f} | {new_q['avg_explanation_length']:.0f} | "
    length_change = new_q['avg_explanation_length'] - old_q['avg_explanation_length']
    md += f"{length_change:+.0f} |\n"

    md += f"| Junk frames | {old_q['junk_slides']} | {new_q['junk_slides']} | "
    md += f"{new_q['junk_slides'] - old_q['junk_slides']:+d} |\n"

    md += "\n"

    # Verdict
    verdict = comparison["verdict"]
    emoji = {
        "improved": "✅",
        "degraded": "❌",
        "mixed": "⚠️",
        "unchanged": "➖"
    }.get(verdict["verdict"], "")

    md += f"## Overall Verdict {emoji}\n\n"
    md += f"**{verdict['summary']}**\n\n"

    # Improvements
    if quality["improvements"]:
        md += "## Improvements ✅\n\n"
        for improvement in quality["improvements"]:
            md += f"- {improvement}\n"
        md += "\n"

    # Regressions
    if quality["regressions"]:
        md += "## Regressions ⚠️\n\n"
        for regression in quality["regressions"]:
            md += f"- {regression}\n"
        md += "\n"

    # Removed slides
    if slides["removed_titles"]:
        md += "## Removed Slides\n\n"
        for i, title in enumerate(slides["removed_titles"][:10], 1):
            md += f"{i}. \"{title}\"\n"
        if slides["total_removed"] > 10:
            md += f"\n... and {slides['total_removed'] - 10} more\n"
        md += "\n"

    # Added slides
    if slides["added_titles"]:
        md += "## Added Slides\n\n"
        for i, title in enumerate(slides["added_titles"][:10], 1):
            md += f"{i}. \"{title}\"\n"
        if slides["total_added"] > 10:
            md += f"\n... and {slides['total_added'] - 10} more\n"
        md += "\n"

    # Changed explanations
    content = comparison["content_changes"]
    if content["changed_explanations"] > 0:
        md += "## Changed Explanations\n\n"
        md += f"Total slides with changed explanations: {content['changed_explanations']}\n\n"

        for example in content["examples"]:
            md += f"### {example['title']}\n\n"
            md += f"**Old ({example['old_length']} chars):** {example['old_explanation']}\n\n"
            md += f"**New ({example['new_length']} chars):** {example['new_explanation']}\n\n"
            md += f"**Assessment:** {example['change_type']}\n\n"
            md += "---\n\n"

    # Write to file
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(md)

    print(f"✓ Markdown report written to: {output_path}")


def generate_json_metrics(comparison: dict, output_path: str):
    """Generate machine-readable JSON metrics."""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)

    print(f"✓ JSON metrics written to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare two knowledge extraction reports"
    )

    parser.add_argument(
        "old_report",
        help="Path to old report directory"
    )

    parser.add_argument(
        "new_report",
        help="Path to new report directory"
    )

    parser.add_argument(
        "--output",
        default=".",
        help="Output directory for comparison files (default: current directory)"
    )

    parser.add_argument(
        "--fail-on-regression",
        action="store_true",
        help="Exit with non-zero code if regressions detected (for CI/CD)"
    )

    args = parser.parse_args()

    try:
        # Run comparison
        print("="*60)
        print("REPORT COMPARISON")
        print("="*60)

        comparison = compare_reports(args.old_report, args.new_report)

        # Generate outputs
        os.makedirs(args.output, exist_ok=True)

        md_path = os.path.join(args.output, "comparison_report.md")
        json_path = os.path.join(args.output, "comparison_metrics.json")

        generate_markdown_report(comparison, md_path)
        generate_json_metrics(comparison, json_path)

        # Print summary
        print("\n" + "="*60)
        print("COMPARISON SUMMARY")
        print("="*60)
        print(f"Verdict: {comparison['verdict']['verdict'].upper()}")
        print(f"Summary: {comparison['verdict']['summary']}")
        print("="*60 + "\n")

        # Check fail-on-regression
        if args.fail_on_regression and comparison["verdict"]["has_regressions"]:
            print("❌ Regressions detected - failing as requested")
            print(f"Regressions: {comparison['quality']['regressions']}")
            return 1

        print("✓ Comparison complete!")
        return 0

    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
