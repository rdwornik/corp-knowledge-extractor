"""
Quality assurance tests for report output.

Automated checks to ensure generated reports meet quality standards:
- Speaker explanations are detailed (not raw transcripts)
- Junk frames are filtered out
- Categories are balanced (not everything in "general")
- Q&A pairs are specific and well-formed

Usage:
    # Test a specific report
    pytest tests/test_quality.py --report-path output/2024-01-15_1430

    # Test latest report
    pytest tests/test_quality.py

    # Generate quality report
    python tests/test_quality.py --report output/2024-01-15_1430
"""

import pytest
import os
import sys
import json
import re
from pathlib import Path
from collections import Counter
from typing import Dict, List, Tuple

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.config_loader import get


class QualityChecker:
    """Automated quality checks for generated reports."""

    def __init__(self, report_path: str):
        """
        Initialize quality checker.

        Args:
            report_path: Path to output folder (e.g., output/2024-01-15_1430)
        """
        self.report_path = report_path
        self.report_md_path = os.path.join(report_path, "report.md")
        self.knowledge_jsonl_path = os.path.join(report_path, "knowledge.jsonl")
        self.metadata_path = os.path.join(report_path, "metadata.json")

        # Load config thresholds
        self.min_explanation_length = get(
            "settings", "limits.min_explanation_length", 30
        )
        self.min_technical_length = get(
            "settings", "limits.min_technical_details_length", 10
        )

    def check_speaker_explanation_quality(self) -> Tuple[bool, str, Dict]:
        """
        Check quality of speaker explanations.

        Ensures:
        - speaker_explanation field is populated
        - Not just raw transcript dumps
        - Educational value present
        - Minimum length threshold met

        Returns:
            (passed, message, metrics)
        """
        if not os.path.exists(self.report_md_path):
            return False, "Report markdown not found", {}

        with open(self.report_md_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Find all speaker explanations
        pattern = r'\*\*Speaker Explanation:\*\* (.+?)(?:\n\n|\*\*|$)'
        explanations = re.findall(pattern, content, re.DOTALL)

        if not explanations:
            return False, "No speaker explanations found in report", {}

        # Analyze explanations
        total = len(explanations)
        too_short = sum(1 for e in explanations if len(e.strip()) < self.min_explanation_length)
        empty = sum(1 for e in explanations if not e.strip())
        generic_phrases = [
            "the speaker discussed",
            "the presenter explained",
            "as mentioned",
            "this slide shows"
        ]
        generic_count = sum(
            1 for e in explanations
            if any(phrase in e.lower() for phrase in generic_phrases)
        )

        # Check for raw transcript indicators (too conversational)
        transcript_indicators = ["um", "uh", "you know", "like,", "okay so"]
        raw_transcript_count = sum(
            1 for e in explanations
            if sum(ind in e.lower() for ind in transcript_indicators) >= 2
        )

        avg_length = sum(len(e) for e in explanations) / total if total > 0 else 0

        metrics = {
            "total_explanations": total,
            "empty": empty,
            "too_short": too_short,
            "generic": generic_count,
            "raw_transcript_suspected": raw_transcript_count,
            "avg_length": round(avg_length, 1),
            "min_length_threshold": self.min_explanation_length
        }

        # Determine pass/fail
        empty_ratio = empty / total if total > 0 else 0
        short_ratio = too_short / total if total > 0 else 0
        generic_ratio = generic_count / total if total > 0 else 0
        raw_ratio = raw_transcript_count / total if total > 0 else 0

        issues = []
        if empty_ratio > 0.1:  # >10% empty
            issues.append(f"{empty_ratio*100:.1f}% explanations are empty")
        if short_ratio > 0.3:  # >30% too short
            issues.append(f"{short_ratio*100:.1f}% explanations are too short")
        if generic_ratio > 0.5:  # >50% generic
            issues.append(f"{generic_ratio*100:.1f}% explanations are generic")
        if raw_ratio > 0.2:  # >20% raw transcript
            issues.append(f"{raw_ratio*100:.1f}% appear to be raw transcript")

        if issues:
            return False, "Speaker explanation quality issues: " + "; ".join(issues), metrics

        return True, "Speaker explanations are high quality", metrics

    def check_no_junk_frames(self) -> Tuple[bool, str, Dict]:
        """
        Verify junk frames are filtered out.

        Checks for common junk patterns:
        - "Loading" slides
        - "Thank you" slides
        - Generic transitions
        - Empty slides

        Returns:
            (passed, message, metrics)
        """
        if not os.path.exists(self.report_md_path):
            return False, "Report markdown not found", {}

        with open(self.report_md_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Load junk patterns from config
        junk_patterns = get("filters", "junk_patterns", [
            "loading",
            "thank you",
            "any questions",
            "q&a",
            "break",
            "transition"
        ])

        # Find all slide titles
        title_pattern = r'^## (.+)$'
        titles = re.findall(title_pattern, content, re.MULTILINE)

        if not titles:
            return False, "No slide titles found in report", {}

        # Check for junk
        junk_found = []
        for title in titles:
            title_lower = title.lower()
            for pattern in junk_patterns:
                if pattern.lower() in title_lower:
                    junk_found.append(title)
                    break

        total = len(titles)
        junk_count = len(junk_found)
        junk_ratio = junk_count / total if total > 0 else 0

        metrics = {
            "total_slides": total,
            "junk_slides": junk_count,
            "junk_ratio": round(junk_ratio, 3),
            "junk_titles": junk_found[:5]  # First 5 examples
        }

        if junk_ratio > 0.1:  # >10% junk
            return False, f"Too many junk slides: {junk_ratio*100:.1f}%", metrics

        return True, f"Junk filtering working well ({junk_count}/{total} junk slides)", metrics

    def check_categories_balanced(self) -> Tuple[bool, str, Dict]:
        """
        Check that categories are balanced.

        Prevents everything from being dumped into "general" category.
        Validates categorization keywords are working.

        Returns:
            (passed, message, metrics)
        """
        if not os.path.exists(self.report_md_path):
            return False, "Report markdown not found", {}

        with open(self.report_md_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Find category headers (e.g., "# 🏗️ Infrastructure & Platform")
        category_pattern = r'^# (.+)$'
        categories = re.findall(category_pattern, content, re.MULTILINE)

        if not categories:
            return False, "No categories found in report", {}

        # Count slides per category (approximate - count ## after each #)
        category_distribution = {}
        lines = content.split('\n')
        current_category = None

        for line in lines:
            if line.startswith('# ') and not line.startswith('## '):
                current_category = line[2:].strip()
                category_distribution[current_category] = 0
            elif line.startswith('## ') and current_category:
                category_distribution[current_category] += 1

        total_slides = sum(category_distribution.values())

        if total_slides == 0:
            return False, "No slides found in report", {}

        # Check for imbalance
        general_keywords = ["general", "miscellaneous", "other", "uncategorized"]
        general_category = None
        general_count = 0

        for cat, count in category_distribution.items():
            if any(keyword in cat.lower() for keyword in general_keywords):
                general_category = cat
                general_count = count
                break

        general_ratio = general_count / total_slides if total_slides > 0 else 0

        metrics = {
            "total_slides": total_slides,
            "total_categories": len(category_distribution),
            "distribution": category_distribution,
            "general_category": general_category,
            "general_count": general_count,
            "general_ratio": round(general_ratio, 3)
        }

        # Check balance
        issues = []

        if len(category_distribution) == 1:
            issues.append("Only one category (no categorization happening)")

        if general_ratio > 0.7:  # >70% in general
            issues.append(f"{general_ratio*100:.1f}% slides in general category")

        if issues:
            return False, "Category imbalance: " + "; ".join(issues), metrics

        return True, f"Categories well balanced ({len(category_distribution)} categories)", metrics

    def check_qa_pairs_quality(self) -> Tuple[bool, str, Dict]:
        """
        Check Q&A pairs quality.

        Validates:
        - Q&A format (question/answer fields present)
        - Specificity (not too generic)
        - Category tagging
        - Source frame references

        Returns:
            (passed, message, metrics)
        """
        if not os.path.exists(self.knowledge_jsonl_path):
            return False, "Knowledge JSONL not found", {}

        qa_pairs = []
        with open(self.knowledge_jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    qa_pairs.append(json.loads(line))

        if not qa_pairs:
            return False, "No Q&A pairs found", {}

        # Analyze Q&A pairs
        total = len(qa_pairs)
        missing_fields = 0
        generic_questions = 0
        generic_answers = 0
        missing_category = 0
        missing_source = 0

        generic_question_patterns = [
            "what is this",
            "what does this show",
            "what is shown",
            "what's this about"
        ]

        generic_answer_patterns = [
            "this shows",
            "this is about",
            "as shown",
            "the slide shows"
        ]

        for qa in qa_pairs:
            # Check required fields
            if "question" not in qa or "answer" not in qa:
                missing_fields += 1
                continue

            question = qa.get("question", "").lower()
            answer = qa.get("answer", "").lower()

            # Check for generic questions
            if any(pattern in question for pattern in generic_question_patterns):
                generic_questions += 1

            # Check for generic answers
            if any(pattern in answer for pattern in generic_answer_patterns):
                generic_answers += 1

            # Check for category
            if "category" not in qa or not qa["category"]:
                missing_category += 1

            # Check for source frame
            if "source" not in qa:
                missing_source += 1

        avg_question_length = sum(len(qa.get("question", "")) for qa in qa_pairs) / total
        avg_answer_length = sum(len(qa.get("answer", "")) for qa in qa_pairs) / total

        metrics = {
            "total_qa_pairs": total,
            "missing_fields": missing_fields,
            "generic_questions": generic_questions,
            "generic_answers": generic_answers,
            "missing_category": missing_category,
            "missing_source": missing_source,
            "avg_question_length": round(avg_question_length, 1),
            "avg_answer_length": round(avg_answer_length, 1)
        }

        # Determine pass/fail
        issues = []

        if missing_fields > 0:
            issues.append(f"{missing_fields} Q&A pairs missing required fields")

        generic_q_ratio = generic_questions / total
        if generic_q_ratio > 0.5:  # >50% generic questions
            issues.append(f"{generic_q_ratio*100:.1f}% questions are generic")

        generic_a_ratio = generic_answers / total
        if generic_a_ratio > 0.5:  # >50% generic answers
            issues.append(f"{generic_a_ratio*100:.1f}% answers are generic")

        missing_cat_ratio = missing_category / total
        if missing_cat_ratio > 0.3:  # >30% missing category
            issues.append(f"{missing_cat_ratio*100:.1f}% missing category tags")

        if issues:
            return False, "Q&A quality issues: " + "; ".join(issues), metrics

        return True, f"Q&A pairs are high quality ({total} pairs)", metrics

    def run_all_checks(self) -> Dict:
        """
        Run all quality checks and return comprehensive report.

        Returns:
            Dictionary with all check results and overall pass/fail
        """
        checks = {
            "speaker_explanation": self.check_speaker_explanation_quality(),
            "junk_frames": self.check_no_junk_frames(),
            "categories": self.check_categories_balanced(),
            "qa_pairs": self.check_qa_pairs_quality()
        }

        results = {
            "report_path": self.report_path,
            "checks": {},
            "overall_pass": True
        }

        for check_name, (passed, message, metrics) in checks.items():
            results["checks"][check_name] = {
                "passed": passed,
                "message": message,
                "metrics": metrics
            }
            if not passed:
                results["overall_pass"] = False

        return results


# Pytest fixtures and tests
@pytest.fixture
def latest_report_path():
    """Find the latest report in output directory."""
    output_dir = get("settings", "output.directory", "output")
    if not os.path.exists(output_dir):
        pytest.skip(f"Output directory {output_dir} does not exist")

    subdirs = [
        os.path.join(output_dir, d)
        for d in os.listdir(output_dir)
        if os.path.isdir(os.path.join(output_dir, d))
    ]

    if not subdirs:
        pytest.skip("No reports found in output directory")

    # Get most recent
    latest = max(subdirs, key=os.path.getmtime)
    return latest


class TestReportQuality:
    """Pytest tests for report quality."""

    def test_speaker_explanations(self, latest_report_path):
        """Test speaker explanation quality."""
        checker = QualityChecker(latest_report_path)
        passed, message, metrics = checker.check_speaker_explanation_quality()

        print(f"\nSpeaker Explanation Check: {message}")
        print(f"Metrics: {json.dumps(metrics, indent=2)}")

        assert passed, f"Speaker explanation quality check failed: {message}"

    def test_junk_filtering(self, latest_report_path):
        """Test junk frame filtering."""
        checker = QualityChecker(latest_report_path)
        passed, message, metrics = checker.check_no_junk_frames()

        print(f"\nJunk Frame Check: {message}")
        print(f"Metrics: {json.dumps(metrics, indent=2)}")

        assert passed, f"Junk frame check failed: {message}"

    def test_category_balance(self, latest_report_path):
        """Test category balance."""
        checker = QualityChecker(latest_report_path)
        passed, message, metrics = checker.check_categories_balanced()

        print(f"\nCategory Balance Check: {message}")
        print(f"Metrics: {json.dumps(metrics, indent=2)}")

        assert passed, f"Category balance check failed: {message}"

    def test_qa_quality(self, latest_report_path):
        """Test Q&A pair quality."""
        checker = QualityChecker(latest_report_path)
        passed, message, metrics = checker.check_qa_pairs_quality()

        print(f"\nQ&A Quality Check: {message}")
        print(f"Metrics: {json.dumps(metrics, indent=2)}")

        assert passed, f"Q&A quality check failed: {message}"


def generate_quality_report(report_path: str) -> None:
    """
    Generate a comprehensive quality report for a given output.

    Args:
        report_path: Path to report directory
    """
    checker = QualityChecker(report_path)
    results = checker.run_all_checks()

    print("\n" + "="*60)
    print(f"QUALITY REPORT: {report_path}")
    print("="*60 + "\n")

    for check_name, check_result in results["checks"].items():
        status = "✓ PASS" if check_result["passed"] else "✗ FAIL"
        print(f"{status} - {check_name.replace('_', ' ').title()}")
        print(f"  {check_result['message']}")

        if check_result["metrics"]:
            print(f"  Metrics: {json.dumps(check_result['metrics'], indent=4)}")
        print()

    print("="*60)
    overall_status = "✓ ALL CHECKS PASSED" if results["overall_pass"] else "✗ SOME CHECKS FAILED"
    print(f"OVERALL: {overall_status}")
    print("="*60 + "\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Quality check for generated reports")
    parser.add_argument(
        "--report",
        help="Path to report directory (e.g., output/2024-01-15_1430)"
    )

    args = parser.parse_args()

    if args.report:
        generate_quality_report(args.report)
    else:
        # Run pytest
        pytest.main([__file__, "-v"])
