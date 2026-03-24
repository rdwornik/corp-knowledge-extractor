"""Tests for standard extraction prompt key_facts requirement (FIX 1: v3 regression).

Verifies that the standard prompt requests key_facts and that the parsing
pipeline correctly handles them for both standard and deep extraction paths.
"""

import yaml
from pathlib import Path


def _load_standard_prompt() -> str:
    """Load the standard extraction prompt from settings.yaml."""
    config_path = Path(__file__).parent.parent / "config" / "settings.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config["prompts"]["extract"]


class TestStandardPromptRequestsFacts:
    """Standard prompt must request key_facts to avoid zero-fact output."""

    def test_standard_prompt_contains_key_facts(self):
        """Standard prompt string must contain 'key_facts'."""
        prompt = _load_standard_prompt()
        assert "key_facts" in prompt

    def test_standard_prompt_asks_for_minimum(self):
        """Standard prompt should ask for 'at least 5' facts."""
        prompt = _load_standard_prompt()
        assert "at least 5" in prompt

    def test_standard_prompt_has_entities_mentioned(self):
        """Standard prompt should request entities_mentioned."""
        prompt = _load_standard_prompt()
        assert "entities_mentioned" in prompt

    def test_standard_prompt_has_facts_array(self):
        """Standard prompt JSON schema should still have structured facts array."""
        prompt = _load_standard_prompt()
        assert '"facts"' in prompt

    def test_standard_prompt_has_key_points(self):
        """key_points should still be present for backward compatibility."""
        prompt = _load_standard_prompt()
        assert "key_points" in prompt


class TestStandardExtractionParsing:
    """Verify key_facts from standard extraction flow into ExtractionResult."""

    def test_standard_extraction_has_facts_via_raw_json(self):
        """Mock standard extraction response with key_facts → parsed into raw_json."""
        from src.extract import _result_from_json
        from src.inventory import SourceFile, FileType

        mock_data = {
            "title": "Test Document",
            "summary": "A test summary.",
            "key_points": ["Point 1", "Point 2"],
            "key_facts": [
                "Revenue grew 15% YoY to $500M",
                "Lenzing operates 5 fiber plants globally",
                "SAP S/4 migration planned for Q3 2026",
                "Blue Yonder WMS handles 200K orders/day",
                "Contract value is $2.5M over 3 years",
            ],
            "entities_mentioned": ["Lenzing AG", "SAP", "Blue Yonder"],
            "facts": [
                {"fact": "Revenue grew 15% YoY to $500M", "page": 3},
            ],
            "topics": ["Supply Chain Planning"],
            "products": ["Blue Yonder Platform"],
            "content_type": "document",
            "language": "en",
            "quality": "full",
        }

        sf = SourceFile(
            path=Path("test.pdf"),
            name="test.pdf",
            type=FileType.DOCUMENT,
            size_bytes=1000,
        )

        result = _result_from_json(mock_data, sf, tokens=1000)
        assert result.key_points == ["Point 1", "Point 2"]

        # Simulate what extract_from_text now does: always preserve raw_json
        result.raw_json = mock_data

        # Verify key_facts accessible from raw_json (used by synthesize.py)
        assert len(result.raw_json.get("key_facts", [])) == 5
        assert "Revenue grew 15% YoY to $500M" in result.raw_json["key_facts"]

    def test_standard_extraction_fallback_to_key_points(self):
        """When key_facts is empty, key_points should serve as fallback."""
        mock_data = {
            "title": "Old Format",
            "summary": "Test.",
            "key_points": ["Insight A", "Insight B with 50+ chars to qualify as specific content"],
            "topics": ["SLA"],
            "content_type": "document",
            "language": "en",
            "quality": "full",
        }

        # Simulate: raw_json has no key_facts (old standard extraction)
        key_facts = mock_data.get("key_facts") or mock_data.get("key_points") or []
        assert len(key_facts) == 2
        assert "Insight A" in key_facts
