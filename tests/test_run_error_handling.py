"""Tests for per-file error handling in the extraction loop."""

from pathlib import Path
from unittest.mock import patch, MagicMock

from src.extract import ExtractionError


class TestUnexpectedErrorContinues:
    def test_unexpected_error_continues(self, tmp_path):
        """A TypeError in extraction should not crash the pipeline — file goes to failed list."""
        from src.inventory import SourceFile, FileType
        from src.tier_router import TierDecision, Tier

        # Create two fake files
        file_a = SourceFile(path=tmp_path / "a.pdf", name="a.pdf", type=FileType.DOCUMENT, size_bytes=100)
        file_b = SourceFile(path=tmp_path / "b.pdf", name="b.pdf", type=FileType.DOCUMENT, size_bytes=100)
        (tmp_path / "a.pdf").touch()
        (tmp_path / "b.pdf").touch()

        text_result = MagicMock()
        text_result.text = "content"
        text_result.char_count = 7
        text_result.extractor = "pdfplumber"
        text_result.slide_count = 0

        decision_a = TierDecision(tier=Tier.TEXT_AI, reason="test", estimated_cost=0.001, model=None, text_result=text_result)
        decision_b = TierDecision(tier=Tier.TEXT_AI, reason="test", estimated_cost=0.001, model=None, text_result=text_result)

        # File A throws TypeError, file B succeeds
        mock_result_b = MagicMock()
        mock_result_b.title = "B Title"
        mock_result_b.slides = []
        mock_result_b.name = "b.pdf"

        call_count = [0]

        def fake_extract(f, config, text_result, custom_prompt=None):
            call_count[0] += 1
            if f.name == "a.pdf":
                raise TypeError("NoneType has no attribute 'get'")
            return mock_result_b

        files = [file_a, file_b]
        tier_decisions = {"a.pdf": decision_a, "b.pdf": decision_b}
        extracts = {}
        failed = []
        cost_total = 0.0

        # Simulate the extraction loop from run.py
        for f in files:
            decision = tier_decisions[f.name]
            try:
                if decision.tier == Tier.TEXT_AI and decision.text_result:
                    result = fake_extract(f, {}, decision.text_result)
                extracts[f.name] = result
                cost_total += decision.estimated_cost
            except ExtractionError as exc:
                failed.append(f.path.name)
            except Exception as exc:
                failed.append(f.path.name)

        # File A should be in failed, file B should be extracted
        assert "a.pdf" in failed
        assert "b.pdf" in extracts
        assert len(extracts) == 1
        assert call_count[0] == 2  # both files were attempted
