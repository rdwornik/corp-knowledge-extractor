"""Tests for dynamic token budget computation."""

from src.extract import compute_token_budget


class TestComputeTokenBudget:
    def test_standard_default(self):
        assert compute_token_budget("standard") == 8192

    def test_standard_from_config(self):
        config = {"llm": {"token_budgets": {"standard": 10000}}}
        assert compute_token_budget("standard", config=config) == 10000

    def test_deep_base(self):
        budget = compute_token_budget("deep", slide_count=10)
        assert budget == 16384  # 10 slides < 20, no extra

    def test_deep_large_deck(self):
        budget = compute_token_budget("deep", slide_count=40)
        assert budget == 16384 + (20 * 256)  # 21504

    def test_deep_capped(self):
        budget = compute_token_budget("deep", slide_count=200)
        assert budget == 24576  # capped at max

    def test_deep_exactly_20_slides(self):
        budget = compute_token_budget("deep", slide_count=20)
        assert budget == 16384  # no extra

    def test_multimodal_base(self):
        budget = compute_token_budget("multimodal", duration_min=10)
        assert budget == 8192  # 10 min < 30, no extra

    def test_multimodal_long_video(self):
        budget = compute_token_budget("multimodal", duration_min=60)
        assert budget == 8192 + (30 * 128)  # 12032

    def test_multimodal_capped(self):
        budget = compute_token_budget("multimodal", duration_min=300)
        assert budget == 16384  # capped

    def test_multimodal_exactly_30_min(self):
        budget = compute_token_budget("multimodal", duration_min=30)
        assert budget == 8192  # no extra

    def test_unknown_depth_falls_to_standard(self):
        budget = compute_token_budget("whatever")
        assert budget == 8192

    def test_config_overrides_defaults(self):
        config = {
            "llm": {
                "token_budgets": {
                    "deep_base": 20000,
                    "deep_per_slide_over_20": 500,
                    "deep_max": 30000,
                }
            }
        }
        budget = compute_token_budget("deep", config=config, slide_count=30)
        assert budget == 20000 + (10 * 500)  # 25000
