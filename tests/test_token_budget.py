"""Tests for dynamic token budget computation."""

from src.extract import compute_token_budget


class TestComputeTokenBudget:
    def test_standard_default(self):
        assert compute_token_budget("standard") == 8192

    def test_standard_from_config(self):
        config = {"llm": {"token_budgets": {"standard": 10000}}}
        assert compute_token_budget("standard", config=config) == 10000

    # --- Proportional PPTX formula: base 8192 + 280 per slide, cap 24576 ---

    def test_budget_pptx_10_slides(self):
        budget = compute_token_budget("deep", slide_count=10)
        assert budget == 8192 + 10 * 280  # 10992

    def test_budget_pptx_60_slides(self):
        budget = compute_token_budget("deep", slide_count=60)
        # 8192 + 60*280 = 24992 → capped at 24576
        assert budget == 24576

    def test_deep_zero_slides(self):
        budget = compute_token_budget("deep", slide_count=0)
        assert budget == 8192  # base only

    def test_deep_capped(self):
        budget = compute_token_budget("deep", slide_count=200)
        assert budget == 24576  # capped at max

    # --- Proportional MP4 formula: base 6144 + 150 per 5min block, cap 16384 ---

    def test_budget_mp4_30min(self):
        budget = compute_token_budget("multimodal", duration_min=30)
        # 6144 + 6*150 = 7044
        assert budget == 6144 + 6 * 150  # 7044

    def test_budget_mp4_90min(self):
        budget = compute_token_budget("multimodal", duration_min=90)
        # 6144 + 18*150 = 8844
        assert budget == 6144 + 18 * 150  # 8844

    def test_multimodal_zero_duration(self):
        budget = compute_token_budget("multimodal", duration_min=0)
        assert budget == 6144  # base only

    def test_multimodal_capped(self):
        budget = compute_token_budget("multimodal", duration_min=600)
        assert budget == 16384  # capped

    def test_unknown_depth_falls_to_standard(self):
        budget = compute_token_budget("whatever")
        assert budget == 8192

    def test_config_overrides_defaults(self):
        config = {
            "llm": {
                "token_budgets": {
                    "deep_base": 10000,
                    "deep_per_slide": 300,
                    "deep_max": 30000,
                }
            }
        }
        budget = compute_token_budget("deep", config=config, slide_count=30)
        assert budget == 10000 + (30 * 300)  # 19000
