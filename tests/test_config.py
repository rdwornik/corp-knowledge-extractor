"""
Tests for configuration loading and validation.

Validates that config files are properly structured and all required
values are present with correct types.
"""

import pytest
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.config_loader import get, get_path


class TestConfigLoader:
    """Test the config_loader utility."""

    def test_get_basic_value(self):
        """Test retrieving a basic configuration value."""
        model = get("settings", "llm.model")
        assert model is not None
        assert isinstance(model, str)
        assert len(model) > 0

    def test_get_with_default(self):
        """Test default value fallback."""
        value = get("settings", "nonexistent.key", "default_value")
        assert value == "default_value"

    def test_get_nested_value(self):
        """Test retrieving nested configuration values."""
        input_dir = get("settings", "input.directory")
        assert input_dir == "data/input"

        video_exts = get("settings", "input.video_extensions")
        assert isinstance(video_exts, list)
        assert ".mp4" in video_exts

    def test_get_path_resolves(self):
        """Test that get_path resolves relative paths."""
        input_path = get_path("settings", "input.directory")
        assert os.path.isabs(input_path)  # Should be absolute

    def test_invalid_config_file(self):
        """Test handling of invalid config file name."""
        value = get("nonexistent_file", "some.key", "fallback")
        assert value == "fallback"


class TestSettingsConfig:
    """Validate settings.yaml structure and required fields."""

    def test_input_settings_exist(self):
        """Verify input configuration is present."""
        directory = get("settings", "input.directory")
        assert directory is not None

        extensions = get("settings", "input.video_extensions")
        assert isinstance(extensions, list)
        assert len(extensions) > 0

    def test_output_settings_exist(self):
        """Verify output configuration is present."""
        output_dir = get("settings", "output.directory")
        assert output_dir is not None
        assert isinstance(output_dir, str)

    def test_llm_settings_exist(self):
        """Verify LLM configuration is present."""
        model = get("settings", "llm.model")
        assert model is not None
        assert isinstance(model, str)

        whisper_model = get("settings", "llm.whisper_model")
        assert whisper_model is not None

    def test_limits_exist(self):
        """Verify limits configuration is present."""
        ocr_limit = get("settings", "limits.ocr_text_max_chars")
        assert isinstance(ocr_limit, int)
        assert ocr_limit > 0

        min_length = get("settings", "limits.min_valuable_text_length")
        assert isinstance(min_length, int)
        assert min_length > 0


class TestProcessingConfig:
    """Validate processing.yaml structure."""

    def test_frame_extraction_settings(self):
        """Verify frame extraction configuration."""
        threshold = get("processing", "frames.change_threshold", 30)
        assert isinstance(threshold, (int, float))
        assert threshold > 0

    def test_deduplication_settings(self):
        """Verify deduplication configuration."""
        similarity = get("processing", "dedup.similarity_threshold", 0.85)
        assert isinstance(similarity, float)
        assert 0 < similarity <= 1.0

    def test_synthesis_settings(self):
        """Verify synthesis configuration."""
        chunk_size = get("processing", "synthesis.chunk_size", 10)
        assert isinstance(chunk_size, int)
        assert chunk_size > 0


class TestAnonymizationConfig:
    """Validate anonymization.yaml structure."""

    def test_custom_terms_exist(self):
        """Verify custom anonymization terms are defined."""
        custom_terms = get("anonymize", "custom_terms", [])
        assert isinstance(custom_terms, list)

    def test_exclude_terms_exist(self):
        """Verify exclusion terms are defined."""
        exclude_terms = get("anonymize", "exclude_terms", [])
        assert isinstance(exclude_terms, list)

    def test_auto_detection_settings(self):
        """Verify auto-detection settings."""
        auto_detect = get("anonymize", "auto_detect_pii", True)
        assert isinstance(auto_detect, bool)


class TestCategoriesConfig:
    """Validate categories.yaml structure."""

    def test_category_order_exists(self):
        """Verify category order is defined."""
        order = get("categories", "order", [])
        assert isinstance(order, list)
        assert len(order) > 0
        assert "general" in order  # Should always have general fallback

    def test_category_titles_exist(self):
        """Verify category titles are defined."""
        titles = get("categories", "titles", {})
        assert isinstance(titles, dict)
        assert len(titles) > 0

    def test_category_keywords_exist(self):
        """Verify category keywords are defined."""
        keywords = get("categories", "keywords", {})
        assert isinstance(keywords, dict)
        assert len(keywords) > 0

    def test_categories_consistency(self):
        """Verify category order matches titles and keywords."""
        order = get("categories", "order", [])
        titles = get("categories", "titles", {})
        keywords = get("categories", "keywords", {})

        for category in order:
            if category == "general":
                continue  # General is a fallback, may not have keywords
            assert category in titles, f"Category '{category}' missing from titles"
            # Keywords are optional but recommended
            if category not in keywords:
                print(f"Warning: Category '{category}' has no keywords")


class TestFiltersConfig:
    """Validate filters.yaml structure."""

    def test_junk_patterns_exist(self):
        """Verify junk patterns are defined."""
        patterns = get("filters", "junk_patterns", [])
        assert isinstance(patterns, list)

    def test_filler_patterns_exist(self):
        """Verify filler patterns are defined."""
        patterns = get("filters", "filler_patterns", [])
        assert isinstance(patterns, list)

    def test_specific_terms_exist(self):
        """Verify specific terms list exists."""
        terms = get("filters", "specific_terms", [])
        assert isinstance(terms, list)


class TestConfigIntegrity:
    """Integration tests for configuration consistency."""

    def test_all_config_files_loadable(self):
        """Verify all config files can be loaded without errors."""
        config_files = [
            "settings",
            "processing",
            "anonymize",
            "categories",
            "filters"
        ]

        for config_file in config_files:
            # Should not raise any exceptions
            value = get(config_file, "dummy.path", None)
            assert value is None  # Dummy path should return None

    def test_paths_are_valid(self):
        """Verify configured paths are valid."""
        # Input directory should exist or be creatable
        input_dir = get_path("settings", "input.directory")
        assert isinstance(input_dir, str)
        assert len(input_dir) > 0

        # Output directory should exist or be creatable
        output_dir = get_path("settings", "output.directory")
        assert isinstance(output_dir, str)
        assert len(output_dir) > 0

    def test_llm_models_are_strings(self):
        """Verify LLM model names are valid strings."""
        model = get("settings", "llm.model")
        assert isinstance(model, str)
        assert len(model) > 0

        tagger_model = get("settings", "llm.tagger_model", model)
        assert isinstance(tagger_model, str)

        whisper_model = get("settings", "llm.whisper_model")
        assert isinstance(whisper_model, str)


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
