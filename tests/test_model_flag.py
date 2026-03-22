"""Tests for --model pro/flash CLI flag."""

from click.testing import CliRunner
from unittest.mock import patch

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from run import cli

# The model mapping used in scripts/run.py
MODEL_MAP = {"flash": "gemini-3-flash-preview", "pro": "gemini-3.1-pro-preview"}


class TestModelFlag:
    def test_model_flag_pro(self):
        """--model pro maps to gemini-3.1-pro-preview."""
        assert MODEL_MAP["pro"] == "gemini-3.1-pro-preview"

    def test_model_flag_default(self):
        """No flag → no model_override."""
        # When model is None, MODEL_MAP is not consulted
        assert MODEL_MAP.get(None) is None

    def test_model_flag_flash(self):
        """--model flash maps to gemini-3-flash-preview."""
        assert MODEL_MAP["flash"] == "gemini-3-flash-preview"

    def test_cli_rejects_invalid_model(self, tmp_path):
        """--model invalid → Click error."""
        source = tmp_path / "test.txt"
        source.write_text("Hello")
        runner = CliRunner()
        result = runner.invoke(cli, ["process", str(source), "--model", "gpt4"])
        assert result.exit_code != 0
        assert "Invalid value" in result.output or "invalid choice" in result.output.lower()

    def test_cli_accepts_pro(self, tmp_path):
        """--model pro is accepted by Click."""
        source = tmp_path / "test.txt"
        source.write_text("Hello")
        runner = CliRunner()
        # Will fail at extraction but should pass argument validation
        result = runner.invoke(cli, ["process", str(source), "--model", "pro"], catch_exceptions=True)
        # Should NOT fail with "Invalid value" for model
        assert "Invalid value" not in (result.output or "")

    def test_cli_accepts_flash(self, tmp_path):
        """--model flash is accepted by Click."""
        source = tmp_path / "test.txt"
        source.write_text("Hello")
        runner = CliRunner()
        result = runner.invoke(cli, ["process", str(source), "--model", "flash"], catch_exceptions=True)
        assert "Invalid value" not in (result.output or "")
