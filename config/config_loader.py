"""
Universal configuration loader with caching.

Usage:
    from config.config_loader import get

    # Get nested value using dot notation
    sample_rate = get("processing", "frames.sample_rate")
    model_name = get("settings", "llm.model")

    # Get entire section
    categories = get("categories", "keywords")
"""

import os
import yaml
from typing import Any
from pathlib import Path


# Cache for loaded configs
_cache: dict[str, dict] = {}


def get(file: str, key: str = None, default: Any = None) -> Any:
    """
    Get configuration value from YAML file.

    Args:
        file: Config file name without extension (e.g., "processing", "settings")
        key: Dot-notation path to value (e.g., "frames.sample_rate")
             If None, returns entire config
        default: Default value if key not found

    Returns:
        Configuration value, entire config dict, or default

    Examples:
        >>> get("processing", "frames.sample_rate")
        1
        >>> get("settings", "llm.model")
        "gemini-2.0-flash"
        >>> get("categories", "order")
        ["infrastructure", "sla", "api", ...]
    """
    # Load config if not cached
    if file not in _cache:
        try:
            _cache[file] = _load_config(file)
        except FileNotFoundError:
            return default

    config = _cache[file]

    # Return entire config if no key specified
    if key is None:
        return config

    # Navigate nested keys using dot notation
    parts = key.split(".")
    value = config

    try:
        for part in parts:
            value = value[part]
        return value
    except (KeyError, TypeError):
        return default


def _load_config(file: str) -> dict:
    """Load YAML config file."""
    # Get config directory (same directory as this file)
    config_dir = Path(__file__).parent
    config_path = config_dir / f"{file}.yaml"

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def reload(file: str = None):
    """
    Reload configuration from disk.

    Args:
        file: Config file to reload. If None, reloads all cached configs.
    """
    if file is None:
        _cache.clear()
    elif file in _cache:
        del _cache[file]


def get_path(file: str, key: str) -> str:
    """
    Get a path from config and resolve it to absolute path.

    Args:
        file: Config file name
        key: Dot-notation path to path value

    Returns:
        Absolute path as string
    """
    path = get(file, key)

    # If already absolute, return as-is
    if os.path.isabs(path):
        return path

    # Resolve relative to project root (parent of config dir)
    project_root = Path(__file__).parent.parent
    return str(project_root / path)