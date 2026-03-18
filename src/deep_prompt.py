"""Deep extraction prompt builder.

Loads the deep_prompt.txt template and injects overlay field definitions
based on the document's doc_type. Falls back to standard prompt for
doc_types without overlay mappings.
"""

import logging
from pathlib import Path

import yaml

log = logging.getLogger(__name__)

_PROMPTS_DIR = Path(__file__).parent.parent / "config" / "prompts"


def _load_overlay_fields() -> dict[str, str]:
    """Load overlay field templates from YAML config."""
    path = _PROMPTS_DIR / "overlay_fields.yaml"
    if not path.exists():
        log.warning("overlay_fields.yaml not found at %s", path)
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def build_deep_prompt(doc_type: str) -> str:
    """Build a deep extraction prompt with overlay fields for the given doc_type.

    Args:
        doc_type: The document type classification (e.g. "architecture", "security")

    Returns:
        The complete prompt string with overlay fields injected.
    """
    template_path = _PROMPTS_DIR / "deep_prompt.txt"
    if not template_path.exists():
        raise FileNotFoundError(f"Deep prompt template not found: {template_path}")

    template = template_path.read_text(encoding="utf-8")
    overlay_fields_map = _load_overlay_fields()

    overlay_fields = overlay_fields_map.get(doc_type, "")
    if not overlay_fields:
        log.warning("No overlay fields defined for doc_type=%s, using empty overlay", doc_type)

    prompt = template.replace("{doc_type}", doc_type).replace("{overlay_fields}", overlay_fields.strip())
    return prompt


def build_deep_multimodal_prompt(doc_type: str) -> str:
    """Build a unified deep multimodal prompt for Tier 3 video/PPTX extraction.

    Combines per-slide narrative analysis with global structured output
    (key_facts, entities, overlay) in a single prompt. Avoids the competing
    system_instruction vs user prompt problem.

    Args:
        doc_type: The document type classification (e.g. "training", "architecture")

    Returns:
        The complete unified prompt string with overlay fields injected.
    """
    template_path = _PROMPTS_DIR / "deep_multimodal.txt"
    if not template_path.exists():
        raise FileNotFoundError(f"Deep multimodal prompt template not found: {template_path}")

    template = template_path.read_text(encoding="utf-8")
    overlay_fields_map = _load_overlay_fields()

    overlay_fields = overlay_fields_map.get(doc_type, "")
    if not overlay_fields:
        log.warning("No overlay fields defined for doc_type=%s, using empty overlay", doc_type)

    prompt = template.replace("{doc_type}", doc_type).replace("{overlay_fields}", overlay_fields.strip())
    return prompt
