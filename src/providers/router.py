"""Route extraction requests to the right provider based on tier and document size."""

import logging
import os
from pathlib import Path

from dotenv import load_dotenv

from src.providers.base import ExtractionProvider

# Global API keys (Documents/.secrets/.env)
_global_env = Path.home() / "Documents" / ".secrets" / ".env"
if _global_env.exists():
    load_dotenv(_global_env, override=False)
# Local .env (project-specific vars only)
load_dotenv(override=False)

logger = logging.getLogger(__name__)

# Model -> provider mapping
ANTHROPIC_MODELS = {"claude-haiku-4-5-20251001", "claude-sonnet-4-6"}
GEMINI_MODELS = {"gemini-3-flash-preview", "gemini-3.1-flash-lite", "gemini-3.1-pro-preview"}

# Default routing rules
DEFAULT_TEXT_MODEL = "claude-haiku-4-5-20251001"
DEFAULT_MULTIMODAL_MODEL = "gemini-3-flash-preview"
DEFAULT_LARGE_CONTEXT_MODEL = "gemini-3-flash-preview"  # >190K tokens
ESCALATION_MODEL = "claude-sonnet-4-6"

# Token threshold for Haiku context limit
HAIKU_TOKEN_LIMIT = 190_000  # 200K context, leave 10K buffer


def _has_anthropic_key() -> bool:
    """Check if ANTHROPIC_API_KEY is available."""
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


def get_provider(model: str) -> ExtractionProvider:
    """Instantiate the correct provider for a given model.

    Falls back to Gemini if Anthropic key is missing.
    """
    if model in ANTHROPIC_MODELS:
        if not _has_anthropic_key():
            logger.warning(
                "ANTHROPIC_API_KEY not set — falling back to Gemini for model %s",
                model,
            )
            from src.providers.gemini_provider import GeminiProvider

            return GeminiProvider()
        from src.providers.anthropic_provider import AnthropicProvider

        return AnthropicProvider()
    elif model in GEMINI_MODELS:
        from src.providers.gemini_provider import GeminiProvider

        return GeminiProvider()
    else:
        raise ValueError(f"Unknown model: {model}. Known: {ANTHROPIC_MODELS | GEMINI_MODELS}")


def select_model(
    file_path: Path,
    file_size: int,
    has_images: bool = False,
    model_override: str | None = None,
) -> tuple[str, str]:
    """Auto-select LLM model based on file type and content.

    Args:
        file_path: Path to the source file
        file_size: File size in bytes
        has_images: Whether the file contains images (relevant for PDFs)
        model_override: Explicit model from --model flag (takes precedence)

    Returns:
        Tuple of (model_name, routing_reason)
    """
    if model_override:
        MODEL_MAP = {"pro": "gemini-3.1-pro-preview", "flash": "gemini-3-flash-preview"}
        resolved = MODEL_MAP.get(model_override, model_override)
        return resolved, "manual_override"

    ext = file_path.suffix.lower()

    if file_size < 5000:
        return "free", "small_file_local"

    if ext in (".pptx", ".mp4", ".mkv", ".avi", ".mov", ".wav"):
        return "gemini-3.1-pro-preview", "pptx_multimodal" if ext == ".pptx" else "video_multimodal"

    if ext == ".pdf" and has_images:
        return "gemini-3.1-pro-preview", "pdf_with_images"

    return "gemini-3-flash-preview", "text_default"


def route_model(
    tier: int,
    text_length: int = 0,
    model_override: str | None = None,
    batch_mode: bool = False,
) -> str:
    """Determine which model to use based on tier and document size.

    Args:
        tier: CKE extraction tier (1=local, 2=text-AI, 3=multimodal)
        text_length: approximate character count of extracted text
        model_override: explicit model from --model flag (takes precedence)
        batch_mode: if True, force Gemini for batch API discount (50% off > Haiku savings)

    Returns:
        Model identifier string
    """
    if model_override:
        return model_override

    # Tier 3 (multimodal) always uses Gemini
    if tier == 3:
        return DEFAULT_MULTIMODAL_MODEL

    # Batch mode forces Gemini (50% batch discount beats Haiku pricing)
    if batch_mode:
        logger.info("Batch mode active, routing to Gemini for batch discount")
        return DEFAULT_LARGE_CONTEXT_MODEL

    # Estimate tokens (rough: 4 chars per token)
    estimated_tokens = text_length // 4

    # Large documents -> Gemini (1M context)
    if estimated_tokens > HAIKU_TOKEN_LIMIT:
        logger.info(
            "Document too large for Haiku (%d est. tokens), routing to Gemini",
            estimated_tokens,
        )
        return DEFAULT_LARGE_CONTEXT_MODEL

    # No Anthropic key -> fall back to Gemini
    if not _has_anthropic_key():
        logger.warning("ANTHROPIC_API_KEY not set — routing to Gemini")
        return DEFAULT_LARGE_CONTEXT_MODEL

    # Default text extraction -> Claude Haiku
    return DEFAULT_TEXT_MODEL
