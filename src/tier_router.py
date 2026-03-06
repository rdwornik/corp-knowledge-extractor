"""
Tiered extraction router — decides the cheapest extraction strategy per file.

Tier 1: Local text extraction only (FREE)
    - Plain text files (TXT, MD, CSV)
    - Well-structured documents with good text extraction quality
    - Still post-processed through corp-os-meta for normalization

Tier 2: Text-only AI — Gemini 2.0 Flash ($0.001/file)
    - Documents with good local text but needing AI for structure/insight
    - PDF, DOCX, PPTX, XLSX with extractable text
    - Sends extracted text to cheaper/faster model

Tier 3: Full multimodal AI — Gemini 2.5 Flash ($0.02-0.05/file)
    - Videos, audio (always need multimodal)
    - Image-heavy documents (scanned PDFs, visual PPTX)
    - Files where local extraction failed or is poor quality

Usage:
    from src.tier_router import route_tier, Tier, TierDecision

    decision = route_tier(source_file)
    print(f"Tier {decision.tier.value}: {decision.reason}")
    print(f"Est. cost: ${decision.estimated_cost:.4f}")
"""

import logging
from dataclasses import dataclass
from enum import IntEnum

from src.inventory import SourceFile, FileType
from src.text_extract import extract_text, TextExtractionResult

log = logging.getLogger(__name__)


class Tier(IntEnum):
    LOCAL = 1       # Free local extraction
    TEXT_AI = 2     # Text-only Gemini 2.0 Flash
    MULTIMODAL = 3  # Full multimodal Gemini 2.5 Flash


# Approximate cost per file by tier
TIER_COSTS = {
    Tier.LOCAL: 0.0,
    Tier.TEXT_AI: 0.001,
    Tier.MULTIMODAL: 0.03,  # avg of $0.02-0.05 range
}

# Model used for each tier
TIER_MODELS = {
    Tier.LOCAL: None,
    Tier.TEXT_AI: "gemini-2.5-flash",
    Tier.MULTIMODAL: "gemini-2.5-flash",
}


@dataclass
class TierDecision:
    """Result of tier routing decision."""
    tier: Tier
    reason: str
    estimated_cost: float
    model: str | None
    text_result: TextExtractionResult | None = None


def route_tier(
    file: SourceFile,
    force_tier: int | None = None,
) -> TierDecision:
    """Decide extraction tier for a file.

    Args:
        file: SourceFile to route
        force_tier: Override tier (1, 2, or 3). None = auto.

    Returns:
        TierDecision with tier, reason, estimated cost, and optional text result
    """
    # Manual override
    if force_tier is not None:
        tier = Tier(force_tier)
        return TierDecision(
            tier=tier,
            reason=f"forced to tier {tier.value}",
            estimated_cost=TIER_COSTS[tier],
            model=TIER_MODELS[tier],
        )

    # Video/Audio always need multimodal (Tier 3)
    if file.type in (FileType.VIDEO, FileType.AUDIO):
        return TierDecision(
            tier=Tier.MULTIMODAL,
            reason=f"{file.type.value} requires multimodal processing",
            estimated_cost=TIER_COSTS[Tier.MULTIMODAL],
            model=TIER_MODELS[Tier.MULTIMODAL],
        )

    # Try local text extraction
    text_result = extract_text(file.path)

    # No text extracted → multimodal
    if text_result.extraction_quality == "none":
        return TierDecision(
            tier=Tier.MULTIMODAL,
            reason=f"local extraction failed ({text_result.error or 'no text'})",
            estimated_cost=TIER_COSTS[Tier.MULTIMODAL],
            model=TIER_MODELS[Tier.MULTIMODAL],
            text_result=text_result,
        )

    # Image-heavy presentations → multimodal (visuals matter)
    if file.type == FileType.SLIDES and text_result.has_images:
        return TierDecision(
            tier=Tier.MULTIMODAL,
            reason="slides with images need visual analysis",
            estimated_cost=TIER_COSTS[Tier.MULTIMODAL],
            model=TIER_MODELS[Tier.MULTIMODAL],
            text_result=text_result,
        )

    # Partial extraction (scanned PDF, sparse text) → multimodal
    if text_result.extraction_quality == "partial":
        return TierDecision(
            tier=Tier.MULTIMODAL,
            reason=f"partial text extraction ({text_result.char_count} chars)",
            estimated_cost=TIER_COSTS[Tier.MULTIMODAL],
            model=TIER_MODELS[Tier.MULTIMODAL],
            text_result=text_result,
        )

    # Good text extraction — plain text files can skip AI entirely (Tier 1)
    if file.type in (FileType.NOTE, FileType.TRANSCRIPT) and text_result.char_count < 5000:
        return TierDecision(
            tier=Tier.LOCAL,
            reason=f"small text file ({text_result.char_count} chars)",
            estimated_cost=TIER_COSTS[Tier.LOCAL],
            model=TIER_MODELS[Tier.LOCAL],
            text_result=text_result,
        )

    # Good text from structured documents → text-only AI (Tier 2)
    return TierDecision(
        tier=Tier.TEXT_AI,
        reason=f"good text extraction ({text_result.char_count} chars, {text_result.extractor})",
        estimated_cost=TIER_COSTS[Tier.TEXT_AI],
        model=TIER_MODELS[Tier.TEXT_AI],
        text_result=text_result,
    )


def estimate_batch_cost(files: list[SourceFile], force_tier: int | None = None) -> dict:
    """Estimate total cost for a batch of files.

    Returns dict with per-tier counts and total estimated cost.
    """
    tiers = {Tier.LOCAL: 0, Tier.TEXT_AI: 0, Tier.MULTIMODAL: 0}
    total_cost = 0.0
    decisions = []

    for f in files:
        decision = route_tier(f, force_tier=force_tier)
        tiers[decision.tier] += 1
        total_cost += decision.estimated_cost
        decisions.append((f, decision))

    return {
        "tier_counts": {t.value: count for t, count in tiers.items()},
        "total_cost": total_cost,
        "decisions": decisions,
        "file_count": len(files),
    }
