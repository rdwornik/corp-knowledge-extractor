"""Deterministic document type classifier.

Classifies files into doc_types based on folder path and filename patterns.
Does NOT use LLM — pure rules. Fast and free.

doc_type determines extraction depth:
- "architecture", "security", "commercial", "product_doc" -> deep extraction
- "rfp_response" -> deep extraction
- "meeting", "training" -> deep extraction (meeting overlay)
- "general" -> standard extraction (base only)
"""

import re
from pathlib import Path

DEEP_DOC_TYPES = {
    "architecture",
    "security",
    "commercial",
    "product_doc",
    "rfp_response",
    "vendor_assessment",
    "discovery",
    "meeting",
    "training",
}
STANDARD_DOC_TYPES = {"general"}

# Filename-based doc_type patterns — checked BEFORE folder/content rules
FILENAME_DOC_TYPE_PATTERNS = [
    (r"(?i)(RFI|RFP|request.for.(information|proposal))", "rfp_response"),
    (r"(?i)(questionnaire|vendor.assessment|security.assessment|VA\b)", "vendor_assessment"),
    (r"(?i)(discovery|requirements)", "discovery"),
]


def classify_from_filename(filename: str) -> str | None:
    """Pre-classify doc_type from filename patterns.

    Returns doc_type string if a pattern matches, None otherwise.
    """
    for pattern, doc_type in FILENAME_DOC_TYPE_PATTERNS:
        if re.search(pattern, filename):
            return doc_type
    return None


def classify_doc_type(filepath: str, folder_context: str | None = None) -> str:
    """Classify a file's doc_type from its path and name.

    Rules (applied in order, first match wins):
    1. Folder path patterns (most reliable)
    2. Filename patterns
    3. Default: "general"
    """
    path_lower = filepath.lower().replace("\\", "/")
    name_lower = Path(filepath).name.lower()

    # --- Filename pattern rules (highest priority) ---
    filename_match = classify_from_filename(Path(filepath).name)
    if filename_match is not None:
        return filename_match

    # --- Folder path rules ---
    if "01_product_docs" in path_lower or "product_docs" in path_lower:
        return "product_doc"
    if "03_competitive" in path_lower:
        return "commercial"
    if "02_training" in path_lower or "training" in path_lower:
        return "training"
    if "certificate" in path_lower or "security" in path_lower or "compliance" in path_lower:
        return "security"
    if "rfp" in path_lower and ("response" in path_lower or "answer" in path_lower or "submission" in path_lower):
        return "rfp_response"
    if "discovery" in path_lower or "meeting" in path_lower or "workshop" in path_lower:
        return "meeting"

    # --- Filename rules ---
    if any(kw in name_lower for kw in ["architecture", "platform", "technical"]):
        return "architecture"
    if any(
        kw in name_lower
        for kw in [
            "sla",
            "soc_2",
            "soc2",
            "iso_27001",
            "iso27001",
            "security",
            "whitepaper",
            "gdpr",
        ]
    ):
        return "security"
    if any(kw in name_lower for kw in ["pricing", "commercial", "service_description", "contract"]):
        return "commercial"
    if any(kw in name_lower for kw in ["rfp_response", "rfp_answer", "rfi_response"]):
        return "rfp_response"
    if any(kw in name_lower for kw in ["meeting", "discovery", "workshop", "notes"]):
        return "meeting"
    if any(kw in name_lower for kw in ["training", "enablement", "bootcamp"]):
        return "training"

    return "general"


def should_extract_deep(doc_type: str) -> bool:
    """Whether this doc_type warrants deep (overlay) extraction."""
    return doc_type in DEEP_DOC_TYPES
