"""
Local text extraction from documents — Tier 1 (FREE, no API calls).

Extracts plain text from PDF, DOCX, PPTX, XLSX, CSV, and text files
using local libraries. Returns structured result with quality metrics
so the tier router can decide whether AI extraction is needed.

Usage:
    from src.text_extract import extract_text, TextExtractionResult

    result = extract_text(Path("report.pdf"))
    if result.extraction_quality == "good":
        # Tier 1 or 2 is sufficient
    else:
        # Fall back to Tier 3 multimodal
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger(__name__)


@dataclass
class TextExtractionResult:
    """Result of local text extraction."""
    text: str
    char_count: int = 0
    page_count: int = 0
    slide_count: int = 0
    has_images: bool = False
    extraction_quality: str = "none"  # "good", "partial", "none"
    extractor: str = "unknown"
    error: str | None = None


def extract_text(path: Path) -> TextExtractionResult:
    """Extract text from a file using the appropriate local extractor.

    Args:
        path: Path to the file to extract text from

    Returns:
        TextExtractionResult with extracted text and quality metrics
    """
    ext = path.suffix.lower()
    extractors = {
        ".pdf": _extract_pdf,
        ".docx": _extract_docx,
        ".pptx": _extract_pptx,
        ".xlsx": _extract_xlsx,
        ".csv": _extract_csv,
        ".txt": _extract_plaintext,
        ".md": _extract_plaintext,
        ".markdown": _extract_plaintext,
        ".rst": _extract_plaintext,
        ".log": _extract_plaintext,
    }

    extractor = extractors.get(ext)
    if not extractor:
        return TextExtractionResult(
            text="",
            extraction_quality="none",
            extractor="unsupported",
            error=f"No local extractor for {ext}",
        )

    try:
        return extractor(path)
    except Exception as exc:
        log.warning("Local text extraction failed for %s: %s", path.name, exc)
        return TextExtractionResult(
            text="",
            extraction_quality="none",
            extractor=ext.lstrip("."),
            error=str(exc),
        )


def _assess_quality(text: str, page_count: int = 0) -> str:
    """Assess extraction quality based on text density."""
    char_count = len(text)
    if char_count == 0:
        return "none"
    # Very short text relative to page count suggests scanned/image-heavy doc
    if page_count > 0 and char_count / page_count < 100:
        return "partial"
    if char_count < 200:
        return "partial"
    return "good"


def _extract_pdf(path: Path) -> TextExtractionResult:
    """Extract text from PDF using pdfplumber."""
    import pdfplumber

    pages_text = []
    has_images = False

    with pdfplumber.open(path) as pdf:
        page_count = len(pdf.pages)
        for page in pdf.pages:
            text = page.extract_text() or ""
            pages_text.append(text)
            if page.images:
                has_images = True

    full_text = "\n\n".join(pages_text)
    return TextExtractionResult(
        text=full_text,
        char_count=len(full_text),
        page_count=page_count,
        has_images=has_images,
        extraction_quality=_assess_quality(full_text, page_count),
        extractor="pdfplumber",
    )


def _extract_docx(path: Path) -> TextExtractionResult:
    """Extract text from DOCX using python-docx."""
    import docx

    doc = docx.Document(path)
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    has_images = any(
        r.element.tag.endswith("}drawing") or r.element.tag.endswith("}pict")
        for p in doc.paragraphs
        for r in p.runs
        for child in r.element
    )

    full_text = "\n\n".join(paragraphs)
    return TextExtractionResult(
        text=full_text,
        char_count=len(full_text),
        page_count=0,  # DOCX doesn't have reliable page count
        has_images=has_images,
        extraction_quality=_assess_quality(full_text),
        extractor="python-docx",
    )


def _extract_pptx(path: Path) -> TextExtractionResult:
    """Extract text from PPTX using python-pptx."""
    from pptx import Presentation

    prs = Presentation(path)
    slides_text = []
    has_images = False

    for slide in prs.slides:
        texts = []
        for shape in slide.shapes:
            if shape.has_text_frame:
                for paragraph in shape.text_frame.paragraphs:
                    text = paragraph.text.strip()
                    if text:
                        texts.append(text)
            if shape.shape_type == 13:  # MSO_SHAPE_TYPE.PICTURE
                has_images = True
        if texts:
            slides_text.append("\n".join(texts))

    full_text = "\n\n---\n\n".join(slides_text)
    return TextExtractionResult(
        text=full_text,
        char_count=len(full_text),
        slide_count=len(prs.slides),
        has_images=has_images,
        extraction_quality=_assess_quality(full_text, len(prs.slides)),
        extractor="python-pptx",
    )


def _extract_xlsx(path: Path) -> TextExtractionResult:
    """Extract text from XLSX using openpyxl."""
    import openpyxl

    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    sheets_text = []

    for sheet_name in wb.sheetnames:
        ws = wb[sheet_name]
        rows = []
        for row in ws.iter_rows(values_only=True):
            cells = [str(c) for c in row if c is not None]
            if cells:
                rows.append(" | ".join(cells))
        if rows:
            sheets_text.append(f"## {sheet_name}\n" + "\n".join(rows))

    wb.close()
    full_text = "\n\n".join(sheets_text)
    return TextExtractionResult(
        text=full_text,
        char_count=len(full_text),
        extraction_quality=_assess_quality(full_text),
        extractor="openpyxl",
    )


def _extract_csv(path: Path) -> TextExtractionResult:
    """Extract text from CSV."""
    import csv

    rows = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f)
        for row in reader:
            if any(cell.strip() for cell in row):
                rows.append(" | ".join(row))

    full_text = "\n".join(rows)
    return TextExtractionResult(
        text=full_text,
        char_count=len(full_text),
        extraction_quality=_assess_quality(full_text),
        extractor="csv",
    )


def _extract_plaintext(path: Path) -> TextExtractionResult:
    """Extract text from plain text files."""
    text = path.read_text(encoding="utf-8", errors="replace")
    return TextExtractionResult(
        text=text,
        char_count=len(text),
        extraction_quality=_assess_quality(text),
        extractor="plaintext",
    )
