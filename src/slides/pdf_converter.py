"""Convert PPTX to PDF for Gemini multimodal extraction.

Converter cascade:
1. PowerPoint COM → PDF export (30s timeout, subprocess isolation)
2. LibreOffice headless → PDF (--macro-security-level=4, --norestore)
3. Fail → return None (caller falls back to text-only)
"""

import logging
import shutil
import subprocess
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

COM_TIMEOUT = 30
LIBREOFFICE_TIMEOUT = 120


def convert_pptx_to_pdf(pptx_path: Path, output_dir: Path) -> Path | None:
    """Convert PPTX to PDF using available converter.

    Returns the PDF path on success, None on failure.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / f"{pptx_path.stem}.pdf"

    # Try PowerPoint COM first
    try:
        result = _convert_via_com(pptx_path, pdf_path)
        if result and result.exists():
            logger.info("Converted %s to PDF via PowerPoint COM", pptx_path.name)
            return result
    except Exception as exc:
        logger.info("PowerPoint COM conversion failed: %s", exc)

    # Try LibreOffice headless
    try:
        result = _convert_via_libreoffice(pptx_path, output_dir)
        if result and result.exists():
            logger.info("Converted %s to PDF via LibreOffice", pptx_path.name)
            return result
    except Exception as exc:
        logger.info("LibreOffice conversion failed: %s", exc)

    logger.warning("All PDF converters failed for %s", pptx_path.name)
    return None


def _convert_via_com(pptx_path: Path, pdf_path: Path) -> Path | None:
    """Convert via PowerPoint COM automation in a subprocess (30s timeout)."""
    import sys

    script = (
        f"import comtypes.client; "
        f"ppt = comtypes.client.CreateObject('PowerPoint.Application'); "
        f"ppt.Visible = 1; "
        f"prs = ppt.Presentations.Open(r'{pptx_path.resolve()}', ReadOnly=True, Untitled=False, WithWindow=False); "
        f"prs.SaveAs(r'{pdf_path.resolve()}', 32); "  # 32 = ppSaveAsPDF
        f"prs.Close()"
    )

    subprocess.run(
        [sys.executable, "-c", script],
        timeout=COM_TIMEOUT,
        check=True,
        capture_output=True,
    )

    return pdf_path if pdf_path.exists() else None


def _convert_via_libreoffice(pptx_path: Path, output_dir: Path) -> Path | None:
    """Convert via LibreOffice headless."""
    lo_path = _find_libreoffice()
    if not lo_path:
        raise FileNotFoundError("LibreOffice not found")

    subprocess.run(
        [
            lo_path,
            "--headless",
            "--norestore",
            "--macro-security-level=4",
            "--convert-to", "pdf",
            "--outdir", str(output_dir),
            str(pptx_path),
        ],
        timeout=LIBREOFFICE_TIMEOUT,
        check=True,
        capture_output=True,
    )

    pdf_path = output_dir / f"{pptx_path.stem}.pdf"
    return pdf_path if pdf_path.exists() else None


def _find_libreoffice() -> str | None:
    """Find LibreOffice executable."""
    for name in ("libreoffice", "soffice"):
        found = shutil.which(name)
        if found:
            return found

    for candidate in [
        r"C:\Program Files\LibreOffice\program\soffice.exe",
        r"C:\Program Files (x86)\LibreOffice\program\soffice.exe",
    ]:
        if Path(candidate).exists():
            return candidate

    return None
