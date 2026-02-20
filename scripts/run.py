"""
Corporate Knowledge Extractor - main entry point.

Pipeline:
    inventory → compress → frame_extract (videos) → extract (Gemini) → correlate → synthesize

Commands:
    process    Process a file or folder (new package)
    reextract  Re-run extraction on existing package
    info       Show package info
"""

import logging
import os
import sys
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))

import click
from dotenv import load_dotenv

load_dotenv()

from config.config_loader import load_config
from src.inventory import scan_input, FileType
from src.extract import extract_knowledge, ExtractionError
from src.correlate import correlate_files
from src.synthesize import build_package
from src.reextract import reextract_package
from src.frames.extractor import extract_frames
from src.compress import compress_video, needs_compression

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# Rich output (optional)
try:
    from rich.console import Console
    console = Console()
    HAS_RICH = True
except ImportError:
    HAS_RICH = False
    console = None


def _print(msg: str) -> None:
    if HAS_RICH:
        console.print(msg)
    else:
        print(msg)


@click.group()
def cli():
    """Corporate Knowledge Extractor — API-first pipeline."""
    pass


@cli.command()
@click.argument("input_path", type=click.Path(exists=True))
@click.option("--output", default="output", show_default=True, help="Output directory")
@click.option("--name", default=None, help="Package name (defaults to input name)")
def process(input_path: str, output: str, name: str | None):
    """Process a file or folder into a knowledge package.

    Pipeline: inventory → compress → frame_extract → extract (Gemini) → correlate → synthesize
    """
    config = load_config()
    input_p = Path(input_path)
    output_p = Path(output)
    package_name = name or input_p.stem or "package"

    # Pre-compute frames directory (inside the package, before it's built)
    frames_dir = output_p / package_name / "source" / "frames"

    _print(f"\n[bold]Corporate Knowledge Extractor[/bold]" if HAS_RICH else "\nCorporate Knowledge Extractor")
    _print(f"Input:   {input_p}")
    _print(f"Output:  {output_p / package_name}")
    _print("")

    # --- 1. Scan ---
    _print("Scanning input files...")
    files = scan_input(input_p, config)
    if not files:
        _print("[red]No supported files found.[/red]" if HAS_RICH else "No supported files found.")
        sys.exit(1)

    _print(f"Found {len(files)} file(s):")
    for f in files:
        size_mb = f.size_bytes / (1024 * 1024)
        _print(f"  {f.type.value:12s}  {f.path.name}  ({size_mb:.1f} MB)")

    # --- 2. Compress videos if needed ---
    compressed_paths: dict[str, Path] = {}  # stem → compressed path
    for f in files:
        if f.type == FileType.VIDEO:
            if needs_compression(f.path, config):
                compressed_out = output_p / package_name / "source" / "video" / f.path.name
                _print(f"\nCompressing {f.path.name}...")
                result_path = compress_video(f.path, compressed_out, config)
                compressed_paths[f.name] = result_path
            else:
                skip_mb = config.get("compression", {}).get("skip_if_under_mb", 50)
                size_mb = f.size_bytes / (1024 * 1024)
                _print(f"Skipping compression: {f.path.name} is {size_mb:.0f}MB (threshold: {skip_mb}MB)")

    # --- 3. Extract frames from videos ---
    frame_results: dict[str, list[Path]] = {}  # stem → list[Path]
    for f in files:
        if f.type == FileType.VIDEO:
            _print(f"\nExtracting frames from {f.path.name}...")
            try:
                frames = extract_frames(f.path, frames_dir, config)
                if frames:
                    frame_results[f.name] = frames
                    _print(f"  → {len(frames)} slide frames extracted")
                else:
                    _print(f"  → No frames extracted")
            except Exception as exc:
                log.warning("Frame extraction failed for %s: %s", f.path.name, exc)
                _print(f"  → Frame extraction failed: {exc}")

    # --- 4. Extract knowledge via Gemini ---
    _print("\nExtracting knowledge with Gemini...")
    extracts: dict = {}
    failed: list[str] = []

    for f in files:
        frames = frame_results.get(f.name)  # None for non-video files
        frame_count = len(frames) if frames else 0
        label = f"{f.path.name}" + (f" + {frame_count} frames" if frame_count else "")
        _print(f"  → {label}")
        try:
            result = extract_knowledge(f, config, frames=frames)
            extracts[f.name] = result
            slide_info = f" | {len(result.slides)} slides" if result.slides else ""
            _print(f"    ✓ {result.title}{slide_info}")
        except ExtractionError as exc:
            log.warning("Skipping %s: %s", f.path.name, exc)
            failed.append(f.path.name)
            _print(f"    ✗ Failed: {exc}")

    if not extracts:
        _print("[red]No files were successfully extracted.[/red]" if HAS_RICH
               else "No files were successfully extracted.")
        sys.exit(1)

    if failed:
        _print(f"\n[yellow]Warning: {len(failed)} file(s) failed extraction.[/yellow]" if HAS_RICH
               else f"\nWarning: {len(failed)} file(s) failed extraction.")

    # --- 5. Correlate ---
    _print("\nGrouping related files...")
    groups = correlate_files(files, extracts)
    _print(f"  {len(groups)} group(s)")

    # --- 6. Build package ---
    _print("\nBuilding package...")
    pkg_path = build_package(groups, extracts, output_p, package_name, config)

    _print(f"\n[green]✓ Done![/green]" if HAS_RICH else "\n✓ Done!")
    _print(f"Package: {pkg_path}")


@cli.command()
@click.argument("package_path", type=click.Path(exists=True))
def reextract(package_path: str):
    """Re-run extraction on an existing package (source/ is preserved)."""
    config = load_config()
    pkg = Path(package_path)

    _print(f"\nRe-extracting package: {pkg}")
    _print("Note: source/ is immutable — only extract/ will be regenerated.\n")

    try:
        reextract_package(pkg, config)
        _print(f"\n[green]✓ Re-extraction complete.[/green]" if HAS_RICH else "\n✓ Re-extraction complete.")
        _print(f"Package: {pkg}")
    except Exception as exc:
        _print(f"[red]✗ Error: {exc}[/red]" if HAS_RICH else f"✗ Error: {exc}")
        log.exception("Re-extraction failed")
        sys.exit(1)


@cli.command()
@click.argument("package_path", type=click.Path(exists=True))
def info(package_path: str):
    """Show info about an existing knowledge package."""
    import yaml

    pkg = Path(package_path)
    meta_path = pkg / "extract" / "_meta.yaml"

    _print(f"\nPackage: {pkg.name}")
    _print(f"Path:    {pkg}\n")

    if meta_path.exists():
        with open(meta_path, encoding="utf-8") as f:
            meta = yaml.safe_load(f)
        _print(f"Extracted at:  {meta.get('extracted_at', 'unknown')}")
        _print(f"Model:         {meta.get('model', 'unknown')}")
        _print(f"Pipeline:      v{meta.get('pipeline_version', 'unknown')}")
        _print(f"Prompt hash:   {meta.get('prompt_hash', 'unknown')}")
        src_files = meta.get("source_files") or []
        _print(f"Source files:  {len(src_files)}")
        for sf in src_files:
            size_mb = (sf.get("size_bytes") or 0) / (1024 * 1024)
            _print(f"  {sf.get('type', '?'):12s}  {sf.get('path', '?')}  ({size_mb:.1f} MB)")
    else:
        _print("  [no _meta.yaml found]")

    extract_dir = pkg / "extract"
    if extract_dir.exists():
        extracts = [p for p in extract_dir.glob("*.md") if p.name != "synthesis.md"]
        _print(f"\nExtract files: {len(extracts)}")
        for e in sorted(extracts):
            _print(f"  {e.name}")

    history_dir = pkg / ".history"
    if history_dir.exists():
        versions = sorted(history_dir.iterdir())
        _print(f"\nHistory: {len(versions)} previous version(s)")
        for v in versions:
            _print(f"  {v.name}")

    frames_dir = pkg / "source" / "frames"
    if frames_dir.exists():
        frame_files = sorted(frames_dir.glob("frame_*.png"))
        _print(f"\nFrames: {len(frame_files)}")


if __name__ == "__main__":
    cli()
