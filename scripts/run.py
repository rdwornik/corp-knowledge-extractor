"""
Corporate Knowledge Extractor - main entry point.

Pipeline:
    inventory → compress → sample_frames → extract (Gemini + frames)
             → keep_slides + cleanup → correlate → synthesize

Commands:
    process    Process a file or folder (new package)
    reextract  Re-run extraction on existing package
    info       Show package info
"""

import logging
import shutil
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
from src.frames.sampler import sample_frames, SampledFrame
from src.compress import compress_video, needs_compression

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

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
        # Strip Rich markup for plain output
        import re
        print(re.sub(r"\[/?[a-z/ ]+\]", "", msg))


def keep_slide_frames(
    all_frames: list[SampledFrame],
    slides: list,
    output_frames_dir: Path,
    config: dict,
) -> None:
    """
    Copy only the frames Gemini identified as unique slides to the output dir.

    Renames them slide_001.png, slide_002.png, ... to match slide_number.
    Deletes the temp sampled frames when cleanup_non_slides is enabled.

    Args:
        all_frames: All SampledFrame objects from sample_frames()
        slides: SlideInfo list from ExtractionResult.slides
        output_frames_dir: Where to write slide_NNN.png files
        config: Unified config dict
    """
    output_frames_dir.mkdir(parents=True, exist_ok=True)

    # Build lookup: frame_index → SampledFrame
    frame_by_index = {f.index: f for f in all_frames}

    for slide in slides:
        frame_idx = slide.frame_index
        source_frame = frame_by_index.get(frame_idx)
        if source_frame and source_frame.path.exists():
            target = output_frames_dir / f"slide_{slide.slide_number:03d}.png"
            shutil.copy2(source_frame.path, target)
            log.debug("Copied frame %d → %s", frame_idx, target.name)
        else:
            log.warning(
                "Slide %d references frame_index=%d but no matching sample found",
                slide.slide_number, frame_idx,
            )

    # Cleanup temp frames
    if config.get("frame_sampling", {}).get("cleanup_non_slides", True):
        for f in all_frames:
            if f.path.exists():
                f.path.unlink()
        # Remove empty temp dir
        try:
            if all_frames:
                all_frames[0].path.parent.rmdir()
        except OSError:
            pass  # Not empty — leave it


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

    Pipeline: inventory → compress → sample_frames → extract (Gemini)
              → keep_slides + cleanup → correlate → synthesize
    """
    config = load_config()
    input_p = Path(input_path)
    output_p = Path(output)
    package_name = name or input_p.stem or "package"
    output_frames_dir = output_p / package_name / "source" / "frames"

    _print(f"\n[bold]Corporate Knowledge Extractor[/bold]" if HAS_RICH
           else "\nCorporate Knowledge Extractor")
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
        _print(f"  {f.type.value:12s}  {f.path.name}  ({f.size_bytes / 1024 / 1024:.1f} MB)")

    # --- 2. Compress videos ---
    for f in files:
        if f.type == FileType.VIDEO:
            if needs_compression(f.path, config):
                compressed_out = output_p / package_name / "source" / "video" / f.path.name
                _print(f"\nCompressing {f.path.name}...")
                compress_video(f.path, compressed_out, config)

    # --- 3. Sample frames from videos ---
    _print("\nSampling frames from videos...")
    sampled: dict[str, list[SampledFrame]] = {}  # stem → frames
    for f in files:
        if f.type == FileType.VIDEO:
            temp_dir = output_p / package_name / "temp_frames" / f.name
            _print(f"  → {f.path.name}")
            frames = sample_frames(f.path, temp_dir, config)
            sampled[f.name] = frames
            _print(f"    {len(frames)} frames sampled")

    # --- 4. Extract knowledge via Gemini ---
    _print("\nExtracting knowledge with Gemini...")
    extracts: dict = {}
    failed: list[str] = []

    for f in files:
        frames = sampled.get(f.name)
        frame_count = len(frames) if frames else 0
        label = f.path.name + (f" + {frame_count} frames" if frame_count else "")
        _print(f"  → {label}")
        try:
            result = extract_knowledge(f, config, sampled_frames=frames)
            extracts[f.name] = result
            slide_info = f" | {len(result.slides)} slides identified" if result.slides else ""
            _print(f"    ✓ {result.title}{slide_info}")
        except ExtractionError as exc:
            log.warning("Skipping %s: %s", f.path.name, exc)
            failed.append(f.path.name)
            _print(f"    ✗ {exc}")

    if not extracts:
        _print("[red]No files were successfully extracted.[/red]" if HAS_RICH
               else "No files were successfully extracted.")
        sys.exit(1)

    if failed:
        _print(f"\n[yellow]Warning: {len(failed)} file(s) failed.[/yellow]" if HAS_RICH
               else f"\nWarning: {len(failed)} file(s) failed.")

    # --- 5. Keep only AI-identified slide frames, clean up temp ---
    _print("\nSelecting unique slide frames...")
    for stem, result in extracts.items():
        if result.slides and stem in sampled:
            keep_slide_frames(sampled[stem], result.slides, output_frames_dir, config)
            _print(f"  {stem}: kept {len(result.slides)} slide frame(s)")
        elif stem in sampled and not result.slides:
            # No slides identified — clean up temp frames
            if config.get("frame_sampling", {}).get("cleanup_non_slides", True):
                for sf in sampled[stem]:
                    if sf.path.exists():
                        sf.path.unlink()
            _print(f"  {stem}: no slides identified, temp frames removed")

    # --- 6. Correlate ---
    _print("\nGrouping related files...")
    groups = correlate_files(files, extracts)
    _print(f"  {len(groups)} group(s)")

    # --- 7. Build package ---
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
    _print(f"\nRe-extracting: {pkg}")
    try:
        reextract_package(pkg, config)
        _print(f"\n[green]✓ Done.[/green]" if HAS_RICH else "\n✓ Done.")
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
        with open(meta_path, encoding="utf-8") as fh:
            meta = yaml.safe_load(fh)
        _print(f"Extracted at: {meta.get('extracted_at', 'unknown')}")
        _print(f"Model:        {meta.get('model', 'unknown')}")
        _print(f"Pipeline:     v{meta.get('pipeline_version', 'unknown')}")
        for sf in meta.get("source_files") or []:
            size_mb = (sf.get("size_bytes") or 0) / (1024 * 1024)
            _print(f"  {sf.get('type', '?'):12s}  {sf.get('path', '?')}  ({size_mb:.1f} MB)")
    else:
        _print("  [no _meta.yaml found]")

    extract_dir = pkg / "extract"
    if extract_dir.exists():
        extracts = [p for p in extract_dir.glob("*.md") if p.name != "synthesis.md"]
        _print(f"\nExtract files: {len(extracts)}")

    frames_dir = pkg / "source" / "frames"
    if frames_dir.exists():
        slide_files = sorted(frames_dir.glob("slide_*.png"))
        _print(f"Slide frames:  {len(slide_files)}")

    history_dir = pkg / ".history"
    if history_dir.exists():
        versions = sorted(history_dir.iterdir())
        _print(f"History:       {len(versions)} version(s)")


if __name__ == "__main__":
    cli()
