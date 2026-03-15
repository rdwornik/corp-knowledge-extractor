"""
Corporate Knowledge Extractor - main entry point.

Pipeline:
    inventory → compress → sample_frames → extract (Gemini + frames)
             ->keep_slides + cleanup → correlate → synthesize

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

import json
import click
from dotenv import load_dotenv

load_dotenv()

from config.config_loader import load_config
from src.inventory import scan_input, FileType
from src.extract import extract_knowledge, extract_from_text, extract_local, extract_pptx_multimodal, ExtractionError
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

    Renames them slide_001.png, slide_002.png, ... sequentially (1-based),
    regardless of the frame_index or slide_number Gemini returned.
    Updates slide.slide_number in place so markdown references stay consistent.
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

    seq = 1  # Sequential counter for output filenames
    for slide in slides:
        frame_idx = slide.frame_index
        source_frame = frame_by_index.get(frame_idx)
        if source_frame and source_frame.path.exists():
            target = output_frames_dir / f"slide_{seq:03d}.png"
            shutil.copy2(source_frame.path, target)
            log.debug("Copied frame %d -> slide_%03d.png", frame_idx, seq)
            slide.slide_number = seq  # Keep slide_number in sync with filename
            seq += 1
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


def _keep_pptx_slides(rendered_slides: list, output_dir: Path) -> None:
    """Copy rendered PPTX slide PNGs to the output package.

    Slides are already named slide_001.png, slide_002.png, etc.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    for rs in rendered_slides:
        target = output_dir / rs.image_path.name
        if rs.image_path.exists():
            shutil.copy2(rs.image_path, target)
    # Cleanup temp rendered slides
    for rs in rendered_slides:
        if rs.image_path.exists():
            rs.image_path.unlink()
    # Remove empty temp dir
    try:
        if rendered_slides:
            rendered_slides[0].image_path.parent.rmdir()
    except OSError:
        pass


@click.group(invoke_without_command=True)
@click.pass_context
def cli(ctx):
    """Corporate Knowledge Extractor — API-first pipeline.

    Run with no arguments to process all files in data/input/ automatically.
    """
    if ctx.invoked_subcommand is None:
        # Default: process the configured input directory
        ctx.invoke(process)


@cli.command()
@click.argument("input_path", type=click.Path(exists=True), default=None, required=False)
@click.option("--output", default="output", show_default=True, help="Output directory")
@click.option("--name", default=None, help="Package name (defaults to input name)")
@click.option("--tier", type=click.IntRange(1, 3), default=None,
              help="Force extraction tier: 1=local, 2=text-AI, 3=multimodal (default: auto)")
@click.option("--dry-run-tiers", is_flag=True, help="Show tier routing + cost estimate without processing")
@click.option("--prompt-file", type=click.Path(exists=True), default=None,
              help="Custom extraction prompt file (replaces default prompt)")
@click.option("--model", default=None, help="Override Gemini model (default from settings.yaml)")
def process(input_path: str | None, output: str, name: str | None, tier: int | None, dry_run_tiers: bool, prompt_file: str | None, model: str | None):
    """Process a file or folder into a knowledge package.

    INPUT_PATH defaults to the data/input/ directory from config if not given.

    Pipeline: inventory > compress > sample_frames > extract (Gemini)
              > keep_slides + cleanup > correlate > synthesize
    """
    config = load_config()

    if model:
        config["model_override"] = model

    # Read custom prompt if provided
    custom_prompt = None
    if prompt_file:
        custom_prompt = Path(prompt_file).read_text(encoding="utf-8")

    # Resolve input path — fall back to configured input directory
    from datetime import datetime
    using_default_input = input_path is None
    if using_default_input:
        input_path = config.get("input", {}).get("directory", "data/input")
        _print(f"No path given — using configured input directory: {input_path}")

    input_p = Path(input_path)
    if not input_p.exists():
        _print(f"Input path does not exist: {input_p}")
        sys.exit(1)
    output_p = Path(output)
    # Use timestamp when processing the default input folder (stem would just be "input")
    if name:
        package_name = name
    elif using_default_input:
        package_name = datetime.now().strftime("%Y-%m-%d_%H%M")
    else:
        package_name = input_p.stem or "package"
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

    # --- 1b. Tier routing ---
    from src.tier_router import route_tier, Tier, estimate_batch_cost

    if dry_run_tiers:
        estimate = estimate_batch_cost(files, force_tier=tier)
        _print("\n=== TIER ROUTING ESTIMATE ===")
        for f, decision in estimate["decisions"]:
            tier_label = {1: "LOCAL (free)", 2: "TEXT-AI ($0.001)", 3: "MULTIMODAL ($0.03)"}
            _print(f"  Tier {decision.tier.value} {tier_label[decision.tier.value]:20s} {f.path.name}")
            _print(f"         {decision.reason}")
        _print(f"\nTier 1 (local):      {estimate['tier_counts'].get(1, 0)} files")
        _print(f"Tier 2 (text-AI):    {estimate['tier_counts'].get(2, 0)} files")
        _print(f"Tier 3 (multimodal): {estimate['tier_counts'].get(3, 0)} files")
        _print(f"Estimated total cost: ${estimate['total_cost']:.4f}")
        return

    tier_decisions = {}
    for f in files:
        decision = route_tier(f, force_tier=tier)
        tier_decisions[f.name] = decision
        _print(f"  Tier {decision.tier.value}: {f.path.name} ({decision.reason})")

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
            _print(f"  -> {f.path.name}")
            frames = sample_frames(f.path, temp_dir, config)
            sampled[f.name] = frames
            _print(f"    {len(frames)} frames sampled")

    # --- 4. Extract knowledge (tiered) ---
    _print("\nExtracting knowledge...")
    extracts: dict = {}
    failed: list[str] = []
    cost_total = 0.0

    rendered_pptx_slides: dict[str, list] = {}  # stem -> RenderedSlide list

    for f in files:
        decision = tier_decisions[f.name]
        frames = sampled.get(f.name)
        frame_count = len(frames) if frames else 0
        tier_label = f"[Tier {decision.tier.value}]"
        label = f"{tier_label} {f.path.name}" + (f" + {frame_count} frames" if frame_count else "")
        _print(f"  -> {label}")
        try:
            if decision.tier == Tier.LOCAL and decision.text_result:
                result = extract_local(f, decision.text_result)
            elif decision.tier == Tier.TEXT_AI and decision.text_result:
                result = extract_from_text(f, config, decision.text_result, custom_prompt=custom_prompt)
            elif f.path.suffix.lower() == ".pptx" and decision.tier == Tier.MULTIMODAL:
                # PPTX multimodal: render slides as PNG, send to Gemini
                from src.slides.renderer import render_slides
                temp_slides_dir = output_p / package_name / "temp_slides" / f.name
                rendered = render_slides(f.path, temp_slides_dir)
                rendered_pptx_slides[f.name] = rendered
                _print(f"    Rendered {len(rendered)} slide images")
                result = extract_pptx_multimodal(f, config, rendered, custom_prompt=custom_prompt)
            else:
                result = extract_knowledge(f, config, sampled_frames=frames, custom_prompt=custom_prompt)
            extracts[f.name] = result
            cost_total += decision.estimated_cost
            slide_info = f" | {len(result.slides)} slides identified" if result.slides else ""
            _print(f"    [OK] {result.title}{slide_info}")
        except ExtractionError as exc:
            log.warning("Skipping %s: %s", f.path.name, exc)
            failed.append(f.path.name)
            _print(f"    [FAIL] {exc}")

    if not extracts:
        _print("[red]No files were successfully extracted.[/red]" if HAS_RICH
               else "No files were successfully extracted.")
        sys.exit(1)

    if failed:
        _print(f"\n[yellow]Warning: {len(failed)} file(s) failed.[/yellow]" if HAS_RICH
               else f"\nWarning: {len(failed)} file(s) failed.")

    # --- 5. Keep only AI-identified slide frames, clean up temp ---
    _print("\nSelecting unique slide frames...")
    output_pptx_slides_dir = output_p / package_name / "source" / "slides"
    for stem, result in extracts.items():
        if result.slides and stem in sampled:
            keep_slide_frames(sampled[stem], result.slides, output_frames_dir, config)
            _print(f"  {stem}: kept {len(result.slides)} slide frame(s)")
        elif stem in rendered_pptx_slides:
            # PPTX multimodal: copy rendered slide PNGs to source/slides/
            _keep_pptx_slides(rendered_pptx_slides[stem], output_pptx_slides_dir)
            _print(f"  {stem}: kept {len(rendered_pptx_slides[stem])} PPTX slide(s)")
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

    # --- 8. Write machine-readable extract.json per file ---
    from datetime import datetime as _dt
    extract_dir = pkg_path / "extract"
    for stem, result in extracts.items():
        extract_json_path = extract_dir / f"{stem}.json"
        if custom_prompt:
            # Custom prompt: save raw Gemini JSON as-is (structure differs from standard)
            extract_data = {
                "source_file": str(result.source_file.path).replace('\\', '/'),
                "processed_at": _dt.now().isoformat(),
                **result.raw_json,
            }
        else:
            extract_data = {
                "schema_version": 2,
                "id": stem,
                "source_file": str(result.source_file.path).replace('\\', '/'),
                "title": result.title,
                "summary": result.summary,
                "topics": result.topics,
                "products": result.products,
                "people": result.people,
                "domains": result.domains,
                "key_points": result.key_points,
                "content_type": result.content_type,
                "source_type": result.source_type,
                "layer": result.layer,
                "confidentiality": result.confidentiality,
                "authority": result.authority,
                "client": result.client,
                "project": result.project,
                "slides_count": len(result.slides),
                "links_line": result.links_line,
                "validation_result": result.validation_result,
                "valid_to": result.raw_json.get("valid_to"),
                "source_date": result.source_date,
                "facts": result.facts,
                "processed_at": _dt.now().isoformat(),
            }
        extract_json_path.write_text(json.dumps(extract_data, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    _print(f"\n[green]Done![/green]" if HAS_RICH else "\nDone!")
    _print(f"Package: {pkg_path}")
    if cost_total > 0:
        _print(f"Estimated API cost: ${cost_total:.4f}")


@cli.command("process-manifest")
@click.argument("manifest_path", type=click.Path(exists=True))
@click.option("--resume", is_flag=True, help="Skip already-completed files")
@click.option("--max-rpm", default=100, show_default=True, help="Max requests per minute to Gemini API")
@click.option("--tier", type=click.IntRange(1, 3), default=None,
              help="Force extraction tier: 1=local, 2=text-AI, 3=multimodal (default: auto)")
@click.option("--batch", is_flag=True,
              help="Use Gemini Batch API (50% cheaper, async processing)")
@click.option("--batch-poll-interval", default=60, show_default=True,
              help="Seconds between batch status checks")
@click.option("--batch-timeout", default=86400, show_default=True,
              help="Max seconds to wait for batch completion")
@click.option("--model", default=None, help="Override Gemini model (default from settings.yaml)")
def process_manifest(manifest_path: str, resume: bool, max_rpm: int, tier: int | None,
                     batch: bool, batch_poll_interval: int, batch_timeout: int, model: str | None):
    """Process multiple files from a JSON manifest.

    Used by corp-project-extractor for batch extraction.

    Default mode: sequential Gemini API calls (immediate results).

    With --batch: submits all Tier 2 requests as a single Gemini Batch API
    job at 50% cost. Tier 1 (local) still runs immediately. Tier 3
    (multimodal) falls back to synchronous calls.

    Examples:

        cke process-manifest manifest.json --resume --max-rpm 80

        cke process-manifest manifest.json --batch --batch-poll-interval 30
    """
    from src.manifest import Manifest

    config = load_config()
    if model:
        config["model_override"] = model
    manifest = Manifest.from_file(Path(manifest_path))

    mode_label = "[bold]Batch API[/bold]" if batch else "[bold]Sequential[/bold]"
    if not HAS_RICH:
        mode_label = "Batch API" if batch else "Sequential"
    _print(f"\n{mode_label} processing: {len(manifest.files)} files from project '{manifest.project}'")
    _print(f"Output:     {manifest.output_dir}")
    if not batch:
        _print(f"Rate limit: {max_rpm} RPM")
    else:
        _print(f"Batch poll: every {batch_poll_interval}s (timeout: {batch_timeout}s)")
    if resume:
        _print("[yellow]Resume mode: skipping completed files[/yellow]" if HAS_RICH
               else "Resume mode: skipping completed files")
    _print("")

    if batch:
        from src.batch_api import BatchJobRunner
        runner = BatchJobRunner(manifest, config, force_tier=tier, resume=resume)
        summary = runner.run(poll_interval=batch_poll_interval, timeout=batch_timeout)
    else:
        from src.batch import BatchProcessor
        processor = BatchProcessor(manifest, config, max_rpm=max_rpm, resume=resume, force_tier=tier)
        summary = processor.process_all()

    _print("")
    done_str = f"[bold green]Done:[/bold green] {summary['done']}" if HAS_RICH else f"Done: {summary['done']}"
    err_str = f"[bold red]Errors:[/bold red] {summary['error']}" if HAS_RICH else f"Errors: {summary['error']}"
    skip_str = f"[bold yellow]Skipped:[/bold yellow] {summary['skipped']}" if HAS_RICH else f"Skipped: {summary['skipped']}"
    _print(done_str)
    _print(err_str)
    _print(skip_str)
    _print(f"Total: {summary['total']}")
    tiers = summary.get("tiers", {})
    if any(tiers.values()):
        _print(f"\nTiers: local={tiers.get(1, 0)}, text-AI={tiers.get(2, 0)}, multimodal={tiers.get(3, 0)}")
        cost = summary.get('cost', 0)
        cost_label = f"${cost:.4f}"
        if batch:
            cost_label += " (50% batch discount applied to Tier 2)"
        _print(f"Estimated API cost: {cost_label}")

    if summary["error"] > 0:
        status_path = manifest.output_dir / "status.json"
        _print(f"\n{'[red]' if HAS_RICH else ''}Check {status_path} for error details{'[/red]' if HAS_RICH else ''}")


@cli.command()
@click.argument("package_path", type=click.Path(exists=True))
def reextract(package_path: str):
    """Re-run extraction on an existing package (source/ is preserved)."""
    config = load_config()
    pkg = Path(package_path)
    _print(f"\nRe-extracting: {pkg}")
    try:
        reextract_package(pkg, config)
        _print(f"\n[green]Done.[/green]" if HAS_RICH else "\nDone.")
        _print(f"Package: {pkg}")
    except Exception as exc:
        _print(f"[red]Error: {exc}[/red]" if HAS_RICH else f"Error: {exc}")
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


@cli.command()
@click.argument("path", type=click.Path(exists=True))
@click.option("--recursive/--no-recursive", default=True, help="Scan subfolders")
@click.option("--output", "-o", default=None, help="Output JSON path (default: stdout)")
@click.option(
    "--exclude", multiple=True,
    default=["80_Archive", ".corp", "_knowledge", ".venv", "__pycache__", ".git"],
    help="Folders to skip",
)
def scan(path: str, recursive: bool, output: str | None, exclude: tuple[str, ...]):
    """Scan files and extract local metadata (Tier 1, no API calls)."""
    from src.scan import scan_path, results_to_json

    input_path = Path(path).resolve()
    _print(f"Scanning: {input_path}")
    _print(f"Recursive: {recursive} | Exclude: {', '.join(exclude)}")

    results = scan_path(input_path, recursive=recursive, exclude=exclude)
    data = results_to_json(results)

    json_str = json.dumps(data, indent=2, ensure_ascii=False, default=str)

    if output:
        out_path = Path(output)
        out_path.write_text(json_str, encoding="utf-8")
        _print(f"\nScanned {data['total_files']} files -> {out_path}")
    else:
        print(json_str)
        print(f"\nScanned {data['total_files']} files", file=sys.stderr)


if __name__ == "__main__":
    cli()
