import sys
import os
import argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config_loader import get, get_path
from src.transcribe.groq_backend import transcribe_groq
from src.transcribe.openai_backend import transcribe_openai, transcribe_openai_with_preprocessing
from src.transcribe.provider_selector import (
    select_provider,
    estimate_transcription,
    print_estimate,
    validate_provider_selection
)
from src.frames.extractor import extract_frames
from src.ocr.reader import read_frames
from src.frames.tagger import tag_frames
from src.align.aligner import align
from src.anonymize.anonymizer import anonymize
from src.synthesize.gemini_backend import GeminiSynthesizer
from src.output.post_processor import post_process
from src.output.generator import generate_output

# Load configuration
INPUT_DIR = get_path("settings", "input.directory")
CUSTOM_TERMS = get("anonymize", "custom_terms", [])
VIDEO_EXTENSIONS = tuple(get("settings", "input.video_extensions", [".mp4", ".mkv", ".avi", ".mov"]))


def process_file(
    file_path: str,
    preset: str = None,
    sample_rate: int = None,
    pixel_threshold: float = None,
    provider: str = "auto",
    estimate_only: bool = False
):
    """
    Process a video file with optional preset configuration.

    Args:
        file_path: Path to video file
        preset: Preset name (powerpoint, excel, demo, audio_only, hybrid)
        sample_rate: Override sample rate
        pixel_threshold: Override pixel threshold
        provider: Transcription provider ("auto", "groq", "openai")
        estimate_only: Only show cost estimate, don't process
    """
    name = os.path.basename(file_path)
    print(f"\n{'='*50}")
    print(f"Processing: {name}")
    print('='*50)

    # Step 0: Provider selection and cost estimation
    print("\nStep 0: Analyzing file and selecting transcription provider...")

    if provider == "auto":
        # Automatic selection based on file analysis
        selected_provider = select_provider(file_path, show_estimate=True)
    else:
        # Manual provider selection - still show estimate
        estimate = estimate_transcription(file_path)
        print_estimate(estimate, file_path)

        # Override with user selection
        selected_provider = provider
        print(f"\n⚙️  User override: Using {selected_provider.upper()}")

    # Validate API key is available
    validate_provider_selection(selected_provider)

    # If estimate-only mode, stop here
    if estimate_only:
        print("\n✓ Estimate complete (--estimate-only mode)")
        return

    # Step 1: Transcription
    print(f"\nStep 1: Transcribing with {selected_provider.upper()}...")

    if selected_provider == "groq":
        t = transcribe_groq(file_path)
    elif selected_provider == "openai":
        # Use OpenAI with preprocessing to reduce costs
        use_preprocessing = get("settings", "transcription.silence_removal.enabled", True)
        t = transcribe_openai_with_preprocessing(
            file_path,
            enable_preprocessing=use_preprocessing
        )
    else:
        raise ValueError(f"Unknown provider: {selected_provider}")

    print(f"  ✓ {len(t)} segments transcribed")

    print("Step 2: Extracting frames...")
    f = extract_frames(
        file_path,
        preset=preset,
        sample_rate=sample_rate,
        threshold=pixel_threshold
    )
    print(f"  {len(f)} frames")

    # If no frames (audio-only mode), skip frame-dependent steps
    if len(f) == 0:
        print("  No frames extracted (audio-only mode)")
        print("Step 3-4: Skipping OCR and tagging (no frames)")
        print("Step 5: Skipping alignment (no frames)")
        print("Step 6: Anonymizing transcript...")

        # Anonymize transcript segments
        for segment in t:
            segment['text'] = anonymize(segment['text'], CUSTOM_TERMS)

        print("Step 7: Synthesizing with Gemini (audio-only mode)...")
        synth = GeminiSynthesizer()
        # TODO: Implement audio-only synthesis mode
        # For now, use regular synthesis with empty frames
        result = synth.synthesize([], t)

        print("Step 8: Post-processing...")
        result = post_process(result, [])

        print("Step 9: Generating output...")
        folder = generate_output(result, [])
        print(f"Done! Output: {folder}")
        return

    print("Step 3: OCR...")
    f = read_frames(f)

    print("Step 4: Tagging frames...")
    f = tag_frames(f)

    print("Step 5: Aligning...")
    aligned = align(t, f)

    print("Step 6: Anonymizing...")
    for item in aligned:
        item['speech'] = anonymize(item['speech'], CUSTOM_TERMS)
        item['slide_text'] = anonymize(item['slide_text'], CUSTOM_TERMS)

    for frame in f:
        frame['text'] = anonymize(frame.get('text', ''), CUSTOM_TERMS)

    print("Step 7: Synthesizing with Gemini...")
    synth = GeminiSynthesizer()
    result = synth.synthesize(f, aligned)

    print("Step 8: Post-processing (dedup, categorize)...")
    result = post_process(result, f)
    print(f"  {len(result['slide_breakdown'])} unique slides, {len(result['qa_pairs'])} Q&A pairs")

    print("Step 9: Generating output...")
    folder = generate_output(result, f)
    print(f"Done! Output: {folder}")


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Extract knowledge from corporate meeting recordings",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Preset Examples:
  python scripts/run.py                          # Default (PowerPoint preset)
  python scripts/run.py --preset excel           # Excel spreadsheet review
  python scripts/run.py --preset demo            # Software demonstration
  python scripts/run.py --preset audio_only      # Audio-only meeting
  python scripts/run.py --preset hybrid          # Auto-adaptive mode

Manual Override:
  python scripts/run.py --sample-rate 10 --pixel-threshold 0.25

Available Presets:
  powerpoint   - Slide presentations with distinct transitions (default)
  excel        - Spreadsheet reviews with scrolling
  demo         - Software demonstrations
  audio_only   - Audio-only meetings (no frames)
  hybrid       - Auto-adaptive (switches between modes)
        """
    )

    parser.add_argument(
        "--preset",
        choices=["powerpoint", "excel", "demo", "audio_only", "hybrid"],
        default=None,
        help="Content type preset (default: powerpoint behavior from config)"
    )

    parser.add_argument(
        "--sample-rate",
        type=int,
        default=None,
        help="Seconds between frame checks (overrides preset)"
    )

    parser.add_argument(
        "--pixel-threshold",
        type=float,
        default=None,
        help="Pixel change threshold 0.0-1.0 (overrides preset)"
    )

    parser.add_argument(
        "--file",
        type=str,
        default=None,
        help="Process specific file instead of all files in input directory"
    )

    parser.add_argument(
        "--provider",
        choices=["auto", "groq", "openai"],
        default="auto",
        help="Transcription provider (default: auto-select based on file size and cost)"
    )

    parser.add_argument(
        "--estimate-only",
        action="store_true",
        help="Show transcription cost estimate without processing"
    )

    args = parser.parse_args()

    # Determine files to process
    if args.file:
        if not os.path.exists(args.file):
            print(f"Error: File not found: {args.file}")
            return
        files = [args.file]
    else:
        files = [
            os.path.join(INPUT_DIR, f)
            for f in os.listdir(INPUT_DIR)
            if f.lower().endswith(VIDEO_EXTENSIONS)
        ]

    if not files:
        print(f"No video files in {INPUT_DIR}/")
        return

    # Show configuration
    print("="*60)
    print("CORPORATE KNOWLEDGE EXTRACTOR")
    print("="*60)
    if args.preset:
        print(f"Preset: {args.preset}")
    if args.sample_rate:
        print(f"Sample rate override: {args.sample_rate}s")
    if args.pixel_threshold:
        print(f"Pixel threshold override: {args.pixel_threshold}")
    print(f"Files to process: {len(files)}")
    print("="*60)

    for file_path in files:
        process_file(
            file_path,
            preset=args.preset,
            sample_rate=args.sample_rate,
            pixel_threshold=args.pixel_threshold,
            provider=args.provider,
            estimate_only=args.estimate_only
        )

    print("\n✓ All done!")


if __name__ == "__main__":
    main()