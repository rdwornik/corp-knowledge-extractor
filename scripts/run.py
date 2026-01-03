import sys
import os
import argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config_loader import get, get_path
from src.transcribe.groq_backend import transcribe_groq
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


def process_file(file_path: str, preset: str = None, sample_rate: int = None, pixel_threshold: float = None):
    """
    Process a video file with optional preset configuration.

    Args:
        file_path: Path to video file
        preset: Preset name (powerpoint, excel, demo, audio_only, hybrid)
        sample_rate: Override sample rate
        pixel_threshold: Override pixel threshold
    """
    name = os.path.basename(file_path)
    print(f"\n{'='*50}")
    print(f"Processing: {name}")
    print('='*50)

    print("Step 1: Transcribing (Groq API)...")
    t = transcribe_groq(file_path)
    print(f"  {len(t)} segments")

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
            pixel_threshold=args.pixel_threshold
        )

    print("\n✓ All done!")


if __name__ == "__main__":
    main()