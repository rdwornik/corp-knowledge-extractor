import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.transcribe.groq_backend import transcribe_groq
from src.frames.extractor import extract_frames
from src.ocr.reader import read_frames
from src.frames.tagger import tag_frames
from src.align.aligner import align
from src.anonymize.anonymizer import anonymize
from src.synthesize.gemini_backend import GeminiSynthesizer
from src.output.post_processor import post_process
from src.output.generator import generate_output

INPUT_DIR = "data/input"
CUSTOM_TERMS = ["Blue Yonder"]
VIDEO_EXTENSIONS = (".mp4", ".mkv", ".avi", ".mov")


def process_file(file_path: str):
    name = os.path.basename(file_path)
    print(f"\n{'='*50}")
    print(f"Processing: {name}")
    print('='*50)

    print("Step 1: Transcribing (Groq API)...")
    t = transcribe_groq(file_path)
    print(f"  {len(t)} segments")

    print("Step 2: Extracting frames...")
    f = extract_frames(file_path)
    print(f"  {len(f)} frames")

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
    files = [
        os.path.join(INPUT_DIR, f)
        for f in os.listdir(INPUT_DIR)
        if f.lower().endswith(VIDEO_EXTENSIONS)
    ]

    if not files:
        print(f"No video files in {INPUT_DIR}/")
        return

    print(f"Found {len(files)} file(s)")
    
    for file_path in files:
        process_file(file_path)

    print("\n✓ All done!")


if __name__ == "__main__":
    main()