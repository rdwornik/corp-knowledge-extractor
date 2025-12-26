import os
from src.transcribe.transcriber import transcribe
from src.frames.extractor import extract_frames
from src.ocr.reader import read_frames
from src.align.aligner import align
from src.anonymize.anonymizer import anonymize
from src.synthesize.ollama_backend import OllamaSynthesizer
from src.output.generator import generate_output

INPUT_DIR = "data/input"
CUSTOM_TERMS = ["Blue Yonder"]  # Add company names to mask
VIDEO_EXTENSIONS = (".mp4", ".mkv", ".avi", ".mov")


def process_file(file_path: str):
    name = os.path.basename(file_path)
    print(f"\n{'='*50}")
    print(f"Processing: {name}")
    print('='*50)

    print("Step 1: Transcribing...")
    t = transcribe(file_path)
    print(f"  {len(t)} segments")

    print("Step 2: Extracting frames...")
    f = extract_frames(file_path)
    print(f"  {len(f)} frames")

    print("Step 3: OCR...")
    f = read_frames(f)

    print("Step 4: Aligning...")
    aligned = align(t, f)

    print("Step 5: Anonymizing...")
    for item in aligned:
        item['speech'] = anonymize(item['speech'], CUSTOM_TERMS)
        item['slide_text'] = anonymize(item['slide_text'], CUSTOM_TERMS)

    print("Step 6: Synthesizing with Mistral...")
    synth = OllamaSynthesizer(model='mistral')
    result = synth.synthesize(aligned)

    print("Step 7: Generating output...")
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