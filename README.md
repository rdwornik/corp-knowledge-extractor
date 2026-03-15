# Corp Knowledge Extractor (CKE)

Extraction engine for the **corp-by-os** ecosystem. Processes corporate files (PDF, PPTX, DOCX, XLSX, CSV, video) into structured Obsidian notes with validated YAML frontmatter and machine-readable JSON. Uses tiered Gemini AI to minimize cost while maximizing extraction quality.

## Features

- **Tiered extraction** — routes files to the cheapest effective tier (free local, ~$0.001 text AI, ~$0.03 multimodal)
- **Batch processing** — manifest-driven batch with resume support and Gemini Batch API (50% cheaper)
- **Schema v2 output** — validated frontmatter via corp-os-meta, Obsidian Dataview compatible
- **Multi-format** — PDF, PPTX, DOCX, XLSX, CSV, MP4, MKV, AVI, MOV, WAV
- **Insight-first** — extracts what speakers SAID, not just what slides SHOW
- **Defense-in-depth taxonomy** — prompt injection + post-process normalization + unknown logging
- **Local scanning** — `cke scan` for metadata-only extraction without API calls
- **Multi-provider AI** — Gemini (primary), Anthropic (experimental)

## Installation

```bash
# Python 3.11+ required
pip install -e ../corp-os-meta    # shared schema + taxonomy
pip install -e .                  # install CKE + all deps

# Create .env
echo "GEMINI_API_KEY=your_key_here" > .env

# Optional: FFmpeg for video compression
# Windows: https://www.gyan.dev/ffmpeg/builds/
```

## Usage

```bash
# Process a single file or folder
cke process report.pdf
cke process ./documents/

# Cost estimate without processing
cke process ./documents/ --dry-run-tiers

# Batch from manifest (used by corp-project-extractor)
cke process-manifest manifest.json --resume --max-rpm 100

# Batch API (50% cheaper, async)
cke process-manifest manifest.json --batch

# Local metadata scan (no API calls, free)
cke scan ./documents/ -o metadata.json

# Force a specific tier or model
cke process report.pdf --tier 2
cke process report.pdf --model gemini-2.5-flash

# Custom extraction prompt
cke process report.pdf --prompt-file config/prompts/product_architecture.md

# Re-run extraction on existing package
cke reextract output/my_package/

# Show package info
cke info output/my_package/
```

## Architecture

Part of the corp-by-os ecosystem:

```
corp-os-meta              — shared schema + taxonomy (imported as library)
corp-knowledge-extractor  — extraction engine (THIS)
corp-project-extractor    — project orchestrator (calls CKE via process-manifest)
corp-by-os                — root orchestrator, sole vault writer
```

CKE is a pure extraction engine. No routing, no vault writes, no orchestration.

### Pipeline

```
Input (file, folder, or JSON manifest)
  -> scan_input()           -> classify files            (src/inventory.py)
  -> route_tier()           -> pick cheapest tier         (src/tier_router.py)
  -> compress_video()       -> compress if needed         (src/compress.py, Tier 3)
  -> sample_frames()        -> extract key frames         (src/frames/sampler.py, Tier 3)
  -> extract_*()            -> Gemini or local            (src/extract.py, src/text_extract.py)
  -> post_process()         -> normalize via corp-os-meta (src/post_process.py)
  -> correlate_files()      -> group related files        (src/correlate.py)
  -> build_package()        -> Obsidian notes + JSON      (src/synthesize.py)
```

### Tiered Extraction

| Tier | Method | Cost | When |
|------|--------|------|------|
| 1 — LOCAL | Local text extraction | FREE | Small text files (<5000 chars) |
| 2 — TEXT_AI | Text sent to Gemini | ~$0.001/file | PDF, PPTX, DOCX, XLSX |
| 3 — MULTIMODAL | Full upload to Gemini | ~$0.03/file | Video, audio, scanned PDFs |

Batch API (`--batch` flag): 50% discount on Tier 2, async processing.
PPTX, XLSX, DOCX are always capped at Tier 2 — Gemini rejects their MIME types.

## Output

Each extraction produces a package directory:

```
output/<package-name>/
  extract/
    <file>.md              — Obsidian note with YAML frontmatter
    <file>.json            — machine-readable extraction data
    _meta.yaml             — extraction metadata
  source/
    frames/                — slide PNGs (video only)
  index.md                 — Dataview-compatible index
```

## Configuration

| File | Purpose |
|------|---------|
| `config/settings.yaml` | Model, prompts, file types, compression settings |
| `config/taxonomy_review.yaml` | Pending unknown terms for review |
| `config/prompts/` | Custom extraction prompts |
| `.env` | `GEMINI_API_KEY` |

Default model: `gemini-3-flash-preview` (since 2026-03-12). Override per-run with `--model`.

## Testing

```bash
python -m pytest              # 324 pass, 4 skip
python -m pytest -v           # verbose
python -m ruff check src/     # lint
```

## Dependencies

**Python (via pyproject.toml):**
corp-os-meta, google-genai, pdfplumber, python-pptx, python-docx, openpyxl,
opencv-python, click, rich, Jinja2, PyYAML, python-dotenv, pydub, numpy, pillow, packaging, requests

**System (optional):**
FFmpeg/ffprobe (video compression and metadata)

## Related repos

- **corp-by-os** — root orchestrator, sole vault writer
- **corp-os-meta** — shared schema + taxonomy
- **corp-project-extractor** — project orchestrator, calls CKE via process-manifest

## License

Internal use only — Blue Yonder Pre-Sales Engineering
