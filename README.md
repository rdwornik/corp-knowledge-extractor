# Corp Knowledge Extractor (CKE)

Extraction engine for the **corp-by-os** ecosystem. Processes PDF, PPTX, DOCX, XLSX, CSV, and video files into structured Obsidian notes with validated frontmatter and machine-readable JSON.

## Quick Start

```bash
pip install -e ../corp-os-meta   # shared schema + taxonomy
pip install -e .
cp .env.example .env             # add GEMINI_API_KEY
cke --help
```

## CLI Commands

```
cke process <path>                     -- extract single file or folder
  --prompt-file <prompt.md>            -- extract with custom prompt
  --model <name>                       -- override Gemini model
  --tier N                             -- force extraction tier (1/2/3)
  --dry-run-tiers                      -- cost estimate without processing

cke process-manifest <manifest.json>   -- batch extraction from manifest
  --resume                             -- skip completed files
  --max-rpm N                          -- rate limit (default: 100)
  --batch                              -- use Gemini Batch API (50% cheaper, async)
  --batch-poll-interval N              -- poll interval seconds (default: 60)
  --batch-timeout N                    -- max wait seconds (default: 86400)
  --model <name>                       -- override model from settings.yaml

cke scan <path>                        -- local metadata scan (no API calls)
  --recursive / --no-recursive         -- scan subfolders (default: recursive)
  -o <output.json>                     -- output file (default: stdout)
  --exclude <folder>                   -- folders to skip (repeatable)

cke reextract <package>                -- re-run extraction on existing package
cke info <package>                     -- show package metadata
```

## Architecture

Part of the corp-by-os ecosystem:

```
corp-os-meta              -- shared schema + taxonomy (imported as library)
corp-knowledge-extractor  -- extraction engine (THIS)
corp-project-extractor    -- project orchestrator (calls CKE via process-manifest)
corp-by-os                -- root orchestrator (calls CKE via corp extract)
```

CKE is a pure extraction engine. No routing, no vault writes, no orchestration.

### Pipeline

```
Input (file, folder, or JSON manifest)
  -> scan_input()           -> classify files           (src/inventory.py)
  -> route_tier()           -> pick cheapest tier        (src/tier_router.py)
  -> extract_*()            -> Gemini or local           (src/extract.py)
  -> post_process()         -> normalize via corp-os-meta (src/post_process.py)
  -> build_package()        -> Obsidian notes + JSON     (src/synthesize.py)
```

## Tiered Extraction

| Tier | Method | Cost | When |
|------|--------|------|------|
| 1 | Local text extraction | FREE | Small text files (<5000 chars) |
| 2 | Text -> Gemini 3 Flash | ~$0.001/file | PDF, PPTX, DOCX, XLSX |
| 3 | Multimodal upload | ~$0.03/file | Video, audio, scanned PDFs |

Batch API (`--batch` flag): 50% discount on Tier 2, async processing.

PPTX, XLSX, DOCX are always capped at Tier 2 -- Gemini rejects their MIME types for upload.

## Configuration

| File | Purpose |
|------|---------|
| `config/settings.yaml` | Model, prompts, file types, compression settings |
| `config/taxonomy_review.yaml` | Pending unknown terms for batch review |
| `config/prompts/` | Custom extraction prompts |
| `.env` | `GEMINI_API_KEY` |

### Default Model

`gemini-3-flash-preview` (since 2026-03-12). Override per-run with `--model`.

## Output

Each extraction produces a package directory:

```
output/<package-name>/
  extract/
    <file>.md              -- Obsidian note with YAML frontmatter
    <file>.json            -- machine-readable extraction data
    _meta.yaml             -- extraction metadata
  source/
    frames/                -- slide PNGs (video only)
  index.md                 -- Dataview-compatible index
```

## Dependencies

**Required (pip):**
corp-os-meta, google-genai, pdfplumber, python-pptx, python-docx, openpyxl,
opencv-python, click, rich, Jinja2, PyYAML, python-dotenv, pydub, numpy, pillow

**Optional (system):**
FFmpeg/ffprobe (video compression and metadata)

## Tests

```bash
pytest                     # run all tests
pytest -v                  # verbose output
```

247 tests covering inventory, correlate, text extraction, tier routing, manifest,
post-processing, batch API, polarity, locator, source date, and scan.

## Environment Setup

```bash
# Python 3.11+
pip install -e ../corp-os-meta
pip install -e .

# .env file
GEMINI_API_KEY=your_key_here

# Optional: FFmpeg for video processing
# Windows: https://www.gyan.dev/ffmpeg/builds/
```
