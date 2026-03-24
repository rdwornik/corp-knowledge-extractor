# Corporate Knowledge Extractor (CKE) v0.3.0

AI-powered extraction engine for corporate presentations, videos, and documents. Converts PPTX, MP4, PDF, DOCX into structured knowledge notes for Obsidian.

## What it does

- **Structured knowledge extraction** — key_facts, entities, overlays from any corporate document
- **Per-slide visual analysis** for presentations (COM→PDF→Gemini multimodal)
- **Per-slide speaker commentary** from video recordings (FFmpeg scene detection + Gemini)
- **Full verbatim transcripts** from video with timestamps
- **Session merge** — correlates PPTX + MP4 from same session into unified training_session note
- **Fact validation** — cross-references extracted numbers against source text, flags hallucinations
- **Deep extraction** with doc-type-specific overlays (training, architecture, security, commercial, meeting)

## Architecture

| Tier | Method | Cost | When |
|------|--------|------|------|
| 1 — LOCAL | Local text extraction | FREE | Small text files (<5000 chars) |
| 2 — TEXT_AI | Text → Gemini deep extraction | ~$0.001/file | PPTX, PDF, DOCX, XLSX |
| 3 — MULTIMODAL | Full upload to Gemini | ~$0.03/file | Video, audio, scanned PDFs |

- **Dual-signal for PPTX:** Gemini multimodal (visual) + text grounding (structured facts)
- **Provider abstraction:** Anthropic (Haiku/Sonnet) + Google (Gemini), automatic routing
- **Dynamic token budgets** scaled by slide count and extraction depth
- **PPTX rendering:** PowerPoint COM → PDF → PyMuPDF → per-slide PNGs
- **Video frames:** FFmpeg scene detection → key frame sampling → Gemini multimodal

### Pipeline

```
Input (file, folder, or JSON manifest)
  → scan_input()           → classify files            (src/inventory.py)
  → route_tier()           → pick cheapest tier         (src/tier_router.py)
  → compress_video()       → compress if needed         (src/compress.py)
  → scene_detect()         → FFmpeg scene frames        (src/frames/scene_detect.py)
  → extract_*()            → Gemini or local            (src/extract.py)
  → fact_validation()      → cross-ref numbers          (src/fact_validation.py)
  → post_process()         → normalize via corp-os-meta (src/post_process.py)
  → correlate_sessions()   → match PPTX+MP4 pairs      (src/correlate_sessions.py)
  → merge_session()        → unified session note       (src/merge_session.py)
  → build_package()        → Obsidian notes + JSON      (src/synthesize.py)
```

## Key metrics (Cognitive Friday S4E1 test set)

| Metric | Score |
|--------|-------|
| Extraction quality (overall) | 24/100 → 92/100 |
| MP4 | 95/100 — 15 key_facts, 20 frames, overlay 6/6, transcript 55k chars |
| PPTX | 85/100 — 18 key_facts, 55 slide PNGs, 29k chars, overlay 6/6 |

## CLI

```bash
cke process <file_or_folder>             # extract knowledge
cke process <folder> --no-compress       # skip video compression
cke process <file> --model <model>       # override LLM model
cke process <file> --tier 3              # force multimodal tier
cke process-manifest manifest.json       # batch from manifest
cke process-manifest manifest.json --batch  # Gemini Batch API (50% cheaper)
cke scan <folder> -o metadata.json       # local metadata scan (no API)
cke reextract output/<package>/          # re-run extraction
cke info output/<package>/               # show package info
```

## Output structure

```
output/{package_name}/
├── extract/
│   ├── {filename}.md                # extraction note (frontmatter + content)
│   ├── {filename}.json              # structured data
│   ├── {filename}_transcript.md     # video transcript (if MP4)
│   └── session_{id}.md              # merged session note (if PPTX+MP4 pair)
├── source/
│   ├── docs/                        # original source files
│   ├── slides/                      # rendered slide PNGs (PPTX)
│   └── frames/                      # video frames (MP4)
├── index.md                         # package index
├── synthesis.md                     # cross-file synthesis
└── _meta.yaml                       # extraction metadata
```

## Requirements

- Python 3.12+
- FFmpeg (video compression + scene detection)
- PowerPoint (PPTX→PDF via COM, Windows) or LibreOffice (fallback)
- API keys: `GEMINI_API_KEY` (required), `ANTHROPIC_API_KEY` (optional)

## Installation

```bash
pip install -e ../corp-os-meta       # shared schema + taxonomy (required)
pip install -e .                     # install CKE + deps
```

API keys are loaded globally from `Documents/.secrets/.env` via PowerShell profile.
See CLAUDE.md for key management commands.

## Tests

```bash
python -m pytest                     # 449 pass, 4 skip
python -m ruff check src/            # lint
python eval_extraction.py output/    # quality evaluation
```

## Tech stack

Python, Click, Rich, PyMuPDF, OpenCV, FFmpeg, comtypes, python-pptx, pdfplumber, google-genai, Jinja2, PyYAML

## Part of corp-by-os ecosystem

```
corp-os-meta              — shared schema + taxonomy
corp-knowledge-extractor  — extraction engine (THIS)
corp-project-extractor    — project orchestrator, calls CKE via process-manifest
corp-by-os                — root orchestrator, sole vault writer
```

CKE is a pure extraction engine. No routing, no vault writes, no orchestration.

## License

Internal use only — Blue Yonder Pre-Sales Engineering
