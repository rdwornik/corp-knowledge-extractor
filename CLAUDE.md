# CLAUDE.md - Corporate Knowledge Extractor

## Project Overview

Python pipeline extracting structured knowledge from corporate files (video, PDF, PPTX, DOCX, XLSX). Produces markdown notes with YAML frontmatter (Obsidian Dataview compatible) and machine-readable extract.json.

**Part of corp-by-os ecosystem:**
```
corp-os-meta (v2.0) — shared schema + taxonomy (pip install -e ../corp-os-meta)
corp-knowledge-extractor (THIS) — extraction engine
corp-project-extractor — project orchestrator, calls CKE via process-manifest
corp-by-os (v0.5.0) — root orchestrator, sole vault writer
```

**Path:** `C:\Users\1028120\Documents\Scripts\corp-knowledge-extractor`
**CLI:** `cke` (entry point: `scripts.run:cli`)

## Architecture

```
Input (file, folder, or JSON manifest)
    |
[1] scan_input()              -> list[SourceFile]           (inventory.py)
    |
[2] route_tier()              -> TierDecision (1/2/3)       (tier_router.py)
    |
[3] compress_video()          -> compressed video            (compress.py, Tier 3 only)
    |
[4] sample_frames()           -> list[SampledFrame]          (frames/sampler.py, Tier 3 only)
    |
[5] extract_{local|from_text|knowledge}()                   (extract.py, text_extract.py)
    |
[6] post_process_extraction() -> normalized frontmatter      (post_process.py -> corp_os_meta)
    |
[7] keep_slide_frames()       -> slide_001.png ...           (run.py)
    |
[8] correlate_files()         -> list[FileGroup]             (correlate.py)
    |
[9] build_package()           -> output directory            (synthesize.py + Jinja2 templates)
```

## Tiered Extraction (Cost Optimization)

| Tier | Method | Cost | When Used |
|------|--------|------|-----------|
| 1 -- LOCAL | Local text extraction only | FREE | Small text files (<5000 chars) |
| 2 -- TEXT_AI | Text sent to Gemini 3 Flash | ~$0.001 | PDFs, PPTX, DOCX, XLSX (upload blocked) |
| 3 -- MULTIMODAL | Full video+frames to Gemini | ~$0.03 | Video/audio, scanned PDFs |

Batch API (`--batch` flag): 50% discount on Tier 2, async processing.

**Key constraint:** PPTX, XLSX, DOCX always capped at Tier 2 -- Gemini rejects their MIME types.

## CLI Commands

```
cke process <path>                     -- process file or folder
  --prompt-file <prompt.md>            -- custom extraction prompt
  --model <name>                       -- override Gemini model
  --tier N                             -- force tier (1/2/3)
  --dry-run-tiers                      -- cost estimate without processing
cke process-manifest <manifest.json>   -- batch from manifest (used by CPE)
  --resume                             -- skip already-completed files
  --max-rpm N                          -- rate limit Gemini calls
  --tier N                             -- force specific tier (1/2/3)
  --batch                              -- Gemini Batch API (50% cheaper, async)
  --batch-poll-interval N              -- poll interval seconds (default: 60)
  --batch-timeout N                    -- max wait seconds (default: 86400)
  --model <name>                       -- override Gemini model
cke scan <path>                        -- local metadata scan (Tier 1, no API)
  --recursive / --no-recursive         -- scan subfolders (default: recursive)
  -o <output.json>                     -- output file (default: stdout)
  --exclude <folder>                   -- folders to skip (repeatable)
cke reextract <package_path>           -- re-run on existing package
cke info <package_path>                -- show package info
```

## Schema v2 (from corp-os-meta)

Every output note has validated frontmatter:
```yaml
# REQUIRED
title: str
date: date
type: enum (presentation|meeting|training|demo|rfp|contract|email|notes|document|report)
topics: list[str]       # max 8, normalized against taxonomy
schema_version: 1
source_tool: "knowledge-extractor"
source_file: str

# OPTIONAL
products: list[str]      # max 4
people: list[str]        # max 3
domains: list[str]       # max 3, from 8 knowledge domains
language: str
quality: enum (full|partial|fragment)
summary: str
source_type: enum (documentation|meeting|workshop|demo)
layer: enum (learning|process|reference|decision)
confidentiality: enum (public|internal|confidential|restricted)
authority: enum (authoritative|approved|tribal)
client: str              # from manifest
project: str             # from manifest
```

## Key Design Decisions

1. **Frame sampling is DUMB, AI is SMART** -- OpenCV samples at fixed intervals, Gemini decides which frames show unique slides
2. **Insight-first output** -- no slide descriptions, focus on what speaker SAID
3. **"So what" + critical notes** on every slide
4. **Deterministic Links: line** from frontmatter (no inline [[links]])
5. **Defense-in-depth taxonomy** -- ~90% from prompt injection, ~8% post-process normalization, ~2% logged as unknown
6. **Cardinality caps** -- max 8 topics, 4 products, 3 people, 3 domains per note
7. **PPTX/XLSX/DOCX always Tier 2** -- Gemini rejects these MIME types
8. **corp_os_meta is shared source of truth** -- taxonomy, normalization, validation, links
9. **gemini-3-flash-preview** for all AI tiers (since 2026-03-12, fallback: gemini-2.5-flash)
10. **tojson_raw** over Jinja2's tojson -- preserve & characters in domain names

## Key Source Files

| File | Purpose |
|------|---------|
| `scripts/run.py` | Click CLI entry point (process, process-manifest, scan, reextract, info) |
| `src/inventory.py` | Scan + classify input files |
| `src/tier_router.py` | Route files to cheapest tier |
| `src/text_extract.py` | Local text extraction (PDF, DOCX, PPTX, XLSX) + source_date |
| `src/extract.py` | Gemini API extraction (Tier 2 + Tier 3) + fact enrichment |
| `src/post_process.py` | Normalize via corp_os_meta |
| `src/batch.py` | Manifest-driven sequential batch processor with resume |
| `src/batch_api.py` | Gemini Batch API integration (async, 50% cheaper) |
| `src/scan.py` | Local metadata scanner (cke scan command) |
| `src/polarity.py` | Deterministic polarity detection for facts (regex, no LLM) |
| `src/correlate.py` | Group related files into FileGroups |
| `src/synthesize.py` | Build output package (Jinja2 templates) |
| `src/manifest.py` | Manifest schema for process-manifest |
| `src/utils.py` | Robust LLM JSON parser (4-strategy) |
| `src/taxonomy_prompt.py` | Inject canonical taxonomy into Gemini prompts |
| `config/settings.yaml` | Main config: model, prompts, file types, compression |
| `config/taxonomy_review.yaml` | Pending unknown terms for review |
| `config/prompts/` | Custom extraction prompts (e.g., product_architecture.md) |

## Dev Standards

- **Shell:** PowerShell on Windows
- **Run:** `cke` CLI or `py -m scripts.run` from project root
- **Paths:** pathlib, forward slashes in output
- **Config:** YAML, no hardcoded values
- **Types:** dataclasses, type hints everywhere
- **Quality:** ruff + mypy + pytest
- **Git:** feature branches, meaningful commits
- **Output:** Rich for CLI, logging (not print)
- **Comments:** explain WHY, not WHAT. No bare except.

## Dependencies

```
corp-os-meta (pip install -e ../corp-os-meta)
google-genai (NOT deprecated google.generativeai)
pdfplumber, python-pptx, python-docx, openpyxl
opencv-python, pydub, numpy, pillow
click, rich, Jinja2, PyYAML, python-dotenv
```

## Known Issues

1. **quality field mismatch** -- Gemini outputs "high", schema expects "full|partial|fragment"
2. **Products pollution** -- "Demand Planning" in both topics AND products, "SAP" as product
3. **People as roles** -- "Project Manager" instead of actual names
4. **Stale tests** -- test_pipeline.py (imports src.align), test_transcription.py (imports src.transcribe) -- modules removed
5. **Working tests:** 247 pass, 1 pre-existing fail (test_comparison.py LocalPath.touch), 4 skipped

## Lenzing Pilot Results

- 143 files -> 141 extracted (98.6%), $0.20 total
- Tier 1: 6, Tier 2: 133, Tier 3: 2
- 123 notes, 992 facts, 5 dashboards

## Environment

```bash
pip install -e ../corp-os-meta
pip install -e .
# .env
GEMINI_API_KEY=your_key_here
```
