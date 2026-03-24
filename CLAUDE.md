# CLAUDE.md — Corporate Knowledge Extractor

## What this repo does

Python pipeline that extracts structured knowledge from corporate files (video, PDF, PPTX, DOCX, XLSX) using tiered Gemini AI. Produces Obsidian-compatible markdown notes with YAML frontmatter and machine-readable extract.json.

## Quick start

```bash
pip install -e ../corp-os-meta    # shared schema dependency
pip install -e .                  # install CKE + deps
cke process <path>                # process a file or folder
python -m pytest                  # run tests (474+ pass, 4 skip)
```

## Architecture

**Part of corp-by-os ecosystem:**
```
corp-os-meta (v2.0) — shared schema + taxonomy
corp-knowledge-extractor (THIS) — extraction engine
corp-project-extractor — project orchestrator, calls CKE via process-manifest
corp-by-os (v0.5.0) — root orchestrator, sole vault writer
```

**Pipeline flow:**
```
Input (file, folder, or JSON manifest)
    |
[1] scan_input()              -> list[SourceFile]           (inventory.py)
[2] route_tier()              -> TierDecision (1/2/3)       (tier_router.py)
[3] compress_video()          -> compressed video            (compress.py, Tier 3 only)
[4] sample_frames()           -> list[SampledFrame]          (frames/sampler.py, Tier 3 only)
[5] extract_{local|from_text|knowledge}()                   (extract.py, text_extract.py)
[6] post_process_extraction() -> normalized frontmatter      (post_process.py -> corp_os_meta)
[7] keep_slide_frames()       -> slide_001.png ...           (run.py)
[8] correlate_files()         -> list[FileGroup]             (correlate.py)
[9] build_package()           -> output directory            (synthesize.py + Jinja2 templates)
```

**Tiered extraction (cost optimization):**

| Tier | Method | Cost | When Used |
|------|--------|------|-----------|
| 1 — LOCAL | Local text extraction only | FREE | Small text files (<5000 chars) |
| 2 — TEXT_AI | Text sent to Gemini | ~$0.001 | PDFs, PPTX, DOCX, XLSX |
| 3 — MULTIMODAL | Full video+frames to Gemini | ~$0.03 | Video/audio, scanned PDFs |

**Key constraint:** PPTX, XLSX, DOCX always capped at Tier 2 — Gemini rejects their MIME types.

## Key source files

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
| `src/deep_prompt.py` | Deep extraction prompt builder |
| `src/doc_type_classifier.py` | Document type classification |
| `src/freshness.py` | Freshness scoring for extracted content |
| `src/providers/` | Multi-provider AI abstraction (Gemini, Anthropic) |
| `src/slides/renderer.py` | PPTX slide rendering to PNG (COM/LibreOffice) |
| `src/frames/` | Video frame sampling, extraction, tagging |
| `config/settings.yaml` | Main config: model, prompts, file types, compression |
| `config/taxonomy_review.yaml` | Pending unknown terms for review |
| `config/prompts/` | Custom extraction prompts |
| `templates/extract.md.j2` | Jinja2 template for output markdown |

## Dev standards

- Python 3.11+, PowerShell on Windows (`py -m`, pathlib)
- `pyproject.toml` as single source of truth for deps
- `ruff` + `pytest` quality gate (mypy optional)
- Feature branches, meaningful commits, no deletions without asking
- Logging not print, dataclasses not dicts, type hints everywhere
- Forward slashes in all output paths
- Click CLI, Rich output
- Comments explain WHY, not WHAT. No bare except.
- YAML config, no hardcoded values

## CLI commands

```bash
cke process <path>                     # process file or folder
  --prompt-file <prompt.md>            # custom extraction prompt
  --model <name>                       # override Gemini model
  --tier N                             # force tier (1/2/3)
  --dry-run-tiers                      # cost estimate without processing

cke process-manifest <manifest.json>   # batch from manifest (used by CPE)
  --resume                             # skip already-completed files
  --max-rpm N                          # rate limit Gemini calls
  --tier N                             # force specific tier (1/2/3)
  --batch                              # Gemini Batch API (50% cheaper, async)
  --batch-poll-interval N              # poll interval seconds (default: 60)
  --batch-timeout N                    # max wait seconds (default: 86400)
  --model <name>                       # override Gemini model

cke scan <path>                        # local metadata scan (Tier 1, no API)
  --recursive / --no-recursive         # scan subfolders (default: recursive)
  -o <output.json>                     # output file (default: stdout)
  --exclude <folder>                   # folders to skip (repeatable)

cke reextract <package_path>           # re-run on existing package
cke info <package_path>                # show package info
```

## Test suite

```bash
python -m pytest                       # 324 pass, 4 skip
python -m pytest --tb=short -q         # quick summary
python -m ruff check src/              # lint check
```

**Tests exist for:** batch_api, comparison, config, correlate, deep_prompt, doc_type_classifier, freshness, inventory, locator, manifest, polarity, post_process, providers, quality, scan, slides, source_date, text_extract, tier_router

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

## Key design decisions

1. **Frame sampling is DUMB, AI is SMART** — OpenCV samples at fixed intervals, Gemini decides which frames show unique slides
2. **Insight-first output** — no slide descriptions, focus on what speaker SAID
3. **"So what" + critical notes** on every slide
4. **Deterministic Links: line** from frontmatter (no inline [[links]])
5. **Defense-in-depth taxonomy** — ~90% from prompt injection, ~8% post-process normalization, ~2% logged as unknown
6. **Cardinality caps** — max 8 topics, 4 products, 3 people, 3 domains per note
7. **PPTX/XLSX/DOCX always Tier 2** — Gemini rejects these MIME types
8. **corp_os_meta is shared source of truth** — taxonomy, normalization, validation, links
9. **gemini-3-flash-preview** for all AI tiers (since 2026-03-12)
10. **tojson_raw** over Jinja2's tojson — preserve & characters in domain names

## API Keys

Keys loaded globally from `Documents/.secrets/.env` via PowerShell profile.
Do NOT add API keys to local `.env`.
Check: `keys list` | Update: `keys set KEY value` | Reload: `keys reload`

This repo uses:
- `GEMINI_API_KEY` (required) — Gemini multimodal extraction, synthesis, transcript, frame tagging
- `ANTHROPIC_API_KEY` (optional) — Claude Haiku/Sonnet text extraction, falls back to Gemini if missing

## Dependencies

```
corp-os-meta (pip install -e ../corp-os-meta)
google-genai (NOT deprecated google.generativeai)
pdfplumber, python-pptx, python-docx, openpyxl
opencv-python, pydub, numpy, pillow
click, rich, Jinja2, PyYAML, python-dotenv
packaging, requests
```

## Integration points

CKE is a PURE extraction engine — input in, structured knowledge out.

- **corp-os-meta**: imports Pydantic schemas, taxonomy, normalization (post_process.py)
- **corp-by-os**: invokes CKE via subprocess `cke process-manifest` (ingest pipeline)
- **corp-project-extractor**: invokes CKE via subprocess `cke process-manifest` (extraction)

CKE does NOT write to the vault. Output goes to staging; corp-by-os copies to vault.

## Related repos

- [ECOSYSTEM.md](../ECOSYSTEM.md) — full ecosystem overview
- [corp-os-meta](../corp-os-meta/) — shared schema (import dependency)
- [corp-by-os](../corp-by-os/) — orchestrator (invokes CKE)
- [corp-project-extractor](../corp-project-extractor/) — project classifier (invokes CKE)

## Known issues

1. **quality field mismatch** — Gemini outputs "high", schema expects "full|partial|fragment"
2. **Products pollution** — "Demand Planning" in both topics AND products, "SAP" as product
3. **People as roles** — "Project Manager" instead of actual names
4. **Modules without dedicated tests:** batch.py, compress.py, extract.py, frames/extractor.py, frames/sampler.py, frames/tagger.py, reextract.py, synthesize.py, taxonomy_prompt.py, utils.py
5. **python-pptx MemoryError** — importing pptx module can cause MemoryError on Python 3.12 (regex compilation bug); tests patch at function level to avoid

## Lenzing Pilot Results

- 143 files -> 141 extracted (98.6%), $0.20 total
- Tier 1: 6, Tier 2: 133, Tier 3: 2
- 123 notes, 992 facts, 5 dashboards


## Global Skills
Before modifying code, consult ~/.claude/skills/gotchas/ for known ecosystem traps.
After pytest passes, check ~/.claude/skills/verify/ for verification scripts.

