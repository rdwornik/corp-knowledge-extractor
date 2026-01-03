# CLAUDE.md - Corporate Knowledge Extractor

## Project Overview

Python pipeline that extracts structured knowledge from corporate meeting recordings (MP4/MKV) with PowerPoint slides. Produces markdown reports with slide images and JSONL Q&A knowledge base for RAG systems.

**Primary Use Case:** After a meeting, generate educational notes that capture not just what's on slides, but what the speaker EXPLAINED about them - the nuances, context, and insider knowledge.

**Current State:** Optimized for Type A meetings (training videos with slides). Types B and C are on the roadmap.

## Meeting Types

The project is designed to support three types of corporate meetings:

### Type A: Internal Training Sessions (CURRENT FOCUS)
**Characteristics:**
- Video recordings with PowerPoint presentations
- Presenter explaining slides with detailed commentary
- Educational content (product training, technical overviews)
- 30-90 minute duration typical
- Large file sizes (2-4GB common)

**Pipeline Requirements:**
- Frame extraction (slide detection)
- OCR (reading slide text)
- Transcription (what speaker said)
- Alignment (match speech to slides)
- Knowledge synthesis (extract insights from speech)

**Output Focus:**
- Detailed educational reports
- Q&A pairs for knowledge base
- Categorized by technical topics

### Type B: Client Meetings (ROADMAP)
**Characteristics:**
- Often audio-only (no presentation)
- Discussion-based, not slide-driven
- Focus on decisions, action items, concerns
- Multiple speakers
- 30-60 minute duration typical

**Pipeline Requirements:**
- Transcription with speaker diarization
- Action item detection
- Decision tracking
- Sentiment analysis
- No frame extraction needed

**Output Focus:**
- Meeting summary
- Action items with owners
- Client concerns/requests
- Follow-up tasks

### Type C: Internal Updates (ROADMAP)
**Characteristics:**
- Mix of formats (video, audio, shared documents)
- Status updates, roadmap reviews
- Cross-functional impact
- What changed, what's new
- Variable duration

**Pipeline Requirements:**
- Multi-format ingestion (video, audio, PDF, DOCX)
- Change detection
- Impact analysis
- Task extraction

**Output Focus:**
- Change summary
- Impact by team/function
- Action items
- Timeline/roadmap updates

## Tech Stack

- **Transcription:** Whisper via Groq API (fast, cheap)
- **Frame Extraction:** OpenCV with pixel-based change detection
- **OCR:** Tesseract (local)
- **Semantic Tagging:** Gemini Flash API (batch processing)
- **Knowledge Synthesis:** Gemini Flash API (chunked processing)
- **Anonymization:** spaCy NER + custom terms
- **Output:** Markdown report + JSONL Q&A pairs

## Project Structure

```
corporate-knowledge-extractor/
├── config/
│   ├── prompts/
│   │   └── knowledge_extraction.txt    # LLM prompt template
│   ├── settings.yaml                   # Main configuration
│   ├── processing.yaml                 # Frame/alignment settings
│   ├── anonymization.yaml              # Redaction terms
│   ├── categories.yaml                 # Report categories
│   ├── filters.yaml                    # Junk/filler patterns
│   └── config_loader.py                # Config loading utility
├── src/
│   ├── transcribe/
│   │   └── groq_backend.py             # Whisper transcription
│   ├── frames/
│   │   ├── extractor.py                # Frame extraction + deduplication
│   │   └── tagger.py                   # Semantic tagging via LLM
│   ├── ocr/
│   │   └── reader.py                   # Tesseract OCR
│   ├── align/
│   │   └── aligner.py                  # Speech-to-frame alignment
│   ├── anonymize/
│   │   └── anonymizer.py               # PII redaction
│   ├── synthesize/
│   │   ├── base.py                     # Base synthesizer class
│   │   └── gemini_backend.py           # Gemini API integration
│   └── output/
│       ├── generator.py                # Markdown/JSONL generation
│       └── post_processor.py           # Deduplication, categorization
├── scripts/
│   └── run.py                          # Main entry point
├── data/
│   └── input/                          # Place video files here
└── output/                             # Generated reports
```

## Pipeline Flow

```
Video File
    ↓
[1] transcribe_groq() → segments[] {start, end, text}
    ↓
[2] extract_frames() → frames[] {timestamp, path}
    ↓
[3] read_frames() → frames[] + {text} (OCR)
    ↓
[4] tag_frames() → frames[] + {tags} (semantic)
    ↓
[5] align() → aligned[] {start, end, speech, slide_text, frame_idx}
    ↓
[6] anonymize() → redacted aligned[] and frames[]
    ↓
[7] synthesize() → {slide_breakdown[], qa_pairs[]}
    ↓
[8] post_process() → deduplicated, categorized synthesis
    ↓
[9] generate_output() → report.md + knowledge.jsonl
```

## Critical Architecture Decisions

### Frame-Speech Alignment
The synthesizer receives BOTH `frames[]` AND `aligned_data[]`:
- `frames[]` = actual PNG images sorted by timestamp
- `aligned_data[]` = transcript segments with speech content

Speech is mapped to frames by timestamp range, NOT by OCR text grouping. This ensures frame images match their descriptions in the report.

### FRAME_ID Consistency
Each frame gets a unique ID (001, 002, 003...) that flows through:
1. Synthesizer prompt → LLM output
2. Generator → image filenames
3. Markdown report → image references

**Never renumber frames mid-pipeline.**

### LLM Output Fields
The LLM produces these fields (defined in knowledge_extraction.txt):
- `frame_id`: String "001", "002", etc.
- `title`: Descriptive title
- `visual_content`: What's shown on slide
- `technical_details`: Numbers, versions, specs
- `speaker_explanation`: **MAIN CONTENT** - what speaker said
- `context_relationships`: Connections to other topics
- `key_terminology`: Term definitions

**The generator MUST use `speaker_explanation`** - this contains the educational value.

## Configuration Files

### settings.yaml
```yaml
input:
  directory: "data/input"
  video_extensions: [".mp4", ".mkv", ".avi", ".mov"]

llm:
  model: "gemini-2.0-flash"
  chunk_size: 10
```

### anonymization.yaml
```yaml
custom_terms:      # Always redact these
  - "Blue Yonder"
  - "ClientName"

exclude_terms:     # Never redact (products misidentified as persons)
  - "WMS"
  - "BYDM"
```

### categories.yaml
```yaml
order: [infrastructure, sla, api, architecture, security, ...]

titles:
  infrastructure: "🏗️ Infrastructure & Platform"
  sla: "📋 Service Level Agreements"
  
keywords:
  infrastructure: [saas, platform, data center, azure]
  sla: [sla, availability, disaster recovery, rto]
```

## Common Issues & Solutions

### Problem: Frame images don't match text descriptions
**Cause:** Mismatch between frame numbering systems
**Solution:** Ensure both synthesizer and generator sort frames by timestamp before assigning IDs

### Problem: Report has empty/generic descriptions
**Cause:** Generator looking for wrong field (e.g., `key_insight` instead of `speaker_explanation`)
**Solution:** Check generator.py `_format_slide()` uses fields that LLM actually produces

### Problem: Too many duplicate slides
**Cause:** Similar OCR text creates near-duplicate frames
**Solution:** Adjust `dedup_similarity` threshold in processing.yaml, or improve post_processor deduplication

### Problem: Speaker explanations are generic summaries
**Cause:** Prompt not emphasizing extraction of actual speech content
**Solution:** Prompt must instruct LLM to capture WHAT speaker said, not summarize THAT they spoke

## Recent Architecture Changes

### Configuration Refactoring (Dec 2025)
Migrated from hardcoded values to centralized YAML configuration:

**Before:** Magic numbers and paths scattered across 9 Python modules
**After:** Domain-separated YAML files with single responsibility

**New config structure:**
- `settings.yaml` - Paths, file extensions, LLM models, tool paths
- `processing.yaml` - Frame extraction, deduplication, alignment parameters
- `anonymization.yaml` - Custom terms, exclusions, auto-detection settings
- `categories.yaml` - Categorization rules and keywords
- `filters.yaml` - Junk patterns, filler content, stop words

**config_loader.py features:**
- Dot-notation access: `get("processing", "frames.sample_rate")`
- In-memory caching for performance
- Path resolution: `get_path("settings", "input.directory")`
- Graceful defaults: `get("settings", "llm.model", "gemini-2.0-flash")`

**Benefits:**
- Zero hardcoded values in source code
- Easy configuration changes without code modifications
- Improved testability and deployment flexibility
- Clear separation between configuration and implementation

## Development Guidelines

### Adding New Config Values
1. Identify the appropriate YAML file (settings/processing/filters/etc.)
2. Add with descriptive comment
3. Update config_loader.py if new file needed
4. Update consuming code to use `get("file", "key.path", default)`
5. Never hardcode values - always use config_loader

### Modifying LLM Prompt
1. Work on `feature/prompt-optimization` branch
2. Make ONE change at a time
3. Test with full pipeline run
4. Compare report quality before/after
5. Commit with descriptive message

### Git Workflow
- `main`: Stable, working code
- `feature/*`: Experimental changes
- Commit after EVERY working change
- Never mix prompt changes with code changes in same commit

## Environment Setup

```bash
# Required
pip install opencv-python pytesseract spacy python-dotenv google-genai

# spaCy model
python -m spacy download en_core_web_sm

# Tesseract (Windows)
# Install from: https://github.com/UB-Mannheim/tesseract/wiki
# Add to PATH: C:\Program Files\Tesseract-OCR

# .env file
GEMINI_API_KEY=your_key_here
GROQ_API_KEY=your_key_here
```

## Running

```bash
# Place video in data/input/
python scripts/run.py

# Output in output/YYYY-MM-DD_HHMM/
#   - report.md
#   - knowledge.jsonl
#   - frames/
#   - metadata.json
```

## Cost Estimate

Per 50-minute meeting:
- Groq API (Whisper): Free tier / ~$0.001
- Gemini API (tagging + synthesis): ~$0.01
- **Total: ~$0.01-0.02**

## Testing & Quality Assurance

### Automated Quality Checks
The `tests/test_quality.py` module provides automated report quality validation:

**check_speaker_explanation_quality():**
- Ensures speaker_explanation field is populated
- Detects raw transcript dumps (red flag)
- Validates educational value extraction
- Min length threshold: 30 chars (configurable)

**check_no_junk_frames():**
- Verifies junk filter patterns work
- Detects "loading", "thank you", generic slides
- Ensures only valuable content in report

**check_categories_balanced():**
- Prevents everything dumping into "general"
- Validates categorization keywords working
- Reports category distribution

**check_qa_pairs_quality():**
- Validates Q&A format (question/answer fields)
- Checks for specificity (not generic)
- Ensures category tagging
- Verifies source frame references

### Manual Review Checklist
Before deploying updated prompts or configs:
1. Process sample training video (known good baseline)
2. Run automated quality tests
3. Manual spot-check:
   - Do frame images match descriptions?
   - Are speaker explanations detailed (not summaries)?
   - Are technical details specific (not generic)?
   - Is PII properly anonymized?
4. Compare metrics to baseline (frames, Q&A count, categories)
5. Git commit with test results in message

### Known Quality Issues
1. **Duplicate slides:** OCR similarity can create near-duplicates
   - Mitigation: Post-processor deduplication
   - Tuning: `processing.yaml` dedup_similarity threshold

2. **Generic explanations:** LLM sometimes summarizes instead of extracting
   - Mitigation: Prompt emphasizes "what speaker SAID"
   - Detection: Automated quality check flags short/generic text

3. **Frame numbering mismatch:** Rare bug where images don't match text
   - Root cause: Sorting inconsistency between synthesizer and generator
   - Both must sort frames by timestamp before ID assignment
   - Detection: Manual verification during spot checks

## Future Improvements

1. **Multi-input support (Type B/C meetings):**
   - Audio-only processing (skip frame extraction)
   - Document ingestion (PDF, DOCX, XLSX)
   - Auto-detect meeting type

2. **Quality improvements:**
   - Prompt optimization for better insight extraction
   - Smarter duplicate detection (semantic, not just OCR)
   - A/B testing framework for prompt changes

3. **Operational:**
   - Video compression before processing (4GB → 500MB)
   - Incremental processing (skip already-processed files)
   - CI/CD with automated quality gates

4. **Features:**
   - Multi-language support (non-English meetings)
   - Speaker diarization (who said what)
   - Web UI (upload video, download report)
   - Batch processing dashboard

## Git

Commit frequently with clear messages. Push after each working feature.

## Model Selection Guide

Use `/model claude-sonnet-4-20250514` (default) for:
- Creating simple files and functions
- Small edits, quick fixes
- Running tests and commands
- Iterative development
- Simple CRUD operations

Use `/model claude-opus-4-20250514` for:
- System architecture decisions
- Complex debugging (errors spanning multiple files)
- Refactoring across multiple files
- Large context analysis (understanding whole codebase)
- Code review and optimization
- When Sonnet fails 2+ times on same task

Rule: Start with Sonnet. Switch to Opus when stuck or task is complex.