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

## Content Type Presets System

### Architecture

The preset system allows customizing frame extraction behavior for different content types (PowerPoint, Excel, Demo, Audio-only, Hybrid).

**Location:** `config/presets/*.yaml`

**Components:**
1. **Preset YAML files** - Configuration for each content type
2. **extractor.py** - `load_preset()` function loads YAML into dict
3. **run.py** - CLI argument `--preset <name>` passes to extractor
4. **AdaptiveFrameTracker** - Class for hybrid mode dynamic switching

### Available Presets

**powerpoint.yaml:**
- Default behavior, optimized for slide-based presentations
- Sample rate: 1s, Pixel threshold: 5%, Max/min: 10
- Use case: Training videos, pitch decks

**excel.yaml:**
- Sparse sampling for spreadsheet scrolling
- Sample rate: 10s, Pixel threshold: 30%, Max/min: 3
- Prevents frame explosion from cell selection/scroll
- Use case: Financial reports, dashboard reviews

**demo.yaml:**
- Time-based sampling for software demonstrations
- Sample rate: 15s, Pixel threshold: 25%, Max/min: 4
- Ignores cursor movement and hover states
- Use case: Feature walkthroughs, UI tutorials

**audio_only.yaml:**
- Disables frame extraction (`frames.enabled: false`)
- Returns empty frame list, skips OCR/tagging/alignment
- Use case: Client calls, phone meetings, interviews

**hybrid.yaml:**
- Adaptive mode with automatic content detection
- Analyzes activity every 60 seconds (`analysis_window`)
- Switches between powerpoint/demo modes based on frame rate
- Logs mode switches for transparency
- Use case: Mixed content (slides + demo + discussion)

### How Presets Work

**1. Loading:**
```python
from src.frames.extractor import load_preset

preset_config = load_preset("excel")  # Loads config/presets/excel.yaml
# Returns dict with:
# - frames: {sample_rate, pixel_threshold, max_per_minute, max_total}
# - deduplication: {enabled, similarity_threshold, pixel_similarity}
# - synthesis: {focus, speaker_explanation_weight}
```

**2. Applying:**
```python
extract_frames(
    video_path,
    preset="excel",  # Uses excel.yaml settings
    sample_rate=None,  # Override if needed
    threshold=None
)
```

**3. Adaptive Mode (Hybrid Preset):**
```python
class AdaptiveFrameTracker:
    # Tracks frame activity in sliding window
    def add_frame(timestamp)
    def should_check_switch(current_time)
    def check_and_switch(total_frames, current_time)
        # Analyzes frames/minute in last 60 seconds
        # High activity (>5 f/min) → switch to demo mode
        # Low activity (<2 f/min) → switch to powerpoint mode
        # Returns new settings dict or None
```

**4. Usage:**
```bash
# Via CLI
python scripts/run.py --preset excel

# Via code
f = extract_frames("video.mp4", preset="excel")
```

### Adding New Presets

1. **Create YAML file** in `config/presets/`:
```yaml
name: "My Custom Preset"
description: "Description of use case"

frames:
  enabled: true
  sample_rate: 5  # Seconds between checks
  pixel_threshold: 0.20  # 0.0-1.0
  max_per_minute: 8
  max_total: 200

  deduplication:
    enabled: true
    similarity_threshold: 0.88
    pixel_similarity: 0.88

synthesis:
  focus: "my_focus_type"
  speaker_explanation_weight: "high"
```

2. **Update run.py** - Add to `choices` list in argparse
3. **Update README.md** - Document the new preset
4. **Test** - Verify with representative video

### Preset Configuration Fields

**frames.enabled:** Boolean - Enable/disable frame extraction
**frames.sample_rate:** Int - Seconds between frame checks
**frames.pixel_threshold:** Float 0.0-1.0 - % of pixels changed = new frame
**frames.max_per_minute:** Int - Prevent frame explosion
**frames.max_total:** Int - Hard limit for very long videos
**frames.mode:** String - "adaptive" for hybrid, omit for static
**frames.deduplication:** Dict - Similarity thresholds for dedup

**Adaptive mode fields (hybrid only):**
**frames.analysis_window:** Int - Seconds in sliding window (60)
**frames.adaptive_rules:** Dict - Thresholds for mode switching
**frames.modes:** Dict - Settings for each sub-mode (powerpoint/demo/excel)

### Design Decisions

**Why presets vs. hardcoded values?**
- Different content types need different strategies
- Excel scrolling generates 10x more frames than PowerPoint
- Demo cursor movement shouldn't trigger frames
- Audio-only needs to skip frame pipeline entirely

**Why YAML vs. Python?**
- Non-technical users can adjust settings
- No code changes needed for tuning
- Version control shows config history
- Easy A/B testing of parameters

**Why adaptive mode?**
- Real meetings mix content types
- Manual mode switching is tedious
- Auto-detection handles 80% of cases
- Logging shows when switches happen (transparency)

## Report Comparison System

### Architecture

The comparison system detects quality regressions by diff'ing two reports.

**Location:** `scripts/compare_reports.py`

**Components:**
1. **load_report_data()** - Loads markdown, JSONL, metadata
2. **compare_*()** functions - Frame/slide/QA/quality comparisons
3. **determine_verdict()** - Overall assessment (improved/degraded/mixed)
4. **generate_markdown_report()** - Human-readable diff
5. **generate_json_metrics()** - Machine-readable metrics

### How It Works

**1. Load Reports:**
```python
old = load_report_data("output/2025-01-01_1200")
new = load_report_data("output/2025-01-03_1430")

# Extracts:
# - Slide titles and explanations from markdown
# - Q&A pairs from knowledge.jsonl
# - Frame count from frames/ directory
# - Quality metrics via QualityChecker
```

**2. Compare Metrics:**
```python
comparison = {
    "frames": {"old_count": 94, "new_count": 87, "change": -7},
    "slides": {"removed_titles": [...], "added_titles": [...]},
    "qa_pairs": {"old_count": 280, "new_count": 310, "change": +30},
    "quality": {
        "improvements": ["Longer explanations (+20%)"],
        "regressions": []
    }
}
```

**3. Determine Verdict:**
```python
def determine_verdict(comparison):
    if regressions and not improvements:
        return {"verdict": "degraded", "has_regressions": True}
    elif improvements and not regressions:
        return {"verdict": "improved", "has_regressions": False}
    elif improvements and regressions:
        return {"verdict": "mixed", "has_regressions": True}
    else:
        return {"verdict": "unchanged", "has_regressions": False}
```

**4. Generate Outputs:**
- **comparison_report.md** - Summary table, improvements/regressions lists, changed explanations
- **comparison_metrics.json** - All metrics for CI/CD integration

### Usage

**Basic Comparison:**
```bash
python scripts/compare_reports.py output/old output/new
# Outputs: comparison_report.md + comparison_metrics.json
```

**CI/CD Integration:**
```bash
# Fail build if regressions detected
python scripts/compare_reports.py \
  tests/fixtures/baseline \
  output/latest \
  --fail-on-regression

# Exit codes:
# 0 = no regressions (pass)
# 1 = regressions detected (fail)
```

**Baseline Workflow:**
```bash
# 1. Establish baseline
python scripts/run.py
cp -r output/latest tests/fixtures/baseline_v1

# 2. Make changes (update prompt, config, code)
# ... edit files ...

# 3. Re-run pipeline
python scripts/run.py

# 4. Compare
python scripts/compare_reports.py \
  tests/fixtures/baseline_v1 \
  output/latest

# 5. Review comparison_report.md
# 6. If improved: update baseline
# 7. If degraded: fix issues
```

### Comparison Metrics

**Frames:**
- Count change (absolute and percentage)
- Indicates if frame extraction improved/degraded

**Slides:**
- Removed titles (slides that disappeared)
- Added titles (new slides detected)
- Helps identify if deduplication changed

**Q&A Pairs:**
- Count change (more Q&A = more knowledge extracted)
- Percentage change

**Quality Metrics:**
- Average explanation length (longer = more detail)
- Empty explanation count (should be low)
- Junk slide count (should decrease over time)
- General category percentage (should decrease = better categorization)

**Content Changes:**
- Slides with changed explanations
- Change type: improved/degraded/rewritten
- Length comparison

### Design Decisions

**Why compare reports vs. raw pipeline output?**
- Reports are the user-facing artifact
- Markdown diffs are human-readable
- Quality metrics in reports reflect actual value

**Why separate markdown and JSON outputs?**
- Markdown for humans (review in GitHub, email)
- JSON for machines (CI/CD, automated alerts)

**Why fail-on-regression flag?**
- Prevents accidental quality degradation
- Blocks PR merges if tests show regression
- Forces explicit override for intentional changes

**Why track removed/added slides?**
- Deduplication changes can remove valid content
- New slides indicate better detection
- Helps debug frame extraction issues

### Integration with Quality Tests

```python
# tests/test_quality.py provides the metrics
from tests.test_quality import QualityChecker

checker = QualityChecker(report_dir)
quality_data = {
    "speaker_explanation": checker.check_speaker_explanation_quality(),
    "junk_frames": checker.check_no_junk_frames(),
    "categories": checker.check_categories_balanced(),
    "qa_pairs": checker.check_qa_pairs_quality()
}

# compare_reports.py uses these metrics for comparison
```

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

## Audio Preprocessing for Large Files

### Problem

Groq API has a 25MB file size limit for transcription. However, corporate meeting recordings (especially 5-hour training sessions) typically produce large audio files:

- **Raw video**: 2-4GB (typical corporate meeting)
- **Extracted audio (compressed)**: 50-80MB (5-hour meeting at 32kbps)
- **After silence removal**: 30-50MB (30-40% reduction)
- **After chunking**: Multiple 24MB chunks

### Solution Architecture

The transcription system uses a 3-stage preprocessing pipeline:

```
Input Video (3GB)
    ↓
[1] Extract Audio → 73MB MP3 (mono, 16kHz, 32kbps)
    ↓
[2] Remove Silence → 45MB MP3 (38% reduction)
    ↓
[3] Split Chunks → 2x 24MB chunks
    ↓
[4] Transcribe → 2 separate API calls
    ↓
[5] Merge Transcripts → Single timeline with corrected timestamps
```

### Stage 1: Audio Extraction & Optimization

**Module:** `src/transcribe/groq_backend.py::extract_audio()`

Extracts audio from video and applies Whisper-optimized settings:

```python
# FFmpeg optimization
-vn             # No video
-ac 1           # Mono (stereo not needed for speech)
-ar 16000       # 16kHz sample rate (Whisper optimal)
-b:a 32k        # 32kbps bitrate (sufficient for speech)
```

**Typical Results:**
- 2GB MP4 → 45MB MP3 (98% reduction)
- Quality: Perfect for Whisper transcription
- No loss of speech intelligibility

### Stage 2: Silence Removal

**Module:** `scripts/preprocess_audio.py::remove_silence()`

Uses FFmpeg `silenceremove` filter to eliminate long pauses while preserving natural speech timing:

```python
# FFmpeg silenceremove filter
silenceremove=
    start_periods=1:               # Remove silence from start
    start_threshold=-40dB:         # Silence = audio below -40dB
    stop_periods=-1:               # Remove all silence
    stop_duration=2.0:             # Only remove pauses > 2 seconds
    detection=peak                 # Peak detection method
```

**Configuration (settings.yaml):**
```yaml
transcription:
  silence_removal:
    enabled: true
    threshold_db: -40          # -30 = aggressive, -50 = conservative
    min_silence_duration: 2.0  # Remove pauses > 2 seconds
```

**Threshold Guidelines:**
- `-30dB`: Aggressive (removes more, risk of clipping quiet speech)
- `-40dB`: **Recommended** (good balance for meetings)
- `-50dB`: Conservative (keeps more content)

**Typical Results:**
- 73MB audio with 40% silence → 45MB (38% reduction)
- 5-hour recording → 3-hour effective speech time
- Preserves all speech content, removes long pauses between topics

**When Silence Removal Helps Most:**
- Training sessions with long pauses
- Presentations with slide transition gaps
- Recordings with dead air at start/end
- Meetings with extended silences

**When to Disable:**
- Music/audio with intentional pauses
- Recordings where timing is critical
- Already heavily edited audio

### Stage 3: Audio Chunking

**Module:** `src/transcribe/chunker.py`

When silence removal alone isn't enough to get below 25MB, the audio is split into chunks.

**Key Features:**

1. **Intelligent Boundary Detection:**
   - Uses pydub to detect silence boundaries
   - Splits at natural pauses (not mid-word)
   - Avoids cutting sentences

2. **Overlap for Context:**
   - 5-second overlap between chunks (configurable)
   - Prevents loss of words at boundaries
   - Overlap segments filtered during merge

3. **Size-Based Splitting:**
   ```python
   # Calculate number of chunks needed
   num_chunks = ceil(file_size_mb / max_chunk_size_mb)

   # Target chunk duration
   target_duration = total_duration / num_chunks

   # Find silence boundary closest to target
   split_point = find_nearest_silence_boundary(target_duration)
   ```

**Example:**
- 45MB file, 24MB max → 2 chunks (22.5MB each)
- Chunk 1: 0:00-2:30 (with 5s extension to 2:35)
- Chunk 2: 2:30-5:00 (starts at 2:25 due to overlap)

### Stage 4: Transcript Merging

**Module:** `src/transcribe/chunker.py::merge_transcripts()`

After transcribing each chunk separately, transcripts are merged with timeline corrections:

```python
# Chunk 1 segments (0:00-2:35)
[
    {"start": 0, "end": 5, "text": "Welcome to the training"},
    {"start": 5, "end": 150, "text": "Let's begin..."},
    {"start": 150, "end": 155, "text": "Next slide"}  # Overlap
]

# Chunk 2 segments (originally 0:00-2:30, offset by 150s)
[
    {"start": 0, "end": 5, "text": "Next slide"},  # SKIP (overlap)
    {"start": 5, "end": 150, "text": "Now we'll cover..."}
]

# Merged result
[
    {"start": 0, "end": 5, "text": "Welcome to the training"},
    {"start": 5, "end": 150, "text": "Let's begin..."},
    {"start": 150, "end": 155, "text": "Next slide"},
    {"start": 155, "end": 300, "text": "Now we'll cover..."}  # Offset applied
]
```

**Merging Logic:**
1. Process chunks sequentially
2. Apply cumulative time offset to each segment
3. Skip segments in overlap region (first N seconds of each chunk)
4. Maintain continuous timeline

### Configuration

**config/settings.yaml:**
```yaml
transcription:
  provider: "groq"
  max_file_size_mb: 25

  silence_removal:
    enabled: true
    threshold_db: -40
    min_silence_duration: 2.0

  chunking:
    enabled: true
    max_chunk_size_mb: 24
    overlap_seconds: 5.0
```

### Usage Examples

**Automatic Preprocessing (Default):**
```python
from src.transcribe.groq_backend import transcribe_groq

# Automatically handles large files
segments = transcribe_groq("5_hour_training.mp4")
# → Extracts audio, removes silence, chunks if needed, merges
```

**Manual Preprocessing:**
```python
from scripts.preprocess_audio import preprocess_for_transcription

# Preprocess separately
output, stats = preprocess_for_transcription(
    "training.mp4",
    remove_silence_enabled=True,
    threshold_db=-40,
    min_silence_duration=2.0
)

print(f"Reduced from {stats['original_size_mb']:.1f}MB to {stats['final_size_mb']:.1f}MB")
# → "Reduced from 73.2MB to 44.8MB"
```

**Disable Preprocessing:**
```python
# For small files or when preprocessing not needed
segments = transcribe_groq(
    "short_meeting.mp4",
    enable_preprocessing=False,
    enable_chunking=False
)
```

### Size Estimates

**Typical 5-Hour Corporate Training:**
- Original video: 3.5GB MP4
- Extracted audio: 73MB MP3 (32kbps, mono, 16kHz)
- After silence removal: 45MB (38% reduction)
- Chunks needed: 2 chunks of 22.5MB each
- API calls: 2 (one per chunk)
- Processing time: ~8-12 minutes (Groq API)
- Cost: Free tier / ~$0.002

**Size Reduction by Meeting Type:**

| Meeting Type | Original | After Silence | Reduction | Reason |
|--------------|----------|---------------|-----------|--------|
| Training (5h) | 73MB | 45MB | 38% | Long pauses between slides |
| Demo (2h) | 30MB | 22MB | 27% | Some pauses, mostly continuous |
| Discussion (1h) | 15MB | 13MB | 13% | Continuous speech, few pauses |
| Webinar (3h) | 50MB | 28MB | 44% | Q&A pauses, intro/outro silence |

**Rule of Thumb:**
- Presentations with slides: 30-40% reduction
- Software demos: 20-30% reduction
- Discussions/meetings: 10-20% reduction
- Webinars with Q&A: 40-50% reduction

### Troubleshooting

**Problem: File still too large after silence removal**
```bash
# Check file size
python -c "import os; print(f'{os.path.getsize('audio.mp3')/1024/1024:.1f}MB')"

# If > 25MB, chunking will automatically activate
# Verify chunking is enabled in settings.yaml
```

**Problem: Speech getting cut off**
```yaml
# Adjust silence threshold (more conservative)
transcription:
  silence_removal:
    threshold_db: -50  # Was -40, now keeps more content
    min_silence_duration: 3.0  # Was 2.0, now only removes longer pauses
```

**Problem: Too many chunks**
```yaml
# Increase chunk size (closer to limit)
transcription:
  chunking:
    max_chunk_size_mb: 24.5  # Was 24, now uses more headroom
```

**Problem: Words lost at chunk boundaries**
```yaml
# Increase overlap
transcription:
  chunking:
    overlap_seconds: 10.0  # Was 5.0, now more context preserved
```

### Testing

Run transcription tests:
```bash
pytest tests/test_transcription.py -v
```

Tests cover:
- Silence removal reduces file size
- Speech content preserved after silence removal
- Audio chunking at correct boundaries
- Transcript merging with correct timestamps
- Overlap handling
- Full preprocessing pipeline

### Performance Metrics

**Processing Time (5-hour meeting):**
1. Extract audio: ~30 seconds (FFmpeg)
2. Remove silence: ~45 seconds (FFmpeg)
3. Split chunks: ~15 seconds (pydub)
4. Transcribe chunk 1: ~3 minutes (Groq API)
5. Transcribe chunk 2: ~3 minutes (Groq API)
6. Merge transcripts: <1 second (Python)

**Total: ~7-8 minutes** (vs. hours with local Whisper)

**API Cost (Groq):**
- 45MB audio = 2 chunks
- 2 API calls × $0.001 = $0.002
- **Effectively free** on Groq's free tier

## Intelligent Provider Selection

### Problem

Corporate meetings vary widely in duration:
- **Short meetings** (< 1h 40min): Groq free tier is perfect
- **Long meetings** (> 1h 40min): Groq free tier has hourly rate limits
- **Very long meetings** (5+ hours): Would require multiple Groq sessions with 20-minute waits

Users shouldn't have to manually calculate which provider to use or deal with rate limit errors mid-processing.

### Solution

Automatic provider selection based on file duration, cost analysis, and API availability.

**Decision Tree:**
```
File Analysis
    ↓
[1] Get duration (e.g., 5h 20min)
    ↓
[2] Check Groq limit (7200s/hour)
    ↓
[3] Estimate costs:
    - Groq: FREE but 3 sessions, ~2.5h total time
    - OpenAI: $1.92, ~8min total time
    ↓
[4] Auto-select: OpenAI (faster, worth the cost)
    ↓
[5] Show estimate, proceed with selected provider
```

### Configuration

**config/settings.yaml:**
```yaml
transcription:
  provider: "auto"  # "auto", "groq", "openai"

  auto_selection:
    groq_max_duration_seconds: 6000  # 1h 40min safe threshold
    prefer_free: true                 # Try Groq first if within limits

  groq:
    model: "whisper-large-v3"
    rate_limit_seconds_per_hour: 7200

  openai:
    model: "whisper-1"
    cost_per_minute: 0.006

  retry:
    enabled: true
    max_retries: 3
    backoff_multiplier: 1.5
```

### Usage Examples

**Example 1: Automatic Selection (Default)**
```bash
python scripts/run.py --file meeting.mp4

# Output:
# === TRANSCRIPTION ESTIMATE ===
# File: meeting.mp4
# Duration: 45min (2700 seconds)
#
# Provider options:
# [1] Groq (FREE) - ✅ Available
#     Total time: ~8 minutes
#
# [2] OpenAI Whisper (Available)
#     Cost: $0.27
#     Total time: ~5 minutes
#
# Auto-selecting: GROQ
# Reason: Within free tier limits
```

**Example 2: Force OpenAI (Paid, Fast)**
```bash
python scripts/run.py --file long_meeting.mp4 --provider openai

# Immediately uses OpenAI regardless of file size
# Good for production when speed matters more than cost
```

**Example 3: Estimate Only**
```bash
python scripts/run.py --file training.mp4 --estimate-only

# Shows cost estimate without processing
# Helps budget decisions for large video archives
```

**Example 4: Large File Auto-Selection**
```bash
python scripts/run.py --file 5_hour_training.mp4

# Output:
# === TRANSCRIPTION ESTIMATE ===
# File: 5_hour_training.mp4
# Duration: 5h 20min (19213 seconds)
#
# Provider options:
# [1] Groq (FREE) - ⚠️ Exceeds hourly limit (7200s)
#     Would require ~3 sessions with 20min waits
#     Total time: ~150 minutes
#
# [2] OpenAI Whisper (✅ Recommended)
#     Cost: $1.92
#     Total time: ~8 minutes
#
# Auto-selecting: OPENAI
# Reason: File too large for Groq free tier (would require 3 sessions with 60min waits)
```

### Provider Comparison

| Provider | Cost | Speed | File Size Limit | Rate Limits | Best For |
|----------|------|-------|-----------------|-------------|----------|
| **Groq** | FREE | ~0.5x realtime | 25MB | 7200s/hour | Short meetings (<1h 40min) |
| **OpenAI** | $0.006/min | ~10x realtime | 25MB* | Generous | Long meetings, production |

*Both providers automatically use chunking for files > 25MB

**Groq Advantages:**
- Completely free
- Good quality (Whisper large-v3)
- Fast for short files

**Groq Disadvantages:**
- 25MB file size limit per request (auto-chunking enabled)
- 7200s/hour rate limit
- Long waits for large files (multiple sessions)

**OpenAI Advantages:**
- 10x faster processing than Groq
- No hourly rate limits (duration-based only)
- Auto-chunking handles large files seamlessly
- More reliable for production use

**OpenAI Disadvantages:**
- Costs $0.006/minute of audio
- Requires API billing setup
- 25MB file size limit per request (auto-chunking enabled)

### Cost Examples

| Meeting Type | Duration | Groq Cost | OpenAI Cost | Recommendation |
|--------------|----------|-----------|-------------|----------------|
| Team standup | 15min | $0 (FREE) | $0.09 | Groq |
| Client call | 1h | $0 (FREE) | $0.36 | Groq |
| Training (short) | 1h 30min | $0 (FREE) | $0.54 | Groq |
| Training (long) | 5h | $0 (3 sessions)* | $1.80 | OpenAI |
| All-day workshop | 8h | $0 (4 sessions)* | $2.88 | OpenAI |

*Groq is free but requires multiple sessions with 20-minute waits between them

**Break-Even Analysis:**
- For files < 1h 40min: Always use Groq (free)
- For files > 1h 40min: OpenAI worth it ($0.006/min vs. waiting)
- For production pipelines: OpenAI (speed + reliability)
- For budget-constrained hobbyists: Groq (patience pays off)

### Automatic Retry Logic

**Groq Rate Limit Handling:**

When Groq returns a rate limit error:
```
RateLimitError: Rate limit reached. Please try again in 19m35.5s.
```

The system automatically:
1. Parses wait time from error message ("19m35.5s" → 1175 seconds)
2. Shows progress: "⚠️ Rate limit reached (retry 1/3). Waiting 19m 36s..."
3. Waits with 30-second progress updates
4. Retries automatically
5. After 3 failed retries, suggests: "Use --provider openai for faster processing"

**Example Output:**
```
  Transcribing chunk_001.mp3 (24.5MB)...

  ⚠️  Rate limit reached (retry 1/3)
  Rate limit. Waiting 19m 36s...
    ... 19m 6s remaining
    ... 18m 36s remaining
    ... 18m 6s remaining
    ...
  ✓ Wait complete, retrying...

  Transcribing chunk_001.mp3 (24.5MB)...
  ✓ Transcribed 1,234 segments
```

### Implementation Architecture

**Files:**

1. **src/transcribe/provider_selector.py** - Intelligence
   - `estimate_transcription()` - Analyze file and calculate costs
   - `select_provider()` - Auto-select best provider
   - `print_estimate()` - Format user-friendly output

2. **src/transcribe/openai_backend.py** - OpenAI integration
   - `transcribe_openai()` - Simple OpenAI transcription
   - `transcribe_openai_with_preprocessing()` - With silence removal (reduces cost)

3. **src/transcribe/groq_backend.py** - Enhanced with retry
   - `parse_wait_time()` - Extract wait time from error message
   - `wait_with_progress()` - User-friendly wait display
   - `_transcribe_chunk()` - Now includes automatic retry logic

4. **scripts/run.py** - CLI integration
   - `--provider {auto,groq,openai}` - Manual override
   - `--estimate-only` - Cost estimation without processing

### API Key Setup

Add both API keys to `.env`:

```bash
# Groq (free tier)
GROQ_API_KEY=gsk_...

# OpenAI (paid)
OPENAI_API_KEY=sk-...
```

**Note:** You only need the API key for the provider you intend to use. The system will validate keys before processing.

### Troubleshooting

**Problem: "GROQ_API_KEY not found in .env"**
```bash
# Solution: Add to .env file
echo "GROQ_API_KEY=your_key_here" >> .env
```

**Problem: "OPENAI_API_KEY not found in .env"**
```bash
# Either add the key OR use Groq instead
python scripts/run.py --provider groq
```

**Problem: Groq rate limit keeps failing**
```bash
# Solution 1: Wait for rate limit to reset (1 hour)
# Solution 2: Use OpenAI instead
python scripts/run.py --provider openai
```

**Problem: OpenAI costs too much for large archive**
```bash
# Use preprocessing to reduce costs by 30-40%
# Already enabled by default in openai_backend.py

# Or process in batches during off-hours with Groq
# Split 100 videos across 10 days = 10 videos/day within free tier
```

### Best Practices

**For Development:**
- Use `--estimate-only` first to understand costs
- Use Groq for short test files (free)
- Use OpenAI when iterating quickly (worth the cost)

**For Production:**
- Set `provider: "openai"` in config for reliability
- Use preprocessing to reduce OpenAI costs
- Monitor spending with OpenAI usage dashboard

**For Large Archives:**
- Run `--estimate-only` on sample files first
- Calculate total cost: `num_files × avg_cost_per_file`
- Batch process with Groq over time (10 files/day free)
- Or budget for OpenAI bulk processing ($50-100 for 100 hours)

**Cost Optimization:**
- Enable silence removal: 30-40% cost reduction
- Process during off-peak: Groq rate limits reset hourly
- Archive compressed versions: Reprocess later if needed

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