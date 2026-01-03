# Corporate Knowledge Extractor

Automated pipeline for extracting structured knowledge from corporate meeting recordings. Converts videos with presentations into searchable markdown reports and Q&A knowledge bases optimized for RAG systems.

## Business Value

**Problem:** Corporate knowledge lives in meeting recordings - training sessions, client calls, internal updates. This knowledge is locked in video files, inaccessible and unsearchable.

**Solution:** Automatically extract, structure, and anonymize meeting content into:
- Markdown reports with slide images and detailed explanations
- JSONL Q&A pairs ready for RAG/chatbot integration
- Categorized knowledge base for easy navigation

**Use Cases:**

### Type A: Internal Training Sessions
**Input:** Product training video with PowerPoint presentation (4GB MP4/MKV)
**Focus:** Extract educational insights - what the speaker EXPLAINED, not just what's on slides
**Output:**
- Detailed markdown report with categorized sections
- Slide images with speaker commentary
- Q&A knowledge base for onboarding new team members

**Example:** 50-minute WMS platform overview with 40 slides → 15-page report with 120 Q&A pairs

### Type B: Client Meetings
**Input:** Audio-only recordings or video calls (often without presentations)
**Focus:** Capture action items, decisions, client concerns, commitments
**Output:**
- Meeting summary with key decisions
- Action item tracking
- Client sentiment analysis (planned)

**Example:** 30-minute discovery call → structured notes with follow-up tasks

### Type C: Internal Updates
**Input:** Mix of video, audio, or shared documents
**Focus:** What changed, what's important, what requires action
**Output:**
- Change summary
- Cross-functional impact analysis
- Action items by team

**Example:** Quarterly roadmap review → digestible update report

**Note:** Currently optimized for Type A (training videos). Types B and C are on the roadmap.

## Tech Stack

| Component | Technology | Why |
|-----------|-----------|-----|
| Transcription | Whisper via Groq API | Fast, accurate, free tier available |
| Frame Extraction | OpenCV | Pixel-based change detection for slide transitions |
| OCR | Tesseract (local) | Free, no API costs, good for presentation text |
| Semantic Tagging | Gemini Flash 2.0 | Batch processing, cheap, fast |
| Knowledge Synthesis | Gemini Flash 2.0 | Long context, structured output |
| Anonymization | spaCy NER + custom terms | PII detection + corporate-specific redaction |
| Output | Markdown + JSONL | Human-readable + machine-parseable |

## Cost Estimate

**Per 50-minute training video:**
- Groq API (Whisper transcription): Free tier or ~$0.001
- Gemini API (tagging + synthesis): ~$0.01-0.02
- **Total: ~$0.01-0.02 per meeting**

Compare to:
- Manual note-taking: 2-3 hours @ $50/hour = $100-150
- Professional transcription service: $75-200

## System Requirements

### Software
- Python 3.9+
- Tesseract OCR 5.0+
- FFmpeg (for video compression - optional)
- 8GB RAM minimum (16GB recommended for large videos)
- 10GB free disk space per video (temp files + output)

### API Keys (Required)
- Gemini API key (Google AI Studio)
- Groq API key (free tier available)

### Operating Systems
- Windows 10/11 (tested)
- Linux (should work, adjust Tesseract path)
- macOS (should work, adjust Tesseract path)

## Installation

### 1. Clone Repository
```bash
git clone https://github.com/yourusername/corporate-knowledge-extractor.git
cd corporate-knowledge-extractor
```

### 2. Install Python Dependencies
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/macOS
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Install Tesseract OCR

**Windows:**
1. Download installer: https://github.com/UB-Mannheim/tesseract/wiki
2. Install to `C:\Program Files\Tesseract-OCR`
3. Add to PATH or update `config/settings.yaml`

**Linux:**
```bash
sudo apt-get install tesseract-ocr
```

**macOS:**
```bash
brew install tesseract
```

### 4. Install spaCy Language Model
```bash
python -m spacy download en_core_web_sm
```

### 5. Configure API Keys

Create `.env` file in project root:
```bash
GEMINI_API_KEY=your_gemini_api_key_here
GROQ_API_KEY=your_groq_api_key_here
```

**Get API keys:**
- Gemini: https://aistudio.google.com/app/apikey
- Groq: https://console.groq.com/keys

## Configuration

### Core Settings (`config/settings.yaml`)

```yaml
input:
  directory: "data/input"
  video_extensions: [".mp4", ".mkv", ".avi", ".mov"]

output:
  directory: "output"

tools:
  tesseract_path: "C:\\Program Files\\Tesseract-OCR\\tesseract.exe"

llm:
  model: "gemini-2.0-flash"
  whisper_model: "medium"
```

### Processing Settings (`config/processing.yaml`)
- Frame extraction thresholds
- Deduplication similarity scores
- Alignment parameters

### Anonymization (`config/anonymization.yaml`)
Add company-specific terms to always redact:
```yaml
custom_terms:
  - "YourCompanyName"
  - "ClientName"
  - "ProjectCodename"

exclude_terms:
  - "WMS"  # Product names mistaken for people
  - "Azure"
```

### Categories (`config/categories.yaml`)
Customize report sections:
```yaml
order: [infrastructure, api, security, architecture]

titles:
  infrastructure: "Infrastructure & Platform"
  api: "API & Integration"

keywords:
  infrastructure: [saas, cloud, azure, platform]
  api: [rest, api, endpoint, webhook]
```

### Content Filters (`config/filters.yaml`)
Define junk patterns to skip:
```yaml
junk_patterns:
  - "^thank you$"
  - "^any questions"
  - "loading slide"
```

## Usage

### Basic Usage

1. Place video file in `data/input/`:
```bash
data/input/training-session-2024.mp4
```

2. Run pipeline:
```bash
python scripts/run.py
```

3. Find output in `output/YYYY-MM-DD_HHMM/`:
```
output/2024-01-15_1430/
├── report.md           # Main report
├── knowledge.jsonl     # Q&A pairs for RAG
├── metadata.json       # Processing metadata
└── frames/            # Extracted slide images
    ├── frame_001.png
    ├── frame_002.png
    └── ...
```

### Processing Multiple Videos

Place multiple files in `data/input/` - they will be processed sequentially.

### Command Examples

```bash
# Standard processing (default PowerPoint behavior)
python scripts/run.py

# Process specific file
python scripts/run.py --file "data/input/training.mp4"

# Compress video first
python scripts/compress_video.py data/input/large-file.mp4
python scripts/run.py
```

## Content Type Presets

The extractor supports different content types through **presets** - optimized configurations for different recording scenarios.

### Available Presets

**PowerPoint (default)** - Slide presentations
```bash
python scripts/run.py --preset powerpoint
# or just: python scripts/run.py
```
- **Use for:** Training videos, pitch decks, technical presentations
- **Behavior:** Detects distinct slide transitions (5% pixel change)
- **Sampling:** Every 1 second
- **Expected output:** 30-60 frames/hour, 25-50 slides in report

**Excel** - Spreadsheet reviews
```bash
python scripts/run.py --preset excel
```
- **Use for:** Financial reports, data analysis, dashboard reviews
- **Behavior:** Sparse sampling to avoid capturing every scroll/cell change
- **Sampling:** Every 10 seconds, 30% pixel change threshold
- **Expected output:** 10-20 frames/hour, 8-15 slides in report
- **Why different:** Scrolling generates many similar frames - aggressive sampling prevents explosion

**Demo** - Software demonstrations
```bash
python scripts/run.py --preset demo
```
- **Use for:** Feature walkthroughs, UI tutorials, workflow training
- **Behavior:** Time-based sampling with moderate change detection
- **Sampling:** Every 15 seconds, 25% pixel change threshold
- **Expected output:** 15-25 frames/hour, 12-20 slides in report
- **Why different:** Cursor movement and hover states shouldn't trigger frames

**Audio Only** - Meetings without video
```bash
python scripts/run.py --preset audio_only
```
- **Use for:** Client calls, phone meetings, interviews
- **Behavior:** No frame extraction, transcription-only processing
- **Output:** Meeting notes format with action items and decisions
- **Expected output:** No slides, conversation-focused Q&A pairs

**Hybrid (Adaptive)** - Mixed content
```bash
python scripts/run.py --preset hybrid
```
- **Use for:** Meetings that mix PowerPoint, demos, and Excel
- **Behavior:** Auto-detects content type every 60 seconds and adjusts sampling
- **Adaptive logic:**
  - High activity (>5 frames/min) → switches to demo mode
  - Low activity (<2 frames/min) → switches to powerpoint mode
- **Expected output:** 20-40 frames/hour, adapts to content
- **Logging:** Mode switches logged to console for transparency

### Preset Comparison Table

| Preset | Sample Rate | Pixel Threshold | Max per Minute | Best For |
|--------|-------------|-----------------|----------------|----------|
| powerpoint | 1s | 5% | 10 | Slide-based presentations |
| excel | 10s | 30% | 3 | Spreadsheet scrolling |
| demo | 15s | 25% | 4 | Software demonstrations |
| audio_only | N/A | N/A | 0 | Voice-only meetings |
| hybrid | Adaptive | Adaptive | Adaptive | Mixed content |

### Manual Parameter Override

Override specific parameters without using presets:

```bash
# Custom sample rate and threshold
python scripts/run.py --sample-rate 5 --pixel-threshold 0.15

# Combine preset with override
python scripts/run.py --preset demo --sample-rate 20
```

### Example Scenarios

**Scenario 1: Financial Report Review**
```bash
# Video shows presenter scrolling through Excel spreadsheet
python scripts/run.py --preset excel data/input/q4-review.mp4

# Result: ~12 frames instead of 90+ (avoids capturing every cell selection)
```

**Scenario 2: Product Feature Demo**
```bash
# Live software walkthrough with UI interactions
python scripts/run.py --preset demo data/input/feature-demo.mp4

# Result: ~20 frames capturing key application states, ignoring cursor moves
```

**Scenario 3: Mixed Training Session**
```bash
# Starts with slides, then switches to live demo, then Q&A
python scripts/run.py --preset hybrid data/input/full-training.mp4

# Result: Automatically adapts - tight sampling for slides, sparse for demo
# Logs mode switches:
#   [01:00] Switching to POWERPOINT mode (low activity)
#   [15:00] Switching to DEMO mode (high activity)
#   [45:00] Switching to POWERPOINT mode (low activity)
```

**Scenario 4: Client Discovery Call**
```bash
# Audio-only recording, no screen sharing
python scripts/run.py --preset audio_only data/input/client-call.m4a

# Result: No frames, focus on action items and decisions
```

## Report Comparison

Compare two reports to detect quality regressions or improvements.

### Basic Comparison

```bash
# Compare old and new reports
python scripts/compare_reports.py output/2025-01-01_1200 output/2025-01-03_1430

# Outputs:
#   comparison_report.md - Human-readable diff
#   comparison_metrics.json - Machine-readable metrics
```

### Comparison Output

**comparison_report.md:**
- Summary table (frames, slides, Q&A counts, quality metrics)
- Overall verdict (improved/degraded/mixed/unchanged)
- Improvements list (e.g., "Longer explanations +20%")
- Regressions list (e.g., "More junk frames +3")
- Removed/added slides
- Changed slide explanations with examples

**comparison_metrics.json:**
```json
{
  "timestamp": "2026-01-03T14:30:00",
  "frames": {"old_count": 94, "new_count": 87, "change": -7},
  "slides": {"old_count": 65, "new_count": 58, "removed_titles": [...]},
  "qa_pairs": {"old_count": 280, "new_count": 310, "change": +30},
  "quality": {
    "improvements": ["Longer explanations (+20%)", "Fewer junk frames (-60%)"],
    "regressions": []
  },
  "verdict": {
    "verdict": "improved",
    "summary": "Quality has improved - no regressions"
  }
}
```

### CI/CD Integration

Use in continuous integration to prevent quality regressions:

```bash
# Fail build if regressions detected
python scripts/compare_reports.py \
  tests/fixtures/baseline_report \
  output/latest \
  --fail-on-regression

# Exit code:
#   0 = no regressions (or improved)
#   1 = regressions detected
```

### Baseline Comparison

```bash
# Establish baseline
python scripts/run.py
cp -r output/latest tests/fixtures/baseline_report

# Later, compare against baseline
python scripts/run.py  # Process video with updated code/prompts
python scripts/compare_reports.py \
  tests/fixtures/baseline_report \
  output/latest
```

## Output Structure

### report.md
Markdown document with:
- Executive summary
- Categorized sections (Infrastructure, Security, API, etc.)
- Slide breakdowns with images
- Each slide includes:
  - Title and visual description
  - Technical details (versions, specs, numbers)
  - Speaker explanation (main content)
  - Context and relationships
  - Key terminology definitions

### knowledge.jsonl
JSONL file with Q&A pairs:
```json
{"question": "What is the RTO for disaster recovery?", "answer": "4 hours", "category": "sla", "source": "frame_015"}
{"question": "Which Azure regions are supported?", "answer": "East US, West Europe, Southeast Asia", "category": "infrastructure", "source": "frame_003"}
```

**Use cases:**
- Import into vector database for RAG
- Feed to chatbot for Q&A
- Search/filter by category
- Track knowledge coverage

### metadata.json
Processing statistics:
```json
{
  "video_filename": "training.mp4",
  "duration_seconds": 3045,
  "frames_extracted": 87,
  "unique_slides": 42,
  "qa_pairs": 156,
  "processing_time_seconds": 245,
  "llm_model": "gemini-3-flash-preview"
}
```

## Pipeline Architecture

```
Video File (MP4/MKV)
    ↓
[1] Transcribe (Groq Whisper)
    → segments[] {start, end, text}
    ↓
[2] Extract Frames (OpenCV)
    → frames[] {timestamp, path}
    ↓
[3] OCR (Tesseract)
    → frames[] + {text}
    ↓
[4] Semantic Tagging (Gemini)
    → frames[] + {tags}
    ↓
[5] Speech-Frame Alignment
    → aligned[] {speech, slide_text, frame_idx}
    ↓
[6] Anonymization (spaCy + custom)
    → redacted content
    ↓
[7] Knowledge Synthesis (Gemini)
    → {slide_breakdown[], qa_pairs[]}
    ↓
[8] Post-Processing
    → deduplicated, categorized
    ↓
[9] Generate Output
    → report.md, knowledge.jsonl, frames/
```

## Troubleshooting

### Problem: "Tesseract not found"
**Solution:**
- Windows: Install from https://github.com/UB-Mannheim/tesseract/wiki
- Update `config/settings.yaml` with correct path
- Verify: `tesseract --version`

### Problem: "GEMINI_API_KEY not found"
**Solution:**
- Create `.env` file in project root
- Add `GEMINI_API_KEY=your_key_here`
- Restart terminal/IDE to load new environment

### Problem: "Out of memory"
**Solution:**
- Reduce video resolution before processing
- Use `scripts/compress_video.py` (planned feature)
- Process shorter segments
- Increase system RAM

### Problem: Too many duplicate slides in report
**Solution:**
- Increase `dedup_similarity` threshold in `config/processing.yaml`
- Current default: 0.85 (85% similar = duplicate)
- Try: 0.90 for stricter deduplication

### Problem: Missing speaker explanations (generic summaries)
**Solution:**
- Check `config/prompts/knowledge_extraction.txt`
- Ensure prompt emphasizes extracting actual speech content
- Verify `speaker_explanation` field is populated in output
- May indicate audio quality issues (check transcription accuracy)

### Problem: Frame images don't match descriptions
**Solution:**
- This indicates frame numbering mismatch
- File issue on GitHub with sample output
- Temporary workaround: manual verification

### Problem: API rate limits exceeded
**Solution:**
- Add delays in `config/processing.yaml`
- Use smaller `chunk_size` for LLM calls (default: 10)
- Groq free tier: 30 requests/minute
- Gemini free tier: 15 requests/minute

### Problem: OCR missing text from slides
**Solution:**
- Check Tesseract language: `tesseract --list-langs`
- Improve video quality (higher resolution)
- Adjust frame extraction threshold in `config/processing.yaml`

### Problem: Personal names not anonymized
**Solution:**
- Add to `config/anonymization.yaml` custom_terms
- Verify spaCy model: `python -m spacy download en_core_web_sm`
- Check anonymization.yaml exclude_terms (may be blocking)

## Development

### Project Structure
```
corporate-knowledge-extractor/
├── config/                 # Configuration files
│   ├── prompts/           # LLM prompt templates
│   ├── settings.yaml      # Paths, tools, models
│   ├── processing.yaml    # Frame extraction, alignment
│   ├── anonymization.yaml # PII redaction rules
│   ├── categories.yaml    # Report categorization
│   ├── filters.yaml       # Content filtering
│   └── config_loader.py   # Config utility
├── src/                   # Source code
│   ├── transcribe/        # Audio → text (Groq)
│   ├── frames/            # Frame extraction + tagging
│   ├── ocr/               # Tesseract integration
│   ├── align/             # Speech-frame alignment
│   ├── anonymize/         # PII redaction
│   ├── synthesize/        # LLM knowledge extraction
│   └── output/            # Report generation
├── scripts/               # Entry points
│   └── run.py            # Main pipeline
├── tests/                 # Test suite (planned)
├── data/input/           # Input videos
└── output/               # Generated reports
```

### Adding New Features

See `ROADMAP.md` for planned improvements.

### Git Workflow
- `main`: Stable, tested code
- `feature/*`: New features
- Commit after each working change
- Never mix prompt changes with code changes

## Contributing

1. Fork repository
2. Create feature branch
3. Make changes
4. Test with real video
5. Submit pull request

## License

MIT License - see LICENSE file

## Support

- GitHub Issues: Report bugs or request features
- Documentation: See `CLAUDE.md` for technical architecture
- Roadmap: See `ROADMAP.md` for planned features

## Acknowledgments

- OpenAI Whisper for transcription model
- Google Gemini for knowledge synthesis
- Groq for fast Whisper API
- Tesseract OCR project
- spaCy for NER

---

**Version:** 1.0.0 (Current - Training Video Focus)
**Last Updated:** January 2025
