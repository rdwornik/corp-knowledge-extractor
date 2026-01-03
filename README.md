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
# Standard processing
python scripts/run.py

# With verbose output
python scripts/run.py --verbose

# Process specific file (planned)
python scripts/run.py --file "training.mp4"

# Compress video first (planned)
python scripts/compress_video.py data/input/large-file.mp4
python scripts/run.py
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
  "llm_model": "gemini-2.0-flash"
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
