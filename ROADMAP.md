# Corporate Knowledge Extractor - Roadmap

**Vision:** Universal corporate knowledge extraction platform supporting all meeting types (training, client calls, internal updates) with multi-format input and automated quality assurance.

**Current State:** v1.0 - Optimized for Type A meetings (training videos with slides)

---

## Phase 1: Stabilization & Testing (CURRENT - Q1 2026)

**Goal:** Ensure Type A (training videos) pipeline is robust, tested, and production-ready.

### 1.1 Automated Quality Diagnostics
- [x] Create `tests/` directory structure
- [ ] Implement `test_quality.py` with diagnostic functions:
  - [ ] `check_speaker_explanation_quality()` - Detect raw transcript vs. insights
  - [ ] `check_no_junk_frames()` - Verify filtering working
  - [ ] `check_categories_balanced()` - Validate categorization
  - [ ] `check_qa_pairs_quality()` - Q&A format and specificity
- [ ] Add quality metrics to `metadata.json`:
  - [ ] Average speaker_explanation length
  - [ ] Category distribution
  - [ ] Junk frame percentage
  - [ ] Q&A specificity score
- [ ] Create baseline dataset (3-5 known-good training videos)
- [ ] Document quality thresholds (when to reject output)

**Success Criteria:**
- Automated tests run on every pipeline execution
- Quality metrics logged to metadata.json
- Baseline dataset established for regression testing

### 1.2 CI/CD & Testing Infrastructure
- [ ] Set up pytest framework
- [ ] Implement `test_config.py`:
  - [ ] Config loader validation
  - [ ] YAML syntax checking
  - [ ] Required field verification
- [ ] Implement `test_pipeline.py`:
  - [ ] End-to-end integration test with sample video
  - [ ] Component unit tests (transcribe, extract, align, etc.)
  - [ ] Mock API calls for faster testing
- [ ] GitHub Actions workflow:
  - [ ] Run tests on PR
  - [ ] Quality gate (block merge if tests fail)
  - [ ] Automated quality report on commit

**Success Criteria:**
- All tests passing
- CI/CD pipeline running on every commit
- Test coverage >70% for core modules

### 1.3 Video Compression & Optimization
**Problem:** 4GB video files are slow to process and expensive to store.

**Solution:** Pre-processing compression pipeline.

- [ ] Create `scripts/compress_video.py`:
  - [ ] FFmpeg wrapper for video compression
  - [ ] Target: 4GB → 500MB-1GB
  - [ ] Preserve quality for OCR (720p minimum)
  - [ ] Options:
    - [ ] Video + audio (standard)
    - [ ] Audio-only extraction (for Type B meetings)
    - [ ] Frame rate reduction (presentation detection)
- [ ] Create `scripts/batch_compress.py`:
  - [ ] Process all videos in directory
  - [ ] Progress tracking
  - [ ] Error handling (skip corrupted files)
  - [ ] Compression report (original size, new size, ratio)
- [ ] Add compression quality verification:
  - [ ] OCR accuracy check (before/after)
  - [ ] Transcription accuracy check
  - [ ] Report degradation if quality drops
- [ ] Update `run.py`:
  - [ ] Optional `--compress` flag
  - [ ] Auto-detect large files (>2GB) and suggest compression

**Success Criteria:**
- 4GB training video compresses to <1GB
- OCR accuracy maintained (>95% same text detected)
- Processing time reduced by 40%+

### 1.4 Documentation & Usability
- [ ] Complete README.md with real examples
- [ ] Add troubleshooting section based on common issues
- [ ] Create video tutorial (5-min quickstart)
- [ ] Example outputs in `/examples` directory
- [ ] Cost calculator (input duration → estimated API cost)

**Deliverables:**
- Robust test suite with automated quality checks
- CI/CD pipeline with quality gates
- Video compression tools (4GB → 500MB)
- Professional documentation

**Timeline:** 4-6 weeks

---

## Phase 2: Multi-Type Meeting Support (Q2 2026)

**Goal:** Support all three meeting types with intelligent auto-detection.

### 2.1 Type A Enhancements (Training Sessions)
**Current Focus - Already Supported**

Improvements:
- [ ] Better duplicate slide detection (semantic similarity, not just OCR)
- [ ] Slide transition detection (animation awareness)
- [ ] Multi-presenter support (speaker diarization)
- [ ] Interactive Q&A extraction (audience questions + answers)

### 2.2 Type B Support (Client Meetings)
**Characteristics:** Audio-only, discussion-based, action items, decisions

**Pipeline Modifications:**
- [ ] Audio-only mode (skip frame extraction):
  - [ ] Detect input type (audio vs. video)
  - [ ] Direct transcription without frame processing
  - [ ] Speaker diarization (who said what)
- [ ] Action item detection:
  - [ ] NER for tasks ("John will send proposal by Friday")
  - [ ] Task extraction prompt template
  - [ ] Owner assignment
  - [ ] Deadline detection
- [ ] Decision tracking:
  - [ ] Identify decision points in discussion
  - [ ] Extract agreed-upon choices
  - [ ] Note dissenting opinions/concerns
- [ ] Sentiment analysis:
  - [ ] Client satisfaction indicators
  - [ ] Risk flags (concerns, objections)
  - [ ] Opportunity signals (interest, enthusiasm)

**Output Format:**
```markdown
# Client Meeting Summary

**Date:** 2026-02-15
**Participants:** John (Client), Sarah (Sales), Mike (Solutions)
**Duration:** 45 minutes

## Key Decisions
1. Move forward with Pilot Phase (approved by John)
2. Timeline: 6 weeks starting March 1st
3. Budget: $50K approved

## Action Items
- [ ] Sarah: Send SOW by Feb 20 (OWNER: Sarah, DUE: 2026-02-20)
- [ ] Mike: Prepare technical architecture diagram (OWNER: Mike, DUE: 2026-02-18)
- [ ] John: Get security team approval (OWNER: John, DUE: 2026-02-25)

## Client Concerns
- Data residency requirements (EU only)
- Integration with legacy SAP system
- Training timeline for 50 users

## Next Steps
- Follow-up call scheduled: March 1st
- Pilot kickoff: March 5th
```

**New Config Files:**
- `config/meeting_types.yaml` - Type detection rules
- `config/action_items.yaml` - Task extraction patterns
- `config/sentiment.yaml` - Sentiment analysis keywords

**Success Criteria:**
- Audio-only processing works (no frames required)
- Action items extracted with 90% accuracy
- Speaker diarization with 85% accuracy
- Sentiment analysis provides useful flags

### 2.3 Type C Support (Internal Updates)
**Characteristics:** Mix of video/audio/documents, change tracking

**Pipeline Modifications:**
- [ ] Multi-format input:
  - [ ] PDF ingestion (meeting notes, reports)
  - [ ] DOCX support (shared documents)
  - [ ] XLSX parsing (roadmaps, timelines)
  - [ ] Video + document combination
- [ ] Change detection:
  - [ ] Compare with previous meetings (delta)
  - [ ] Highlight "what's new"
  - [ ] Deprecated features/decisions
- [ ] Impact analysis:
  - [ ] Tag by team/function (Engineering, Sales, Support)
  - [ ] Cross-functional dependencies
  - [ ] Timeline implications

**Output Format:**
```markdown
# Internal Update Summary

**Meeting:** Q1 Roadmap Review
**Date:** 2026-03-15

## What's New
- Feature X moved to Q2 (was Q1)
- New hire: Senior Backend Engineer starting April 1
- Budget increase: +$200K for cloud infrastructure

## Changes from Last Update
- API v2 timeline delayed 3 weeks (dependency on Feature Y)
- Mobile app priority increased (customer feedback)

## Impact by Team
### Engineering
- Additional backend capacity (new hire)
- Refactor database schema for Feature X

### Sales
- Update pitch decks (Feature X timeline change)
- New pricing tier approved

### Support
- Training needed for Feature X (Q2)
- Knowledge base update required

## Action Items
- [Engineering] Complete database refactor by April 15
- [Sales] Update sales collateral by March 20
```

**Success Criteria:**
- PDF/DOCX/XLSX ingestion working
- Change detection identifies deltas
- Impact analysis tags relevant teams

### 2.4 Auto-Detection & Routing
**Goal:** Automatically determine meeting type and apply appropriate pipeline.

- [ ] Meeting type classifier:
  - [ ] Input analysis (has video? has documents? speaker count?)
  - [ ] Content-based hints (keywords: "training", "client", "update")
  - [ ] Duration heuristics
  - [ ] User override option
- [ ] Dynamic pipeline routing:
  - [ ] Type A → frame extraction + knowledge synthesis
  - [ ] Type B → audio + action items + sentiment
  - [ ] Type C → multi-input + change detection
- [ ] Unified output interface:
  - [ ] Common metadata format
  - [ ] Type-specific sections
  - [ ] Consistent JSONL for all types

**New Scripts:**
- `scripts/detect_type.py` - Analyze input and suggest type
- `scripts/run_auto.py` - Auto-detect and route to appropriate pipeline

**Success Criteria:**
- 95% accurate type detection on test set
- All three types produce quality output
- Unified interface for all meeting types

**Deliverables:**
- Type A enhancements (better deduplication, multi-presenter)
- Type B full support (audio-only, action items, sentiment)
- Type C full support (multi-input, change detection, impact)
- Auto-detection and routing

**Timeline:** 8-10 weeks

---

## Phase 3: Quality & Intelligence (Q3 2026)

**Goal:** Improve output quality, add intelligence features, optimize costs.

### 3.1 Advanced Quality Metrics
- [ ] Report quality scoring:
  - [ ] Insight density (specific facts per slide)
  - [ ] Junk ratio (valuable vs. filler content)
  - [ ] Categorization accuracy
  - [ ] PII leakage detection
- [ ] Automated quality reports:
  - [ ] Quality score in metadata.json
  - [ ] Recommendations for improvement
  - [ ] Flag low-quality outputs for review
- [ ] Quality trends dashboard:
  - [ ] Track quality over time
  - [ ] Identify degradation (model updates, config changes)
  - [ ] A/B test prompt variations

### 3.2 Prompt Optimization & A/B Testing
**Problem:** Hard to know if prompt changes improve quality.

**Solution:** Systematic A/B testing framework.

- [ ] Prompt versioning:
  - [ ] Store prompts in `config/prompts/` with version numbers
  - [ ] Track which version produced which output
  - [ ] Easy rollback to previous versions
- [ ] A/B testing infrastructure:
  - [ ] Process same video with Prompt A and Prompt B
  - [ ] Compare quality metrics
  - [ ] Statistical significance testing
  - [ ] Winner selection automation
- [ ] Prompt templates library:
  - [ ] Best practices collection
  - [ ] Domain-specific variations (technical, business, general)
  - [ ] Example few-shot prompts
- [ ] LLM model comparison:
  - [ ] Test Gemini Flash vs. Gemini Pro
  - [ ] Cost/quality tradeoffs
  - [ ] Speed benchmarks

**Tools:**
- `tests/test_prompt_quality.py` - Compare prompt versions
- `scripts/ab_test.py` - Run A/B tests on prompts
- `scripts/benchmark_models.py` - Compare different LLMs

**Success Criteria:**
- Systematic prompt improvement process
- Data-driven prompt selection
- Quality improvements measured quantitatively

### 3.3 Semantic Deduplication
**Problem:** Current deduplication uses OCR text similarity, misses semantic duplicates.

**Solution:** Embedding-based semantic comparison.

- [ ] Generate embeddings for slides:
  - [ ] Use sentence-transformers (local, no API cost)
  - [ ] Embed visual_content + speaker_explanation
  - [ ] Store embeddings for comparison
- [ ] Semantic similarity clustering:
  - [ ] Cosine similarity threshold (e.g., >0.90 = duplicate)
  - [ ] Merge semantically identical slides
  - [ ] Keep most detailed version
- [ ] Smart merging:
  - [ ] Combine complementary information
  - [ ] Preserve all unique insights
  - [ ] Note merged sources

**Success Criteria:**
- Reduce duplicate slides by 30-40%
- Preserve all unique information
- No increase in processing time

### 3.4 Benchmark Dataset & Regression Testing
**Problem:** No systematic way to measure improvement or catch regressions.

**Solution:** Curated benchmark dataset with ground truth.

- [ ] Create benchmark set:
  - [ ] 10 diverse training videos (different topics, lengths, quality)
  - [ ] Manual annotation (ground truth):
    - [ ] Expected number of slides
    - [ ] Expected categories
    - [ ] Sample Q&A pairs
    - [ ] Key terminology
- [ ] Automated regression suite:
  - [ ] Process benchmark set with current pipeline
  - [ ] Compare output to ground truth
  - [ ] Score accuracy (precision/recall for Q&A, categories)
  - [ ] Fail CI if regression detected
- [ ] Continuous improvement tracking:
  - [ ] Log benchmark scores over time
  - [ ] Visualize improvements
  - [ ] Celebrate wins

**Success Criteria:**
- Benchmark dataset of 10 videos with ground truth
- Automated regression testing on every release
- No quality regressions introduced

**Deliverables:**
- Advanced quality metrics and scoring
- A/B testing framework for prompts
- Semantic deduplication (embedding-based)
- Benchmark dataset with regression testing

**Timeline:** 6-8 weeks

---

## Phase 4: Production & Scale (Q4 2026)

**Goal:** Deploy at scale, handle high volume, add enterprise features.

### 4.1 Performance & Scalability
- [ ] Batch processing optimization:
  - [ ] Parallel video processing
  - [ ] Queue management (prioritization)
  - [ ] Resource limits (max concurrent jobs)
- [ ] Incremental processing:
  - [ ] Track processed videos (hash-based)
  - [ ] Skip re-processing (cache results)
  - [ ] Delta updates (new meetings only)
- [ ] Cloud deployment:
  - [ ] Containerization (Docker)
  - [ ] Kubernetes orchestration
  - [ ] Auto-scaling workers
  - [ ] Object storage integration (S3, Azure Blob)
- [ ] Cost optimization:
  - [ ] API call batching
  - [ ] Caching (transcripts, embeddings)
  - [ ] Rate limit handling (retry with backoff)

### 4.2 Enterprise Features
- [ ] User management:
  - [ ] Multi-user support
  - [ ] Role-based access (uploader, reviewer, admin)
  - [ ] Audit logs (who processed what, when)
- [ ] Organization features:
  - [ ] Team workspaces
  - [ ] Shared knowledge bases
  - [ ] Custom category taxonomies per team
- [ ] Integration APIs:
  - [ ] REST API for programmatic access
  - [ ] Webhooks (processing complete notifications)
  - [ ] Export to Confluence, Notion, SharePoint
  - [ ] SSO integration (SAML, OAuth)

### 4.3 Web Interface
**Goal:** Non-technical users can upload videos and download reports.

- [ ] Upload interface:
  - [ ] Drag-and-drop video upload
  - [ ] Progress tracking
  - [ ] Metadata input (title, meeting type, participants)
- [ ] Processing dashboard:
  - [ ] Job queue status
  - [ ] Real-time progress updates
  - [ ] Error notifications
- [ ] Results viewer:
  - [ ] Inline report preview
  - [ ] Download markdown/JSONL
  - [ ] Share link generation
  - [ ] Search across reports
- [ ] Admin panel:
  - [ ] Usage analytics
  - [ ] Cost tracking
  - [ ] Quality metrics dashboard
  - [ ] User management

**Tech Stack (Proposed):**
- Frontend: React + TypeScript
- Backend: FastAPI (Python)
- Database: PostgreSQL (metadata, user data)
- Storage: S3/Azure Blob (videos, reports)
- Queue: Redis/Celery (job processing)

### 4.4 Advanced Analytics
- [ ] Knowledge base search:
  - [ ] Vector search over all Q&A pairs
  - [ ] Semantic similarity (find related topics)
  - [ ] Full-text search
  - [ ] Filter by category, date, meeting type
- [ ] Insights dashboard:
  - [ ] Most-discussed topics (across all meetings)
  - [ ] Knowledge gaps (questions without answers)
  - [ ] Trend analysis (what's hot this quarter)
  - [ ] Duplicate knowledge detection (redundant meetings)
- [ ] Recommendations:
  - [ ] Suggest related meetings
  - [ ] Identify missing documentation
  - [ ] Auto-tag expertise (who knows what)

### 4.5 Multi-Language Support
- [ ] Transcription:
  - [ ] Whisper supports 90+ languages
  - [ ] Language detection
  - [ ] Translation option (to English)
- [ ] LLM synthesis:
  - [ ] Gemini supports multiple languages
  - [ ] Language-specific prompts
  - [ ] Cultural context awareness
- [ ] OCR:
  - [ ] Tesseract language packs
  - [ ] Multi-language slide detection

**Deliverables:**
- Cloud-native deployment (Docker, Kubernetes)
- Web interface for uploads and results
- Enterprise features (SSO, multi-tenant, APIs)
- Advanced analytics and search
- Multi-language support

**Timeline:** 10-12 weeks

---

## Beyond Phase 4: Future Exploration

**Potential Features (Not Prioritized):**

### Real-Time Processing
- Live meeting transcription and note-taking
- Real-time action item detection during calls
- Live Q&A suggestion (chatbot in meeting)

### Intelligence Augmentation
- Automatic meeting preparation (previous context)
- Participant expertise matching (who should attend)
- Meeting outcome prediction (will this be valuable?)

### Integrations
- Calendar integration (auto-process recorded meetings)
- Slack/Teams bots (post summaries to channels)
- CRM integration (link client meetings to accounts)
- LMS integration (training content to learning platforms)

### Advanced AI
- Multimodal understanding (understand diagrams, charts)
- Video understanding (gestures, expressions, engagement)
- Predictive analytics (meeting effectiveness scoring)

---

## Success Metrics

**Phase 1 (Stabilization):**
- Test coverage >70%
- CI/CD pipeline operational
- Video compression working (4GB → <1GB)
- Quality metrics baseline established

**Phase 2 (Multi-Type):**
- All 3 meeting types supported
- Auto-detection 95% accurate
- Type B action items 90% accurate
- Type C change detection functional

**Phase 3 (Quality):**
- Prompt A/B testing framework working
- Benchmark dataset created (10 videos)
- Semantic deduplication reduces duplicates 30%+
- Quality scores tracked over time

**Phase 4 (Production):**
- Web UI deployed and functional
- 100+ videos processed per week
- <5% error rate
- User satisfaction >4/5

---

## Resource Requirements

**Development Time (Estimated):**
- Phase 1: 4-6 weeks (1 developer)
- Phase 2: 8-10 weeks (1-2 developers)
- Phase 3: 6-8 weeks (1 developer)
- Phase 4: 10-12 weeks (2-3 developers + 1 designer)

**Total:** 7-9 months

**Infrastructure Costs (Estimated Monthly at Scale):**
- API Costs (100 videos/month): $10-20
- Cloud Hosting (AWS/Azure): $100-200
- Storage (S3/Blob): $20-50
- Total: $130-270/month

**Break-Even Analysis:**
- Manual note-taking cost: $100-150 per meeting
- Automated cost: ~$0.01-0.02 per meeting
- Break-even: 2-3 meetings (ROI immediate)

---

## Risk Mitigation

**Technical Risks:**
1. **LLM quality regression** (API changes)
   - Mitigation: Benchmark suite detects regressions immediately
   - Fallback: Pinned model versions, rollback capability

2. **API rate limits at scale**
   - Mitigation: Queue system with backoff
   - Alternative: Self-hosted models (Whisper, LLaMA)

3. **Large file processing failures**
   - Mitigation: Compression pipeline, chunked processing
   - Monitoring: Error tracking and alerting

**Operational Risks:**
1. **PII leakage in reports**
   - Mitigation: Enhanced anonymization, pre-release review
   - Audit: Regular PII detection scans

2. **Cost overruns (API usage)**
   - Mitigation: Budget alerts, rate limiting
   - Optimization: Caching, model downgrades where possible

**User Adoption Risks:**
1. **Too complex for non-technical users**
   - Mitigation: Web UI (Phase 4), video tutorials
   - Support: Documentation, examples, quick-start guide

2. **Output quality not meeting expectations**
   - Mitigation: Quality metrics, A/B testing, continuous improvement
   - Feedback loop: User ratings, iterative refinement

---

## Version History

- **v1.0 (Current):** Type A support (training videos)
- **v1.1 (Phase 1):** Testing + compression
- **v2.0 (Phase 2):** Multi-type support (A, B, C)
- **v2.5 (Phase 3):** Quality improvements + benchmarks
- **v3.0 (Phase 4):** Production deployment + web UI

---

**Last Updated:** January 2026
**Owner:** Pre-Sales Engineering Team
**Status:** Phase 1 In Progress
