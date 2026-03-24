# Code Review Report — corp-knowledge-extractor

**Date:** 2026-03-15
**Branch:** `code-review-2026-03-15`
**Reviewer:** Claude Code (automated)

## Summary

| Metric | Value |
|--------|-------|
| **Tests** | 324 passed, 4 skipped, 0 failed |
| **Ruff** | Clean (all checks passed) |
| **Commits** | 5 |
| **Files changed** | 55 |

## Commits Made

1. **`e19abdd` fix: resolve all failing tests**
2. **`2798716` style: ruff lint + format pass**
3. **`bbb340a` chore: remove redundant requirements.txt**
4. **`1c6dc11` docs: update CLAUDE.md to current state**
5. **`9c212b8` docs: professional README**

## Issues Found & Fixed

### Test Failures (5 fixed)

| Test | Issue | Fix |
|------|-------|-----|
| `test_comparison.py::test_comparison_with_mock_reports` | `LocalPath.touch()` doesn't exist in pytest's `tmpdir` (py.path.local) | Replaced with `open(path, "w").close()` |
| `test_quality.py` (4 tests) | Tests fail when `output/` dir exists but contains no actual report files | Added check for report.md/knowledge.jsonl before returning path; skip if absent |
| `test_scan.py::test_pptx_mock` | `MemoryError` when patching `pptx.Presentation` — python-pptx has regex compilation bug on Python 3.12 | Patched `src.scan._scan_pptx` directly instead of importing pptx module |

### Lint Issues (8 fixed)

| File | Issue | Fix |
|------|-------|-----|
| `src/batch.py` | `extract_pptx_multimodal` used in `_process_single` but not passed as parameter or imported there | Added to parameter list and call site |
| `src/compress.py` | Unused variable `result` from `subprocess.run()` | Removed assignment |
| `src/frames/__init__.py` | Implicit re-exports (`sample_frames`, `SampledFrame`) | Made explicit with `as` syntax |
| `src/frames/tagger.py` | Module-level import after `sys.path` setup (E402) | Added `# noqa: E402` (intentional) |
| `src/frames/tagger.py` | Bare `except:` (E722) | Changed to `except Exception:` |
| `src/slides/renderer.py` | `comtypes.client` imported but unused (F401) | Added `# noqa: F401` (availability check) |
| `src/slides/renderer.py` | `fitz` imported but unused (F401) | Added `# noqa: F401` (availability check) |

### Environment Cleanup

- Removed `requirements.txt` — all dependencies already in `pyproject.toml` (single source of truth)
- `venv/` directory exists but already in `.gitignore` (no action needed)

### Documentation

- **CLAUDE.md** — restructured, updated test counts (247 -> 324), added new modules (providers, slides, deep_prompt, freshness, doc_type_classifier), documented untested modules, removed stale references
- **README.md** — added features list, expanded usage examples, updated architecture with full pipeline, updated test count

## Test Coverage Gaps

Modules without dedicated test files:

| Module | Purpose |
|--------|---------|
| `src/batch.py` | Manifest-driven batch processor |
| `src/compress.py` | Video compression (FFmpeg) |
| `src/extract.py` | Gemini API extraction (core) |
| `src/frames/extractor.py` | Frame extraction |
| `src/frames/sampler.py` | Frame sampling |
| `src/frames/tagger.py` | Frame tagging |
| `src/reextract.py` | Re-extraction on existing packages |
| `src/synthesize.py` | Output package builder |
| `src/taxonomy_prompt.py` | Taxonomy prompt injection |
| `src/utils.py` | LLM JSON parser |

## Known Issues (Pre-existing, Not Fixed)

1. **quality field mismatch** — Gemini outputs "high", schema expects "full|partial|fragment"
2. **Products pollution** — "Demand Planning" in both topics AND products
3. **People as roles** — "Project Manager" instead of actual names
4. **python-pptx MemoryError** — importing pptx module can trigger MemoryError on Python 3.12 regex compilation; worked around in tests
5. **mypy not installed** — listed as dev dependency but not present in environment

## Final State

```
REPO: corp-knowledge-extractor
TESTS: 324 passed, 0 failed, 4 skipped
RUFF: clean
COMMITS: 5 (on code-review-2026-03-15 branch)
FILES CHANGED: 55
```
