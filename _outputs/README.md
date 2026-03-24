# CKE Extraction Outputs — Archive

All extraction outputs consolidated here for reference, analysis, and
pattern learning. NO output should be deleted without explicit approval.

## v2/ (formerly output_v2/)
- **CKE version:** v0.4.0
- **Tags:** NO (pre-tag system)
- **Provenance:** NO
- **File naming:** old format ({source_stem}.md)
- **Packages:** projects (128 notes), rfp (45), source_library (276), templates (43)
- **Status:** Historical baseline. DO NOT ingest to vault.
- **Use for:** comparing v0.4 vs v0.5+ quality improvement, understanding
  extraction evolution

## v3/ (formerly output_v3/)
- **CKE version:** v0.5.0
- **Tags:** YES (11 prefixes)
- **Provenance:** YES (source_path, source_hash, extracted_at)
- **File naming:** new format ({date}_{name}_{hash4}.md)
- **Packages:** projects (167 notes), source_library (42 notes)
- **Status:** Batch B (aborted). Valid extractions but no user context.
  Source paths point to OneDrive (may be stale after MyWork restructure).
- **Use for:** tag analysis, classifier training data, quality benchmarking,
  cherry-pick valuable notes through magistrala

## test/ (formerly output/test/ + output_test/)
- **Packages:** test, pdf_test, pptx_test, project_test
- **Use for:** regression testing, format verification

## golden_set/ (formerly output/golden_set/)
- **Packages:** golden_set (21 human-verified notes), golden_set_review (review files)
- **Use for:** eval_extraction.py benchmarking, quality gate calibration

## misc/ (formerly output/)
- **Packages:** 01_Product_Docs, Cognitive_Friday
- 01_Product_Docs: raw source library (no _meta.yaml, pre-pipeline)
- Cognitive_Friday: v0.5+ extraction with provenance (model: gemini-3-flash-preview)
- **Use for:** source material reference, Cognitive Friday analysis
