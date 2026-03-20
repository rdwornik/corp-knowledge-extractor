# Python Environment — corp-knowledge-extractor

- Python >=3.11, target 3.12
- Virtual env: .venv\Scripts\Activate.ps1
- Install: pip install -e ../corp-os-meta && pip install -e ".[dev]"
- This is a PURE EXTRACTION ENGINE — no routing, no vault writes, no orchestration
- corp-os-meta is an import dependency (Pydantic schemas, taxonomy)
- Invoked by corp-by-os and corp-project-extractor via subprocess
- PPTX/XLSX/DOCX always capped at Tier 2 (Gemini rejects MIME types)
- Forward slashes in all output paths
