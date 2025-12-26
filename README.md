# Corporate Knowledge Extractor

Extract structured knowledge from meeting recordings and PowerPoints.

## Setup
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Modules

- `src/transcribe/` — Video/audio → transcript (Whisper)
- `src/extract/` — PPTX → text
- `src/synthesize/` — LLM synthesis