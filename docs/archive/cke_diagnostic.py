"""CKE Diagnostic — run from corp-knowledge-extractor folder."""
import os
import sys

print("=== CKE DIAGNOSTIC ===\n")

# 1. doc_type classifier
try:
    from src.doc_type_classifier import classify_doc_type, should_extract_deep
    dtype = classify_doc_type("60_Source_Library/02_Training_Enablement/test.pptx")
    deep = should_extract_deep(dtype)
    print(f"1. doc_type classifier: OK (type={dtype}, deep={deep})")
except Exception as e:
    print(f"1. doc_type classifier: FAILED ({e})")

# 2. Provider router
try:
    from src.providers.router import route_model
    model = route_model(tier=2, text_length=5000)
    print(f"2. Provider router: OK (model={model})")
except Exception as e:
    print(f"2. Provider router: FAILED ({e})")

# 3. ANTHROPIC_API_KEY
key = os.environ.get("ANTHROPIC_API_KEY", "")
if key:
    print(f"3. ANTHROPIC_API_KEY: SET ({key[:8]}...)")
else:
    print("3. ANTHROPIC_API_KEY: NOT SET — Haiku won't work!")

# 4. Slide renderer
try:
    from src.slides.renderer import can_render
    print(f"4. Slide renderer: can_render={can_render()}")
except Exception as e:
    print(f"4. Slide renderer: FAILED ({e})")

# 5. Freshness module
try:
    from src.freshness import compute_source_hash
    print("5. Freshness module: OK")
except Exception as e:
    print(f"5. Freshness module: FAILED ({e})")

# 6. Deep prompt
from pathlib import Path
dp = Path("config/prompts/deep_prompt.txt")
print(f"6. Deep prompt file: {'EXISTS' if dp.exists() else 'MISSING'}")

# 7. Overlay fields
of = Path("config/prompts/overlay_fields.yaml")
print(f"7. Overlay fields file: {'EXISTS' if of.exists() else 'MISSING'}")

# 8. Check extract.py for deep extraction wiring
try:
    code = Path("src/extract.py").read_text(encoding="utf-8")
    has_deep = "deep_prompt" in code or "_apply_deep_fields" in code
    has_provider = "providers" in code or "AnthropicProvider" in code
    has_pptx_multi = "extract_pptx_multimodal" in code or "pptx_multimodal" in code
    print(f"8. extract.py wiring: deep={has_deep}, providers={has_provider}, pptx_multi={has_pptx_multi}")
except Exception as e:
    print(f"8. extract.py: FAILED ({e})")

print("\n=== END DIAGNOSTIC ===")
