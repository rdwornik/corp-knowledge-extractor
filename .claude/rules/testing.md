# Testing Standards

- pytest for all tests, never unittest
- Test files: tests/test_*.py
- Run: python -m pytest
- 480+ tests passing
- No silent failures — log warnings, raise on errors
- python-pptx MemoryError: patch at function level in tests (regex compilation bug on 3.12)
