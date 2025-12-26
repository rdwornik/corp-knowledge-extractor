import re
import spacy

# Load model once
nlp = spacy.load("en_core_web_sm")


def anonymize(
    text: str,
    custom_terms: list[str] = None,
    auto_detect_names: bool = True,
    mask_emails: bool = True,
    mask_phones: bool = True
) -> str:
    """
    Mask personal/sensitive data in text.
    
    Args:
        text: Input text
        custom_terms: Company names, project names to mask
        auto_detect_names: Use NER to find person names
        mask_emails: Mask email addresses
        mask_phones: Mask phone numbers
    
    Returns:
        Anonymized text
    """
    # Auto-detect names
    if auto_detect_names:
        doc = nlp(text)
        for ent in reversed(doc.ents):  # Reverse to preserve indices
            if ent.label_ == "PERSON":
                text = text[:ent.start_char] + "[PERSON]" + text[ent.end_char:]
    
    # Emails
    if mask_emails:
        text = re.sub(
            r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
            '[EMAIL]',
            text
        )
    
    # Phone numbers
    if mask_phones:
        text = re.sub(
            r'\+?[\d\s\-\(\)]{10,}',
            '[PHONE]',
            text
        )
    
    # Custom terms
    if custom_terms:
        for term in custom_terms:
            text = re.sub(
                re.escape(term),
                '[REDACTED]',
                text,
                flags=re.IGNORECASE
            )
    
    return text