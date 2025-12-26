import re


def anonymize(text: str, custom_terms: list[str] = None) -> str:
    """
    Mask personal/sensitive data in text.
    
    Args:
        text: Input text
        custom_terms: Company names, project names to mask
    
    Returns:
        Anonymized text
    """
    # Emails
    text = re.sub(
        r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
        '[EMAIL]',
        text
    )
    
    # Phone numbers (various formats)
    text = re.sub(
        r'\+?[\d\s\-\(\)]{10,}',
        '[PHONE]',
        text
    )
    
    # Custom terms (company names, etc.)
    if custom_terms:
        for term in custom_terms:
            text = re.sub(
                re.escape(term),
                '[REDACTED]',
                text,
                flags=re.IGNORECASE
            )
    
    return text