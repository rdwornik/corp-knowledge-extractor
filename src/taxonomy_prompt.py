"""Generate canonical taxonomy terms for injection into extraction prompts."""

from corp_os_meta.normalize import load_taxonomy


def get_taxonomy_for_prompt() -> str:
    """Generate a formatted taxonomy list for inclusion in extraction prompts.

    Returns a compact string listing canonical topic and product names
    that the LLM should use in its output.
    """
    taxonomy = load_taxonomy()

    topics = [entry["name"] for entry in taxonomy.get("topics", [])]
    products = [entry["name"] for entry in taxonomy.get("products", [])]

    lines = []
    lines.append("CANONICAL TOPICS (use these exact names when applicable):")
    lines.append(", ".join(topics))
    lines.append("")
    lines.append("CANONICAL PRODUCTS (use these exact names when applicable):")
    lines.append(", ".join(products))
    lines.append("")
    lines.append(
        "If a concept does not match any canonical term, use your best short "
        "descriptive name. Do NOT invent variations of existing terms."
    )

    return "\n".join(lines)
