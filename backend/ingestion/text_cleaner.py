
import re


def clean_text(text: str) -> str:
    """Clean extracted PDF text while preserving paragraph boundaries."""
    if not text or not text.strip():
        return ""

    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Normalize whitespace on each line.
    lines = [line.strip() for line in text.split("\n")]
    text = "\n".join(lines)

    # Collapse excessive blank lines.
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Normalize spaces and tabs within lines.
    text = re.sub(r"[ \t]+", " ", text)

    return text.strip()
