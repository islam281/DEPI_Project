"""Conservative text cleaning that does not depend on book titles."""
import re
import unicodedata


def clean_text(text: str) -> str:
    """Normalize Unicode and whitespace without discarding meaningful headings."""
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"<!--\s*image\s*-->", "", text, flags=re.IGNORECASE)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
