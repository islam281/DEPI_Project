"""Load page-level JSON or prepare it from a searchable PDF.

The existing Salah al-Din geometry JSON is used as-is. New PDFs are extracted
once and cached. Image-only PDFs require OCR/manual preparation; we never
invent text from pages that have no valid text layer.
"""
from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from pathlib import Path

from src.ingestion.text_preprocessing import clean_text

EXTRACTOR_VERSION = "nateq-arabic-geometry-v1"
ARABIC_RE = re.compile(r"[\u0621-\u064a\u064b-\u065f\u0670\u0671-\u06d3]+")


def _arabic_geometry_text(page) -> str:
    """Reconstruct an RTL text layer from positioned PDF characters.

    This follows the geometry approach used for the original historical PDF.
    It is intended for single-column Arabic text PDFs, not scanned images.
    """
    from pdfplumber.utils import extract_text

    chars = page.chars
    if not chars:
        return ""

    def is_mark(char: dict) -> bool:
        value = char.get("text", "")
        return bool(value) and all(
            unicodedata.category(letter).startswith("M") for letter in value
        )

    base_chars = [ch for ch in chars if not is_mark(ch) and not ch["text"].isspace()]
    positioned_chars = []

    for char in chars:
        if is_mark(char):
            matches = [
                base for base in base_chars
                if base["x0"] - 0.5 <= char["x0"] <= base["x1"] + 0.5
                and abs(base["top"] - char["top"]) < base["size"] * 0.65
            ]
            if matches:
                nearest = min(matches, key=lambda base: abs(base["top"] - char["top"]))
                char = {**char, **{k: nearest[k] for k in ("x0", "x1", "top", "bottom", "doctop")}}
                char["x0"] -= 0.001
        positioned_chars.append(char)

    text = extract_text(
        positioned_chars,
        x_tolerance_ratio=0.08,
        char_dir="rtl",
        char_dir_render="ltr",
    ) or ""
    # Correct logical direction of numeric strings in the RTL extraction.
    text = re.sub(
        r"[0-9٠-٩۰-۹]+(?:[.,٫٬][0-9٠-٩۰-۹]+)*",
        lambda match: match.group(0)[::-1],
        text,
    )
    text = text.translate(str.maketrans("()[]{}«»", ")(][}{»«"))
    return clean_text(text)


def _quality_status(text: str) -> str:
    """Flag suspicious extraction instead of silently indexing corrupted text."""
    if sum(ch.isalnum() for ch in text) < 20:
        return "low_text"
    runs = ARABIC_RE.findall(text)
    letters = sum(sum(c.isalpha() for c in run) for run in runs)
    long_runs = sum(sum(c.isalpha() for c in run) >= 25 for run in runs)
    merged_words = letters >= 120 and long_runs >= 3 and long_runs / max(len(runs), 1) >= 0.08
    if "\ufffd" in text or "(cid:" in text or merged_words:
        return "review"
    return "ok"


def extract_pdf_to_json(
    pdf_path: str | Path,
    output_json: str | Path,
    *,
    book_id: str,
    title: str,
    source_label: str | None = None,
) -> Path:
    """Create a cached, page-level JSON with original physical PDF page numbers.

    The cache uses PDF content hash, preventing accidental use of stale text.
    Pages flagged as suspicious remain in the JSON for manual inspection.
    """
    import pdfplumber

    pdf_path = Path(pdf_path).resolve()
    output_json = Path(output_json)
    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    digest = hashlib.sha256(pdf_path.read_bytes()).hexdigest()

    if output_json.exists():
        cached = json.loads(output_json.read_text(encoding="utf-8"))
        cfg = cached.get("config", {}) if isinstance(cached, dict) else {}
        if (cfg.get("pdf_sha256") == digest and cfg.get("extraction_method") == EXTRACTOR_VERSION
                and cfg.get("document_id") == book_id and cfg.get("document_title") == title):
            return output_json

    pages = []
    readable_count = 0
    review_count = 0
    with pdfplumber.open(pdf_path) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            # Use positioned Arabic glyphs for Arabic; native text for other scripts.
            native = page.extract_text() or ""
            normalized_native = unicodedata.normalize("NFKC", native)
            arabic_share = sum("\u0600" <= c <= "\u06ff" for c in normalized_native) / max(len(normalized_native), 1)
            text = _arabic_geometry_text(page) if arabic_share > 0.25 else clean_text(native)
            status = _quality_status(text)
            if status == "ok":
                readable_count += 1
            elif status == "review":
                review_count += 1
            pages.append({
                "page_content": text,
                "metadata": {
                    "source": source_label or str(pdf_path),
                    "document_id": book_id,
                    "document_title": title,
                    "page_number": page_number,
                    "extraction_method": EXTRACTOR_VERSION,
                    "quality_status": status,
                },
            })

    if readable_count == 0:
        raise ValueError(
            "No reliably readable pages were extracted. The PDF may be scanned "
            "or need OCR/manual review. Please provide a verified extracted JSON."
        )
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps({
        "config": {
            "pdf_sha256": digest,
            "document_id": book_id,
            "document_title": title,
            "extraction_method": EXTRACTOR_VERSION,
        },
        "total_pages": len(pages),
        "review_pages": [p["metadata"]["page_number"] for p in pages
                         if p["metadata"]["quality_status"] == "review"],
        "pages": pages,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Extracted {readable_count}/{len(pages)} readable pages from {title}; "
          f"review flagged pages: {review_count}")
    return output_json


def load_source_docs_from_json(
    json_path: str | Path,
    *,
    book_id: str | None = None,
    title: str | None = None,
    source_label: str | None = None,
) -> list:
    """Load original book pages, assigning trusted book identity metadata.

    A prepared NATEQ geometry JSON or a page-list JSON is accepted. Historical
    numbering remains physical PDF page numbering, as stored in the JSON.
    """
    from langchain_core.documents import Document

    payload = json.loads(Path(json_path).read_text(encoding="utf-8"))
    rows = payload.get("pages", []) if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise ValueError("Expected a JSON list of pages or {'pages': [...]} object")

    documents = []
    for row in rows:
        metadata = dict(row.get("metadata") or {})
        if metadata.get("quality_status", "ok") != "ok":
            continue
        text = clean_text(str(row.get("page_content") or ""))
        if not text:
            continue
        page = metadata.get("page_number")
        if page is None:
            raise ValueError("Each source page must include page_number metadata")
        # The selected registry entry is the authority for book identity.
        if book_id is not None:
            metadata["document_id"] = book_id
        if title is not None:
            metadata["document_title"] = title
        if source_label is not None:
            metadata["source"] = source_label
        if not metadata.get("document_id") or not metadata.get("document_title"):
            raise ValueError("Book metadata must include document_id and document_title")
        metadata["page_number"] = int(page)
        documents.append(Document(page_content=text, metadata=metadata))

    if not documents:
        raise ValueError("No valid text pages found. Check PDF extraction quality.")
    return documents
