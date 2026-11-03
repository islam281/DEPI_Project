"""Prepare cited RAG passages for the command line, notebooks, and frontend.

The LLM returns reference identifiers (for example, "E1") but must not
invent page numbers or document names. This module resolves the references
against trusted retrieval metadata after structural citation validation.
"""
from __future__ import annotations

from pathlib import PurePosixPath, PureWindowsPath


def _safe_source_filename(source_path: object) -> str:
    """Return only the filename, never a local machine's full filesystem path."""
    raw_path = str(source_path or "").strip()
    if not raw_path:
        return ""
    if "\\" in raw_path:
        return PureWindowsPath(raw_path).name
    return PurePosixPath(raw_path).name


def build_cited_sources(
    citations: list[str],
    sources: list[dict],
    *,
    preview_chars: int = 500,
) -> list[dict]:
    """Resolve only cited references into user-facing source objects.

    Each returned object provides:
      - reference: the exact E-number from the model's citations array
      - title: the document title from the retrieved page's metadata
      - page_number: the page number from the original source document
      - content: the full *focused* passage supplied to the model
      - excerpt: a shorter preview for compact user interfaces
      - file_name: only the source PDF filename (no private local paths)

    The original `sources` array remains available separately to callers.
    Invalid/unknown references are never added to this list.
    """
    if preview_chars <= 0:
        raise ValueError("preview_chars must be positive")

    sources_by_reference = {
        source.get("reference"): source
        for source in sources
        if source.get("reference")
    }
    displayed_sources = []
    seen = set()

    for citation in citations:
        if citation in seen:
            continue
        seen.add(citation)

        source = sources_by_reference.get(citation)
        if source is None:
            continue

        metadata = source.get("metadata") or {}
        text = str(source.get("content") or "").strip()
        page = source.get("page_number", metadata.get("page_number", metadata.get("page")))
        title = str(
            metadata.get("document_title")
            or metadata.get("title")
            or metadata.get("document_id")
            or "مصدر غير معنون"
        )
        excerpt = text[:preview_chars]
        if len(text) > preview_chars:
            excerpt = excerpt.rstrip() + "…"

        displayed_sources.append({
            "reference": citation,
            "title": title,
            "page_number": page,
            "file_name": _safe_source_filename(metadata.get("source")),
            "document_id": metadata.get("document_id"),
            "book_id": metadata.get("document_id"),
            "excerpt": excerpt,
            "content": text,
        })

    return displayed_sources


def format_answer_with_sources(answer: str, cited_sources: list[dict]) -> str:
    """Format an Arabic answer and its *actual cited* passages for display.

    This is presentation only. It neither verifies semantic support nor
    edits the model's answer or the underlying cited source passages.
    """
    answer = str(answer or "").strip()
    if not cited_sources:
        return answer

    lines = [answer, "", "المصادر:"]
    for source in cited_sources:
        page_label = source.get("page_number")
        page_text = f"صفحة {page_label}" if page_label is not None else "صفحة غير معروفة"
        lines.append(
            f"[{source['reference']}] {source['title']} — {page_text}"
        )
        lines.append(f"النص المستشهد به: {source['content']}")
        lines.append("")

    return "\n".join(lines).rstrip()
