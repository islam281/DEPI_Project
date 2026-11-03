"""Offline tests for switching books, source identity and isolated indexes."""
from dataclasses import dataclass
import json
from pathlib import Path

import pytest

from src.Config.settings import NateqSettings
from src.ingestion.book_registry import BookRegistry
from src.application.book_manager import BookManager


@pytest.fixture
def catalog(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    (data / "books.json").write_text(json.dumps({"default_book": "alpha", "books": {
        "alpha": {"title": "كتاب أول", "json": "books/alpha/extracted.json", "pdf": None}
    }}, ensure_ascii=False), encoding="utf8")
    (data / "books/alpha").mkdir(parents=True)
    (data / "books/alpha/extracted.json").write_text(json.dumps({"pages": [{
        "page_content": "Old text", "metadata": {"page_number": 1, "quality_status": "ok"}
    }]}), encoding="utf8")
    return data


def test_catalog_add_list_default_and_duplicate(catalog, tmp_path):
    registry = BookRegistry(catalog)
    sample_json = tmp_path / "new.json"
    sample_json.write_text('{"pages": []}', encoding="utf8")
    second = registry.register_book(book_id="beta", title="كتاب ثان", json_path=sample_json)
    assert second["title"] == "كتاب ثان"
    assert len(registry.list_books()) == 2
    assert registry.default_book_id() == "alpha"
    registry.set_default_book("beta")
    assert BookRegistry(catalog).default_book_id() == "beta"
    with pytest.raises(ValueError, match="already exists"):
        registry.register_book(book_id="beta", title="Duplicate", json_path=sample_json)
    with pytest.raises(ValueError):
        registry.register_book(book_id="../escape", title="Unsafe", json_path=sample_json)


def test_index_paths_are_book_and_content_specific(catalog, tmp_path):
    s = NateqSettings(index_root=str(tmp_path / "indexes"))
    manager = BookManager(data_dir=catalog, settings=s)
    file_a = catalog / "books/alpha/extracted.json"
    file_b = tmp_path / "second.json"
    file_b.write_text(file_a.read_text(encoding="utf8"), encoding="utf8")
    index_a = manager._versioned_index_root("alpha", file_a)
    index_b = manager._versioned_index_root("beta", file_b)
    assert index_a != index_b
    file_b.write_text('{"pages": []}', encoding="utf8")
    assert manager._versioned_index_root("beta", file_b) != index_b


def test_multiple_book_response_never_crosses_sources(catalog, tmp_path, monkeypatch):
    data_b = tmp_path / "beta.json"
    data_b.write_text(json.dumps({"pages": [{"page_content": "New text", "metadata": {
        "page_number": 1, "quality_status": "ok"}}]}), encoding="utf8")
    registry = BookRegistry(catalog)
    registry.register_book(book_id="beta", title="كتاب ثان", json_path=data_b)

    # Avoid downloading BGE-M3, Chroma, or Groq in a unit test.
    monkeypatch.setattr("src.ingestion.book_registry.load_source_docs_from_json", lambda *_args, book_id, title, **_kwargs: [
        type("Page", (), {"page_content": "text", "metadata": {
            "document_id": book_id, "document_title": title, "page_number": 1}})()
    ])
    monkeypatch.setattr("src.retrival.embedding.make_embeddings", lambda *_args: object())
    monkeypatch.setattr("src.generation.LLM.make_llm", lambda *_args: object())

    class FakeRag:
        def __init__(self, docs, **kwargs):
            self.doc = docs[0]
            self.index_root = kwargs["settings"].index_root
            self._cross_encoder = None
        def ask(self, question):
            return {"answer": "ok", "cited_sources": [{
                "document_id": self.doc.metadata["document_id"],
                "title": self.doc.metadata["document_title"]
            }]}

    manager = BookManager(data_dir=catalog, settings=NateqSettings(index_root=str(tmp_path / "indexes")), service_factory=FakeRag)
    first = manager.ask("Test", book_id="alpha")
    second = manager.ask("Test", book_id="beta")
    assert first["book_id"] == "alpha"
    assert second["book_id"] == "beta"
    assert first["cited_sources"][0]["document_id"] == "alpha"
    assert second["cited_sources"][0]["document_id"] == "beta"
    assert manager.get_rag("alpha").index_root != manager.get_rag("beta").index_root
    assert manager.get_rag("alpha") is manager.get_rag("alpha")


def test_arabic_pdf_geometry_extracts_logical_sentence(tmp_path):
    """Regression test for presentation-form Arabic glyphs in the bundled book."""
    from src.ingestion.pdf_loader import _arabic_geometry_text, extract_pdf_to_json
    import pdfplumber

    pdf_path = Path(__file__).resolve().parents[1] / "data/books/salah_al_din/book.pdf"
    with pdfplumber.open(pdf_path) as pdf:
        page = pdf.pages[14]
        native = page.extract_text() or ""
        # The raw PDF has an unusual font with Arabic presentation-form glyphs.
        assert "وطلبها منه" not in native
        assert "وفتح آمد وطلبها منه" in _arabic_geometry_text(page)


def test_metadata_book_id_and_title_are_not_hardcoded(tmp_path, monkeypatch):
    """Override stale source metadata from a prepared historical JSON."""
    import sys
    from types import ModuleType
    from src.ingestion.pdf_loader import load_source_docs_from_json

    fake_root = ModuleType("langchain_core")
    fake_documents = ModuleType("langchain_core.documents")
    class Doc:
        def __init__(self, page_content, metadata):
            self.page_content, self.metadata = page_content, metadata
    fake_documents.Document = Doc
    monkeypatch.setitem(sys.modules, "langchain_core", fake_root)
    monkeypatch.setitem(sys.modules, "langchain_core.documents", fake_documents)

    prepared = tmp_path / "old.json"
    prepared.write_text(json.dumps({"pages": [{
        "page_content": "العلم مفيد", "metadata": {
            "document_id": "old", "document_title": "Old Book", "page_number": 7,
            "source": "old.pdf", "quality_status": "ok"
        }
    }]}), encoding="utf8")
    docs = load_source_docs_from_json(prepared, book_id="new_book", title="كتاب آخر", source_label="books/new_book/book.pdf")
    assert docs[0].metadata["document_id"] == "new_book"
    assert docs[0].metadata["document_title"] == "كتاب آخر"
    assert docs[0].metadata["source"] == "books/new_book/book.pdf"
