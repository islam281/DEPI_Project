"""Switch between independent book-specific RAG services without mixing evidence."""
from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

from src.Config.settings import NateqSettings
from src.application.rag_service_provider import RAGService
from src.ingestion.book_registry import BookRegistry


class BookManager:
    """Keep one retriever and one Chroma index per selected book.

    The embedding model, Cross-Encoder, and Groq client may be shared in memory.
    Chroma, BM25, parents, children, and citations are book-isolated.
    """

    def __init__(
        self,
        *,
        data_dir: str | Path = "data",
        settings: NateqSettings | None = None,
        service_factory=None,
    ) -> None:
        self.registry = BookRegistry(data_dir)
        self.settings = settings or NateqSettings()
        self._services = {}
        self._embedding = None
        self._cross_encoder = None
        self._llm = None
        self._factory = service_factory or RAGService

    def list_books(self) -> list[dict]:
        return self.registry.list_books()

    def add_book(self, **kwargs) -> dict:
        return self.registry.register_book(**kwargs)

    def set_default_book(self, book_id: str) -> None:
        self.registry.set_default_book(book_id)

    def _versioned_index_root(self, book_id: str, prepared_json: Path) -> Path:
        """Build a unique path for this book and its current indexing settings."""
        fingerprint = {
            "text_sha256": hashlib.sha256(prepared_json.read_bytes()).hexdigest(),
            "parent_size": self.settings.parent_size,
            "parent_overlap": self.settings.parent_overlap,
            "child_size": self.settings.child_size,
            "child_overlap": self.settings.child_overlap,
            "embedding": self.settings.embedding_name,
        }
        version = hashlib.sha256(json.dumps(fingerprint, sort_keys=True).encode()).hexdigest()[:14]
        return Path(self.settings.index_root).resolve() / book_id / version

    def get_rag(self, book_id: str | None = None):
        """Load or reuse a per-book service. Switching never reuses another book's evidence."""
        key = book_id or self.registry.default_book_id()
        if key in self._services:
            return self._services[key]

        book, docs, prepared = self.registry.load_documents(key)
        book_settings = replace(
            self.settings,
            index_root=str(self._versioned_index_root(key, prepared)),
        )
        # Keep already downloaded and initialized models shared across books.
        from src.retrival.embedding import make_embeddings
        from src.generation.LLM import make_llm

        if self._embedding is None:
            self._embedding = make_embeddings(book_settings)
        if self._llm is None:
            self._llm = make_llm(book_settings)

        service = self._factory(
            docs,
            settings=book_settings,
            embedding_model=self._embedding,
            llm=self._llm,
            cross_encoder=self._cross_encoder,
        )
        self._services[key] = service
        return service

    def ask(self, question: str, *, book_id: str | None = None) -> dict:
        """Answer from the requested book, not from a merged multi-book index."""
        selected_book = book_id or self.registry.default_book_id()
        rag = self.get_rag(selected_book)
        result = rag.ask(question)
        if getattr(rag, "_cross_encoder", None) is not None:
            self._cross_encoder = rag._cross_encoder
        result["book_id"] = selected_book
        result["book_title"] = self.registry.get_book(selected_book)["title"]
        return result
