"""Book catalog for registering, listing, and loading independent sources."""
from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

from src.ingestion.pdf_loader import extract_pdf_to_json, load_source_docs_from_json

BOOK_ID = re.compile(r"^[a-z][a-z0-9_-]{1,63}$")


class BookRegistry:
    """Manage a portable data/books.json registry with per-book files."""

    def __init__(self, data_dir: str | Path = "data") -> None:
        self.data_dir = Path(data_dir).resolve()
        self.registry_file = self.data_dir / "books.json"
        if not self.registry_file.exists():
            raise FileNotFoundError(f"Missing book catalog: {self.registry_file}")
        self._read()  # Validate on construction.

    def _read(self) -> dict:
        data = json.loads(self.registry_file.read_text(encoding="utf8"))
        if not isinstance(data.get("books"), dict):
            raise ValueError("books.json must contain a books dictionary")
        return data

    def _write(self, data: dict) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        temp = self.registry_file.with_suffix(".json.tmp")
        temp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf8")
        temp.replace(self.registry_file)

    def list_books(self) -> list[dict]:
        """Return available books, including the configured default selection."""
        data = self._read()
        return [{"book_id": key, "title": entry["title"],
                 "is_default": key == data.get("default_book")}
                for key, entry in sorted(data["books"].items())]

    def default_book_id(self) -> str:
        data = self._read()
        key = data.get("default_book")
        if key not in data["books"]:
            raise ValueError("Default book is not registered")
        return key

    def get_book(self, book_id: str | None = None) -> dict:
        data = self._read()
        book_id = book_id or self.default_book_id()
        entry = data["books"].get(book_id)
        if entry is None:
            raise KeyError(f"Unknown book '{book_id}'. Run --list-books first.")
        return {"book_id": book_id, **entry}

    def set_default_book(self, book_id: str) -> None:
        data = self._read()
        if book_id not in data["books"]:
            raise KeyError(f"Unknown book: {book_id}")
        data["default_book"] = book_id
        self._write(data)

    def register_book(
        self, *, book_id: str, title: str,
        pdf_path: str | Path | None = None,
        json_path: str | Path | None = None,
    ) -> dict:
        """Copy source files into a portable isolated book folder.

        At least one of PDF and pre-extracted JSON is required. Explicit book
        IDs prevent collisions, and existing books are never overwritten.
        """
        if not BOOK_ID.fullmatch(book_id):
            raise ValueError("book_id must use lowercase ASCII letters, digits, - or _")
        if not title.strip():
            raise ValueError("A nonempty book title is required")
        if pdf_path is None and json_path is None:
            raise ValueError("Provide --pdf or --json when registering a book")
        data = self._read()
        if book_id in data["books"]:
            raise ValueError(f"Book '{book_id}' already exists. Use another ID.")

        dest = self.data_dir / "books" / book_id
        if dest.exists():
            raise ValueError(f"Book folder already exists: {dest}")
        pdf = Path(pdf_path).resolve() if pdf_path else None
        prepared = Path(json_path).resolve() if json_path else None
        for source in (pdf, prepared):
            if source and not source.is_file():
                raise FileNotFoundError(source)
        dest.mkdir(parents=True)
        try:
            if pdf:
                shutil.copy2(pdf, dest / "book.pdf")
            if prepared:
                shutil.copy2(prepared, dest / "extracted.json")
            entry = {
                "title": title.strip(),
                "pdf": f"books/{book_id}/book.pdf" if pdf else None,
                "json": f"books/{book_id}/extracted.json",
                "text_mode": "provided_json" if prepared else "auto_pdf",
            }
            data["books"][book_id] = entry
            self._write(data)
        except Exception:
            shutil.rmtree(dest, ignore_errors=True)
            raise
        return self.get_book(book_id)

    def load_documents(self, book_id: str | None = None) -> tuple[dict, list, Path]:
        """Read the selected book; automatically prepare uncached PDFs once."""
        book = self.get_book(book_id)
        prepared = self.data_dir / book["json"]
        # For automatically extracted PDFs, always check the source hash. The
        # extractor returns the cached JSON immediately when nothing changed.
        if book.get("text_mode") == "auto_pdf":
            extract_pdf_to_json(
                self.data_dir / book["pdf"], prepared,
                book_id=book["book_id"], title=book["title"],
                source_label=book["pdf"],
            )
        elif not prepared.exists():
            raise FileNotFoundError(f"Missing prepared JSON: {prepared}")
        # User-prepared JSON remains authoritative: never overwrite it.
        source_label = book.get("pdf") or book["json"]
        documents = load_source_docs_from_json(
            prepared, book_id=book["book_id"], title=book["title"],
            source_label=source_label,
        )
        return book, documents, prepared
