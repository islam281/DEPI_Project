"""CLI for listing, adding, selecting, and querying books with NATEQ.

Examples:
  python run.py --list-books
  python run.py --book salah_al_din --question "أين ولد صلاح الدين؟"
  python run.py --add-book "C:/Books/al_raheeq.pdf" --book al_raheeq --title "الرحيق المختوم"
  python run.py --book al_raheeq --question "ما أبرز أحداث الهجرة؟"
  python run.py --set-default-book al_raheeq
"""
import argparse
from pathlib import Path

from src.application.book_manager import BookManager


def main() -> None:
    parser = argparse.ArgumentParser(description="NATEQ multilingual-book RAG")
    parser.add_argument("--data-dir", default="data", help="Portable book catalog folder")
    parser.add_argument("--list-books", action="store_true", help="List registered books")
    parser.add_argument("--add-book", metavar="PDF", help="Register a new PDF")
    parser.add_argument("--prepared-json", help="Optional page-level JSON to register")
    parser.add_argument("--title", help="Title for a new registered book")
    parser.add_argument("--book", help="Book ID to add or ask from")
    parser.add_argument("--set-default-book", metavar="ID", help="Set default book selection")
    parser.add_argument("--question", help="Arabic question to ask the selected book")
    args = parser.parse_args()
    manager = BookManager(data_dir=Path(args.data_dir))

    if args.list_books:
        for book in manager.list_books():
            marker = "(default)" if book["is_default"] else ""
            print(f"{book['book_id']}: {book['title']} {marker}")
        return
    if args.set_default_book:
        manager.set_default_book(args.set_default_book)
        print(f"Default book: {args.set_default_book}")
        return
    if args.add_book or args.prepared_json:
        if not args.book or not args.title:
            parser.error("Registering a book requires --book ID and --title TITLE")
        registered = manager.add_book(
            book_id=args.book,
            title=args.title,
            pdf_path=args.add_book,
            json_path=args.prepared_json,
        )
        print(f"Registered: {registered['book_id']} — {registered['title']}")
        print("The index will be built when the book is queried for the first time.")
        return
    if not args.question:
        parser.error("Provide --question, --list-books or --add-book")

    result = manager.ask(args.question, book_id=args.book)
    print(f"Book: {result['book_title']} ({result['book_id']})")
    print(result["answer_with_sources"])
    print("\nStatus:", result["status"])
    print("Citation audit:", result["citation_audit"]["source_integrity"])
    print("Timings:", result["timings"])


if __name__ == "__main__":
    main()
