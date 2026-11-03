# Book data

- `books.json` lists the registered books and the default book ID.
- `books/<book_id>/book.pdf` contains the source PDF when provided.
- `books/<book_id>/extracted.json` contains prepared page-level text. This is
  used directly when supplied. For a newly registered PDF, it is created on
  first use using PDF character geometry and cached for later starts.
- `page_number` is the original physical PDF page number, not necessarily the
  number printed in the book footer.
- If a PDF is scanned or its text layer is corrupted, supply a verified JSON
  extraction or run OCR outside the RAG service. The loader will not invent text.

Do not mix content from different books into a single vector index.
