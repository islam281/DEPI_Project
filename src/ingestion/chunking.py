"""Create contextual parent and child documents from complete source pages.

A parent and its children never span two original book pages; this
preserves the source page metadata needed for correct citations.
"""
import hashlib

SEPARATORS = ["\n\n", "\n", "؟", "؛", ".", "،", " ", ""]

def _page(doc):
    try:
        return int(doc.metadata.get("page_number", doc.metadata.get("page")))
    except (ValueError, TypeError):
        return None

def _source_pages_only(page_docs):
    """Normalize existing pages without editing the original objects."""
    pages = list(page_docs)
    good = [p for p in pages if p.page_content.strip() and _page(p) is not None]
    if not good:
        raise ValueError("docs must contain original page Documents with page_number")
    return good

def build_parent_children(page_docs, *, parent_size=2000, parent_overlap=200,
                          child_size=400, child_overlap=80, contextual=False):
    """Build parents page-by-page, never mixing pages/citations."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_core.documents import Document

    if parent_size <= parent_overlap or child_size <= child_overlap:
        raise ValueError("chunk_size must be greater than overlap")
    if parent_size <= child_size:
        raise ValueError("Parent size must exceed child size")
    parents = {}
    children = []

    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=parent_size, chunk_overlap=parent_overlap,
        separators=SEPARATORS, length_function=len,
    )

    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=child_size, chunk_overlap=child_overlap,
        separators=SEPARATORS, length_function=len,
    )

    for original in _source_pages_only(page_docs):
        pg = _page(original)
        document_id = str(original.metadata["document_id"])
        title = str(original.metadata["document_title"])
        for pi, body in enumerate(parent_splitter.split_text(original.page_content)):


            pid = hashlib.sha256(
                f"{document_id}|{pg}|{pi}|{body}".encode("utf8")
            ).hexdigest()[:22]
            meta = {**original.metadata, "parent_id": pid,
                    "parent_index": pi, "page_number": pg}
            parents[pid] = Document(page_content=body, metadata=meta)

            for ci, child_text in enumerate(child_splitter.split_text(body)):
                cid = f"{pid}_c{ci:03d}"
                text = child_text


                if contextual:
                    # Grounded, document-wide header only; NEVER add answer/gold terms.
                    text = f"المصدر: {title}\nرقم الصفحة: {pg}\n\n{child_text}"
                child_meta = {**meta, "child_id": cid, "child_index": ci,
                              "contextual_header": bool(contextual)}
                children.append(Document(page_content=text, metadata=child_meta))
    if not parents or not children:
        raise ValueError("Parent/child split yielded no documents")
    return parents, children
