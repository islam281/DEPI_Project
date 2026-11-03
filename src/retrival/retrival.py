"""Dense + Arabic BM25 candidate retrieval.

The folder name "retrival" intentionally matches the user's original
src archive. Both retrievers operate on the SAME contextual children.
"""
import hashlib
import re
import unicodedata

MARKS = re.compile(r"[\u064b-\u065f\u0670\u06d6-\u06ed\u0640]")
ARABIC_TOKEN = re.compile(r"[\u0621-\u064a\u0660-\u0669\u06f0-\u06f9]+|[0-9]+")
SPACES = re.compile(r"\s+")

def normalize_ar(text):
    s = unicodedata.normalize("NFKC", str(text or ""))
    s = MARKS.sub("", s)
    for a, b in (("أ", "ا"), ("إ", "ا"), ("آ", "ا"), ("ٱ", "ا"), ("ى", "ي")):
        s = s.replace(a, b)
    return s

def tokenize_ar(text):
    return ARABIC_TOKEN.findall(normalize_ar(text).lower())

class ArabicBM25:
    """Matches earlier NATEQ ArabicBM25 normalization and tie-breaking."""
    
    
    def __init__(self, chunks):
        from rank_bm25 import BM25Okapi
        self.docs = list(chunks)
        self.index = BM25Okapi([tokenize_ar(d.page_content) for d in self.docs])

    
    
    def invoke(self, question, k=20):
        tokens = tokenize_ar(question)
        if not tokens:
            return []
        scores = self.index.get_scores(tokens)
        ranks = sorted(range(len(self.docs)), key=lambda i: (-float(scores[i]), i))
        return [self.docs[i] for i in ranks[:k]]

def _norm_space(text):
    return SPACES.sub(" ", unicodedata.normalize("NFKC", str(text or ""))).strip()

def _doc_key(doc):
    # Deduplicate within the SAME retrieval stream; prefer stable child IDs.
    cid = doc.metadata.get("child_id")
    if cid:
        return str(cid)
    pg = doc.metadata.get("page_number", doc.metadata.get("page", ""))
    return hashlib.sha256(f"{pg}|{_norm_space(doc.page_content)}".encode("utf8")).hexdigest()[:20]

def _quota_union(dense_docs, arabic_docs, dense_quota=20, total=25):
    """Use dense results first, then fill remaining slots with unique BM25 results."""
    output, seen = [], set()
    
    def add(items, limit):
        for doc in items:
            if len(output) >= limit:
                return
            key = _doc_key(doc)
            if key not in seen:
                output.append(doc)
                seen.add(key)
    add(dense_docs, dense_quota)
    add(arabic_docs, total)
    add(dense_docs, total)
    add(arabic_docs, total)
    return output[:total]
