"""Cross-Encoder reranking combined with retrieval ranks.

Score: 0.35/(60 + retrieval_rank) + 0.65/(60 + cross_encoder_rank).
The caller supplies the weights from NateqSettings.
"""
import re
import unicodedata

def rerank_with_cross_encoder(
    question: str,
    docs: list,
    cross_encoder,
    *,
    top_k: int,
    retrieval_weight: float,
    reranker_weight: float,
) -> list:
    """Rerank retrieved Children using sentence-transformers CrossEncoder.

    Retrieval rank is the position in the Dense20+BM25 quota union.
    The score formula is: rw/(60 + retrieval_rank) + cw/(60 + ce_rank).
    This intentionally matches the Phase 8 recipe; it is not a multi-list RRF.
    """
    from langchain_core.documents import Document
    if top_k < 1:
        raise ValueError("top_k must be >= 1")

    candidates = []
    seen_text = set()
    for original_position, doc in enumerate(docs, start=1):
        text = re.sub(r"\s+", " ", unicodedata.normalize("NFKC", doc.page_content)).strip()
        # Keep the original notebook's minimum number of Arabic words.
        if len(re.findall(r"[\u0621-\u064A]{2,}", text)) < 15:
            continue
        if text in seen_text:
            continue
        seen_text.add(text)
        candidates.append((doc, text, original_position))

    if not candidates:
        return []

    scores = cross_encoder.predict(
        [(question, text) for _, text, _ in candidates],
        batch_size=2,
        show_progress_bar=False,
    )
    # CE rank 1 is the highest model score.
    indices_by_ce = sorted(range(len(candidates)), key=lambda i: float(scores[i]), reverse=True)
    ce_rank = {position: rank for rank, position in enumerate(indices_by_ce, start=1)}

    reranked = []
    for i, (doc, _text, retrieval_rank) in enumerate(candidates):
        score = (
            retrieval_weight / (60 + retrieval_rank)
            + reranker_weight / (60 + ce_rank[i])
        )
        reranked.append(Document(
            page_content=doc.page_content,
            metadata={
                **doc.metadata,
                "retrieval_rank": retrieval_rank,
                "reranker_rank": ce_rank[i],
                "rerank_score": float(scores[i]),
                "fusion_score": float(score),
            },
        ))
    reranked.sort(key=lambda doc: doc.metadata["fusion_score"], reverse=True)
    return reranked[:top_k]
