"""End-to-end NATEQ RAG service.

The application layer orchestrates ingestion, indexing, retrieval,
Cross-Encoder ranking, focused-parent selection, generation, and
structural citation validation. It does not implement those steps itself.
"""
import hashlib
import json
import time
from pathlib import Path

from src.Config.settings import NateqSettings
from src.ingestion.chunking import build_parent_children
from src.retrival.embedding import make_embeddings
from src.retrival.retrival import ArabicBM25, _quota_union
from src.retrival.vector_database import open_or_create_child_chroma
from src.retrival.focused_parent import _to_parents, select_focused_parents
from src.retrival.reranker import rerank_with_cross_encoder
from src.generation.LLM import make_llm
from src.generation.prompt import get_rag_prompt
from src.application.citation_validation import validate_structural_citations
from src.application.source_display import build_cited_sources, format_answer_with_sources

class RAGService:
    """Answer questions using only the selected book and its Chroma index."""

    def __init__(
        self,
        source_docs: list,
        *,
        settings: NateqSettings | None = None,
        embedding_model=None,
        llm=None,
        cross_encoder=None,
    ) -> None:
        self.settings = settings or NateqSettings()
        self.source_docs = list(source_docs)
        if not self.source_docs:
            raise ValueError("source_docs must contain all original book pages")

        # 1. Build page-local parent/child chunks with stable IDs.
        self.parents, self.children = build_parent_children(
            self.source_docs,
            parent_size=self.settings.parent_size,
            parent_overlap=self.settings.parent_overlap,
            child_size=self.settings.child_size,
            child_overlap=self.settings.child_overlap,
            contextual=True,
        )

        # 2. Reopen a validated BGE-M3 Chroma index or create it once.
        self.embedding_model = embedding_model if embedding_model is not None else make_embeddings(self.settings)
        embedding_signature = {
            "model_name": str(getattr(self.embedding_model, "model_name", "unspecified")),
            "encode_kwargs": str(getattr(self.embedding_model, "encode_kwargs", {})),
            "model_kwargs": str(getattr(self.embedding_model, "model_kwargs", {})),
        }
        index_config = {
            "parent_size": self.settings.parent_size,
            "parent_overlap": self.settings.parent_overlap,
            "child_size": self.settings.child_size,
            "child_overlap": self.settings.child_overlap,
            "count_parents": len(self.parents),
            "embedding": embedding_signature,
            "contextual": True,
        }
        self.vector_store = open_or_create_child_chroma(
            self.children,
            self.embedding_model,
            index_dir=Path(self.settings.index_root) / "contextual_children",
            collection_name="nateq_phase6_contextual_children",
            config=index_config,
        )

        # 3. Keep lexical and dense retrieval on the same children.
        self.bm25 = ArabicBM25(self.children)
        self.dense_retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": self.settings.dense_k},
        )

        self.llm = llm if llm is not None else make_llm(self.settings)
        self.prompt = get_rag_prompt()
        self._cross_encoder = cross_encoder

    @property
    def cross_encoder(self):
        """Load the CPU Cross-Encoder only on the first retrieval request."""
        if self._cross_encoder is None:
            from sentence_transformers import CrossEncoder
            Path(self.settings.models_dir).mkdir(parents=True, exist_ok=True)
            self._cross_encoder = CrossEncoder(
                model_name_or_path=self.settings.reranker_name,
                device="cpu",
                max_length=384,
                cache_folder=self.settings.models_dir,
            )
        return self._cross_encoder

    def retrieval(self, question: str) -> list:
        """Select Dense top 20 and fill up to 25 with Arabic-aware BM25."""
        dense_docs = self.dense_retriever.invoke(question)[: self.settings.dense_k]
        bm25_docs = self.bm25.invoke(
            question,
            k=self.settings.dense_k + self.settings.arabic_quota,
        )
        return _quota_union(
            dense_docs,
            bm25_docs,
            dense_quota=self.settings.dense_k,
            total=self.settings.dense_k + self.settings.arabic_quota,
        )

    def rerank(self, question: str, candidates: list) -> list:
        """Apply the original Cross-Encoder + 35/65 fusion ranking."""
        if not candidates:
            return []
        return rerank_with_cross_encoder(
            question,
            candidates,
            self.cross_encoder,
            top_k=len(candidates),
            retrieval_weight=self.settings.retrieval_weight,
            reranker_weight=self.settings.ce_weight,
        )

    def focused_parent_retrieval(self, question: str) -> tuple[list, dict]:
        """Retrieve children, select unique parents, then shorten each parent."""
        t0 = time.perf_counter()
        candidates = self.retrieval(question)
        retrieval_seconds = time.perf_counter() - t0

        t1 = time.perf_counter()
        ranked_children = self.rerank(question, candidates)
        rerank_seconds = time.perf_counter() - t1

        full_parents = _to_parents(
            ranked_children,
            self.parents,
            max_parents=self.settings.final_k,
            max_context_chars=self.settings.max_parent_context_chars,
        )
        focused_parents, windows = select_focused_parents(
            ranked_children,
            full_parents,
            focus_budget_chars=self.settings.focus_budget_chars,
            min_chars=self.settings.focus_min_per_parent,
            before_ratio=self.settings.before_ratio,
        )
        diagnostics = {
            "candidate_count": len(candidates),
            "selected_parents": len(full_parents),
            "context_chars": sum(len(doc.page_content) for doc in focused_parents),
            "retrieval_seconds": round(retrieval_seconds, 4),
            "rerank_seconds": round(rerank_seconds, 4),
            "focused_windows": windows,
        }
        return focused_parents, diagnostics

    @staticmethod
    def format_context(documents: list) -> str:
        """Give the LLM only numbered source passages and their true pages."""
        evidence_blocks = []
        for number, doc in enumerate(documents, start=1):
            metadata = doc.metadata
            page = metadata.get("page_number", metadata.get("page", "unknown"))
            evidence_blocks.append(
                "<evidence>\n"
                f"reference: E{number}\n"
                f"source: {metadata.get('source', metadata.get('document_title', 'unknown'))}\n"
                f"page: {page}\n"
                f"chunk_id: {metadata.get('focus_child_id', metadata.get('parent_id', 'unknown'))}\n"
                "content:\n"
                f"{doc.page_content}\n"
                "</evidence>"
            )
        return "\n\n".join(evidence_blocks)

    def generate(self, question: str, documents: list) -> dict:
        """Generate a strictly source-grounded JSON response with LangChain."""
        from langchain_core.output_parsers import StrOutputParser

        chain = self.prompt | self.llm | StrOutputParser()
        raw = chain.invoke({
            "context": self.format_context(documents),
            "question": question,
        })
        try:
            output = json.loads(raw)
        except (TypeError, json.JSONDecodeError):
            return {
                "status": "insufficient_evidence",
                "answer": "تعذر تجهيز إجابة موثوقة من المصادر المتاحة.",
                "citations": [],
            }
        if not isinstance(output, dict):
            return {
                "status": "insufficient_evidence",
                "answer": "صيغة الإجابة غير صالحة.",
                "citations": [],
            }
        return output

    def ask(self, question: str, *, question_id: str | None = None) -> dict:
        """Answer one question and validate citations before returning it."""
        if not isinstance(question, str) or not question.strip():
            raise ValueError("question must be a nonempty string")
        question = question.strip()
        started = time.perf_counter()

        focused_docs, diagnostics = self.focused_parent_retrieval(question)
        sources = [
            {
                "id": number,
                "reference": f"E{number}",
                "content": doc.page_content,
                "page_number": doc.metadata.get("page_number", doc.metadata.get("page")),
                "metadata": {
                    **doc.metadata,
                    "page": doc.metadata.get("page_number", doc.metadata.get("page")),
                },
            }
            for number, doc in enumerate(focused_docs, start=1)
        ]

        t0 = time.perf_counter()
        if focused_docs:
            generated = self.generate(question, focused_docs)
        else:
            generated = {
                "status": "insufficient_evidence",
                "answer": "لا تتوفر معلومات كافية في المصادر.",
                "citations": [],
            }
        generation_seconds = time.perf_counter() - t0

        # Preserve the original JSON contract, but reject malformed answers.
        status = generated.get("status")
        answer = generated.get("answer")
        cited_ids = generated.get("citations")
        if (
            status not in {"answered", "insufficient_evidence"}
            or not isinstance(answer, str)
            or not isinstance(cited_ids, list)
            or any(not isinstance(ref, str) for ref in cited_ids)
        ):
            status = "insufficient_evidence"
            answer = "تعذر التحقق من صيغة الإجابة أو مراجعها."
            cited_ids = []

        if status == "insufficient_evidence":
            cited_ids = []

        source_metadata = (
            focused_docs[0].metadata if focused_docs
            else (getattr(self, "source_docs", [None])[0].metadata
                  if getattr(self, "source_docs", None) else {})
        )
        result = {
            "id": question_id or hashlib.sha256(question.encode("utf-8")).hexdigest()[:12],
            "question": question,
            "answer": answer,
            "status": status,
            "citations": cited_ids,
            "sources": sources,
            "book_id": source_metadata.get("document_id"),
            "book_title": source_metadata.get("document_title"),
        }
        audit = validate_structural_citations(result)
        if not audit["valid"]:
            # Never expose a response with invalid or fabricated references.
            result.update({
                "answer": "تعذر عرض الإجابة بسبب مشكلة في مراجع المصادر.",
                "status": "insufficient_evidence",
                "citations": [],
            })

        # Resolve citations only AFTER the structural validation has passed.
        # Non-cited retrieved passages remain in `sources` for diagnostics,
        # but must not appear as evidence for the student-facing answer.
        result["cited_sources"] = build_cited_sources(
            result["citations"], result["sources"]
        )
        result["answer_with_sources"] = format_answer_with_sources(
            result["answer"], result["cited_sources"]
        )

        result["citation_audit"] = audit
        result["timings"] = {
            "retrieval_seconds": diagnostics["retrieval_seconds"],
            "rerank_seconds": diagnostics["rerank_seconds"],
            "generation_seconds": round(generation_seconds, 4),
            "total_seconds": round(time.perf_counter() - started, 4),
            "context_chars": diagnostics["context_chars"],
        }
        return result
