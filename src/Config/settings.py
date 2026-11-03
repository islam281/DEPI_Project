"""Runtime settings shared by all books.

Retrieval values are unchanged from the NATEQ Phase 8 baseline. Book selection,
source paths, and Chroma locations belong to BookManager, not to this class.
"""
from dataclasses import dataclass


@dataclass
class NateqSettings:
    """Configuration of retrieval, ranking, context and generation."""
    parent_size: int = 2000
    parent_overlap: int = 200
    child_size: int = 400
    child_overlap: int = 80
    dense_k: int = 20
    arabic_quota: int = 5
    final_k: int = 5
    retrieval_weight: float = 0.35
    ce_weight: float = 0.65
    max_parent_context_chars: int = 10500
    focus_budget_chars: int = 4500
    focus_min_per_parent: int = 450
    before_ratio: float = 0.60
    index_root: str = "indexes"
    embedding_name: str = "BAAI/bge-m3"
    reranker_name: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
    groq_model_name: str = "openai/gpt-oss-120b"
    models_dir: str = "models"

    def __post_init__(self) -> None:
        if self.parent_size <= self.parent_overlap or self.child_size <= self.child_overlap:
            raise ValueError("Chunk sizes must exceed their overlaps")
        if self.parent_size <= self.child_size:
            raise ValueError("Parent size must exceed child size")
        if self.dense_k < 1 or self.final_k < 1 or self.arabic_quota < 0:
            raise ValueError("Invalid retrieval quotas")
        if abs(self.retrieval_weight + self.ce_weight - 1.0) > 1e-8:
            raise ValueError("Retrieval and Cross-Encoder weights must sum to one")
        if not 0 < self.before_ratio < 1:
            raise ValueError("before_ratio must be between zero and one")
        if self.focus_budget_chars > self.max_parent_context_chars:
            raise ValueError("Focused context cannot exceed full parent budget")
