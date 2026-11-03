"""BGE-M3 embeddings with normalized vectors on CPU."""
from pathlib import Path

from src.Config.settings import NateqSettings

def make_embeddings(settings: NateqSettings):
    """BAAI/bge-m3 on CPU; exactly the previous embedding configuration."""
    from langchain_huggingface import HuggingFaceEmbeddings
    Path(settings.models_dir).mkdir(parents=True, exist_ok=True)
    return HuggingFaceEmbeddings(
        model_name=settings.embedding_name,
        cache_folder=settings.models_dir,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
