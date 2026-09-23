"""Sparse, dense, and fusion retrieval."""

from .rrf import RRF_K, fuse_ranked_lists

__all__ = ["BM25Retriever", "DEFAULT_MODEL", "DenseRetriever", "RRF_K", "fuse_ranked_lists"]


def __getattr__(name: str):
    """Load optional model-backed retrievers only when callers request them."""
    if name == "BM25Retriever":
        from .bm25 import BM25Retriever

        return BM25Retriever
    if name in {"DEFAULT_MODEL", "DenseRetriever"}:
        from .dense import DEFAULT_MODEL, DenseRetriever

        return {"DEFAULT_MODEL": DEFAULT_MODEL, "DenseRetriever": DenseRetriever}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
