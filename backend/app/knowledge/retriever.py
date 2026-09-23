"""Query -> embedding -> vector search. Independent of the LLM."""

import logging
import time
from dataclasses import dataclass

from app.knowledge.embeddings import EmbeddingProvider
from app.knowledge.models import RetrievedChunk
from app.knowledge.vector_store import VectorStore

logger = logging.getLogger(__name__)

MAX_TOP_K = 20


@dataclass(frozen=True)
class RetrievalResult:
    chunks: list[RetrievedChunk]
    embedding_model: str
    query_tokens: int
    latency_ms: int


class KnowledgeRetriever:
    def __init__(self, embedder: EmbeddingProvider, store: VectorStore) -> None:
        self._embedder = embedder
        self._store = store

    def search(self, query: str, top_k: int = 5) -> RetrievalResult:
        query = query.strip()
        if not query:
            raise ValueError("query must not be empty")
        if not 1 <= top_k <= MAX_TOP_K:
            raise ValueError(f"top_k must be between 1 and {MAX_TOP_K}")

        started = time.perf_counter()
        embedding = self._embedder.embed([query])
        chunks = self._store.search(embedding.vectors[0], top_k)
        latency_ms = round((time.perf_counter() - started) * 1000)

        logger.info(
            "knowledge_search",
            extra={
                "top_k": top_k,
                "results": len(chunks),
                "top_score": round(chunks[0].score, 4) if chunks else None,
                "embedding_model": embedding.model,
                "query_tokens": embedding.input_tokens,
                "latency_ms": latency_ms,
            },
        )
        return RetrievalResult(
            chunks=chunks,
            embedding_model=embedding.model,
            query_tokens=embedding.input_tokens,
            latency_ms=latency_ms,
        )
