from typing import Any

from pydantic import BaseModel, Field

from app.knowledge.retriever import MAX_TOP_K


class KnowledgeSearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=MAX_TOP_K)


class KnowledgeChunk(BaseModel):
    text: str
    score: float
    source: str
    article_title: str
    source_url: str
    document_id: str
    chunk_id: str
    chunk_index: int
    metadata: dict[str, Any]


class KnowledgeSearchResponse(BaseModel):
    query: str
    top_k: int
    embedding_model: str
    query_tokens: int
    latency_ms: int
    results: list[KnowledgeChunk]
