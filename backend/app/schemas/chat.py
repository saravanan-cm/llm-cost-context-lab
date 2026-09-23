from typing import Literal

from pydantic import BaseModel, Field

from app.schemas.common import DecimalStr


class ChatRequest(BaseModel):
    conversation_id: str = Field(..., min_length=1, max_length=128)
    message: str = Field(..., min_length=1, max_length=8000)


class TokenUsage(BaseModel):
    input_tokens: int
    output_tokens: int
    total_tokens: int


class CostBreakdown(BaseModel):
    """Cost in USD, serialized as exact decimal strings."""

    currency: str = "USD"
    input_cost: DecimalStr
    output_cost: DecimalStr
    total_cost: DecimalStr


class CreditInfo(BaseModel):
    consumed: DecimalStr
    remaining: DecimalStr


class Source(BaseModel):
    """A knowledge-base document used to ground the answer (deduplicated per article)."""

    title: str
    source: str
    url: str
    score: float


class RetrievedChunkDebug(BaseModel):
    rank: int
    article_title: str
    url: str
    chunk_index: int
    score: float
    used: bool
    """False when the chunk fell below the relevance threshold."""


class RetrievalDebug(BaseModel):
    top_k: int
    min_relevance_score: float | None
    retrieval_latency_ms: int | None
    query_tokens: int
    chunks: list[RetrievedChunkDebug]


class ChatResponse(BaseModel):
    conversation_id: str
    assistant_message: str
    answer_type: Literal["grounded", "no_context", "direct"]
    """grounded: answered from retrieved knowledge; no_context: nothing relevant was retrieved;
    direct: RAG disabled."""
    model: str | None
    """None when no LLM call was made."""
    sources: list[Source] = []
    usage: TokenUsage
    cost: CostBreakdown
    credits: CreditInfo
    debug: RetrievalDebug | None = None
    """Only populated when RAG_DEBUG=true."""
