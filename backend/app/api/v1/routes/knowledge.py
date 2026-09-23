from dataclasses import asdict
from typing import Annotated

from fastapi import APIRouter, Depends

from app.api.deps import get_retriever
from app.knowledge.retriever import KnowledgeRetriever
from app.schemas.common import ErrorResponse
from app.schemas.knowledge import KnowledgeChunk, KnowledgeSearchRequest, KnowledgeSearchResponse

router = APIRouter(prefix="/knowledge", tags=["knowledge (dev)"])


@router.post(
    "/search",
    response_model=KnowledgeSearchResponse,
    responses={503: {"model": ErrorResponse, "description": "Knowledge base unavailable or not built"}},
)
def search(
    request: KnowledgeSearchRequest,
    retriever: Annotated[KnowledgeRetriever, Depends(get_retriever)],
) -> KnowledgeSearchResponse:
    """Development endpoint for testing retrieval. Does not call the LLM."""
    result = retriever.search(request.query, request.top_k)
    return KnowledgeSearchResponse(
        query=request.query,
        top_k=request.top_k,
        embedding_model=result.embedding_model,
        query_tokens=result.query_tokens,
        latency_ms=result.latency_ms,
        results=[KnowledgeChunk(**asdict(chunk)) for chunk in result.chunks],
    )
