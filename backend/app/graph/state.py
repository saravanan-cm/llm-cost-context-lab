"""Chat graph state: data flowing between nodes (no clients, sessions or services)."""

import operator
from decimal import Decimal
from typing import Annotated, Literal, TypedDict

from app.knowledge.models import RetrievedChunk
from app.rag.context_builder import LLMPrompt
from app.rag.sources import SourceReference
from app.services.llm.base import LLMUsage
from app.services.pricing import Cost

GraphStatus = Literal[
    "in_progress",
    "invalid_question",  # empty after normalisation -> END
    "no_knowledge",  # nothing relevant retrieved -> no_knowledge -> END
    "insufficient_credits",  # credit check failed -> END (no LLM call)
    "answered",  # LLM answered and usage was recorded -> END
]
AnswerType = Literal["grounded", "no_context", "direct"]


class ChatState(TypedDict, total=False):
    # Input
    user_id: str
    conversation_id: str
    user_message: str

    # Control
    status: GraphStatus
    graph_path: Annotated[list[str], operator.add]
    """Nodes executed so far, in order (appended by each node)."""

    # analyze_question
    question: str
    retrieval_query: str
    use_retrieval: bool

    # retrieve_knowledge
    retrieved_documents: list[RetrievedChunk]
    """Top-k chunks before relevance filtering (kept for debugging)."""
    relevant_documents: list[RetrievedChunk]
    retrieval_latency_ms: int
    query_tokens: int

    # build_context
    answer_type: AnswerType
    context: LLMPrompt
    sources: list[SourceReference]

    # generate_answer
    assistant_message: str
    model: str | None
    provider: str
    provider_request_id: str | None
    usage: LLMUsage
    llm_latency_ms: int

    # usage_accounting / no_knowledge
    cost: Cost
    credits_consumed: Decimal
    credits_remaining: Decimal


def initial_state(*, user_id: str, conversation_id: str, user_message: str) -> ChatState:
    return ChatState(
        user_id=user_id,
        conversation_id=conversation_id,
        user_message=user_message,
        status="in_progress",
        graph_path=[],
    )
