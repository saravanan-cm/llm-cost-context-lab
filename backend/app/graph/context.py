"""Per-request runtime context for the chat graph (LangGraph ``context_schema``).

Services live here, not in the graph state: the compiled graph is shared across requests,
while these objects (e.g. the usage meter's DB session) are request-scoped.
"""

from dataclasses import dataclass
from typing import Literal

from app.core.config import Settings
from app.knowledge.retriever import KnowledgeRetriever
from app.rag.context_builder import RAGContextBuilder
from app.services.llm.base import LLMProvider
from app.services.metering import UsageMeter


@dataclass(frozen=True)
class ChatGraphSettings:
    rag_enabled: bool = True
    top_k: int = 5
    min_relevance_score: float | None = None
    """Chunks scoring below this are not sent to the LLM. None disables filtering."""
    no_context_mode: Literal["fixed_response", "llm"] = "fixed_response"
    no_context_message: str = "The knowledge base does not currently contain relevant information."
    max_output_tokens: int = 1024

    @classmethod
    def from_settings(cls, settings: Settings) -> "ChatGraphSettings":
        return cls(
            rag_enabled=settings.rag_enabled,
            top_k=settings.rag_top_k,
            min_relevance_score=settings.rag_min_relevance_score,
            no_context_mode=settings.rag_no_context_mode,
            no_context_message=settings.rag_no_context_message,
            max_output_tokens=settings.llm_max_output_tokens,
        )


@dataclass(frozen=True)
class ChatGraphContext:
    retriever: KnowledgeRetriever
    builder: RAGContextBuilder
    llm: LLMProvider
    meter: UsageMeter
    settings: ChatGraphSettings
