"""RAG orchestration: retrieve -> filter by relevance -> build prompt -> LLM.

Two phases so the caller can apply the credit check between them, once the real prompt size
is known and before any LLM spend:

    prepared = rag.prepare(question)     # retrieval + prompt (or a fixed no-context answer)
    answer   = rag.generate(prepared)    # LLM call

Pricing, credits and usage recording are not done here (see ``app.services.metering``).
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Literal

from app.knowledge.models import RetrievedChunk
from app.knowledge.retriever import KnowledgeRetriever
from app.rag.context_builder import LLMPrompt, RAGContextBuilder
from app.rag.sources import SourceReference, group_by_document
from app.services.llm.base import LLMProvider, LLMResult

logger = logging.getLogger(__name__)

AnswerMode = Literal["grounded", "no_context", "direct"]
NoContextMode = Literal["fixed_response", "llm"]


@dataclass(frozen=True)
class RAGSettings:
    enabled: bool = True
    top_k: int = 5
    min_relevance_score: float | None = None
    """Chunks scoring below this are not sent to the LLM. None disables filtering."""
    no_context_mode: NoContextMode = "fixed_response"
    no_context_message: str = "The knowledge base does not currently contain relevant information."


@dataclass(frozen=True)
class PreparedAnswer:
    mode: AnswerMode
    prompt: LLMPrompt | None
    """None when the answer is ``fixed_response`` (no LLM call)."""
    fixed_response: str | None = None
    sources: list[SourceReference] = field(default_factory=list)
    retrieved: list[RetrievedChunk] = field(default_factory=list)
    """All top-k chunks before relevance filtering (for debugging)."""
    retrieval_latency_ms: int | None = None
    query_tokens: int = 0

    @property
    def top_score(self) -> float | None:
        return self.retrieved[0].score if self.retrieved else None


@dataclass(frozen=True)
class GeneratedAnswer:
    result: LLMResult
    latency_ms: int


class RAGService:
    def __init__(
        self,
        *,
        retriever: KnowledgeRetriever,
        builder: RAGContextBuilder,
        llm: LLMProvider,
        settings: RAGSettings,
    ) -> None:
        self._retriever = retriever
        self._builder = builder
        self._llm = llm
        self.settings = settings

    @property
    def model(self) -> str:
        return self._llm.model

    def prepare(self, question: str) -> PreparedAnswer:
        """Retrieve and build the prompt. Retrieval errors propagate (no LLM call follows)."""
        if not self.settings.enabled:
            return PreparedAnswer(mode="direct", prompt=RAGContextBuilder.build_direct(question))

        retrieval = self._retriever.search(question, self.settings.top_k)
        threshold = self.settings.min_relevance_score
        relevant = [c for c in retrieval.chunks if threshold is None or c.score >= threshold]
        common = {
            "retrieved": retrieval.chunks,
            "retrieval_latency_ms": retrieval.latency_ms,
            "query_tokens": retrieval.query_tokens,
        }

        if not relevant:
            if self.settings.no_context_mode == "llm":
                return PreparedAnswer(
                    mode="no_context", prompt=self._builder.build_without_context(question), **common
                )
            return PreparedAnswer(
                mode="no_context", prompt=None, fixed_response=self.settings.no_context_message, **common
            )

        sources = group_by_document(relevant)
        return PreparedAnswer(
            mode="grounded", prompt=self._builder.build(question, sources), sources=sources, **common
        )

    def generate(self, prepared: PreparedAnswer, *, max_output_tokens: int) -> GeneratedAnswer:
        if prepared.prompt is None:
            raise ValueError("prepared answer does not require an LLM call")
        started = time.perf_counter()
        result = self._llm.generate(
            prepared.prompt.input,
            max_output_tokens=max_output_tokens,
            instructions=prepared.prompt.instructions,
        )
        return GeneratedAnswer(result=result, latency_ms=round((time.perf_counter() - started) * 1000))
