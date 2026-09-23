"""Chat use case: run the chat graph and map its final state to the API response.

All orchestration (analysis, retrieval, prompt, credit check, LLM, metering) happens in the
LangGraph workflow (``app.graph``). This class invokes it, logs the run and turns terminal
statuses into application errors.
"""

import logging
import time

from langgraph.graph.state import CompiledStateGraph

from app.core.errors import AppError, InsufficientCreditsError, InvalidQuestionError
from app.graph.context import ChatGraphContext
from app.graph.state import ChatState, initial_state
from app.schemas.chat import (
    ChatDebug,
    ChatRequest,
    ChatResponse,
    CostBreakdown,
    CreditInfo,
    RetrievedChunkDebug,
    Source,
    TokenUsage,
)
from app.services.metering import format_amount

logger = logging.getLogger(__name__)


class ChatService:
    def __init__(self, *, graph: CompiledStateGraph, context: ChatGraphContext, debug: bool = False) -> None:
        self._graph = graph
        self._context = context
        self._debug = debug

    def reply(self, user_id: str, request: ChatRequest) -> ChatResponse:
        started = time.perf_counter()
        base = {"conversation_id": request.conversation_id}
        logger.info("chat_graph_start", extra=base)

        state: ChatState = initial_state(
            user_id=user_id, conversation_id=request.conversation_id, user_message=request.message
        )
        try:
            # stream_mode="values" yields the full state after each step, so the path taken so
            # far is known even if a node fails.
            for state in self._graph.stream(state, context=self._context, stream_mode="values"):
                pass
        except AppError as exc:
            logger.warning(
                "chat_graph_end",
                extra={**base, **_path(state), "outcome": "failure", "error_code": exc.code, "latency_ms": _ms(started)},
            )
            raise

        status = state["status"]
        summary = {**base, **_path(state), **_metrics(state), "status": status, "latency_ms": _ms(started)}
        if status == "invalid_question":
            logger.info("chat_graph_end", extra={**summary, "outcome": "rejected"})
            raise InvalidQuestionError()
        if status == "insufficient_credits":
            logger.info("chat_graph_end", extra={**summary, "outcome": "rejected"})
            raise InsufficientCreditsError()

        logger.info("chat_graph_end", extra={**summary, "outcome": "success"})
        return self._to_response(state)

    def _to_response(self, state: ChatState) -> ChatResponse:
        usage = state["usage"]
        cost = state["cost"]
        return ChatResponse(
            conversation_id=state["conversation_id"],
            assistant_message=state["assistant_message"],
            answer_type=state["answer_type"],
            model=state["model"],
            sources=[
                Source(title=ref.title, source=ref.source, url=ref.url, score=round(ref.score, 4))
                for ref in state.get("sources", [])
            ],
            usage=TokenUsage(
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                total_tokens=usage.total_tokens,
            ),
            cost=CostBreakdown(input_cost=cost.input_cost, output_cost=cost.output_cost, total_cost=cost.total_cost),
            credits=CreditInfo(consumed=state["credits_consumed"], remaining=state["credits_remaining"]),
            debug=self._debug_info(state) if self._debug else None,
        )

    def _debug_info(self, state: ChatState) -> ChatDebug:
        settings = self._context.settings
        threshold = settings.min_relevance_score
        retrieved = state.get("retrieved_documents", [])
        return ChatDebug(
            graph_path=state["graph_path"],
            top_k=settings.top_k,
            min_relevance_score=threshold,
            retrieval_latency_ms=state.get("retrieval_latency_ms"),
            llm_latency_ms=state.get("llm_latency_ms"),
            query_tokens=state.get("query_tokens", 0),
            chunks=[
                RetrievedChunkDebug(
                    rank=rank,
                    article_title=chunk.article_title,
                    url=chunk.source_url,
                    chunk_index=chunk.chunk_index,
                    score=chunk.score,
                    used=threshold is None or chunk.score >= threshold,
                )
                for rank, chunk in enumerate(retrieved, start=1)
            ],
        )


def _path(state: ChatState) -> dict[str, object]:
    return {"graph_path": ">".join(state.get("graph_path", []))}


def _metrics(state: ChatState) -> dict[str, object]:
    metrics: dict[str, object] = {}
    if "retrieved_documents" in state:
        retrieved = state["retrieved_documents"]
        metrics.update(
            retrieval_latency_ms=state.get("retrieval_latency_ms"),
            retrieved=len(retrieved),
            relevant=len(state.get("relevant_documents", [])),
            top_score=round(retrieved[0].score, 4) if retrieved else None,
        )
    if "llm_latency_ms" in state:
        usage = state["usage"]
        metrics.update(
            model=state.get("model"),
            provider_request_id=state.get("provider_request_id"),
            llm_latency_ms=state["llm_latency_ms"],
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            total_tokens=usage.total_tokens,
        )
    if "cost" in state and state.get("status") == "answered":
        metrics.update(
            total_cost_usd=format_amount(state["cost"].total_cost),
            credits_consumed=format_amount(state["credits_consumed"]),
        )
    return metrics


def _ms(started: float) -> int:
    return round((time.perf_counter() - started) * 1000)
