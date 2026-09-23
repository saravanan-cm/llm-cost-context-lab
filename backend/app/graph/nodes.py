"""Chat graph nodes and routing functions.

Nodes orchestrate existing services only: retrieval (``KnowledgeRetriever``), prompt
construction (``RAGContextBuilder``), LLM calls (``LLMProvider``) and metering
(``UsageMeter`` -> PricingService -> CreditPolicy -> UsageService). They contain no vector-search,
SDK, pricing or SQL code.

Expected outcomes (empty question, no knowledge, insufficient credits) set ``status`` and are
routed explicitly. Infrastructure failures (Qdrant, LLM, database) raise the existing typed
errors, which stop the graph before any later node (e.g. no usage is recorded after an LLM failure).
"""

import logging
import time
from collections.abc import Iterator
from contextlib import contextmanager
from decimal import Decimal
from typing import Literal

from langgraph.runtime import Runtime

from app.core.errors import AppError, InsufficientCreditsError
from app.graph.context import ChatGraphContext
from app.graph.state import ChatState
from app.rag.sources import group_by_document
from app.services.llm.base import LLMResult, LLMUsage
from app.services.metering import estimate_prompt_tokens, format_amount
from app.services.pricing import Cost

logger = logging.getLogger(__name__)

ANALYZE_QUESTION = "analyze_question"
PRECHECK_CREDITS = "precheck_credits"
RETRIEVE_KNOWLEDGE = "retrieve_knowledge"
NO_KNOWLEDGE = "no_knowledge"
BUILD_CONTEXT = "build_context"
CHECK_CREDITS = "check_credits"
GENERATE_ANSWER = "generate_answer"
USAGE_ACCOUNTING = "usage_accounting"


@contextmanager
def _traced(node: str) -> Iterator[dict[str, object]]:
    """Log one ``graph_node`` event per node execution with duration, outcome and metrics."""
    fields: dict[str, object] = {}
    started = time.perf_counter()
    try:
        yield fields
    except AppError as exc:
        logger.warning(
            "graph_node",
            extra={"node": node, "outcome": "failure", "error_code": exc.code, "duration_ms": _ms(started)},
        )
        raise
    except Exception as exc:
        logger.error(
            "graph_node",
            extra={"node": node, "outcome": "failure", "error_type": type(exc).__name__, "duration_ms": _ms(started)},
        )
        raise
    logger.info("graph_node", extra={"node": node, "outcome": "ok", "duration_ms": _ms(started), **fields})


# -- nodes ----------------------------------------------------------------------------------


def analyze_question(state: ChatState, runtime: Runtime[ChatGraphContext]) -> ChatState:
    """Normalise the question and decide whether retrieval is needed. No LLM call."""
    with _traced(ANALYZE_QUESTION):
        question = " ".join(state["user_message"].split())
        if not question:
            return {"graph_path": [ANALYZE_QUESTION], "status": "invalid_question"}
        rag_enabled = runtime.context.settings.rag_enabled
        return {
            "graph_path": [ANALYZE_QUESTION],
            "question": question,
            "retrieval_query": question,
            "use_retrieval": rag_enabled,
            "answer_type": "grounded" if rag_enabled else "direct",
        }


def retrieve_knowledge(state: ChatState, runtime: Runtime[ChatGraphContext]) -> ChatState:
    """Retriever -> Qdrant -> chunks, filtered by the configured relevance threshold."""
    with _traced(RETRIEVE_KNOWLEDGE) as trace:
        settings = runtime.context.settings
        result = runtime.context.retriever.search(state["retrieval_query"], settings.top_k)
        threshold = settings.min_relevance_score
        relevant = [c for c in result.chunks if threshold is None or c.score >= threshold]

        update: ChatState = {
            "graph_path": [RETRIEVE_KNOWLEDGE],
            "retrieved_documents": result.chunks,
            "relevant_documents": relevant,
            "retrieval_latency_ms": result.latency_ms,
            "query_tokens": result.query_tokens,
        }
        if not relevant:
            update["answer_type"] = "no_context"
            if settings.no_context_mode == "fixed_response":
                update["status"] = "no_knowledge"
        trace.update(
            retrieved=len(result.chunks),
            relevant=len(relevant),
            top_score=round(result.chunks[0].score, 4) if result.chunks else None,
            retrieval_latency_ms=result.latency_ms,
        )
        return update


def no_knowledge(state: ChatState, runtime: Runtime[ChatGraphContext]) -> ChatState:
    """Controlled answer when the knowledge base has nothing relevant. No LLM call, no charge."""
    with _traced(NO_KNOWLEDGE):
        zero = Decimal(0)
        return {
            "graph_path": [NO_KNOWLEDGE],
            "status": "no_knowledge",
            "assistant_message": runtime.context.settings.no_context_message,
            "model": None,
            "sources": [],
            "usage": LLMUsage(0, 0, 0),
            "cost": Cost(zero, zero, zero),
            "credits_consumed": zero,
            "credits_remaining": runtime.context.meter.remaining_credits(state["user_id"]),
        }


def build_context(state: ChatState, runtime: Runtime[ChatGraphContext]) -> ChatState:
    """Retrieved chunks -> numbered sources -> LLM prompt (via RAGContextBuilder)."""
    with _traced(BUILD_CONTEXT) as trace:
        builder = runtime.context.builder
        question = state["question"]
        answer_type = state["answer_type"]
        sources = []
        if answer_type == "direct":
            prompt = builder.build_direct(question)
        elif answer_type == "grounded":
            sources = group_by_document(state["relevant_documents"])
            prompt = builder.build(question, sources)
        else:  # no_context with RAG_NO_CONTEXT_MODE=llm
            prompt = builder.build_without_context(question)
        trace.update(sources=len(sources), prompt_chars=len(prompt.input) + len(prompt.instructions or ""))
        return {"graph_path": [BUILD_CONTEXT], "context": prompt, "sources": sources}


def _credit_check(node: str, state: ChatState, runtime: Runtime[ChatGraphContext]) -> ChatState:
    """Existing worst-case affordability check (UsageMeter). Estimates from the full prompt when it
    has been built, otherwise from the bare question."""
    with _traced(node) as trace:
        prompt = state.get("context")
        texts = (prompt.instructions, prompt.input) if prompt is not None else (state["question"],)
        try:
            runtime.context.meter.ensure_can_afford(
                state["user_id"],
                model=runtime.context.llm.model,
                estimated_input_tokens=estimate_prompt_tokens(*texts),
                max_output_tokens=runtime.context.settings.max_output_tokens,
            )
        except InsufficientCreditsError:
            trace.update(sufficient=False)
            return {"graph_path": [node], "status": "insufficient_credits"}
        trace.update(sufficient=True)
        return {"graph_path": [node]}


def precheck_credits(state: ChatState, runtime: Runtime[ChatGraphContext]) -> ChatState:
    """Cheap check on the question alone, so users without credits don't trigger retrieval."""
    return _credit_check(PRECHECK_CREDITS, state, runtime)


def check_credits(state: ChatState, runtime: Runtime[ChatGraphContext]) -> ChatState:
    """Accurate check on the full prompt (instructions + context + question) before the LLM call."""
    return _credit_check(CHECK_CREDITS, state, runtime)


def generate_answer(state: ChatState, runtime: Runtime[ChatGraphContext]) -> ChatState:
    """Prompt -> LLMProvider -> answer + provider-reported usage."""
    with _traced(GENERATE_ANSWER) as trace:
        prompt = state["context"]
        started = time.perf_counter()
        result = runtime.context.llm.generate(
            prompt.input,
            max_output_tokens=runtime.context.settings.max_output_tokens,
            instructions=prompt.instructions,
        )
        latency_ms = _ms(started)
        trace.update(
            model=result.model,
            llm_latency_ms=latency_ms,
            input_tokens=result.usage.input_tokens,
            output_tokens=result.usage.output_tokens,
        )
        return {
            "graph_path": [GENERATE_ANSWER],
            "assistant_message": result.text,
            "model": result.model,
            "provider": result.provider,
            "provider_request_id": result.request_id,
            "usage": result.usage,
            "llm_latency_ms": latency_ms,
        }


def usage_accounting(state: ChatState, runtime: Runtime[ChatGraphContext]) -> ChatState:
    """Actual usage -> PricingService -> CreditPolicy -> UsageService (ledger + deduction)."""
    with _traced(USAGE_ACCOUNTING) as trace:
        metered = runtime.context.meter.record(
            user_id=state["user_id"],
            conversation_id=state["conversation_id"],
            result=LLMResult(
                text=state["assistant_message"],
                model=state["model"] or runtime.context.llm.model,
                usage=state["usage"],
                provider=state["provider"],
                request_id=state.get("provider_request_id"),
            ),
            latency_ms=state["llm_latency_ms"],
        )
        trace.update(
            total_cost_usd=format_amount(metered.cost.total_cost),
            credits_consumed=format_amount(metered.credits_consumed),
        )
        return {
            "graph_path": [USAGE_ACCOUNTING],
            "status": "answered",
            "cost": metered.cost,
            "credits_consumed": metered.credits_consumed,
            "credits_remaining": metered.credits_remaining,
        }


# -- routing --------------------------------------------------------------------------------


def route_after_analysis(state: ChatState) -> Literal["end", "precheck_credits"]:
    return "end" if state["status"] == "invalid_question" else PRECHECK_CREDITS


def route_after_precheck(state: ChatState) -> Literal["end", "retrieve_knowledge", "build_context"]:
    if state["status"] == "insufficient_credits":
        return "end"
    return RETRIEVE_KNOWLEDGE if state["use_retrieval"] else BUILD_CONTEXT


def route_after_retrieval(state: ChatState) -> Literal["no_knowledge", "build_context"]:
    return NO_KNOWLEDGE if state["status"] == "no_knowledge" else BUILD_CONTEXT


def route_after_credit_check(state: ChatState) -> Literal["end", "generate_answer"]:
    return "end" if state["status"] == "insufficient_credits" else GENERATE_ANSWER


def _ms(started: float) -> int:
    return round((time.perf_counter() - started) * 1000)
