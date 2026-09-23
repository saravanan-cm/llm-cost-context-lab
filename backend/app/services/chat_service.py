"""Chat use case: credit pre-check -> RAG (retrieval + prompt + LLM) -> metering -> response.

Each step is delegated: retrieval/prompting/LLM to ``RAGService``, pricing/credits/ledger to
``UsageMeter``. This class only sequences them, logs the request and shapes the response.
"""

import logging
import time
from decimal import Decimal

from app.core.errors import AppError
from app.rag.service import GeneratedAnswer, PreparedAnswer, RAGService
from app.schemas.chat import (
    ChatRequest,
    ChatResponse,
    CostBreakdown,
    CreditInfo,
    RetrievalDebug,
    RetrievedChunkDebug,
    Source,
    TokenUsage,
)
from app.services.metering import MeteredUsage, UsageMeter, estimate_prompt_tokens, format_amount

logger = logging.getLogger(__name__)


class ChatService:
    def __init__(
        self,
        *,
        rag: RAGService,
        meter: UsageMeter,
        max_output_tokens: int,
        debug: bool = False,
    ) -> None:
        self._rag = rag
        self._meter = meter
        self._max_output_tokens = max_output_tokens
        self._debug = debug

    def reply(self, user_id: str, request: ChatRequest) -> ChatResponse:
        started = time.perf_counter()
        log: dict[str, object] = {"conversation_id": request.conversation_id, "model": self._rag.model}
        stage = "credit_check"
        prepared: PreparedAnswer | None = None
        try:
            # Cheap early check so users without credits don't trigger retrieval at all.
            self._ensure_can_afford(user_id, request.message)

            stage = "retrieval"
            prepared = self._rag.prepare(request.message)
            log.update(_retrieval_fields(prepared))
            if prepared.prompt is None:
                log.update(outcome="no_context", latency_ms=_elapsed_ms(started))
                logger.info("chat_request", extra=log)
                return self._fixed_response(user_id, request, prepared)

            # Accurate check now that the full prompt (context + instructions) is known.
            stage = "credit_check"
            self._ensure_can_afford(user_id, prepared.prompt.instructions, prepared.prompt.input)

            stage = "llm"
            answer = self._rag.generate(prepared, max_output_tokens=self._max_output_tokens)

            stage = "metering"
            metered = self._meter.record(
                user_id=user_id,
                conversation_id=request.conversation_id,
                result=answer.result,
                latency_ms=answer.latency_ms,
            )
        except AppError as exc:
            log.update(outcome="failure", stage=stage, error_code=exc.code, latency_ms=_elapsed_ms(started))
            logger.warning("chat_request", extra=log)
            raise

        log.update(
            outcome="success",
            model=answer.result.model,
            provider_request_id=answer.result.request_id,
            llm_latency_ms=answer.latency_ms,
            input_tokens=answer.result.usage.input_tokens,
            output_tokens=answer.result.usage.output_tokens,
            total_tokens=answer.result.usage.total_tokens,
            total_cost_usd=format_amount(metered.cost.total_cost),
            credits_consumed=format_amount(metered.credits_consumed),
            latency_ms=_elapsed_ms(started),
        )
        logger.info("chat_request", extra=log)
        return self._llm_response(request, prepared, answer, metered)

    def _ensure_can_afford(self, user_id: str, *texts: str | None) -> None:
        self._meter.ensure_can_afford(
            user_id,
            model=self._rag.model,
            estimated_input_tokens=estimate_prompt_tokens(*texts),
            max_output_tokens=self._max_output_tokens,
        )

    def _llm_response(
        self,
        request: ChatRequest,
        prepared: PreparedAnswer,
        answer: GeneratedAnswer,
        metered: MeteredUsage,
    ) -> ChatResponse:
        usage = answer.result.usage
        return ChatResponse(
            conversation_id=request.conversation_id,
            assistant_message=answer.result.text,
            answer_type=prepared.mode,
            model=answer.result.model,
            sources=_sources(prepared),
            usage=TokenUsage(
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                total_tokens=usage.total_tokens,
            ),
            cost=CostBreakdown(
                input_cost=metered.cost.input_cost,
                output_cost=metered.cost.output_cost,
                total_cost=metered.cost.total_cost,
            ),
            credits=CreditInfo(consumed=metered.credits_consumed, remaining=metered.credits_remaining),
            debug=self._debug_info(prepared),
        )

    def _fixed_response(self, user_id: str, request: ChatRequest, prepared: PreparedAnswer) -> ChatResponse:
        zero = Decimal(0)
        return ChatResponse(
            conversation_id=request.conversation_id,
            assistant_message=prepared.fixed_response or "",
            answer_type=prepared.mode,
            model=None,
            usage=TokenUsage(input_tokens=0, output_tokens=0, total_tokens=0),
            cost=CostBreakdown(input_cost=zero, output_cost=zero, total_cost=zero),
            credits=CreditInfo(consumed=zero, remaining=self._meter.remaining_credits(user_id)),
            debug=self._debug_info(prepared),
        )

    def _debug_info(self, prepared: PreparedAnswer) -> RetrievalDebug | None:
        if not self._debug or prepared.mode == "direct":
            return None
        settings = self._rag.settings
        threshold = settings.min_relevance_score
        return RetrievalDebug(
            top_k=settings.top_k,
            min_relevance_score=threshold,
            retrieval_latency_ms=prepared.retrieval_latency_ms,
            query_tokens=prepared.query_tokens,
            chunks=[
                RetrievedChunkDebug(
                    rank=rank,
                    article_title=chunk.article_title,
                    url=chunk.source_url,
                    chunk_index=chunk.chunk_index,
                    score=chunk.score,
                    used=threshold is None or chunk.score >= threshold,
                )
                for rank, chunk in enumerate(prepared.retrieved, start=1)
            ],
        )


def _sources(prepared: PreparedAnswer) -> list[Source]:
    return [
        Source(title=ref.title, source=ref.source, url=ref.url, score=round(ref.score, 4))
        for ref in prepared.sources
    ]


def _retrieval_fields(prepared: PreparedAnswer) -> dict[str, object]:
    if prepared.mode == "direct":
        return {"rag": False}
    return {
        "rag": True,
        "retrieval_latency_ms": prepared.retrieval_latency_ms,
        "chunks_retrieved": len(prepared.retrieved),
        "chunks_used": sum(len(ref.chunks) for ref in prepared.sources),
        "sources": len(prepared.sources),
        "top_score": round(prepared.top_score, 4) if prepared.top_score is not None else None,
    }


def _elapsed_ms(started: float) -> int:
    return round((time.perf_counter() - started) * 1000)
