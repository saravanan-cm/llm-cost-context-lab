"""Usage metering: LLM usage -> PricingService -> CreditPolicy -> UsageService.

Knows nothing about RAG or prompts; it meters whatever the LLM provider reports.
"""

import logging
import math
from dataclasses import dataclass
from decimal import Decimal

from app.services.credits import CreditPolicy
from app.services.llm.base import LLMResult
from app.services.pricing import Cost, PricingService
from app.services.usage_service import UsageEntry, UsageService

logger = logging.getLogger(__name__)

# Conservative pre-flight estimate: ~3 characters per token plus fixed overhead for message
# framing. Only used to decide whether to allow a request; actual charges always use
# provider-reported usage.
_CHARS_PER_TOKEN_ESTIMATE = 3
_PROMPT_OVERHEAD_TOKENS = 100


def estimate_prompt_tokens(*texts: str | None) -> int:
    chars = sum(len(text) for text in texts if text)
    return math.ceil(chars / _CHARS_PER_TOKEN_ESTIMATE) + _PROMPT_OVERHEAD_TOKENS


@dataclass(frozen=True)
class MeteredUsage:
    cost: Cost
    credits_consumed: Decimal
    credits_remaining: Decimal


class UsageMeter:
    def __init__(self, *, pricing: PricingService, credits: CreditPolicy, usage: UsageService) -> None:
        self._pricing = pricing
        self._credits = credits
        self._usage = usage

    def ensure_can_afford(
        self, user_id: str, *, model: str, estimated_input_tokens: int, max_output_tokens: int
    ) -> None:
        """Raise ``InsufficientCreditsError`` unless the worst-case cost is covered.

        Also raises ``PricingNotConfiguredError`` for an unpriced model, before any spend.
        """
        worst_case = self._pricing.calculate(model, estimated_input_tokens, max_output_tokens)
        self._usage.ensure_can_afford(user_id, self._credits.credits_for_cost(worst_case.total_cost))

    def remaining_credits(self, user_id: str) -> Decimal:
        return self._usage.get_remaining_credits(user_id)

    def record(
        self, *, user_id: str, conversation_id: str, result: LLMResult, latency_ms: int
    ) -> MeteredUsage:
        """Price the provider-reported usage, convert to credits, write ledger + deduct."""
        cost = self._pricing.calculate(result.model, result.usage.input_tokens, result.usage.output_tokens)
        credits_consumed = self._credits.credits_for_cost(cost.total_cost)
        entry = UsageEntry(
            user_id=user_id,
            conversation_id=conversation_id,
            provider=result.provider,
            model=result.model,
            provider_request_id=result.request_id,
            input_tokens=result.usage.input_tokens,
            output_tokens=result.usage.output_tokens,
            total_tokens=result.usage.total_tokens,
            input_cost=cost.input_cost,
            output_cost=cost.output_cost,
            total_cost=cost.total_cost,
            credits_consumed=credits_consumed,
            latency_ms=latency_ms,
        )
        try:
            remaining = self._usage.record_usage(entry)
        except Exception:
            # The provider already charged us; log enough to reconcile the ledger manually.
            logger.exception(
                "usage_record_failed",
                extra={
                    "user_id": user_id,
                    "conversation_id": conversation_id,
                    "provider_request_id": result.request_id,
                    "total_cost_usd": format_amount(cost.total_cost),
                },
            )
            raise
        return MeteredUsage(cost=cost, credits_consumed=credits_consumed, credits_remaining=remaining)


def format_amount(amount: Decimal) -> str:
    return format(amount.normalize(), "f")
