"""Model pricing and cost calculation.

PRICING UNIT: US dollars per 1,000,000 tokens (USD / 1M tokens), separately for input
(prompt) and output (completion) tokens. This is the unit OpenAI publishes.

To add a model, add an entry to ``MODEL_PRICING``. Prices are list prices for standard
(non-cached, non-batch) usage; verify against https://openai.com/api/pricing before relying
on them. Cached-input discounts are not modelled yet.
"""

import re
from dataclasses import dataclass
from decimal import Decimal

from app.core.errors import PricingNotConfiguredError
from app.core.money import quantize_amount

TOKENS_PER_PRICING_UNIT = Decimal(1_000_000)


@dataclass(frozen=True)
class ModelPrice:
    input_per_million: Decimal
    output_per_million: Decimal


@dataclass(frozen=True)
class Cost:
    input_cost: Decimal
    output_cost: Decimal
    total_cost: Decimal


MODEL_PRICING: dict[str, ModelPrice] = {
    "gpt-4.1-mini": ModelPrice(Decimal("0.40"), Decimal("1.60")),
    "gpt-4.1-nano": ModelPrice(Decimal("0.10"), Decimal("0.40")),
    "gpt-4o-mini": ModelPrice(Decimal("0.15"), Decimal("0.60")),
    "gpt-5-mini": ModelPrice(Decimal("0.25"), Decimal("2.00")),
    "gpt-5-nano": ModelPrice(Decimal("0.05"), Decimal("0.40")),
    # Embedding models: input only (output is not billed).
    "text-embedding-3-small": ModelPrice(Decimal("0.02"), Decimal("0")),
    "text-embedding-3-large": ModelPrice(Decimal("0.13"), Decimal("0")),
    "text-embedding-ada-002": ModelPrice(Decimal("0.10"), Decimal("0")),
    # Fictional price so the offline mock provider exercises the metering pipeline.
    "mock-echo": ModelPrice(Decimal("1.00"), Decimal("2.00")),
}

# Providers may report dated snapshots, e.g. "gpt-4.1-mini-2025-04-14".
_SNAPSHOT_SUFFIX = re.compile(r"-\d{4}-\d{2}-\d{2}$")


class PricingService:
    def __init__(self, catalog: dict[str, ModelPrice] | None = None) -> None:
        self._catalog = catalog if catalog is not None else MODEL_PRICING

    def price_for(self, model: str) -> ModelPrice:
        price = self._catalog.get(model) or self._catalog.get(_SNAPSHOT_SUFFIX.sub("", model))
        if price is None:
            raise PricingNotConfiguredError(f"Pricing is not configured for model '{model}'.")
        return price

    def calculate(self, model: str, input_tokens: int, output_tokens: int) -> Cost:
        price = self.price_for(model)
        input_cost = quantize_amount(input_tokens * price.input_per_million / TOKENS_PER_PRICING_UNIT)
        output_cost = quantize_amount(output_tokens * price.output_per_million / TOKENS_PER_PRICING_UNIT)
        return Cost(input_cost, output_cost, input_cost + output_cost)
