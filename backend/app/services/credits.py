"""Conversion from LLM cost (USD) to application credits.

credits = cost_usd * CREDITS_PER_DOLLAR

Credits are deliberately decoupled from tokens and from USD so the product can change
models, margins or pricing without changing user-facing budgets.
"""

from decimal import Decimal

from app.core.money import quantize_amount


class CreditPolicy:
    def __init__(self, credits_per_dollar: Decimal) -> None:
        if credits_per_dollar <= 0:
            raise ValueError("credits_per_dollar must be positive")
        self.credits_per_dollar = credits_per_dollar

    def credits_for_cost(self, cost_usd: Decimal) -> Decimal:
        return quantize_amount(cost_usd * self.credits_per_dollar)
