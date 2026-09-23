"""Monetary/credit amount helpers.

All USD costs and credit amounts are ``decimal.Decimal`` — never ``float`` — and are
quantized to 12 decimal places, enough for a single token at sub-cent-per-million prices.
"""

from decimal import ROUND_HALF_EVEN, Decimal

AMOUNT_SCALE = 12
AMOUNT_QUANTUM = Decimal(1).scaleb(-AMOUNT_SCALE)


def quantize_amount(value: Decimal) -> Decimal:
    return value.quantize(AMOUNT_QUANTUM, rounding=ROUND_HALF_EVEN)
