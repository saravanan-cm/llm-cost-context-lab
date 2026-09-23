from decimal import Decimal

import pytest

from app.core.errors import PricingNotConfiguredError
from app.services.credits import CreditPolicy
from app.services.pricing import ModelPrice, PricingService


def test_cost_uses_usd_per_million_tokens():
    cost = PricingService().calculate("gpt-4.1-mini", input_tokens=1_000_000, output_tokens=1_000_000)

    assert cost.input_cost == Decimal("0.40")
    assert cost.output_cost == Decimal("1.60")
    assert cost.total_cost == Decimal("2.00")


def test_small_costs_keep_precision():
    cost = PricingService().calculate("gpt-5-nano", input_tokens=1, output_tokens=1)

    assert cost.input_cost == Decimal("0.00000005")
    assert cost.output_cost == Decimal("0.0000004")
    assert cost.total_cost == Decimal("0.00000045")


def test_no_float_accumulation_error():
    pricing = PricingService()
    total = sum(
        (pricing.calculate("gpt-4.1-mini", 1, 1).total_cost for _ in range(1_000)), Decimal(0)
    )

    assert total == Decimal("0.002")


@pytest.mark.parametrize(
    ("model", "expected_total"),
    [
        ("gpt-4.1-mini", Decimal("0.000336")),  # 120*0.40 + 180*1.60
        ("gpt-4.1-nano", Decimal("0.000084")),  # 120*0.10 + 180*0.40
        ("gpt-4o-mini", Decimal("0.000126")),  # 120*0.15 + 180*0.60
        ("gpt-5-mini", Decimal("0.00039")),  # 120*0.25 + 180*2.00
    ],
)
def test_pricing_for_multiple_models(model, expected_total):
    assert PricingService().calculate(model, 120, 180).total_cost == expected_total


def test_custom_catalog_with_different_input_output_prices():
    pricing = PricingService(
        {"model-a": ModelPrice(Decimal("1"), Decimal("1")), "model-b": ModelPrice(Decimal("2"), Decimal("8"))}
    )

    assert pricing.calculate("model-a", 1_000_000, 1_000_000).total_cost == Decimal("2")
    assert pricing.calculate("model-b", 1_000_000, 1_000_000).total_cost == Decimal("10")


def test_dated_snapshot_resolves_to_base_model_price():
    pricing = PricingService()

    assert pricing.price_for("gpt-4.1-mini-2025-04-14") == pricing.price_for("gpt-4.1-mini")


def test_unknown_model_raises():
    with pytest.raises(PricingNotConfiguredError):
        PricingService().calculate("no-such-model", 1, 1)


def test_credits_from_cost():
    policy = CreditPolicy(credits_per_dollar=Decimal("1000"))

    assert policy.credits_for_cost(Decimal("0.01")) == Decimal("10")
    assert Decimal(1000) - policy.credits_for_cost(Decimal("0.01")) == Decimal("990")


def test_credit_conversion_is_configurable():
    assert CreditPolicy(Decimal("250")).credits_for_cost(Decimal("0.01")) == Decimal("2.5")


def test_credit_policy_rejects_non_positive_rate():
    with pytest.raises(ValueError):
        CreditPolicy(Decimal(0))
