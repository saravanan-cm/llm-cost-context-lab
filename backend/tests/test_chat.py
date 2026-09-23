from decimal import Decimal

import pytest

from app.core.errors import LLMProviderError, LLMResponseError, LLMTimeoutError

PAYLOAD = {"conversation_id": "c1", "message": "Hello"}

# gpt-4.1-mini: $0.40 / 1M input, $1.60 / 1M output; FakeProvider reports 120 in / 180 out.
EXPECTED_INPUT_COST = "0.000048"
EXPECTED_OUTPUT_COST = "0.000288"
EXPECTED_TOTAL_COST = "0.000336"
EXPECTED_CREDITS = "0.336"  # 0.000336 USD * 1000 credits/USD


def test_chat_returns_llm_reply(client, provider):
    response = client.post("/api/v1/chat", json=PAYLOAD)

    assert response.status_code == 200
    body = response.json()
    assert body["conversation_id"] == "c1"
    assert body["assistant_message"] == "reply to: Hello"
    assert body["model"] == "gpt-4.1-mini"
    assert provider.calls == ["Hello"]


def test_chat_returns_provider_token_usage(client):
    body = client.post("/api/v1/chat", json=PAYLOAD).json()

    assert body["usage"] == {"input_tokens": 120, "output_tokens": 180, "total_tokens": 300}


def test_chat_returns_cost_and_credits(client):
    body = client.post("/api/v1/chat", json=PAYLOAD).json()

    assert body["cost"] == {
        "currency": "USD",
        "input_cost": EXPECTED_INPUT_COST,
        "output_cost": EXPECTED_OUTPUT_COST,
        "total_cost": EXPECTED_TOTAL_COST,
    }
    assert body["credits"] == {"consumed": EXPECTED_CREDITS, "remaining": "999.664"}


def test_successful_request_deducts_credits(client):
    client.post("/api/v1/chat", json=PAYLOAD)
    client.post("/api/v1/chat", json=PAYLOAD)

    usage = client.get("/api/v1/usage").json()
    assert Decimal(usage["credits_used"]) == Decimal(EXPECTED_CREDITS) * 2
    assert Decimal(usage["credits_remaining"]) == Decimal(1000) - Decimal(EXPECTED_CREDITS) * 2


def test_insufficient_credits_rejected_without_calling_llm(client, provider, set_credits):
    set_credits(Decimal("0.5"))  # less than the worst-case estimate for 1024 output tokens

    response = client.post("/api/v1/chat", json=PAYLOAD)

    assert response.status_code == 402
    assert response.json()["error"]["code"] == "insufficient_credits"
    assert provider.calls == []


def test_zero_credits_rejected(client, provider, set_credits):
    set_credits(Decimal(0))

    assert client.post("/api/v1/chat", json=PAYLOAD).status_code == 402
    assert provider.calls == []


@pytest.mark.parametrize(
    ("error", "status", "code"),
    [
        (LLMTimeoutError(), 504, "llm_timeout"),
        (LLMProviderError(), 502, "llm_provider_error"),
        (LLMResponseError(), 502, "llm_invalid_response"),
    ],
)
def test_llm_failure_does_not_deduct_credits(client, provider, error, status, code):
    provider.error = error

    response = client.post("/api/v1/chat", json=PAYLOAD)

    assert response.status_code == status
    assert response.json()["error"]["code"] == code
    usage = client.get("/api/v1/usage").json()
    assert usage["credits_used"] == "0"
    assert usage["request_count"] == 0


def test_unknown_model_pricing_returns_clean_error(client, provider):
    provider.model = "unknown-model"

    response = client.post("/api/v1/chat", json=PAYLOAD)

    assert response.status_code == 500
    assert response.json()["error"]["code"] == "pricing_not_configured"
    assert provider.calls == []


def test_chat_rejects_empty_message(client):
    response = client.post("/api/v1/chat", json={"conversation_id": "c1", "message": ""})

    assert response.status_code == 422


def test_chat_requires_conversation_id(client):
    response = client.post("/api/v1/chat", json={"message": "Hello"})

    assert response.status_code == 422
