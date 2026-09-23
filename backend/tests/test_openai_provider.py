"""OpenAIProvider tests using a stub client. No network calls are made."""

from types import SimpleNamespace

import httpx2
import openai
import pytest
from pydantic import SecretStr

from app.core.errors import (
    LLMConfigurationError,
    LLMProviderError,
    LLMRateLimitedError,
    LLMResponseError,
    LLMTimeoutError,
)
from app.services.llm.openai_provider import OpenAIProvider

REQUEST = httpx2.Request("POST", "https://api.openai.com/v1/responses")


class StubResponses:
    def __init__(self, result=None, error=None):
        self.result, self.error, self.kwargs = result, error, None

    def create(self, **kwargs):
        self.kwargs = kwargs
        if self.error:
            raise self.error
        return self.result


def make_provider(result=None, error=None, api_key="sk-test"):
    stub = StubResponses(result, error)
    provider = OpenAIProvider(
        api_key=SecretStr(api_key) if api_key else None,
        model="gpt-4.1-mini",
        instructions="Be brief.",
        timeout_seconds=5,
        max_retries=0,
        client=SimpleNamespace(responses=stub),
    )
    return provider, stub


def response(**overrides):
    base = dict(
        id="resp_123",
        model="gpt-4.1-mini-2025-04-14",
        status="completed",
        output_text="Hi!",
        usage=SimpleNamespace(input_tokens=10, output_tokens=5, total_tokens=15),
    )
    return SimpleNamespace(**(base | overrides))


def status_error(cls, status):
    return cls("boom", response=httpx2.Response(status, request=REQUEST), body=None)


def test_maps_response_to_result():
    provider, stub = make_provider(response())

    result = provider.generate("Hello", max_output_tokens=50)

    assert result.text == "Hi!"
    assert result.model == "gpt-4.1-mini-2025-04-14"
    assert (result.usage.input_tokens, result.usage.output_tokens, result.usage.total_tokens) == (10, 5, 15)
    assert result.request_id == "resp_123"
    assert stub.kwargs == {
        "model": "gpt-4.1-mini",
        "instructions": "Be brief.",
        "input": "Hello",
        "max_output_tokens": 50,
    }


def test_missing_api_key_raises_configuration_error():
    provider = OpenAIProvider(
        api_key=None, model="gpt-4.1-mini", instructions="", timeout_seconds=5, max_retries=0
    )

    with pytest.raises(LLMConfigurationError, match="OPENAI_API_KEY"):
        provider.generate("Hello", max_output_tokens=50)


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (openai.APITimeoutError(request=REQUEST), LLMTimeoutError),
        (openai.APIConnectionError(request=REQUEST), LLMProviderError),
        (status_error(openai.AuthenticationError, 401), LLMConfigurationError),
        (status_error(openai.NotFoundError, 404), LLMConfigurationError),
        (status_error(openai.RateLimitError, 429), LLMRateLimitedError),
        (status_error(openai.InternalServerError, 500), LLMProviderError),
    ],
)
def test_maps_sdk_errors(error, expected):
    provider, _ = make_provider(error=error)

    with pytest.raises(expected) as exc_info:
        provider.generate("Hello", max_output_tokens=50)
    assert "boom" not in exc_info.value.message


@pytest.mark.parametrize(
    "bad",
    [
        response(usage=None),
        response(output_text=""),
        response(usage=SimpleNamespace(input_tokens=None, output_tokens=1, total_tokens=1)),
    ],
)
def test_unexpected_response_raises(bad):
    provider, _ = make_provider(bad)

    with pytest.raises(LLMResponseError):
        provider.generate("Hello", max_output_tokens=50)


def test_missing_key_via_api_returns_clean_503(client, settings):
    from app.services.llm import get_llm_provider

    no_key = OpenAIProvider(
        api_key=None, model="gpt-4.1-mini", instructions="", timeout_seconds=5, max_retries=0
    )
    client.app.dependency_overrides[get_llm_provider] = lambda: no_key

    response = client.post("/api/v1/chat", json={"conversation_id": "c1", "message": "Hi"})

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "llm_not_configured"
    assert client.get("/api/v1/usage").json()["credits_used"] == "0"
