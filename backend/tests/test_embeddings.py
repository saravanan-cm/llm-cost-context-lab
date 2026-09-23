"""OpenAIEmbeddingProvider tests with a stub client. No network calls."""

from types import SimpleNamespace

import httpx2
import openai
import pytest
from pydantic import SecretStr

from app.core.errors import EmbeddingConfigurationError, EmbeddingDimensionError, EmbeddingProviderError
from app.knowledge.openai_embeddings import OpenAIEmbeddingProvider, resolve_dimension

REQUEST = httpx2.Request("POST", "https://api.openai.com/v1/embeddings")


class StubEmbeddings:
    def __init__(self, dimension: int, error: Exception | None = None, shuffle: bool = False):
        self.dimension, self.error, self.shuffle = dimension, error, shuffle
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        data = [
            SimpleNamespace(index=i, embedding=[float(i)] * self.dimension)
            for i, _ in enumerate(kwargs["input"])
        ]
        if self.shuffle:
            data.reverse()
        return SimpleNamespace(data=data, usage=SimpleNamespace(prompt_tokens=3 * len(kwargs["input"])))


def make_provider(stub, *, model="text-embedding-3-small", dimensions=None, batch_size=2, api_key="sk-test"):
    return OpenAIEmbeddingProvider(
        api_key=SecretStr(api_key) if api_key else None,
        model=model,
        dimensions=dimensions,
        batch_size=batch_size,
        timeout_seconds=5,
        max_retries=0,
        client=SimpleNamespace(embeddings=stub) if stub else None,
    )


def test_batches_requests_and_preserves_order():
    stub = StubEmbeddings(1536, shuffle=True)
    provider = make_provider(stub, batch_size=2)

    batch = provider.embed(["a", "b", "c", "d", "e"])

    assert [len(call["input"]) for call in stub.calls] == [2, 2, 1]
    assert len(batch.vectors) == 5
    assert batch.vectors[0][0] == 0.0 and batch.vectors[1][0] == 1.0  # sorted by index
    assert batch.input_tokens == 15
    assert batch.model == "text-embedding-3-small"


def test_empty_input_makes_no_calls():
    stub = StubEmbeddings(1536)

    assert make_provider(stub).embed([]).vectors == []
    assert stub.calls == []


def test_dimension_resolved_from_model():
    assert resolve_dimension("text-embedding-3-small", None) == 1536
    assert resolve_dimension("text-embedding-3-large", None) == 3072
    assert resolve_dimension("text-embedding-3-large", 256) == 256


def test_unknown_model_requires_explicit_dimension():
    with pytest.raises(EmbeddingConfigurationError, match="OPENAI_EMBEDDING_DIMENSIONS"):
        resolve_dimension("some-new-model", None)
    assert resolve_dimension("some-new-model", 768) == 768


def test_invalid_dimension_override_rejected():
    with pytest.raises(EmbeddingConfigurationError):
        resolve_dimension("text-embedding-3-small", 4096)
    with pytest.raises(EmbeddingConfigurationError):
        resolve_dimension("text-embedding-ada-002", 512)


def test_custom_dimension_is_sent_to_api():
    stub = StubEmbeddings(256)
    make_provider(stub, dimensions=256).embed(["x"])

    assert stub.calls[0]["dimensions"] == 256


def test_wrong_vector_dimension_raises():
    provider = make_provider(StubEmbeddings(10))

    with pytest.raises(EmbeddingDimensionError):
        provider.embed(["x"])


def test_missing_api_key_raises_configuration_error():
    provider = make_provider(None, api_key=None)

    with pytest.raises(EmbeddingConfigurationError, match="OPENAI_API_KEY"):
        provider.embed(["x"])


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (openai.APITimeoutError(request=REQUEST), EmbeddingProviderError),
        (openai.APIConnectionError(request=REQUEST), EmbeddingProviderError),
        (
            openai.AuthenticationError("bad key sk-secret", response=httpx2.Response(401, request=REQUEST), body=None),
            EmbeddingConfigurationError,
        ),
        (
            openai.RateLimitError("slow down", response=httpx2.Response(429, request=REQUEST), body=None),
            EmbeddingProviderError,
        ),
    ],
)
def test_sdk_errors_are_mapped_without_leaking_details(error, expected):
    provider = make_provider(StubEmbeddings(1536, error=error))

    with pytest.raises(expected) as exc_info:
        provider.embed(["x"])
    assert "sk-secret" not in exc_info.value.message
