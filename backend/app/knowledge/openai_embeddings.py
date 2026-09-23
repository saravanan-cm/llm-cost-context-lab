"""OpenAI implementation of ``EmbeddingProvider``."""

import logging
from collections.abc import Sequence
from typing import Any

import openai
from pydantic import SecretStr

from app.core.errors import (
    EmbeddingConfigurationError,
    EmbeddingDimensionError,
    EmbeddingProviderError,
)
from app.knowledge.embeddings import EmbeddingBatch

logger = logging.getLogger(__name__)

# Native output dimension per model. The vector collection is sized from this, so an
# unknown model must be given an explicit OPENAI_EMBEDDING_DIMENSIONS.
NATIVE_DIMENSIONS: dict[str, int] = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}
# Models that accept the ``dimensions`` request parameter (Matryoshka truncation).
_SUPPORTS_DIMENSIONS = {"text-embedding-3-small", "text-embedding-3-large"}


def resolve_dimension(model: str, override: int | None) -> int:
    native = NATIVE_DIMENSIONS.get(model)
    if override is None:
        if native is None:
            raise EmbeddingConfigurationError(
                f"Unknown dimension for embedding model '{model}'. Set OPENAI_EMBEDDING_DIMENSIONS."
            )
        return native
    if native is not None and model not in _SUPPORTS_DIMENSIONS and override != native:
        raise EmbeddingConfigurationError(f"Embedding model '{model}' does not support custom dimensions.")
    if native is not None and override > native:
        raise EmbeddingConfigurationError(
            f"OPENAI_EMBEDDING_DIMENSIONS={override} exceeds the native dimension {native} of '{model}'."
        )
    return override


class OpenAIEmbeddingProvider:
    name = "openai"

    def __init__(
        self,
        *,
        api_key: SecretStr | None,
        model: str,
        dimensions: int | None,
        batch_size: int,
        timeout_seconds: float,
        max_retries: int,
        client: Any | None = None,
    ) -> None:
        self.model = model
        self.dimension = resolve_dimension(model, dimensions)
        self._send_dimensions = dimensions is not None and model in _SUPPORTS_DIMENSIONS
        self._batch_size = batch_size
        self._api_key = api_key
        self._timeout = timeout_seconds
        self._max_retries = max_retries
        self._client = client

    def _get_client(self) -> Any:
        if self._client is None:
            if self._api_key is None:
                raise EmbeddingConfigurationError(
                    "The embedding provider is not configured: OPENAI_API_KEY is missing."
                )
            self._client = openai.OpenAI(
                api_key=self._api_key.get_secret_value(),
                timeout=self._timeout,
                max_retries=self._max_retries,
            )
        return self._client

    def embed(self, texts: Sequence[str]) -> EmbeddingBatch:
        if not texts:
            return EmbeddingBatch(vectors=[], model=self.model, input_tokens=0)
        client = self._get_client()
        vectors: list[list[float]] = []
        tokens = 0
        for start in range(0, len(texts), self._batch_size):
            batch = list(texts[start : start + self._batch_size])
            response = self._create(client, batch)
            vectors.extend(self._vectors(response, len(batch)))
            usage = getattr(response, "usage", None)
            tokens += int(getattr(usage, "prompt_tokens", 0) or 0)
        return EmbeddingBatch(vectors=vectors, model=self.model, input_tokens=tokens)

    def _create(self, client: Any, batch: list[str]) -> Any:
        kwargs: dict[str, Any] = {"model": self.model, "input": batch}
        if self._send_dimensions:
            kwargs["dimensions"] = self.dimension
        try:
            return client.embeddings.create(**kwargs)
        except openai.APITimeoutError as exc:
            raise EmbeddingProviderError("The embedding provider timed out.") from exc
        except (openai.AuthenticationError, openai.PermissionDeniedError) as exc:
            raise EmbeddingConfigurationError(
                "The embedding provider rejected the configured credentials."
            ) from exc
        except openai.NotFoundError as exc:
            raise EmbeddingConfigurationError(
                f"Embedding model '{self.model}' is not available."
            ) from exc
        except openai.RateLimitError as exc:
            raise EmbeddingProviderError("The embedding provider is rate limited.") from exc
        except openai.APIStatusError as exc:
            logger.error("openai_embedding_error", extra={"status": exc.status_code})
            raise EmbeddingProviderError() from exc
        except openai.APIConnectionError as exc:
            raise EmbeddingProviderError("Could not reach the embedding provider.") from exc

    def _vectors(self, response: Any, expected: int) -> list[list[float]]:
        data = getattr(response, "data", None)
        if not isinstance(data, list) or len(data) != expected:
            raise EmbeddingProviderError("The embedding provider returned an unexpected response.")
        vectors = [list(item.embedding) for item in sorted(data, key=lambda item: item.index)]
        for vector in vectors:
            if len(vector) != self.dimension:
                raise EmbeddingDimensionError(
                    f"Expected {self.dimension}-dimensional embeddings from '{self.model}', "
                    f"got {len(vector)}."
                )
        return vectors
