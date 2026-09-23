"""Provider-agnostic embedding interface."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class EmbeddingBatch:
    vectors: list[list[float]]
    """One vector per input text, in input order."""
    model: str
    input_tokens: int
    """Provider-reported tokens across all API calls (for cost tracking)."""


class EmbeddingProvider(Protocol):
    name: str
    model: str
    dimension: int
    """Length of every returned vector. Used to size the vector collection."""

    def embed(self, texts: Sequence[str]) -> EmbeddingBatch:
        """Embed texts, batching API calls internally.

        Raises ``EmbeddingConfigurationError``, ``EmbeddingProviderError`` or
        ``EmbeddingDimensionError``.
        """
        ...
