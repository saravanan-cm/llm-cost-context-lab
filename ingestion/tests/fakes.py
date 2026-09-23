"""Test doubles for ingestion tests. No network calls."""

import hashlib
import math
import re
from collections.abc import Sequence
from datetime import UTC, datetime

from app.core.errors import EmbeddingConfigurationError
from app.knowledge.embeddings import EmbeddingBatch
from app.knowledge.models import Document
from ingestion.errors import DocumentNotFoundError

FETCHED_AT = datetime(2026, 1, 1, tzinfo=UTC)


class HashingEmbedder:
    """Deterministic bag-of-words embedding: similar wording -> similar vectors."""

    name = "fake"
    model = "fake-embedding"

    def __init__(self, dimension: int = 64, fail: bool = False) -> None:
        self.dimension = dimension
        self.fail = fail
        self.calls: list[list[str]] = []

    def embed(self, texts: Sequence[str]) -> EmbeddingBatch:
        if self.fail:
            raise EmbeddingConfigurationError("OPENAI_API_KEY is missing.")
        self.calls.append(list(texts))
        return EmbeddingBatch(
            vectors=[self._vector(t) for t in texts],
            model=self.model,
            input_tokens=sum(len(t.split()) for t in texts),
        )

    def _vector(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        for word in re.findall(r"[a-z0-9]+", text.lower()):
            vector[int(hashlib.md5(word.encode()).hexdigest(), 16) % self.dimension] += 1.0
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        return [v / norm for v in vector]


class FakeSource:
    """In-memory KnowledgeSource keyed by title."""

    name = "wikipedia"

    def __init__(self, articles: dict[str, tuple[int, str]]) -> None:
        self.articles = articles  # title -> (page_id, text)

    def fetch(self, reference: str) -> Document:
        if reference not in self.articles:
            raise DocumentNotFoundError(f"Wikipedia article not found: '{reference}'")
        page_id, text = self.articles[reference]
        return Document(
            document_id=f"wikipedia:en:{page_id}",
            text=text,
            title=reference,
            source="wikipedia",
            source_url=f"https://en.wikipedia.org/wiki/{reference.replace(' ', '_')}",
            language="en",
            fetched_at=FETCHED_AT,
            metadata={"page_id": page_id},
        )
