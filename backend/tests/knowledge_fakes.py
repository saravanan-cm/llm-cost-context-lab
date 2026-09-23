"""Test doubles for knowledge components. No network calls."""

import hashlib
import math
import re
from collections.abc import Sequence

from app.knowledge.embeddings import EmbeddingBatch
from app.knowledge.models import Chunk, make_chunk_id


class HashingEmbedder:
    """Deterministic bag-of-words embedding: similar wording -> similar vectors."""

    name = "fake"
    model = "fake-embedding"

    def __init__(self, dimension: int = 64) -> None:
        self.dimension = dimension
        self.calls: list[list[str]] = []

    def embed(self, texts: Sequence[str]) -> EmbeddingBatch:
        self.calls.append(list(texts))
        return EmbeddingBatch(
            vectors=[self._vector(text) for text in texts],
            model=self.model,
            input_tokens=sum(len(text.split()) for text in texts),
        )

    def _vector(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        for word in re.findall(r"[a-z0-9]+", text.lower()):
            vector[int(hashlib.md5(word.encode()).hexdigest(), 16) % self.dimension] += 1.0
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        return [v / norm for v in vector]


def make_chunks(document_id: str, title: str, texts: list[str]) -> list[Chunk]:
    return [
        Chunk(
            chunk_id=make_chunk_id(document_id, i),
            document_id=document_id,
            chunk_index=i,
            text=text,
            source="wikipedia",
            article_title=title,
            source_url=f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}",
            metadata={"language": "en"},
        )
        for i, text in enumerate(texts)
    ]
