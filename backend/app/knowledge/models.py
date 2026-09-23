"""Source- and store-agnostic knowledge models.

Nothing here knows about Wikipedia, OpenAI or Qdrant.
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

# Fixed namespace so chunk IDs are deterministic across runs (enables idempotent upserts).
_CHUNK_NAMESPACE = uuid.UUID("0b6c6f3e-5a0e-4f5e-9d0a-6c1c2b7f4a11")


def make_chunk_id(document_id: str, chunk_index: int) -> str:
    return str(uuid.uuid5(_CHUNK_NAMESPACE, f"{document_id}#{chunk_index}"))


@dataclass(frozen=True)
class Document:
    document_id: str
    """Stable, source-scoped ID, e.g. ``wikipedia:en:23862``."""
    text: str
    title: str
    source: str
    source_url: str
    language: str
    fetched_at: datetime
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    document_id: str
    chunk_index: int
    text: str
    source: str
    article_title: str
    source_url: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RetrievedChunk:
    text: str
    score: float
    """Cosine similarity (higher is more similar)."""
    chunk_id: str
    document_id: str
    chunk_index: int
    source: str
    article_title: str
    source_url: str
    metadata: dict[str, Any] = field(default_factory=dict)
