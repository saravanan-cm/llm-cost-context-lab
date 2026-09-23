"""Vendor-agnostic vector store interface."""

from collections.abc import Sequence
from typing import Protocol

from app.knowledge.models import Chunk, RetrievedChunk


class VectorStore(Protocol):
    def ensure_collection(self, dimension: int) -> None:
        """Create the collection if missing. Raise ``VectorStoreConfigurationError`` if it
        exists with a different dimension."""
        ...

    def get_document_fingerprint(self, document_id: str) -> str | None:
        """Fingerprint stored with a document's chunks, or None if not stored."""
        ...

    def upsert_document(
        self,
        document_id: str,
        chunks: Sequence[Chunk],
        vectors: Sequence[Sequence[float]],
        fingerprint: str,
    ) -> int:
        """Replace all stored chunks of a document. Returns the number of vectors written."""
        ...

    def search(self, vector: Sequence[float], top_k: int) -> list[RetrievedChunk]:
        ...

    def count(self) -> int:
        ...
