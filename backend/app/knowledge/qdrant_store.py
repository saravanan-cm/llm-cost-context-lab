"""Qdrant implementation of ``VectorStore``. The only module that imports qdrant_client.

Point layout: id = deterministic chunk UUID; vector = embedding (cosine distance);
payload = chunk text + flat metadata fields (document_id, chunk_id, chunk_index, source,
article_title, source_url, fingerprint) + a nested ``metadata`` object.
"""

import logging
from collections.abc import Callable, Sequence
from typing import Any, TypeVar

from pydantic import SecretStr
from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

from app.core.errors import (
    KnowledgeBaseNotReadyError,
    VectorStoreConfigurationError,
    VectorStoreUnavailableError,
)
from app.knowledge.models import Chunk, RetrievedChunk

logger = logging.getLogger(__name__)

T = TypeVar("T")

_UPSERT_BATCH_SIZE = 256
_KEYWORD_INDEXES = ("document_id", "source")


class QdrantVectorStore:
    def __init__(self, client: QdrantClient, collection: str) -> None:
        self._client = client
        self.collection = collection
        self._dimension: int | None = None

    @classmethod
    def from_url(
        cls, url: str, collection: str, *, api_key: SecretStr | None = None, timeout: int = 10
    ) -> "QdrantVectorStore":
        # check_compatibility=False: don't hit the server at construction time.
        client = QdrantClient(
            url=url,
            api_key=api_key.get_secret_value() if api_key else None,
            timeout=timeout,
            check_compatibility=False,
        )
        return cls(client, collection)

    # -- VectorStore -------------------------------------------------------------------

    def ensure_collection(self, dimension: int) -> None:
        if self._call(self._client.collection_exists, self.collection):
            self._check_dimension(self._collection_dimension(), dimension)
            return
        self._call(
            self._client.create_collection,
            collection_name=self.collection,
            vectors_config=models.VectorParams(size=dimension, distance=models.Distance.COSINE),
        )
        for field in _KEYWORD_INDEXES:
            self._call(
                self._client.create_payload_index,
                collection_name=self.collection,
                field_name=field,
                field_schema=models.PayloadSchemaType.KEYWORD,
            )
        self._call(
            self._client.create_payload_index,
            collection_name=self.collection,
            field_name="chunk_index",
            field_schema=models.PayloadSchemaType.INTEGER,
        )
        self._dimension = dimension
        logger.info("qdrant_collection_created", extra={"collection": self.collection, "dimension": dimension})

    def get_document_fingerprint(self, document_id: str) -> str | None:
        points, _ = self._call(
            self._client.scroll,
            collection_name=self.collection,
            scroll_filter=_document_filter(document_id),
            limit=1,
            with_payload=["fingerprint"],
            with_vectors=False,
        )
        if not points or not points[0].payload:
            return None
        return points[0].payload.get("fingerprint")

    def upsert_document(
        self,
        document_id: str,
        chunks: Sequence[Chunk],
        vectors: Sequence[Sequence[float]],
        fingerprint: str,
    ) -> int:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")
        points = [
            models.PointStruct(id=chunk.chunk_id, vector=list(vector), payload=_payload(chunk, fingerprint))
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]
        # Upsert first (deterministic IDs overwrite in place), then remove chunks beyond the
        # new count, so the document is never missing from search mid-update.
        for start in range(0, len(points), _UPSERT_BATCH_SIZE):
            self._call(
                self._client.upsert,
                collection_name=self.collection,
                points=points[start : start + _UPSERT_BATCH_SIZE],
                wait=True,
            )
        self._call(
            self._client.delete,
            collection_name=self.collection,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        *_document_filter(document_id).must,  # type: ignore[misc]
                        models.FieldCondition(key="chunk_index", range=models.Range(gte=len(points))),
                    ]
                )
            ),
            wait=True,
        )
        return len(points)

    def search(self, vector: Sequence[float], top_k: int) -> list[RetrievedChunk]:
        self._check_dimension(self._collection_dimension(), len(vector))
        response = self._call(
            self._client.query_points,
            collection_name=self.collection,
            query=list(vector),
            limit=top_k,
            with_payload=True,
        )
        return [_to_retrieved(point) for point in response.points]

    def count(self) -> int:
        return self._call(self._client.count, collection_name=self.collection, exact=True).count

    # -- helpers ------------------------------------------------------------------------

    def _collection_dimension(self) -> int:
        if self._dimension is None:
            if not self._call(self._client.collection_exists, self.collection):
                raise KnowledgeBaseNotReadyError()
            info = self._call(self._client.get_collection, self.collection)
            vectors = info.config.params.vectors
            if not isinstance(vectors, models.VectorParams):
                raise VectorStoreConfigurationError(
                    f"Collection '{self.collection}' does not use a single unnamed vector."
                )
            self._dimension = vectors.size
        return self._dimension

    def _check_dimension(self, collection_dimension: int, dimension: int) -> None:
        if collection_dimension != dimension:
            raise VectorStoreConfigurationError(
                f"Collection '{self.collection}' stores {collection_dimension}-dimensional vectors but "
                f"the embedding model produces {dimension}. Use a different QDRANT_COLLECTION or "
                "re-ingest with --recreate."
            )

    def recreate(self, dimension: int) -> None:
        """Drop and recreate the collection (used by the ingestion CLI's --recreate)."""
        if self._call(self._client.collection_exists, self.collection):
            self._call(self._client.delete_collection, self.collection)
        self._dimension = None
        self.ensure_collection(dimension)

    def _call(self, fn: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        try:
            return fn(*args, **kwargs)
        except ResponseHandlingException as exc:
            logger.error("qdrant_unreachable", extra={"operation": fn.__name__, "error": type(exc).__name__})
            raise VectorStoreUnavailableError() from exc
        except UnexpectedResponse as exc:
            logger.error("qdrant_error", extra={"operation": fn.__name__, "status": exc.status_code})
            if exc.status_code == 404:
                raise KnowledgeBaseNotReadyError() from exc
            raise VectorStoreUnavailableError() from exc
        except (ConnectionError, TimeoutError, OSError) as exc:
            logger.error("qdrant_unreachable", extra={"operation": fn.__name__, "error": type(exc).__name__})
            raise VectorStoreUnavailableError() from exc


def _document_filter(document_id: str) -> models.Filter:
    return models.Filter(
        must=[models.FieldCondition(key="document_id", match=models.MatchValue(value=document_id))]
    )


def _payload(chunk: Chunk, fingerprint: str) -> dict[str, Any]:
    return {
        "text": chunk.text,
        "document_id": chunk.document_id,
        "chunk_id": chunk.chunk_id,
        "chunk_index": chunk.chunk_index,
        "source": chunk.source,
        "article_title": chunk.article_title,
        "source_url": chunk.source_url,
        "fingerprint": fingerprint,
        "metadata": chunk.metadata,
    }


def _to_retrieved(point: Any) -> RetrievedChunk:
    payload = point.payload or {}
    return RetrievedChunk(
        text=payload.get("text", ""),
        score=float(point.score),
        chunk_id=str(payload.get("chunk_id", point.id)),
        document_id=payload.get("document_id", ""),
        chunk_index=int(payload.get("chunk_index", 0)),
        source=payload.get("source", ""),
        article_title=payload.get("article_title", ""),
        source_url=payload.get("source_url", ""),
        metadata=payload.get("metadata") or {},
    )
