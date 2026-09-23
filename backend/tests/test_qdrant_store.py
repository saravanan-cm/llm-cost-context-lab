"""QdrantVectorStore tests using qdrant-client's in-process ``:memory:`` mode (no server)."""

from unittest.mock import MagicMock

import pytest
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import ResponseHandlingException

from app.core.errors import (
    KnowledgeBaseNotReadyError,
    VectorStoreConfigurationError,
    VectorStoreUnavailableError,
)
from app.knowledge.qdrant_store import QdrantVectorStore
from tests.knowledge_fakes import HashingEmbedder, make_chunks

DOC = "wikipedia:en:1"


@pytest.fixture
def embedder() -> HashingEmbedder:
    return HashingEmbedder()


@pytest.fixture
def store(embedder) -> QdrantVectorStore:
    store = QdrantVectorStore(QdrantClient(":memory:"), "test_collection")
    store.ensure_collection(embedder.dimension)
    return store


def _upsert(store, embedder, texts, doc=DOC, title="Apache Kafka", fingerprint="fp1"):
    chunks = make_chunks(doc, title, texts)
    return store.upsert_document(doc, chunks, embedder.embed(texts).vectors, fingerprint)


def test_upsert_and_search_returns_payload(store, embedder):
    _upsert(store, embedder, ["Kafka consumer groups share partitions", "Redis is an in-memory store"])

    results = store.search(embedder.embed(["kafka consumer groups"]).vectors[0], top_k=2)

    top = results[0]
    assert top.text == "Kafka consumer groups share partitions"
    assert top.article_title == "Apache Kafka"
    assert top.source == "wikipedia"
    assert top.source_url.endswith("Apache_Kafka")
    assert top.document_id == DOC and top.chunk_index == 0
    assert top.metadata == {"language": "en"}
    assert results[0].score > results[1].score


def test_reingest_same_document_does_not_duplicate(store, embedder):
    _upsert(store, embedder, ["a b", "c d", "e f"])
    _upsert(store, embedder, ["a b", "c d", "e f"])

    assert store.count() == 3


def test_reingest_with_fewer_chunks_removes_stale_chunks(store, embedder):
    _upsert(store, embedder, ["a b", "c d", "e f"])
    _upsert(store, embedder, ["a b"], fingerprint="fp2")

    assert store.count() == 1
    assert store.get_document_fingerprint(DOC) == "fp2"


def test_fingerprint_absent_for_unknown_document(store):
    assert store.get_document_fingerprint("wikipedia:en:missing") is None


def test_existing_collection_with_other_dimension_is_rejected(store):
    with pytest.raises(VectorStoreConfigurationError, match="--recreate"):
        store.ensure_collection(128)


def test_query_vector_dimension_mismatch_is_rejected(store):
    with pytest.raises(VectorStoreConfigurationError):
        store.search([0.1] * 10, top_k=1)


def test_search_before_ingestion_reports_not_ready():
    store = QdrantVectorStore(QdrantClient(":memory:"), "never_created")

    with pytest.raises(KnowledgeBaseNotReadyError):
        store.search([0.1] * 64, top_k=1)


def test_recreate_drops_existing_points(store, embedder):
    _upsert(store, embedder, ["a b"])
    store.recreate(embedder.dimension)

    assert store.count() == 0


def test_unreachable_qdrant_maps_to_unavailable():
    client = MagicMock()
    client.collection_exists.side_effect = ResponseHandlingException(ConnectionError("refused"))
    client.collection_exists.__name__ = "collection_exists"
    store = QdrantVectorStore(client, "kb")

    with pytest.raises(VectorStoreUnavailableError):
        store.ensure_collection(64)
