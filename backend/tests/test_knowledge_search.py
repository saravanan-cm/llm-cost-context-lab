"""Retriever + /api/v1/knowledge/search tests (fake embedder, in-memory Qdrant)."""

import pytest
from qdrant_client import QdrantClient

from app.api.deps import get_retriever
from app.core.errors import VectorStoreUnavailableError
from app.knowledge.qdrant_store import QdrantVectorStore
from app.knowledge.retriever import KnowledgeRetriever
from tests.knowledge_fakes import HashingEmbedder, make_chunks

KAFKA = [
    "A Kafka consumer group lets multiple consumers share the partitions of a topic.",
    "Kafka brokers store records in partitioned, replicated logs.",
]
REDIS = ["Redis is an in-memory key-value data store often used as a cache."]


@pytest.fixture
def embedder() -> HashingEmbedder:
    return HashingEmbedder()


@pytest.fixture
def store(embedder) -> QdrantVectorStore:
    store = QdrantVectorStore(QdrantClient(":memory:"), "kb")
    store.ensure_collection(embedder.dimension)
    for doc, title, texts in [("wikipedia:en:1", "Apache Kafka", KAFKA), ("wikipedia:en:2", "Redis", REDIS)]:
        store.upsert_document(doc, make_chunks(doc, title, texts), embedder.embed(texts).vectors, "fp")
    return store


@pytest.fixture
def api(client, embedder, store):
    client.app.dependency_overrides[get_retriever] = lambda: KnowledgeRetriever(embedder, store)
    return client


def test_retriever_returns_most_relevant_chunks(embedder, store):
    result = KnowledgeRetriever(embedder, store).search("How does a Kafka consumer group work?", top_k=2)

    assert [c.article_title for c in result.chunks] == ["Apache Kafka", "Apache Kafka"]
    assert "consumer group" in result.chunks[0].text
    assert result.embedding_model == "fake-embedding"
    assert embedder.calls[-1] == ["How does a Kafka consumer group work?"]


@pytest.mark.parametrize(("query", "top_k"), [("  ", 5), ("kafka", 0), ("kafka", 21)])
def test_retriever_validates_input(embedder, store, query, top_k):
    with pytest.raises(ValueError):
        KnowledgeRetriever(embedder, store).search(query, top_k)


def test_search_endpoint_returns_chunks_with_metadata(api):
    response = api.post("/api/v1/knowledge/search", json={"query": "Kafka consumer group", "top_k": 3})

    assert response.status_code == 200
    body = response.json()
    assert body["top_k"] == 3
    assert len(body["results"]) == 3
    top = body["results"][0]
    assert top["article_title"] == "Apache Kafka"
    assert top["source"] == "wikipedia"
    assert top["source_url"] == "https://en.wikipedia.org/wiki/Apache_Kafka"
    assert 0 < top["score"] <= 1
    assert {"text", "chunk_id", "document_id", "chunk_index", "metadata"} <= top.keys()


def test_search_endpoint_does_not_call_llm(api, provider):
    api.post("/api/v1/knowledge/search", json={"query": "Redis cache"})

    assert provider.calls == []


@pytest.mark.parametrize("body", [{"query": ""}, {"query": "x", "top_k": 0}, {"query": "x", "top_k": 50}])
def test_search_endpoint_validates_request(api, body):
    assert api.post("/api/v1/knowledge/search", json=body).status_code == 422


def test_search_before_ingestion_returns_clean_503(client, embedder):
    empty = QdrantVectorStore(QdrantClient(":memory:"), "empty")
    client.app.dependency_overrides[get_retriever] = lambda: KnowledgeRetriever(embedder, empty)

    response = client.post("/api/v1/knowledge/search", json={"query": "kafka"})

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "knowledge_base_not_ready"


def test_qdrant_down_returns_clean_503(client, embedder):
    class DownStore:
        def search(self, vector, top_k):
            raise VectorStoreUnavailableError()

    client.app.dependency_overrides[get_retriever] = lambda: KnowledgeRetriever(embedder, DownStore())

    response = client.post("/api/v1/knowledge/search", json={"query": "kafka"})

    assert response.status_code == 503
    assert response.json()["error"] == {
        "code": "vector_store_unavailable",
        "message": "The knowledge base is currently unavailable.",
    }
