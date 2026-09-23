"""POST /api/v1/chat with RAG enabled (fake retriever / in-memory Qdrant, fake LLM)."""

from decimal import Decimal

import pytest
from qdrant_client import QdrantClient

from app.api.deps import get_retriever
from app.core.errors import KnowledgeBaseNotReadyError, LLMProviderError, VectorStoreUnavailableError
from app.knowledge.qdrant_store import QdrantVectorStore
from app.knowledge.retriever import KnowledgeRetriever
from tests.conftest import make_chunk
from tests.knowledge_fakes import HashingEmbedder, make_chunks

QUESTION = {"conversation_id": "c1", "message": "How does a Kafka consumer group work?"}
KAFKA_CHUNKS = [
    make_chunk("Apache Kafka", 1, 0.62, "A consumer group shares a topic's partitions among consumers."),
    make_chunk("Apache Kafka", 2, 0.61),
    make_chunk("Distributed computing", 4, 0.33),
    make_chunk("Redis", 3, 0.21),
]


@pytest.fixture
def settings(settings):
    return settings.model_copy(update={"rag_enabled": True})


@pytest.fixture
def retriever(retriever):
    retriever.chunks = list(KAFKA_CHUNKS)
    return retriever


def usage(client):
    return client.get("/api/v1/usage").json()


def test_grounded_answer_with_deduplicated_sources(client, provider):
    response = client.post("/api/v1/chat", json=QUESTION)

    assert response.status_code == 200
    body = response.json()
    assert body["answer_type"] == "grounded"
    assert body["assistant_message"].startswith("reply to:")
    assert body["model"] == "gpt-4.1-mini"
    assert body["sources"] == [
        {"title": "Apache Kafka", "source": "wikipedia", "url": "https://en.wikipedia.org/wiki/Apache_Kafka", "score": 0.62},
        {
            "title": "Distributed computing",
            "source": "wikipedia",
            "url": "https://en.wikipedia.org/wiki/Distributed_computing",
            "score": 0.33,
        },
    ]
    assert body["debug"] is None  # RAG_DEBUG off
    assert "consumer group shares a topic's partitions" in provider.calls[0]


def test_actual_provider_usage_is_metered(client, provider):
    provider.input_tokens, provider.output_tokens = 850, 220  # context makes input large

    body = client.post("/api/v1/chat", json=QUESTION).json()

    # gpt-4.1-mini: 850 * $0.40/1M + 220 * $1.60/1M
    assert body["usage"] == {"input_tokens": 850, "output_tokens": 220, "total_tokens": 1070}
    assert body["cost"]["input_cost"] == "0.00034"
    assert body["cost"]["output_cost"] == "0.000352"
    assert body["cost"]["total_cost"] == "0.000692"
    assert body["credits"] == {"consumed": "0.692", "remaining": "999.308"}
    summary = usage(client)
    assert (summary["total_input_tokens"], summary["total_output_tokens"]) == (850, 220)
    assert summary["total_cost"] == "0.000692"
    assert summary["credits_used"] == "0.692"
    assert summary["request_count"] == 1


def test_no_relevant_knowledge_returns_controlled_answer_without_llm(client, provider, retriever):
    retriever.chunks = [make_chunk("Redis", 0, 0.2)]

    body = client.post("/api/v1/chat", json={"conversation_id": "c1", "message": "What is quantum mechanics?"}).json()

    assert body["answer_type"] == "no_context"
    assert "knowledge base" in body["assistant_message"]
    assert body["model"] is None and body["sources"] == []
    assert body["usage"]["total_tokens"] == 0 and body["cost"]["total_cost"] == "0"
    assert body["credits"] == {"consumed": "0", "remaining": "1000"}
    assert provider.calls == []
    assert usage(client)["request_count"] == 0


@pytest.mark.parametrize(
    ("error", "status", "code"),
    [
        (VectorStoreUnavailableError(), 503, "vector_store_unavailable"),
        (KnowledgeBaseNotReadyError(), 503, "knowledge_base_not_ready"),
    ],
)
def test_retrieval_failure_prevents_llm_call_and_charges(client, provider, retriever, error, status, code):
    retriever.error = error

    response = client.post("/api/v1/chat", json=QUESTION)

    assert response.status_code == status
    assert response.json()["error"]["code"] == code
    assert provider.calls == []
    assert usage(client)["credits_used"] == "0"


def test_llm_failure_after_retrieval_does_not_deduct(client, provider, retriever):
    provider.error = LLMProviderError()

    response = client.post("/api/v1/chat", json=QUESTION)

    assert response.status_code == 502
    assert retriever.calls  # retrieval happened
    summary = usage(client)
    assert summary["credits_used"] == "0" and summary["request_count"] == 0


def test_no_credits_rejected_before_retrieval(client, provider, retriever, set_credits):
    set_credits(Decimal(0))

    assert client.post("/api/v1/chat", json=QUESTION).status_code == 402
    assert retriever.calls == [] and provider.calls == []


def test_credit_check_accounts_for_retrieved_context(client, provider, retriever, set_credits):
    # Enough for the bare question, not for question + ~24k chars of context.
    retriever.chunks = [make_chunk("Apache Kafka", i, 0.6, "x" * 6000) for i in range(4)]
    set_credits(Decimal("3"))  # bare question ~1.7 credits worst case; with context ~5

    response = client.post("/api/v1/chat", json=QUESTION)

    assert response.status_code == 402
    assert retriever.calls and provider.calls == []


def test_debug_mode_exposes_retrieval_details(client, settings, retriever):
    settings.rag_debug = True

    debug = client.post("/api/v1/chat", json=QUESTION).json()["debug"]

    assert debug["top_k"] == 5 and debug["min_relevance_score"] == 0.3
    assert debug["retrieval_latency_ms"] == 12
    assert [(c["article_title"], c["used"]) for c in debug["chunks"]] == [
        ("Apache Kafka", True),
        ("Apache Kafka", True),
        ("Distributed computing", True),
        ("Redis", False),
    ]
    assert debug["chunks"][0]["rank"] == 1 and debug["chunks"][0]["score"] == 0.62


def test_rag_with_in_memory_qdrant(client, provider):
    """Real KnowledgeRetriever + QdrantVectorStore (in-process) + deterministic embedder."""
    embedder = HashingEmbedder()
    store = QdrantVectorStore(QdrantClient(":memory:"), "kb")
    store.ensure_collection(embedder.dimension)
    texts = ["Kafka consumer group members split the partitions of a topic between them."]
    store.upsert_document("wikipedia:en:1", make_chunks("wikipedia:en:1", "Apache Kafka", texts), embedder.embed(texts).vectors, "fp")
    client.app.dependency_overrides[get_retriever] = lambda: KnowledgeRetriever(embedder, store)

    body = client.post("/api/v1/chat", json={"conversation_id": "c1", "message": "kafka consumer group partitions"}).json()

    assert body["answer_type"] == "grounded"
    assert body["sources"][0]["title"] == "Apache Kafka"
    assert "split the partitions" in provider.calls[0]


def test_response_carries_request_id(client):
    response = client.post("/api/v1/chat", json=QUESTION, headers={"X-Request-ID": "abc-123"})

    assert response.headers["x-request-id"] == "abc-123"
    assert len(client.get("/api/v1/health").headers["x-request-id"]) == 32  # generated


def test_blank_question_returns_clean_422(client, provider, retriever):
    response = client.post("/api/v1/chat", json={"conversation_id": "c1", "message": "   "})

    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_question", "message": "The question is empty."}}
    assert retriever.calls == [] and provider.calls == []


def test_debug_mode_exposes_graph_path(client, settings):
    settings.rag_debug = True

    debug = client.post("/api/v1/chat", json=QUESTION).json()["debug"]

    assert debug["graph_path"] == [
        "analyze_question",
        "precheck_credits",
        "retrieve_knowledge",
        "build_context",
        "check_credits",
        "generate_answer",
        "usage_accounting",
    ]
    assert debug["llm_latency_ms"] is not None


def test_graph_state_is_not_exposed_without_debug(client):
    body = client.post("/api/v1/chat", json=QUESTION).json()

    assert body["debug"] is None
    assert set(body) == {
        "conversation_id", "assistant_message", "answer_type", "model", "sources", "usage", "cost", "credits", "debug",
    }
