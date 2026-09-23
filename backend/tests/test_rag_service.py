"""RAGService unit tests with a fake retriever and fake LLM (no network)."""

import pytest

from app.core.errors import LLMTimeoutError, VectorStoreUnavailableError
from app.rag.context_builder import RAGContextBuilder
from app.rag.service import RAGService, RAGSettings
from tests.conftest import FakeProvider, FakeRetriever, make_chunk

QUESTION = "How does a Kafka consumer group work?"
KAFKA_CHUNKS = [
    make_chunk("Apache Kafka", 1, 0.62, "A consumer group shares a topic's partitions among consumers."),
    make_chunk("Apache Kafka", 2, 0.61),
    make_chunk("Distributed computing", 4, 0.33),
    make_chunk("Redis", 3, 0.21),  # below threshold
]


def make_service(retriever, llm=None, **settings):
    return RAGService(
        retriever=retriever,
        builder=RAGContextBuilder(),
        llm=llm or FakeProvider(),
        settings=RAGSettings(**{"top_k": 5, "min_relevance_score": 0.3, **settings}),
    )


def test_retrieves_with_configured_top_k():
    retriever = FakeRetriever(KAFKA_CHUNKS)

    make_service(retriever, top_k=3).prepare(QUESTION)

    assert retriever.calls == [(QUESTION, 3)]


def test_filters_by_relevance_and_deduplicates_sources():
    prepared = make_service(FakeRetriever(KAFKA_CHUNKS)).prepare(QUESTION)

    assert prepared.mode == "grounded"
    assert [s.title for s in prepared.sources] == ["Apache Kafka", "Distributed computing"]
    assert len(prepared.retrieved) == 4  # all kept for debugging
    assert "Redis" not in prepared.prompt.input
    assert "consumer group shares a topic's partitions" in prepared.prompt.input


def test_no_threshold_uses_all_chunks():
    prepared = make_service(FakeRetriever(KAFKA_CHUNKS), min_relevance_score=None).prepare(QUESTION)

    assert [s.title for s in prepared.sources][-1] == "Redis"


def test_calls_retriever_before_llm_and_passes_context():
    events: list[str] = []
    retriever, llm = FakeRetriever(KAFKA_CHUNKS), FakeProvider()
    retriever.events = llm.events = events
    service = make_service(retriever, llm)

    answer = service.generate(service.prepare(QUESTION), max_output_tokens=256)

    assert events == ["retrieve", "llm"]
    assert "consumer group shares a topic's partitions" in llm.calls[0]
    assert llm.calls[0].rstrip().endswith(f"Question: {QUESTION}")
    assert "knowledge context" in llm.instructions[0]
    assert answer.result.text.startswith("reply to:")
    assert answer.result.usage.input_tokens == 120


def test_no_relevant_chunks_returns_fixed_response_without_llm():
    llm = FakeProvider()
    service = make_service(FakeRetriever([make_chunk("Redis", 0, 0.2)]), llm, no_context_message="Nothing here.")

    prepared = service.prepare("What is quantum mechanics?")

    assert prepared.mode == "no_context"
    assert prepared.prompt is None and prepared.fixed_response == "Nothing here."
    assert prepared.sources == []
    with pytest.raises(ValueError):
        service.generate(prepared, max_output_tokens=256)
    assert llm.calls == []


def test_empty_retrieval_is_no_context():
    assert make_service(FakeRetriever([])).prepare("anything").mode == "no_context"


def test_no_context_llm_mode_builds_flagged_prompt():
    prepared = make_service(FakeRetriever([]), no_context_mode="llm").prepare("What is quantum mechanics?")

    assert prepared.mode == "no_context"
    assert prepared.prompt is not None
    assert "knowledge base does not contain information" in prepared.prompt.instructions


def test_retrieval_failure_propagates_before_llm():
    retriever, llm = FakeRetriever(KAFKA_CHUNKS), FakeProvider()
    retriever.error = VectorStoreUnavailableError()

    with pytest.raises(VectorStoreUnavailableError):
        make_service(retriever, llm).prepare(QUESTION)
    assert llm.calls == []


def test_llm_failure_propagates():
    llm = FakeProvider()
    llm.error = LLMTimeoutError()
    service = make_service(FakeRetriever(KAFKA_CHUNKS), llm)

    with pytest.raises(LLMTimeoutError):
        service.generate(service.prepare(QUESTION), max_output_tokens=256)


def test_disabled_rag_skips_retrieval():
    retriever = FakeRetriever(KAFKA_CHUNKS)

    prepared = make_service(retriever, enabled=False).prepare("Hi")

    assert prepared.mode == "direct"
    assert prepared.prompt.input == "Hi" and prepared.prompt.instructions is None
    assert retriever.calls == []
