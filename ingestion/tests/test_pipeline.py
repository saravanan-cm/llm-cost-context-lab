"""End-to-end pipeline tests with a fake source, fake embedder and in-memory Qdrant."""

import pytest
from qdrant_client import QdrantClient

from app.knowledge.qdrant_store import QdrantVectorStore
from app.knowledge.retriever import KnowledgeRetriever
from ingestion.chunking import TextChunker
from ingestion.cleaning import TextCleaner
from ingestion.pipeline import IngestionPipeline
from tests.fakes import FakeSource, HashingEmbedder

KAFKA_TEXT = "\n\n".join(
    [
        "Apache Kafka is a distributed event streaming platform.",
        "== Consumers ==",
        "A consumer group is a set of consumers that cooperate to consume a topic. "
        "Each partition is consumed by exactly one consumer in the group.",
        "== References ==",
        "^ citation needed",
    ]
)
REDIS_TEXT = "Redis is an in-memory key-value store. It is often used as a cache and message broker."
ARTICLES = {"Apache Kafka": (1, KAFKA_TEXT), "Redis": (2, REDIS_TEXT), "Empty": (3, "== References ==\n^ x")}


@pytest.fixture
def store() -> QdrantVectorStore:
    return QdrantVectorStore(QdrantClient(":memory:"), "kb")


def make_pipeline(store, embedder=None, articles=ARTICLES, force=False, dry_run=False):
    embedder = embedder or HashingEmbedder()
    return IngestionPipeline(
        source=FakeSource(dict(articles)),
        cleaner=TextCleaner(),
        chunker=TextChunker(120, 20),
        embedder=None if dry_run else embedder,
        store=None if dry_run else store,
        force=force,
    )


def test_ingests_documents_into_vector_store(store):
    report = make_pipeline(store).run(["Apache Kafka", "Redis"])

    assert report.requested == 2 and report.fetched == 2 and report.ingested == 2
    assert report.chunks_created == report.embeddings_generated == report.vectors_stored
    assert report.vectors_stored == store.count() > 2
    assert report.failures == []


def test_ingested_content_is_retrievable(store):
    embedder = HashingEmbedder()
    make_pipeline(store, embedder).run(["Apache Kafka", "Redis"])

    # The bag-of-words fake embedder needs lexical overlap; real embeddings are semantic.
    result = KnowledgeRetriever(embedder, store).search("consumer group of consumers consume a topic", 1)

    top = result.chunks[0]
    assert top.article_title == "Apache Kafka"
    assert "consumer group" in top.text
    all_text = [p.payload["text"] for p in store._client.scroll("kb", limit=100)[0]]
    assert not any("citation" in text for text in all_text)  # cleaned before chunking


def test_reingesting_unchanged_documents_is_skipped(store):
    make_pipeline(store).run(["Apache Kafka", "Redis"])
    count = store.count()
    embedder = HashingEmbedder()

    report = make_pipeline(store, embedder).run(["Apache Kafka", "Redis"])

    assert report.unchanged == 2 and report.ingested == 0
    assert embedder.calls == []  # no embedding cost on re-run
    assert store.count() == count


def test_force_reembeds_without_duplicating(store):
    make_pipeline(store).run(["Apache Kafka"])
    count = store.count()

    report = make_pipeline(store, force=True).run(["Apache Kafka"])

    assert report.ingested == 1 and report.embeddings_generated == count
    assert store.count() == count


def test_changed_document_is_reingested(store):
    make_pipeline(store).run(["Redis"])

    changed = {**ARTICLES, "Redis": (2, REDIS_TEXT + " It supports persistence.")}
    report = make_pipeline(store, articles=changed).run(["Redis"])

    assert report.ingested == 1 and report.unchanged == 0


def test_duplicate_titles_in_one_run_are_ingested_once(store):
    articles = {**ARTICLES, "Kafka (software)": ARTICLES["Apache Kafka"]}

    report = make_pipeline(store, articles=articles).run(["Apache Kafka", "Kafka (software)"])

    assert report.ingested == 1 and report.duplicates == 1


def test_partial_failure_continues_and_is_reported(store):
    report = make_pipeline(store).run(["Missing Article", "Empty", "Redis"])

    assert report.ingested == 1
    assert [f.reference for f in report.failures] == ["Missing Article", "Empty"]
    assert "not found" in report.failures[0].reason
    assert "no usable text" in report.failures[1].reason
    assert report.aborted is None


def test_fatal_error_aborts_remaining_documents(store):
    report = make_pipeline(store, HashingEmbedder(fail=True)).run(["Apache Kafka", "Redis"])

    assert report.aborted is not None
    assert [f.reference for f in report.failures] == ["Apache Kafka", "Redis"]
    assert report.failures[1].reason == "not attempted"
    assert store.count() == 0


def test_dry_run_chunks_without_embedding_or_storage(store):
    report = make_pipeline(store, dry_run=True).run(["Apache Kafka", "Redis"])

    assert report.chunks_created > 0
    assert report.embeddings_generated == report.vectors_stored == 0
    assert not store._client.collection_exists("kb")
