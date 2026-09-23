"""Live integration test: Wikipedia + OpenAI embeddings + a running Qdrant.

Opt-in only:  RUN_INTEGRATION=1 python -m pytest -m integration   (from ingestion/)
Uses a throwaway collection, deleted afterwards. Costs a fraction of a cent in embeddings.
"""

import os
import uuid

import pytest

from app.core.config import get_settings
from app.knowledge.factory import build_embedding_provider, build_vector_store
from app.knowledge.retriever import KnowledgeRetriever
from ingestion.chunking import TextChunker
from ingestion.cleaning import TextCleaner
from ingestion.pipeline import IngestionPipeline
from ingestion.sources.wikipedia import WikipediaSource

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(os.getenv("RUN_INTEGRATION") != "1", reason="set RUN_INTEGRATION=1 to run"),
]


def test_ingest_and_retrieve_kafka():
    settings = get_settings()
    embedder = build_embedding_provider(settings)
    store = build_vector_store(settings, f"it_{uuid.uuid4().hex[:8]}")
    source = WikipediaSource(language="en", user_agent="LLMCostContextLab/0.3 (integration test)")
    try:
        report = IngestionPipeline(
            source=source,
            cleaner=TextCleaner(),
            chunker=TextChunker(1200, 200),
            embedder=embedder,
            store=store,
        ).run(["Apache Kafka", "Redis"])

        assert report.failures == []
        assert store.count() == report.vectors_stored > 0

        result = KnowledgeRetriever(embedder, store).search("How does a Kafka consumer group work?", 3)
        assert result.chunks[0].article_title == "Apache Kafka"
    finally:
        source.close()
        store._client.delete_collection(store.collection)
