"""Live RAG check: real Qdrant (ingested knowledge base) + OpenAI embeddings + OpenAI LLM.

Opt-in only (costs ~$0.001):   RUN_INTEGRATION=1 python -m pytest tests/integration
Prerequisites: `docker compose up -d qdrant`, `python -m ingestion.ingest`, OPENAI_API_KEY set.
Uses a throwaway SQLite database, so real credits in app.db are untouched.
"""

import os
from decimal import Decimal

import pytest
from fastapi.testclient import TestClient

from app.core.config import get_settings
from app.db.session import get_engine, get_sessionmaker
from app.knowledge.factory import get_embedding_provider, get_vector_store
from app.main import create_app
from app.services.llm.factory import get_llm_provider

pytestmark = pytest.mark.skipif(os.getenv("RUN_INTEGRATION") != "1", reason="set RUN_INTEGRATION=1 to run")

CACHED = (get_settings, get_engine, get_sessionmaker, get_llm_provider, get_embedding_provider, get_vector_store)


@pytest.fixture
def live_client(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'it.db'}")
    monkeypatch.setenv("RAG_ENABLED", "true")
    monkeypatch.setenv("RAG_DEBUG", "true")
    for cached in CACHED:
        cached.cache_clear()
    with TestClient(create_app()) as client:  # lifespan: migrate + seed dev user
        yield client
    for cached in CACHED:
        cached.cache_clear()
    get_engine().dispose()


def test_kafka_question_is_grounded_metered_and_charged(live_client):
    before = live_client.get("/api/v1/usage").json()

    response = live_client.post(
        "/api/v1/chat", json={"conversation_id": "it", "message": "How does a Kafka consumer group work?"}
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["answer_type"] == "grounded"
    assert body["debug"]["graph_path"] == [
        "analyze_question", "precheck_credits", "retrieve_knowledge", "build_context",
        "check_credits", "generate_answer", "usage_accounting",
    ]
    assert body["sources"][0]["title"] == "Apache Kafka"
    assert body["usage"]["input_tokens"] > 500  # retrieved context is part of the input
    assert Decimal(body["cost"]["total_cost"]) > 0
    after = live_client.get("/api/v1/usage").json()
    assert Decimal(after["credits_remaining"]) < Decimal(before["credits_remaining"])
    assert after["total_tokens"] == before["total_tokens"] + body["usage"]["total_tokens"]


def test_out_of_scope_question_is_handled_without_llm(live_client):
    body = live_client.post(
        "/api/v1/chat", json={"conversation_id": "it", "message": "What is quantum mechanics?"}
    ).json()

    assert body["answer_type"] == "no_context"
    assert body["debug"]["graph_path"][-1] == "no_knowledge"
    assert body["model"] is None and body["sources"] == []
    assert body["credits"]["consumed"] == "0"
    assert all(not chunk["used"] for chunk in body["debug"]["chunks"])
