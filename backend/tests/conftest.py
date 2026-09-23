from collections.abc import Callable, Iterator
from decimal import Decimal

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.core.config import Settings, get_settings
from app.db.models import Base
from app.api.deps import get_retriever
from app.db.session import get_db
from app.knowledge.factory import get_embedding_provider, get_vector_store
from app.knowledge.models import RetrievedChunk
from app.knowledge.retriever import RetrievalResult
from app.main import create_app
from app.services.llm import LLMResult, LLMUsage, get_llm_provider
from app.services.usage_service import UsageService

DEV_USER = "dev-user"


class FakeProvider:
    """Test double for LLMProvider. Never makes network calls."""

    name = "fake"

    def __init__(self, model: str = "gpt-4.1-mini", input_tokens: int = 120, output_tokens: int = 180):
        self.model = model
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.error: Exception | None = None
        self.calls: list[str] = []
        self.instructions: list[str | None] = []
        self.events: list[str] | None = None  # shared call log for ordering assertions

    def generate(
        self, message: str, *, max_output_tokens: int, instructions: str | None = None
    ) -> LLMResult:
        self.calls.append(message)
        self.instructions.append(instructions)
        if self.events is not None:
            self.events.append("llm")
        if self.error is not None:
            raise self.error
        return LLMResult(
            text=f"reply to: {message}",
            model=self.model,
            usage=LLMUsage(self.input_tokens, self.output_tokens, self.input_tokens + self.output_tokens),
            provider=self.name,
            request_id="resp_test",
        )


class FakeRetriever:
    """Test double for KnowledgeRetriever returning preset chunks."""

    def __init__(self, chunks: list[RetrievedChunk] | None = None) -> None:
        self.chunks = chunks or []
        self.error: Exception | None = None
        self.calls: list[tuple[str, int]] = []
        self.events: list[str] | None = None

    def search(self, query: str, top_k: int = 5) -> RetrievalResult:
        self.calls.append((query, top_k))
        if self.events is not None:
            self.events.append("retrieve")
        if self.error is not None:
            raise self.error
        return RetrievalResult(
            chunks=self.chunks[:top_k], embedding_model="fake-embedding", query_tokens=7, latency_ms=12
        )


def make_chunk(title: str, index: int, score: float, text: str | None = None) -> RetrievedChunk:
    slug = title.replace(" ", "_")
    return RetrievedChunk(
        text=text or f"{title} chunk {index} text.",
        score=score,
        chunk_id=f"{slug}-{index}",
        document_id=f"wikipedia:en:{slug}",
        chunk_index=index,
        source="wikipedia",
        article_title=title,
        source_url=f"https://en.wikipedia.org/wiki/{slug}",
    )


def _unexpected_real_dependency():
    raise AssertionError("test attempted to use a real embedding provider / vector store")


@pytest.fixture
def settings() -> Settings:
    return Settings(
        _env_file=None,
        llm_provider="mock",
        database_url="sqlite://",
        db_auto_migrate=False,
        credits_per_dollar=Decimal("1000"),
        dev_user_id=DEV_USER,
        dev_user_initial_credits=Decimal("1000"),
        llm_max_output_tokens=1024,
        # Step 2 tests exercise the direct (non-RAG) path; RAG tests opt in via `rag_settings`.
        rag_enabled=False,
        rag_top_k=5,
        rag_min_relevance_score=0.3,
        rag_debug=False,
    )


@pytest.fixture
def session_factory(settings: Settings) -> Iterator[sessionmaker[Session]]:
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    with factory() as session:
        UsageService(session).ensure_user(DEV_USER, settings.dev_user_initial_credits)
    yield factory
    engine.dispose()


@pytest.fixture
def provider() -> FakeProvider:
    return FakeProvider()


@pytest.fixture
def retriever() -> FakeRetriever:
    return FakeRetriever()


@pytest.fixture
def client(
    settings: Settings,
    session_factory: sessionmaker[Session],
    provider: FakeProvider,
    retriever: FakeRetriever,
) -> Iterator[TestClient]:
    app = create_app(settings, run_startup=False)

    def _db() -> Iterator[Session]:
        with session_factory() as session:
            yield session

    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_db] = _db
    app.dependency_overrides[get_llm_provider] = lambda: provider
    app.dependency_overrides[get_retriever] = lambda: retriever
    # Guard: never touch real OpenAI embeddings or Qdrant from unit tests.
    app.dependency_overrides[get_embedding_provider] = _unexpected_real_dependency
    app.dependency_overrides[get_vector_store] = _unexpected_real_dependency
    with TestClient(app, raise_server_exceptions=False) as test_client:
        yield test_client


@pytest.fixture
def set_credits(session_factory: sessionmaker[Session]) -> Callable[[Decimal], None]:
    from app.db.models import User

    def _set(total: Decimal) -> None:
        with session_factory() as session:
            user = session.get(User, DEV_USER)
            assert user is not None
            user.total_credits = total
            session.commit()

    return _set
