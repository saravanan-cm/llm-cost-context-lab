"""Builds configured knowledge components. Shared by the API and the ingestion CLI."""

from functools import lru_cache

from app.core.config import Settings, get_settings
from app.knowledge.embeddings import EmbeddingProvider
from app.knowledge.openai_embeddings import OpenAIEmbeddingProvider
from app.knowledge.qdrant_store import QdrantVectorStore


def build_embedding_provider(settings: Settings) -> EmbeddingProvider:
    if settings.embedding_provider == "openai":
        return OpenAIEmbeddingProvider(
            api_key=settings.openai_api_key,
            model=settings.openai_embedding_model,
            dimensions=settings.openai_embedding_dimensions,
            batch_size=settings.embedding_batch_size,
            timeout_seconds=settings.openai_timeout_seconds,
            max_retries=settings.openai_max_retries,
        )
    raise ValueError(f"Unsupported embedding provider: {settings.embedding_provider}")


def build_vector_store(settings: Settings, collection: str | None = None) -> QdrantVectorStore:
    return QdrantVectorStore.from_url(
        settings.qdrant_url,
        collection or settings.qdrant_collection,
        api_key=settings.qdrant_api_key,
        timeout=settings.qdrant_timeout_seconds,
    )


@lru_cache
def get_embedding_provider() -> EmbeddingProvider:
    return build_embedding_provider(get_settings())


@lru_cache
def get_vector_store() -> QdrantVectorStore:
    return build_vector_store(get_settings())
