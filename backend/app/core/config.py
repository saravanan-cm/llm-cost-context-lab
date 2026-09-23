"""Application configuration loaded from environment variables / .env."""

from decimal import Decimal
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# backend/.env, resolved absolutely so the ingestion CLI (run from the repo root) shares it.
ENV_FILE = Path(__file__).resolve().parents[2] / ".env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ENV_FILE, env_file_encoding="utf-8", extra="ignore")

    app_name: str = "LLM Cost & Context Lab"
    app_version: str = "0.4.0"
    environment: Literal["local", "dev", "staging", "prod"] = "local"
    log_level: str = "INFO"
    log_json: bool = False
    api_v1_prefix: str = "/api/v1"
    cors_origins: list[str] = ["http://localhost:5173"]

    # LLM
    llm_provider: Literal["openai", "mock"] = "openai"
    openai_api_key: SecretStr | None = None
    openai_model: str = "gpt-4.1-mini"
    openai_timeout_seconds: float = Field(default=30.0, gt=0)
    openai_max_retries: int = Field(default=2, ge=0)
    llm_max_output_tokens: int = Field(default=1024, gt=0)
    system_prompt: str = "You are a helpful, concise assistant."

    # Database
    database_url: str = "sqlite:///./app.db"
    db_auto_migrate: bool = True

    # Credits. Conversion from USD cost to application credits (see docs/architecture.md).
    credits_per_dollar: Decimal = Field(default=Decimal("1000"), gt=0)
    dev_user_id: str = "dev-user"
    dev_user_initial_credits: Decimal = Field(default=Decimal("1000"), ge=0)

    # Knowledge base: embeddings + vector store (shared by the API and the ingestion CLI)
    embedding_provider: Literal["openai"] = "openai"
    openai_embedding_model: str = "text-embedding-3-small"
    # Optional reduced dimension (text-embedding-3-* only). Unset = the model's native dimension.
    openai_embedding_dimensions: int | None = Field(default=None, gt=0)
    embedding_batch_size: int = Field(default=100, gt=0, le=2048)
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: SecretStr | None = None
    qdrant_collection: str = Field(default="interview_knowledge", pattern=r"^[A-Za-z0-9_-]{1,255}$")
    qdrant_timeout_seconds: int = Field(default=10, gt=0)

    # RAG (Step 4). Relevance threshold is cosine similarity for the configured embedding
    # model; 0.30 was calibrated for text-embedding-3-small on the current knowledge base
    # (out-of-domain questions scored <= 0.27, in-domain >= 0.31). Blank disables filtering.
    rag_enabled: bool = True
    rag_top_k: int = Field(default=5, ge=1, le=20)
    rag_min_relevance_score: float | None = Field(default=0.30, ge=-1, le=1)
    # "fixed_response": answer without calling the LLM; "llm": let the LLM answer, flagged as
    # not grounded in the knowledge base.
    rag_no_context_mode: Literal["fixed_response", "llm"] = "fixed_response"
    rag_no_context_message: str = (
        "I couldn't find relevant information about this in the knowledge base, so I can't give a "
        "grounded answer. The knowledge base currently covers software engineering topics such as "
        "programming languages, distributed systems, databases, cloud and LLMs."
    )
    # Include retrieved chunks and scores in chat responses (development only).
    rag_debug: bool = False

    @field_validator(
        "openai_api_key",
        "qdrant_api_key",
        "openai_embedding_dimensions",
        "rag_min_relevance_score",
        mode="before",
    )
    @classmethod
    def _blank_is_missing(cls, value: object) -> object:
        return None if isinstance(value, str) and not value.strip() else value


@lru_cache
def get_settings() -> Settings:
    return Settings()
