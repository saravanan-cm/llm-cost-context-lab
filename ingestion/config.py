"""Ingestion-only settings. Shared knowledge settings (embedding model, Qdrant) live in
``app.core.config.Settings``; both read ``backend/.env``."""

from pathlib import Path
from typing import Annotated

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

from app.core.config import ENV_FILE

DEFAULT_ARTICLES_FILE = Path(__file__).resolve().parent / "articles.txt"


class IngestionSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ENV_FILE, env_file_encoding="utf-8", extra="ignore")

    wikipedia_language: str = Field(default="en", pattern=r"^[a-z][a-z0-9-]{1,15}$")
    # Comma-separated titles. When empty, titles are read from WIKIPEDIA_ARTICLES_FILE.
    wikipedia_articles: Annotated[list[str], NoDecode] = []
    wikipedia_articles_file: Path = DEFAULT_ARTICLES_FILE
    # Wikimedia asks API clients to identify themselves: https://meta.wikimedia.org/wiki/User-Agent_policy
    wikipedia_user_agent: str = "LLMCostContextLab/0.3 (educational portfolio project)"
    wikipedia_timeout_seconds: float = Field(default=20.0, gt=0)

    # Chunk size and overlap are measured in characters (~4 characters per token for English).
    chunk_size: int = Field(default=1200, ge=200, le=8000)
    chunk_overlap: int = Field(default=200, ge=0)

    @field_validator("wikipedia_articles", mode="before")
    @classmethod
    def _split_titles(cls, value: object) -> object:
        if isinstance(value, str):
            return parse_titles(value.split(","))
        return value

    @model_validator(mode="after")
    def _overlap_smaller_than_size(self) -> "IngestionSettings":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")
        return self

    def article_titles(self) -> list[str]:
        if self.wikipedia_articles:
            return self.wikipedia_articles
        return load_titles_file(self.wikipedia_articles_file)


def parse_titles(lines: list[str]) -> list[str]:
    """Strip whitespace, drop blanks and ``#`` comments, de-duplicate preserving order."""
    titles: list[str] = []
    for line in lines:
        title = line.split("#", 1)[0].strip()
        if title and title not in titles:
            titles.append(title)
    return titles


def load_titles_file(path: Path) -> list[str]:
    if not path.is_file():
        raise ValueError(f"Articles file not found: {path}")
    return parse_titles(path.read_text(encoding="utf-8").splitlines())
