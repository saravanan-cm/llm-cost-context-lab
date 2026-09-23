import pytest
from pydantic import ValidationError

from ingestion.config import DEFAULT_ARTICLES_FILE, IngestionSettings, load_titles_file, parse_titles


def test_articles_from_env_are_comma_separated(monkeypatch):
    monkeypatch.setenv("WIKIPEDIA_ARTICLES", "Apache Kafka, Redis ,,Redis")

    settings = IngestionSettings(_env_file=None)

    assert settings.article_titles() == ["Apache Kafka", "Redis"]


def test_default_articles_file_is_used_when_env_empty(monkeypatch):
    monkeypatch.delenv("WIKIPEDIA_ARTICLES", raising=False)

    titles = IngestionSettings(_env_file=None).article_titles()

    assert titles == load_titles_file(DEFAULT_ARTICLES_FILE)
    assert "Apache Kafka" in titles and "Retrieval-augmented generation" in titles


def test_parse_titles_ignores_comments_and_blanks():
    assert parse_titles(["# comment", "", "  SQL  ", "NoSQL # inline", "SQL"]) == ["SQL", "NoSQL"]


def test_overlap_must_be_smaller_than_chunk_size():
    with pytest.raises(ValidationError, match="CHUNK_OVERLAP"):
        IngestionSettings(_env_file=None, chunk_size=500, chunk_overlap=500)


def test_invalid_language_rejected():
    with pytest.raises(ValidationError):
        IngestionSettings(_env_file=None, wikipedia_language="EN; drop")
