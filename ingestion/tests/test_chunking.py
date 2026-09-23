from datetime import UTC, datetime

import pytest

from app.knowledge.models import Document, make_chunk_id
from ingestion.chunking import TextChunker

FETCHED_AT = datetime(2026, 1, 1, tzinfo=UTC)


def _doc(text: str) -> Document:
    return Document(
        document_id="wikipedia:en:42",
        text=text,
        title="Apache Kafka",
        source="wikipedia",
        source_url="https://en.wikipedia.org/wiki/Apache_Kafka",
        language="en",
        fetched_at=FETCHED_AT,
        metadata={"page_id": 42, "revision_id": 7},
    )


def _paragraphs(n: int, sentences: int = 4) -> str:
    return "\n\n".join(
        " ".join(f"Paragraph {p} sentence {s} talks about topic {p}." for s in range(sentences))
        for p in range(n)
    )


def test_short_text_is_single_chunk():
    assert TextChunker(500, 50).split_text("One short paragraph.") == ["One short paragraph."]


def test_chunks_respect_size_limit():
    chunks = TextChunker(300, 50).split_text(_paragraphs(20))

    assert len(chunks) > 5
    assert all(len(c) <= 300 for c in chunks)


def test_prefers_paragraph_boundaries():
    text = _paragraphs(6, sentences=2)  # each paragraph ~100 chars
    chunks = TextChunker(250, 0).split_text(text)

    for chunk in chunks:
        assert chunk.startswith("Paragraph")
        assert chunk.endswith(".")


def test_long_paragraph_split_on_sentences():
    paragraph = " ".join(f"Sentence number {i} is here." for i in range(40))
    chunks = TextChunker(200, 0).split_text(paragraph)

    assert all(c.startswith("Sentence number") and c.endswith("is here.") for c in chunks)


def test_overlap_repeats_trailing_content():
    paragraph = " ".join(f"Sentence number {i} is here." for i in range(40))
    chunks = TextChunker(200, 60).split_text(paragraph)

    for previous, current in zip(chunks, chunks[1:]):
        first_sentence = current.split(". ")[0] + "."
        assert previous.endswith(first_sentence) or f"{first_sentence} " in previous


def test_overlong_word_is_hard_split():
    chunks = TextChunker(200, 0).split_text("x" * 450)

    assert [len(c) for c in chunks] == [200, 200, 50]


def test_invalid_configuration_rejected():
    with pytest.raises(ValueError):
        TextChunker(100, 100)
    with pytest.raises(ValueError):
        TextChunker(0, 0)


def test_chunk_metadata_is_preserved():
    chunks = TextChunker(300, 50).chunk(_doc(_paragraphs(10)))

    assert len(chunks) > 1
    for index, chunk in enumerate(chunks):
        assert chunk.chunk_index == index
        assert chunk.chunk_id == make_chunk_id("wikipedia:en:42", index)
        assert chunk.document_id == "wikipedia:en:42"
        assert chunk.source == "wikipedia"
        assert chunk.article_title == "Apache Kafka"
        assert chunk.source_url == "https://en.wikipedia.org/wiki/Apache_Kafka"
        assert chunk.metadata["page_id"] == 42
        assert chunk.metadata["revision_id"] == 7
        assert chunk.metadata["language"] == "en"
        assert chunk.metadata["fetched_at"] == FETCHED_AT.isoformat()
        assert chunk.metadata["chunk_count"] == len(chunks)


def test_chunk_ids_are_deterministic():
    first = TextChunker(300, 50).chunk(_doc(_paragraphs(10)))
    second = TextChunker(300, 50).chunk(_doc(_paragraphs(10)))

    assert [c.chunk_id for c in first] == [c.chunk_id for c in second]
    assert len({c.chunk_id for c in first}) == len(first)
