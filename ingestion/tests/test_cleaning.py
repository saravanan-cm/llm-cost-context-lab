from datetime import UTC, datetime

from app.knowledge.models import Document
from ingestion.cleaning import TextCleaner

RAW = """Redis is an in-memory   data store.


== History ==
Redis was created in 2009.

=== Early years ===
It started as a side project.

== See also ==
* Memcached

== References ==
^ Some citation
=== Sub of references ===
more refs

== Performance ==
Latency is {\\displaystyle O(1)} for most operations."""


def test_removes_reference_sections_and_keeps_content():
    text = TextCleaner().clean_text(RAW)

    assert "Memcached" not in text
    assert "citation" not in text
    assert "more refs" not in text  # subsections of excluded sections are skipped too
    assert "Redis was created in 2009." in text
    assert "It started as a side project." in text
    assert "Latency is" in text


def test_converts_headings_to_plain_paragraphs():
    text = TextCleaner().clean_text(RAW)

    assert "==" not in text
    assert "\n\nHistory\n\n" in text
    assert "\n\nEarly years\n\n" in text


def test_normalises_whitespace_and_strips_math_markup():
    text = TextCleaner().clean_text(RAW)

    assert "in-memory data store" in text
    assert "\n\n\n" not in text
    assert "displaystyle" not in text


def test_clean_returns_new_document_with_same_metadata():
    doc = Document(
        document_id="wikipedia:en:1",
        text=RAW,
        title="Redis",
        source="wikipedia",
        source_url="u",
        language="en",
        fetched_at=datetime(2026, 1, 1, tzinfo=UTC),
        metadata={"page_id": 1},
    )

    cleaned = TextCleaner().clean(doc)

    assert cleaned.document_id == doc.document_id and cleaned.metadata == doc.metadata
    assert cleaned.text != doc.text


def test_empty_input_cleans_to_empty():
    assert TextCleaner().clean_text("== References ==\n^ only refs") == ""
