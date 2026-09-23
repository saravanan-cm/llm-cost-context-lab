"""Group retrieved chunks by source document (one citation per article)."""

from dataclasses import dataclass

from app.knowledge.models import RetrievedChunk


@dataclass(frozen=True)
class SourceReference:
    document_id: str
    title: str
    source: str
    url: str
    score: float
    """Best similarity score among this document's chunks."""
    chunks: tuple[RetrievedChunk, ...]
    """This document's chunks in reading order (by chunk_index)."""


def group_by_document(chunks: list[RetrievedChunk]) -> list[SourceReference]:
    """Deduplicate chunks into one reference per document, ordered by best score."""
    grouped: dict[str, list[RetrievedChunk]] = {}
    for chunk in chunks:
        grouped.setdefault(chunk.document_id, []).append(chunk)

    references = [
        SourceReference(
            document_id=document_id,
            title=items[0].article_title,
            source=items[0].source,
            url=items[0].source_url,
            score=max(item.score for item in items),
            chunks=tuple(sorted(items, key=lambda item: item.chunk_index)),
        )
        for document_id, items in grouped.items()
    ]
    return sorted(references, key=lambda ref: ref.score, reverse=True)
