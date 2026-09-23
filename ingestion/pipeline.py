"""Source -> Document -> Cleaner -> Chunker -> Embedding -> Vector store.

Idempotency: chunk IDs are deterministic (document_id + chunk_index), so re-ingesting
overwrites points instead of duplicating them. Each document's chunks also carry a
fingerprint of (text, chunking config, embedding model); if it matches what is stored, the
document is skipped and no embedding API calls are made.
"""

import hashlib
import logging
from collections.abc import Iterable
from dataclasses import dataclass, field

from app.core.errors import (
    AppError,
    EmbeddingConfigurationError,
    VectorStoreConfigurationError,
    VectorStoreUnavailableError,
)
from app.knowledge.embeddings import EmbeddingProvider
from app.knowledge.models import Document
from app.knowledge.vector_store import VectorStore
from ingestion.chunking import TextChunker
from ingestion.cleaning import TextCleaner
from ingestion.errors import EmptyDocumentError, IngestionError
from ingestion.sources.base import KnowledgeSource

logger = logging.getLogger(__name__)

# Errors that will affect every remaining document, so continuing is pointless.
FATAL_ERRORS = (EmbeddingConfigurationError, VectorStoreUnavailableError, VectorStoreConfigurationError)


@dataclass
class Failure:
    reference: str
    reason: str


@dataclass
class IngestionReport:
    requested: int = 0
    fetched: int = 0
    ingested: int = 0
    unchanged: int = 0
    duplicates: int = 0
    chunks_created: int = 0
    embeddings_generated: int = 0
    embedding_tokens: int = 0
    vectors_stored: int = 0
    failures: list[Failure] = field(default_factory=list)
    aborted: str | None = None


class IngestionPipeline:
    def __init__(
        self,
        *,
        source: KnowledgeSource,
        cleaner: TextCleaner,
        chunker: TextChunker,
        embedder: EmbeddingProvider | None,
        store: VectorStore | None,
        force: bool = False,
    ) -> None:
        """``embedder``/``store`` may be None for a dry run (fetch, clean and chunk only)."""
        self._source = source
        self._cleaner = cleaner
        self._chunker = chunker
        self._embedder = embedder
        self._store = store
        self._force = force

    @property
    def dry_run(self) -> bool:
        return self._embedder is None or self._store is None

    def run(self, references: Iterable[str]) -> IngestionReport:
        references = list(references)
        report = IngestionReport(requested=len(references))
        if not self.dry_run:
            assert self._store is not None and self._embedder is not None
            self._store.ensure_collection(self._embedder.dimension)

        seen: set[str] = set()
        for index, reference in enumerate(references):
            try:
                self._ingest_one(reference, report, seen)
            except FATAL_ERRORS as exc:
                report.failures.append(Failure(reference, exc.message))
                report.aborted = exc.message
                report.failures.extend(Failure(ref, "not attempted") for ref in references[index + 1 :])
                logger.error("ingestion_aborted", extra={"reference": reference, "error": exc.code})
                break
            except (IngestionError, AppError) as exc:
                reason = exc.message if isinstance(exc, AppError) else str(exc)
                report.failures.append(Failure(reference, reason))
                logger.warning("document_failed", extra={"reference": reference, "reason": reason})
        return report

    def _ingest_one(self, reference: str, report: IngestionReport, seen: set[str]) -> None:
        document = self._source.fetch(reference)
        report.fetched += 1
        if document.document_id in seen:
            report.duplicates += 1
            logger.info("document_duplicate", extra={"reference": reference, "document_id": document.document_id})
            return
        seen.add(document.document_id)

        document = self._cleaner.clean(document)
        if not document.text.strip():
            raise EmptyDocumentError(f"'{document.title}' has no usable text after cleaning")
        chunks = self._chunker.chunk(document)
        report.chunks_created += len(chunks)
        if self.dry_run:
            return
        assert self._store is not None and self._embedder is not None

        fingerprint = self.fingerprint(document)
        if not self._force and self._store.get_document_fingerprint(document.document_id) == fingerprint:
            report.unchanged += 1
            logger.info("document_unchanged", extra={"document_id": document.document_id})
            return

        batch = self._embedder.embed([chunk.text for chunk in chunks])
        report.embeddings_generated += len(batch.vectors)
        report.embedding_tokens += batch.input_tokens
        report.vectors_stored += self._store.upsert_document(
            document.document_id, chunks, batch.vectors, fingerprint
        )
        report.ingested += 1
        logger.info(
            "document_ingested",
            extra={
                "document_id": document.document_id,
                "title": document.title,
                "chunks": len(chunks),
                "embedding_tokens": batch.input_tokens,
            },
        )

    def fingerprint(self, document: Document) -> str:
        assert self._embedder is not None
        material = "\x00".join(
            [
                document.text,
                self._chunker.config_signature,
                f"{self._embedder.model}:{self._embedder.dimension}",
            ]
        )
        return hashlib.sha256(material.encode("utf-8")).hexdigest()
