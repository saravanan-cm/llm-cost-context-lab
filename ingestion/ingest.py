"""Ingestion CLI.

Usage (from the repository root, with the backend virtualenv active):

    python -m ingestion.ingest                      # articles from WIKIPEDIA_ARTICLES / articles.txt
    python -m ingestion.ingest --articles "Apache Kafka,Redis"
    python -m ingestion.ingest --dry-run            # fetch + clean + chunk only (no API cost)
    python -m ingestion.ingest --force              # re-embed even if unchanged
    python -m ingestion.ingest --recreate           # drop and recreate the collection first

Exit codes: 0 success, 1 some documents failed, 2 aborted / configuration error.
"""

import argparse
import logging
import sys
from pathlib import Path

from pydantic import ValidationError

from app.core.config import Settings, get_settings
from app.core.errors import AppError
from app.core.logging import configure_logging
from app.knowledge.factory import build_embedding_provider, build_vector_store
from app.services.pricing import PricingService
from ingestion.chunking import TextChunker
from ingestion.cleaning import TextCleaner
from ingestion.config import IngestionSettings, load_titles_file, parse_titles
from ingestion.pipeline import IngestionPipeline, IngestionReport
from ingestion.sources.wikipedia import WikipediaSource


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        settings = get_settings()
        ingestion_settings = IngestionSettings()
        titles = _resolve_titles(args, ingestion_settings)
    except (ValidationError, ValueError) as exc:
        print(f"Configuration error:\n{exc}", file=sys.stderr)
        return 2
    if not titles:
        print("No articles configured (WIKIPEDIA_ARTICLES or articles file).", file=sys.stderr)
        return 2

    configure_logging(args.log_level or settings.log_level, settings.log_json)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    source = WikipediaSource(
        language=ingestion_settings.wikipedia_language,
        user_agent=ingestion_settings.wikipedia_user_agent,
        timeout_seconds=ingestion_settings.wikipedia_timeout_seconds,
    )
    chunker = TextChunker(ingestion_settings.chunk_size, ingestion_settings.chunk_overlap)
    collection = args.collection or settings.qdrant_collection

    try:
        embedder = store = None
        if not args.dry_run:
            embedder = build_embedding_provider(settings)
            store = build_vector_store(settings, collection)
            if args.recreate:
                store.recreate(embedder.dimension)
        _print_header(settings, ingestion_settings, collection, embedder, args.dry_run)
        pipeline = IngestionPipeline(
            source=source,
            cleaner=TextCleaner(),
            chunker=chunker,
            embedder=embedder,
            store=store,
            force=args.force,
        )
        report = pipeline.run(titles)
        total_vectors = store.count() if store is not None else None
    except AppError as exc:
        print(f"\nIngestion aborted: {exc.message}", file=sys.stderr)
        _print_hint(exc, settings)
        return 2
    finally:
        source.close()

    _print_report(report, settings, total_vectors, args.dry_run)
    if report.aborted:
        return 2
    return 1 if report.failures else 0


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m ingestion.ingest", description=__doc__.split("\n")[0])
    parser.add_argument("--articles", help="Comma-separated Wikipedia titles (overrides configuration)")
    parser.add_argument("--articles-file", type=Path, help="File with one title per line")
    parser.add_argument("--collection", help="Qdrant collection (default: QDRANT_COLLECTION)")
    parser.add_argument("--dry-run", action="store_true", help="Fetch, clean and chunk only")
    parser.add_argument("--force", action="store_true", help="Re-embed documents even if unchanged")
    parser.add_argument("--recreate", action="store_true", help="Drop and recreate the collection first")
    parser.add_argument("--log-level", help="Override LOG_LEVEL")
    return parser.parse_args(argv)


def _resolve_titles(args: argparse.Namespace, settings: IngestionSettings) -> list[str]:
    if args.articles:
        return parse_titles(args.articles.split(","))
    if args.articles_file:
        return load_titles_file(args.articles_file)
    return settings.article_titles()


def _print_header(settings: Settings, ing: IngestionSettings, collection: str, embedder, dry_run: bool) -> None:
    print(f"Source: Wikipedia ({ing.wikipedia_language})")
    print(f"Chunking: {ing.chunk_size} chars, {ing.chunk_overlap} overlap")
    if dry_run:
        print("Mode: DRY RUN (no embeddings, no vector store)")
    else:
        print(f"Embedding model: {embedder.model} ({embedder.dimension} dimensions)")
        print(f"Qdrant: {settings.qdrant_url} / collection '{collection}'")
    print()


def _print_report(report: IngestionReport, settings: Settings, total_vectors: int | None, dry_run: bool) -> None:
    rows = [
        ("Wikipedia articles", report.requested),
        ("Documents fetched", report.fetched),
        ("Chunks created", report.chunks_created),
    ]
    if not dry_run:
        cost = PricingService().calculate(settings.openai_embedding_model, report.embedding_tokens, 0).total_cost
        rows += [
            ("Documents ingested", report.ingested),
            ("Unchanged (skipped)", report.unchanged),
            ("Embeddings generated", report.embeddings_generated),
            ("Embedding tokens", report.embedding_tokens),
            ("Embedding cost (USD)", f"{cost.normalize():f}"),
            ("Vectors stored", report.vectors_stored),
            ("Vectors in collection", total_vectors),
        ]
    if report.duplicates:
        rows.append(("Duplicate titles", report.duplicates))
    rows.append(("Failures", len(report.failures)))

    width = max(len(label) for label, _ in rows) + 2
    print("Ingestion summary")
    print("-" * (width + 12))
    for label, value in rows:
        print(f"{label + ':':<{width}}{value}")
    for failure in report.failures:
        print(f"  - {failure.reference}: {failure.reason}")
    if report.aborted:
        print(f"\nAborted: {report.aborted}")


def _print_hint(exc: AppError, settings: Settings) -> None:
    hints = {
        "vector_store_unavailable": f"Is Qdrant running at {settings.qdrant_url}? Try: docker compose up -d qdrant",
        "embedding_not_configured": "Set OPENAI_API_KEY in backend/.env (or use --dry-run).",
        "vector_store_misconfigured": "Use a new --collection or re-run with --recreate.",
    }
    if hint := hints.get(exc.code):
        print(f"Hint: {hint}", file=sys.stderr)


if __name__ == "__main__":
    sys.exit(main())
