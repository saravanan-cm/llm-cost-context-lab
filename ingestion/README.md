# Ingestion

Builds the knowledge base: **Source → Document → Cleaner → Chunker → Embedding → Vector store (Qdrant)**.

It is separate from the API. It reuses the backend package (`app.knowledge.*`) for the document
models, the embedding provider and the vector store, so there is one implementation of each.

## Run

From the repository root, with the backend virtualenv (which has `ingestion/requirements.txt` installed):

```bash
docker compose up -d qdrant                       # Qdrant on :6333
python -m ingestion.ingest --dry-run              # fetch + clean + chunk only, no API cost
python -m ingestion.ingest                        # full ingestion
python -m ingestion.ingest --articles "Apache Kafka,Redis"
python -m ingestion.ingest --force                # re-embed even if unchanged
python -m ingestion.ingest --recreate             # drop + recreate the collection (e.g. new embedding model)
```

Exit codes: `0` success, `1` some documents failed, `2` aborted or configuration error.

## Configuration (`backend/.env`)

| Variable | Default | |
|----------|---------|---|
| `WIKIPEDIA_LANGUAGE` | `en` | Wikipedia language edition |
| `WIKIPEDIA_ARTICLES` | empty | Comma-separated titles. When empty, [`articles.txt`](articles.txt) is used. |
| `WIKIPEDIA_USER_AGENT` | generic | Add a contact per the [Wikimedia UA policy](https://meta.wikimedia.org/wiki/User-Agent_policy) |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `1200` / `200` | Characters |
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-3-small` | Shared with the API |
| `QDRANT_URL` / `QDRANT_COLLECTION` | `http://localhost:6333` / `interview_knowledge` | Shared with the API |

To change the article list, edit `articles.txt` (one title per line). No code changes needed.

## Behaviour

- **Wikipedia source.** Uses the official MediaWiki Action API (`prop=extracts&explaintext=1`), not HTML
  scraping. Redirects are followed. Missing articles and disambiguation pages are reported as failures.
  Requests are spaced at least 1 second apart, and HTTP 429/503 responses are retried with backoff
  that honours `Retry-After`.
- **Cleaning.** Drops References, See also, External links and similar sections. Keeps headings as
  their own paragraphs, normalises whitespace and strips math markup.
- **Chunking.** Splits at paragraph boundaries first, then sentences, then words, up to `CHUNK_SIZE`
  characters. Each new chunk repeats up to `CHUNK_OVERLAP` characters from the end of the previous one.
  Every chunk carries its document's metadata.
- **Idempotency.** Chunk IDs are deterministic UUIDs of `document_id#chunk_index`, so re-runs overwrite
  points instead of duplicating them. Stale trailing chunks are deleted. A per-document fingerprint of
  text, chunking config and embedding model means unchanged documents are skipped with **no embedding
  calls**.
- **Failures.** A per-article failure (not found, empty, Wikipedia error) is recorded and ingestion
  continues. Errors that would affect every article (missing API key, Qdrant unreachable, dimension
  mismatch) abort the run with a hint.

## Adding another source

Implement `KnowledgeSource` ([sources/base.py](sources/base.py)): `fetch(reference) -> Document`.
Cleaning, chunking, embedding and storage are source-independent.

## Tests

```bash
cd ingestion
python -m pytest                                  # unit tests: no network
RUN_INTEGRATION=1 python -m pytest -m integration # live: Wikipedia + OpenAI + running Qdrant
```
