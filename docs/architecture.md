# Architecture

This document describes the current system and the **target** architecture. Components marked
**NOT IMPLEMENTED** are shown to guide later steps.

## Request flow: RAG chat (implemented, Step 4)

> LangGraph, conversation history, topic switching and context compression/summarisation are
> **future steps**. Each chat turn is answered independently from the knowledge base.

```
              User Question
                   |
                   v
              FastAPI API                 routes/chat.py: HTTP only
                   |
                   v
              ChatService                 sequencing, logging, response shaping
                   |
                   v
               RAG Service                app/rag/service.py
                /       \
               /         \
         Retriever       LLM Provider
            |              |
            v              v
         Qdrant          OpenAI
            |
      Relevant Chunks     (top-k, filtered by RAG_MIN_RELEVANCE_SCORE)
            |
            +------→ Context Builder      app/rag/context_builder.py
                          |
                          ↓
                         LLM
                          |
                          ↓
                     Answer + Sources     (deduplicated per article)
                          |
                          ↓
                   Usage / Cost / Credits  UsageMeter → PricingService → CreditPolicy → UsageService
```

### `POST /chat` sequence

1. **Early credit check** on the bare question. A user with no credits is rejected with **402**
   before any retrieval or embedding spend. An unpriced model also fails here.
2. **Retrieve** (`RAGService.prepare`). The question is embedded and the top `RAG_TOP_K` chunks are
   fetched from Qdrant. Chunks below `RAG_MIN_RELEVANCE_SCORE` are dropped. If Qdrant or embeddings
   fail, the request returns 503/502 and **no LLM call or charge happens**.
3. **No relevant chunks.** With `RAG_NO_CONTEXT_MODE=fixed_response` (default), the request returns
   a controlled "the knowledge base doesn't cover this" answer: `answer_type: "no_context"`, no LLM
   call, zero cost and no ledger row. With `llm`, the LLM answers under instructions to say up front
   that the knowledge base lacks the topic.
4. **Build the prompt** (`RAGContextBuilder`). Grounding instructions go in the system instructions.
   Numbered source blocks (one per article, chunks in reading order, with title, source and URL)
   are wrapped in `<knowledge_context>`, followed by the question.
5. **Accurate credit check** on the full prompt (instructions + context + question) plus
   `LLM_MAX_OUTPUT_TOKENS`, before the LLM call.
6. **LLM call** (`RAGService.generate`). Failures map to typed errors, and **no credits are
   deducted**.
7. **Metering** (`UsageMeter.record`) on **provider-reported** tokens. Retrieved context,
   instructions and the question are all in the input tokens the provider reports, so cost and
   credits reflect the real prompt size. The ledger row is written and credits are deducted in one
   transaction.
8. **Response**: `assistant_message`, `answer_type`, `model`, `sources[]` (title, source, url, best
   score), `usage`, `cost`, `credits`, and `debug` (retrieved chunks with scores and used/filtered
   flags) only when `RAG_DEBUG=true`.

With `RAG_ENABLED=false`, chat falls back to the Step 2 behaviour: `answer_type: "direct"`, the
question is sent as-is with `SYSTEM_PROMPT`, and there is no retrieval.

### RAG configuration

| Variable | Default | Notes |
|----------|---------|-------|
| `RAG_ENABLED` | `true` | `false` means direct LLM chat |
| `RAG_TOP_K` | `5` | Chunks retrieved per question (1 to 20) |
| `RAG_MIN_RELEVANCE_SCORE` | `0.30` | Cosine similarity; blank disables filtering. Calibrated for `text-embedding-3-small` on this knowledge base: 13 in-domain questions scored 0.31–0.76 top-1, and 10 out-of-domain questions (quantum mechanics, Roman Empire, ...) scored at most 0.27. **Re-calibrate if the embedding model or corpus changes.** |
| `RAG_NO_CONTEXT_MODE` | `fixed_response` | or `llm` |
| `RAG_NO_CONTEXT_MESSAGE` | (text) | Reply used by `fixed_response` |
| `RAG_DEBUG` | `false` | Adds retrieval details to chat responses and the UI |

The env names are prefixed `RAG_` (rather than bare `TOP_K`) to avoid clashing with the ingestion
and knowledge-search settings.

### Module responsibilities

| Module | Responsibility |
|--------|---------------|
| `api/v1/routes/*` | HTTP only |
| `api/deps.py` | Builds services from settings. `get_current_user_id` is the future auth seam (returns `dev-user`). |
| `api/errors.py` | `AppError`, `SQLAlchemyError` and unexpected errors to `{"error": {code, message}}` |
| `services/chat_service.py` | Sequences credit check → RAG → metering. Logs `chat_request` and builds the response. No retrieval, prompt, pricing or SQL logic. |
| `rag/service.py` | `RAGService`: retrieve → relevance filter → no-context policy → prompt → LLM |
| `rag/context_builder.py` | `RAGContextBuilder`: the only place RAG prompt text lives |
| `rag/sources.py` | Deduplicates chunks into one `SourceReference` per article |
| `knowledge/retriever.py` | Retrieval only (Step 3, unchanged) |
| `services/llm/` | `LLMProvider` Protocol (now with an optional `instructions` override), `OpenAIProvider`, mock |
| `services/metering.py` | `UsageMeter`: pre-flight affordability check and metering of actual usage |
| `services/pricing.py` | `MODEL_PRICING` catalog and `PricingService` |
| `services/credits.py` | `CreditPolicy` (USD to credits) |
| `services/usage_service.py` | Ledger writes, balance checks, usage summary |
| `core/request_context.py` | `X-Request-ID` middleware; the ID is attached to every log line of the request |
| `db/` | SQLAlchemy models, `DecimalAmount` column type, sessions, Alembic runner |

## Tokens vs cost vs credits

These three quantities are **intentionally separate**:

| Concept | Unit | Source of truth | Example |
|---------|------|-----------------|---------|
| **LLM tokens** | tokens (input and output counted separately) | Reported by the provider per request | 120 in / 180 out |
| **LLM cost** | USD | `tokens x model price` (`PricingService`, USD per 1M tokens) | $0.000336 |
| **Application credits** | credits | `cost x CREDITS_PER_DOLLAR` (`CreditPolicy`) | 0.336 credits |

Why they are separate:

- **Tokens are not comparable across models.** A `gpt-5-mini` output token costs 20x a
  `gpt-4.1-nano` input token. Charging "1 credit = 1 token" would make the budget meaningless when
  the model changes.
- **Input and output are priced differently.** Output is typically 4 to 8 times more expensive, so
  totals must be priced per direction.
- **Credits are a product decision, cost is a vendor fact.** Keeping a conversion layer lets the
  product add margin, run promotions, change providers or absorb vendor price changes without changing
  user budgets or historical ledger data. The ledger stores all three values per request.

### Pricing unit and precision

- Prices are **USD per 1,000,000 tokens**, stored as `Decimal` in `MODEL_PRICING`. Input and output
  prices are separate per model. To add a model, add one catalog entry. Dated snapshot names such as
  `gpt-4.1-mini-2025-04-14` resolve to the base entry.
- All money and credit values are `Decimal`, quantized to 12 decimal places. No `float` is used
  anywhere in the pipeline.
- Storage uses `DecimalAmount`: `NUMERIC(38,12)` on PostgreSQL and a scaled `BIGINT` on SQLite (which
  has no exact decimal type), so `SUM()` is exact on both.
- The API returns amounts as decimal strings (`"0.000336"`) so JSON clients don't round them.
- Catalog prices are list prices; verify them against OpenAI's pricing page. Cached-input and batch
  discounts are not yet modelled.

## Data model (implemented)

- `users`: `id`, `total_credits`, `credits_used`, `created_at`. Remaining = total − used.
- `usage_records` (append-only): `user_id`, `conversation_id`, `created_at`, `provider`, `model`,
  `provider_request_id`, `input_tokens`, `output_tokens`, `total_tokens`, `input_cost`, `output_cost`,
  `total_cost`, `credits_consumed`, `latency_ms`.

Migrations are managed by Alembic (`backend/migrations`).

## Known limitations (by design for this step)

- **Check-then-deduct is not strictly atomic.** Concurrent requests can each pass the pre-check and
  push the balance slightly negative. The deduction itself never loses updates. Future options: a
  guarded `UPDATE ... WHERE total_credits - credits_used >= :x`, or a reserve-then-settle hold.
  Distributed locking is deliberately deferred.
- Actual usage can exceed the pre-flight estimate for unusually tokenized input. The actual cost is
  always charged.
- Only the current question (plus retrieved context) is sent to the LLM. There is no conversation
  history yet, so follow-up questions like "and how does it scale?" are not resolved.
- The query-embedding cost of retrieval (about 10 tokens, roughly $0.0000002) is logged but not
  charged to user credits. Only LLM usage is metered.
- Chunks overlap by up to 200 characters, so adjacent chunks of one article can repeat a little text
  in the prompt.
- Sync route + sync SDK/DB, run in FastAPI's threadpool. This is fine at this scale. It can move to
  async clients later without changing the provider interface's shape.

## Observability

Every request gets an `X-Request-ID` (a well-formed incoming one is reused, otherwise one is
generated). It is echoed in the response header and attached as `request_id` to every log line
written while handling the request.

Each chat turn logs one `chat_request` event:

| Field | Meaning |
|-------|---------|
| `outcome` | `success`, `no_context` or `failure` (+ `stage`: `credit_check`, `retrieval`, `llm`, `metering`; + `error_code`) |
| `conversation_id`, `request_id` | Correlation |
| `rag`, `retrieval_latency_ms`, `chunks_retrieved`, `chunks_used`, `sources`, `top_score` | Retrieval quality and latency |
| `model`, `provider_request_id`, `llm_latency_ms` | LLM call |
| `input_tokens`, `output_tokens`, `total_tokens`, `total_cost_usd`, `credits_consumed` | Metering |
| `latency_ms` | End-to-end |

The retriever also logs `knowledge_search` (query-embedding tokens and latency). Output is `key=value`
by default or JSON lines with `LOG_JSON=true`. **Never logged:** API keys, prompts, retrieved text,
user questions or answers.

## Knowledge base: ingestion + retrieval (implemented, Step 3)

> Since Step 4 this retriever also powers RAG in `/chat` (see above). The development search endpoint
> and UI remain for inspecting retrieval without calling the LLM.

```
             Knowledge Sources
                   |
             Wikipedia API              ingestion/sources/wikipedia.py (MediaWiki Action API, plain-text extracts)
                   |
                   ↓
                Document                app/knowledge/models.py (source-agnostic)
                   |
                   ↓
            Cleaner + Chunker           ingestion/cleaning.py, ingestion/chunking.py
                   |
                   ↓
             Embedding Provider         app/knowledge/embeddings.py (Protocol) -> openai_embeddings.py
                   |
                   ↓
                 Qdrant                 app/knowledge/vector_store.py (Protocol) -> qdrant_store.py
                   |
                   ↓
               Retriever                app/knowledge/retriever.py (query -> embedding -> vector search)
                   |
                   ↓
         Knowledge Search API           POST /api/v1/knowledge/search (dev only, no LLM)
```

Runtime topology for local development:

```
FastAPI ──┬── SQLite   (users, usage ledger)
          └── Qdrant   (knowledge vectors; docker compose up -d qdrant)
ingestion CLI ──► Wikipedia API, OpenAI embeddings, Qdrant
```

### Independence of concerns

| Concern | Module | Knows about |
|---------|--------|-------------|
| Knowledge source | `ingestion/sources/*` (`KnowledgeSource` Protocol) | Wikipedia only |
| Document processing | `ingestion/cleaning.py` | text |
| Chunking | `ingestion/chunking.py` | `Document` to `Chunk` |
| Embedding | `app/knowledge/*embeddings.py` (`EmbeddingProvider`) | OpenAI SDK only in `openai_embeddings.py` |
| Vector storage | `app/knowledge/*store.py` (`VectorStore`) | `qdrant_client` only in `qdrant_store.py` |
| Retrieval | `app/knowledge/retriever.py` | the two Protocols above |
| LLM | `app/services/llm/*` | unaware of the knowledge base (for now) |

`ingestion/` is a separate package and CLI. It imports the backend package (installed editable) so
models, embeddings and the vector store have exactly one implementation, shared with the API.

### Key decisions

- **Wikipedia via the official API.** `action=query&prop=extracts&explaintext=1` returns clean text
  with `== Heading ==` markers, with no HTML scraping. Disambiguation pages are rejected. Requests are
  spaced at 1 per second and retry 429/503 with `Retry-After`. The first live run hit HTTP 429 without
  this.
- **Document IDs** are `wikipedia:{lang}:{page_id}`, stable across title changes and redirects.
- **Chunking** uses characters rather than tokens: paragraph → sentence → word splitting, greedy
  packing to `CHUNK_SIZE`, with `CHUNK_OVERLAP` carried over. It is simple, deterministic and
  model-independent. Defaults are 1200/200, about 300 tokens per chunk.
- **Vector dimension comes from the embedding model, never a hard-coded constant.**
  `resolve_dimension()` maps known models to their native size (`text-embedding-3-small` is 1536,
  `-3-large` is 3072). `OPENAI_EMBEDDING_DIMENSIONS` can request a smaller size (3-series only). An
  unknown model requires an explicit dimension. The collection is created with that size and cosine
  distance (OpenAI embeddings are normalised). If an existing collection has a different size,
  ingestion and search fail with an actionable error instead of corrupting data. Switching models
  therefore means a new `QDRANT_COLLECTION` or `--recreate`.
- **Idempotent ingestion.** Point ID = `uuid5(document_id#chunk_index)`. Re-ingesting upserts in
  place, then deletes chunks with a higher index than the new count. Each point stores a fingerprint of
  text, chunking config and embedding model, and unchanged documents are skipped before any embedding
  call, so re-runs cost nothing.
- **Batching.** Embeddings use up to `EMBEDDING_BATCH_SIZE` (100) inputs per API call. Qdrant upserts
  in batches of 256.
- **Costs are visible.** Embedding tokens are provider-reported, and ingestion prints the embedding
  cost using the same `PricingService` as chat. Query embeddings on the dev search endpoint are
  logged but **not** charged to user credits.
- **Partial failure.** A per-article error is recorded and the run continues. Errors that would fail
  every article (no API key, Qdrant down, dimension mismatch) abort with a hint. The CLI exits 0, 1 or
  2.

### Qdrant payload

`text`, `document_id`, `chunk_id`, `chunk_index`, `source`, `article_title`, `source_url`,
`fingerprint`, and `metadata` (`page_id`, `revision_id`, `requested_title`, `language`, `fetched_at`,
`chunk_count`). Payload indexes: `document_id`, `source` (keyword), `chunk_index` (integer).

### Error mapping (knowledge endpoints)

| Condition | HTTP | Code |
|-----------|------|------|
| Qdrant unreachable | 503 | `vector_store_unavailable` |
| Collection not created yet | 503 | `knowledge_base_not_ready` |
| Collection/model dimension mismatch | 500 | `vector_store_misconfigured` |
| Missing/invalid OpenAI key or model | 503 | `embedding_not_configured` |
| OpenAI embedding failure/timeout | 502 | `embedding_provider_error` |

## Target overview

```
User
 ↓
FastAPI
 ↓
LangGraph                         [NOT IMPLEMENTED; today RAGService (Step 4) orchestrates]
 ├── Context Manager              [NOT IMPLEMENTED]
 ├── Retriever                    [IMPLEMENTED, used by RAGService]
 │      ↓
 │    Qdrant                      [IMPLEMENTED]
 └── LLM                          [IMPLEMENTED]
 ↓
Response
 ↓
Token Metering → Credits          [IMPLEMENTED]

Deployment: WAF → ALB → ECS/Fargate, RDS, managed Qdrant, Redis   [NOT IMPLEMENTED]
```

## Components

| Component | Status | Notes |
|-----------|--------|-------|
| React chat UI + usage indicator | IMPLEMENTED | Refreshes `/usage` after each chat request |
| React knowledge search (dev) | IMPLEMENTED | `#/dev/knowledge`, dev builds only |
| FastAPI `/health`, `/chat`, `/usage`, `/knowledge/search` | IMPLEMENTED | |
| `LLMProvider` + OpenAI + mock | IMPLEMENTED | Selected by `LLM_PROVIDER` |
| Pricing, credits, usage ledger | IMPLEMENTED | Described above |
| Relational DB | IMPLEMENTED (SQLite) | PostgreSQL/RDS later via `DATABASE_URL` |
| Authentication | NOT IMPLEMENTED | Fixed `dev-user` |
| Conversation history | NOT IMPLEMENTED | |
| Embedding provider abstraction + OpenAI | IMPLEMENTED | `app/knowledge/` |
| Vector store (Qdrant) | IMPLEMENTED | Local via Docker. Qdrant Cloud or self-hosted on AWS later. |
| Wikipedia ingestion CLI | IMPLEMENTED | `ingestion/` |
| Retriever + dev search API/UI | IMPLEMENTED | |
| RAG in `/chat` (grounded answers, sources, no-context handling) | IMPLEMENTED | Step 4 |
| React sources + retrieval debug panel | IMPLEMENTED | Debug only with `RAG_DEBUG=true` |
| Conversation history / context management | NOT IMPLEMENTED | Topic switching, summarisation, compression |
| LangGraph orchestration | NOT IMPLEMENTED | Topic switching, context compression, RAG |
| Streaming (SSE/WebSocket) | NOT IMPLEMENTED | |
| Redis | NOT IMPLEMENTED | Caching, rate limiting, distributed state |
| AWS (ECS/Fargate, ALB, WAF, RDS) | NOT IMPLEMENTED | `infrastructure/` |

## Design principles

- **Loose coupling via interfaces.** Vendor SDKs are confined to one provider module.
- **Thin routes, logic in services.** Dependencies are injected with FastAPI `Depends` and overridden in tests.
- **Exact money.** `Decimal` everywhere, exact storage, string serialization.
- **Stateless API containers.** State lives in the database, so the API can scale horizontally.
