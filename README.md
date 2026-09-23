# LLM Cost & Context Lab

A portfolio project that will become a SaaS-style RAG chatbot where every user has a credit budget.
The goal is to make **LLM cost and context management visible**. It will track input/output/total tokens,
per-request cost, remaining credits, conversation context, topic switches, RAG retrieval and context compression.

> **Status: Step 4, RAG.** Every chat question is answered from a Wikipedia-based knowledge base in
> Qdrant (Step 3), with source attribution. Actual LLM token usage is metered into cost and credits
> (Step 2). Questions outside the knowledge base get a controlled answer without an LLM call.
> No LangGraph, conversation memory, auth, Redis or streaming yet.

## Repository layout

```
backend/            FastAPI service (Python 3.11+)
  app/
    api/            Routes (/api/v1), dependency wiring, error handlers
    core/           Settings, logging, typed errors, decimal helpers
    db/             SQLAlchemy models, session, exact-decimal column type
    schemas/        Pydantic request/response models
    services/
      llm/          LLMProvider interface, OpenAIProvider, MockLLMProvider
      pricing.py    Model price catalog + cost calculation
      credits.py    USD -> credits conversion policy
      usage_service.py  Usage ledger + credit balance
      chat_service.py   Chat use case: credit check -> RAG -> metering -> response
      metering.py       UsageMeter: affordability check + pricing/credits/ledger of actual usage
    rag/            RAGService, RAGContextBuilder (prompt), source deduplication
    knowledge/      Document/Chunk models, EmbeddingProvider (OpenAI), VectorStore (Qdrant), Retriever
  migrations/       Alembic migrations
  tests/            pytest (no real OpenAI calls)
frontend/           React + TypeScript (Vite) chat UI with usage indicator + dev knowledge search
ingestion/          Wikipedia -> clean -> chunk -> embed -> Qdrant pipeline and CLI (see ingestion/README.md)
infrastructure/     IaC / AWS deployment, NOT IMPLEMENTED
docs/               Architecture docs
```

## Architecture (current)

```
React ──► FastAPI (/api/v1/chat) ──► ChatService ──► RAGService ──┬─► Retriever ──► Qdrant
                                          │                        ├─► RAGContextBuilder
                                          │                        └─► LLMProvider ──► OpenAI
                                          ▼
          LLM usage ──► UsageMeter ──► PricingService ──► CreditPolicy ──► UsageService ──► Database
```

Knowledge base ingestion:

```
Wikipedia API ──► Document ──► Cleaner ──► Chunker ──► EmbeddingProvider ──► Qdrant
                                                                                │
POST /api/v1/knowledge/search ──► Retriever ──► EmbeddingProvider + VectorStore ◄┘
```

The route only validates input and delegates. It has no SDK calls, pricing formulas, credit maths or SQL.
See [docs/architecture.md](docs/architecture.md) for the request flow, the target architecture and the
distinction between tokens, cost and credits.

## Current capabilities

| Endpoint | Description |
|----------|-------------|
| `GET /api/v1/health` | Status, version, environment |
| `POST /api/v1/chat` | `{conversation_id, message}` returns `{assistant_message, answer_type, model, sources, usage, cost, credits, debug}` (RAG) |
| `GET /api/v1/usage` | Credit totals, token totals and total cost for the dev user |
| `POST /api/v1/knowledge/search` | **Dev only.** `{query, top_k}` returns the top chunks with score and metadata. No LLM call. |

- OpenAI via the official SDK (Responses API). Model is configurable (`OPENAI_MODEL`, default `gpt-4.1-mini`).
- Token counts come from the provider's reported usage, not estimates.
- Costs use `Decimal` end to end and are stored exactly (12 dp). JSON returns amounts as decimal strings.
- Fixed development user `dev-user` with 1000 credits (no auth yet).
- A request is rejected with **402** before the LLM is called if the worst-case cost exceeds the remaining credits.
- Credits are deducted only after a successful LLM call.
- Errors are returned as clean JSON: `{"error": {"code", "message"}}`. No secrets or raw provider errors.
- Structured `chat_request` log per turn: request ID, retrieval latency, chunks, top score, LLM latency,
  tokens, cost, credits and outcome. Prompts and user content are not logged. Every response carries `X-Request-ID`.
- UI shows credits remaining, total tokens and the last request's cost (top right).
- Knowledge base: 14 curated Wikipedia articles (software engineering / interview prep), 336 chunks,
  `text-embedding-3-small` (1536 dims) in Qdrant collection `interview_knowledge`. Ingestion is
  idempotent; unchanged articles are skipped without embedding calls.
- Dev knowledge search UI at `http://localhost:5173/#/dev/knowledge` (dev builds only).
- **RAG chat**: top-5 chunks above a 0.30 similarity threshold are sent to the LLM as numbered sources.
  Answers show clickable source links. `RAG_DEBUG=true` adds a "Retrieved context" panel with similarity scores.

Example chat response (real run, "How does a Kafka consumer group work?"):

```json
{
  "conversation_id": "verify-step4",
  "assistant_message": "A Kafka consumer group is a mechanism that allows multiple consumers to cooperate ... [1]",
  "answer_type": "grounded",
  "model": "gpt-4.1-mini-2025-04-14",
  "sources": [
    { "title": "Apache Kafka", "source": "wikipedia", "url": "https://en.wikipedia.org/wiki/Apache_Kafka", "score": 0.6244 },
    { "title": "Distributed computing", "source": "wikipedia", "url": "https://en.wikipedia.org/wiki/Distributed_computing", "score": 0.3301 }
  ],
  "usage": { "input_tokens": 1224, "output_tokens": 346, "total_tokens": 1570 },
  "cost": { "currency": "USD", "input_cost": "0.0004896", "output_cost": "0.0005536", "total_cost": "0.0010432" },
  "credits": { "consumed": "1.0432", "remaining": "997.526" },
  "debug": null
}
```

Out-of-scope questions (e.g. "What is quantum mechanics?", best similarity 0.27) return
`answer_type: "no_context"` with a fixed explanatory message, `model: null`, zero cost and no credit deduction.

## Running locally

Prerequisites: Python 3.11+, Node 20+.

### Backend

```bash
cd backend
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env              # then set OPENAI_API_KEY (or LLM_PROVIDER=mock to run offline)
uvicorn app.main:app --reload     # http://localhost:8000, docs at /docs
pytest                            # run tests
```

On startup the app applies migrations (`DB_AUTO_MIGRATE=true`) to `backend/app.db` and seeds `dev-user`.
To run migrations manually: `alembic upgrade head`. To reset local usage: delete `app.db`.

Without `OPENAI_API_KEY` the app still starts and logs a warning. `POST /chat` then returns
`503 llm_not_configured`.

### Knowledge base (Qdrant + ingestion)

```bash
docker compose up -d qdrant                        # Qdrant on :6333 (dashboard: /dashboard)
backend/.venv/Scripts/pip install -r ingestion/requirements.txt   # once (macOS/Linux: .venv/bin/pip)
backend/.venv/Scripts/python -m ingestion.ingest   # from the repo root; requires OPENAI_API_KEY
```

Then search at `http://localhost:5173/#/dev/knowledge`, or:

```bash
curl -X POST localhost:8000/api/v1/knowledge/search -H "Content-Type: application/json" -d '{"query": "How does a Kafka consumer group work?", "top_k": 5}'
```

Without Qdrant the API still starts. Knowledge search returns `503 vector_store_unavailable`, and chat is unaffected.
Edit `ingestion/articles.txt` to change the article list. See [ingestion/README.md](ingestion/README.md).

### Frontend

```bash
cd frontend
npm install
npm run dev                       # http://localhost:5173 (proxies /api to :8000)
npm run build                     # typecheck + production build
```

### VS Code

- **Terminal > Run Task**: `backend: dev`, `frontend: dev`, `backend: test`, `dev: all`,
  `qdrant: up`, `ingestion: run`, `ingestion: dry run`, `ingestion: test`.
- **Run and Debug**: `Backend: FastAPI`.
- Select the interpreter at `backend/.venv`.

### Docker (optional)

```bash
docker compose up --build         # qdrant :6333, backend :8000, frontend :8080
```

## Configuration

| Variable | Default | Purpose |
|----------|---------|---------|
| `LLM_PROVIDER` | `openai` | `openai` or `mock` (offline, fictional pricing) |
| `OPENAI_API_KEY` | none | Required for `openai` |
| `OPENAI_MODEL` | `gpt-4.1-mini` | Must have an entry in `app/services/pricing.py` |
| `OPENAI_TIMEOUT_SECONDS` / `OPENAI_MAX_RETRIES` | `30` / `2` | SDK timeout and retries |
| `LLM_MAX_OUTPUT_TOKENS` | `1024` | Output cap; also used for the pre-flight credit check |
| `DATABASE_URL` | `sqlite:///./app.db` | SQLAlchemy URL |
| `CREDITS_PER_DOLLAR` | `1000` | Credit conversion rate |
| `DEV_USER_INITIAL_CREDITS` | `1000` | Budget for `dev-user` |
| `LOG_JSON` | `false` | JSON log lines instead of `key=value` |
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-3-small` | Collection dimension is derived from it |
| `OPENAI_EMBEDDING_DIMENSIONS` | native | Optional smaller size (3-series only) |
| `QDRANT_URL` / `QDRANT_COLLECTION` | `http://localhost:6333` / `interview_knowledge` | Vector store |
| `WIKIPEDIA_LANGUAGE` / `WIKIPEDIA_ARTICLES` | `en` / `ingestion/articles.txt` | Ingestion source |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `1200` / `200` | Characters |
| `RAG_ENABLED` | `true` | `false` means plain LLM chat (Step 2 behaviour) |
| `RAG_TOP_K` | `5` | Chunks retrieved per question |
| `RAG_MIN_RELEVANCE_SCORE` | `0.30` | Cosine threshold (calibrated for `text-embedding-3-small`); blank means no filter |
| `RAG_NO_CONTEXT_MODE` | `fixed_response` | or `llm` (answer anyway, flagged as not from the knowledge base) |
| `RAG_DEBUG` | `false` | Include retrieved chunks and scores in chat responses and UI |

## Planned phases

| Step | Scope |
|------|-------|
| 1 | Foundation: monorepo, FastAPI, React chat, mock LLM **(done)** |
| 2 | Real LLM provider, token usage, pricing, credits, usage ledger **(done)** |
| 3 | Wikipedia ingestion, embeddings abstraction, Qdrant, standalone retrieval **(done)** |
| 4 | RAG in chat: grounded answers, sources, no-context handling, metering of context tokens **(done)** |
| 5 | LangGraph orchestration, conversation history, context management: topic switching, compression |
| 6 | Auth, Redis caching/distributed state, streaming |
| 7 | AWS deployment: ECS/Fargate, ALB, WAF, RDS |
