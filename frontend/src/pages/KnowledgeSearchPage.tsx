import { useState, type FormEvent } from "react";
import { searchKnowledge } from "../api/knowledge";
import type { KnowledgeSearchResponse } from "../types/knowledge";

const TOP_K_OPTIONS = [1, 3, 5, 10, 20];

/** Development-only page for testing retrieval in isolation (no LLM involved). */
export default function KnowledgeSearchPage() {
  const [query, setQuery] = useState("How does a Kafka consumer group work?");
  const [topK, setTopK] = useState(5);
  const [pending, setPending] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [response, setResponse] = useState<KnowledgeSearchResponse | null>(null);

  async function handleSubmit(event: FormEvent) {
    event.preventDefault();
    const trimmed = query.trim();
    if (!trimmed) return;
    setPending(true);
    setError(null);
    try {
      setResponse(await searchKnowledge({ query: trimmed, top_k: topK }));
    } catch (err) {
      setResponse(null);
      setError(err instanceof Error ? err.message : "Unknown error");
    } finally {
      setPending(false);
    }
  }

  return (
    <main className="chat">
      <header className="chat__header">
        <h1>Knowledge search (dev)</h1>
      </header>
      <form className="input" onSubmit={handleSubmit}>
        <input
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Search the knowledge base..."
          aria-label="Query"
          disabled={pending}
        />
        <label className="topk">
          Top-K
          <select value={topK} onChange={(e) => setTopK(Number(e.target.value))} disabled={pending}>
            {TOP_K_OPTIONS.map((k) => (
              <option key={k} value={k}>
                {k}
              </option>
            ))}
          </select>
        </label>
        <button type="submit" disabled={pending || !query.trim()}>
          {pending ? "Searching..." : "Search"}
        </button>
      </form>

      {error && <p className="error">{error}</p>}
      {response && (
        <p className="status">
          {response.results.length} results · {response.embedding_model} · {response.query_tokens} query
          tokens · {response.latency_ms} ms
        </p>
      )}

      <ol className="results">
        {response?.results.map((r) => (
          <li key={r.chunk_id} className="result">
            <div className="result__meta">
              <span className="result__score">{r.score.toFixed(3)}</span>
              <a href={r.source_url} target="_blank" rel="noreferrer">
                {r.article_title}
              </a>
              <span className="result__chunk">
                {r.source} · chunk {r.chunk_index}
              </span>
            </div>
            <p className="result__text">{r.text}</p>
          </li>
        ))}
      </ol>
    </main>
  );
}
