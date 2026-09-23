import type { RetrievalDebug } from "../types/chat";

interface Props {
  debug: RetrievalDebug;
}

/** Shown only when the backend returns debug data (RAG_DEBUG=true). */
export default function RetrievalDebugPanel({ debug }: Props) {
  const threshold = debug.min_relevance_score;
  return (
    <details className="debug">
      <summary>
        Retrieved context · top {debug.top_k}
        {threshold !== null && ` · min score ${threshold}`}
        {debug.retrieval_latency_ms !== null && ` · ${debug.retrieval_latency_ms} ms`}
      </summary>
      <ol>
        {debug.chunks.map((c) => (
          <li key={`${c.url}#${c.chunk_index}`} className={c.used ? undefined : "debug__unused"}>
            {c.article_title} <span className="debug__meta">chunk {c.chunk_index}</span> · similarity{" "}
            {c.score.toFixed(3)}
            {!c.used && " (below threshold, not sent)"}
          </li>
        ))}
      </ol>
    </details>
  );
}
