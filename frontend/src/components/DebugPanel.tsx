import type { ChatDebug } from "../types/chat";

interface Props {
  debug: ChatDebug;
}

/** Shown only when the backend returns debug data (RAG_DEBUG=true). */
export default function DebugPanel({ debug }: Props) {
  const threshold = debug.min_relevance_score;
  return (
    <details className="debug">
      <summary>
        Debug · {debug.chunks.length} retrieved
        {threshold !== null && ` · min score ${threshold}`}
        {debug.retrieval_latency_ms !== null && ` · retrieval ${debug.retrieval_latency_ms} ms`}
        {debug.llm_latency_ms !== null && ` · LLM ${debug.llm_latency_ms} ms`}
      </summary>
      <p className="debug__path">Graph: {debug.graph_path.join(" → ")}</p>
      {debug.chunks.length > 0 && (
        <ol>
          {debug.chunks.map((c) => (
            <li key={`${c.url}#${c.chunk_index}`} className={c.used ? undefined : "debug__unused"}>
              {c.article_title} <span className="debug__meta">chunk {c.chunk_index}</span> · similarity{" "}
              {c.score.toFixed(3)}
              {!c.used && " (below threshold, not sent)"}
            </li>
          ))}
        </ol>
      )}
    </details>
  );
}
