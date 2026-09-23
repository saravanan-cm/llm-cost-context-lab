export interface KnowledgeSearchRequest {
  query: string;
  top_k: number;
}

export interface KnowledgeChunk {
  text: string;
  score: number;
  source: string;
  article_title: string;
  source_url: string;
  document_id: string;
  chunk_id: string;
  chunk_index: number;
  metadata: Record<string, unknown>;
}

export interface KnowledgeSearchResponse {
  query: string;
  top_k: number;
  embedding_model: string;
  query_tokens: number;
  latency_ms: number;
  results: KnowledgeChunk[];
}
