export type Role = "user" | "assistant";

export type AnswerType = "grounded" | "no_context" | "direct";

export interface ChatMessage {
  id: string;
  role: Role;
  content: string;
  answerType?: AnswerType;
  sources?: Source[];
  debug?: RetrievalDebug | null;
}

export interface ChatRequest {
  conversation_id: string;
  message: string;
}

/** Monetary and credit amounts are exact decimal strings, e.g. "0.000336". */
export type DecimalString = string;

export interface TokenUsage {
  input_tokens: number;
  output_tokens: number;
  total_tokens: number;
}

export interface CostBreakdown {
  currency: string;
  input_cost: DecimalString;
  output_cost: DecimalString;
  total_cost: DecimalString;
}

export interface Source {
  title: string;
  source: string;
  url: string;
  score: number;
}

export interface RetrievedChunkDebug {
  rank: number;
  article_title: string;
  url: string;
  chunk_index: number;
  score: number;
  used: boolean;
}

export interface RetrievalDebug {
  top_k: number;
  min_relevance_score: number | null;
  retrieval_latency_ms: number | null;
  query_tokens: number;
  chunks: RetrievedChunkDebug[];
}

export interface ChatResponse {
  conversation_id: string;
  assistant_message: string;
  answer_type: AnswerType;
  /** null when no LLM call was made (e.g. nothing relevant in the knowledge base). */
  model: string | null;
  sources: Source[];
  usage: TokenUsage;
  cost: CostBreakdown;
  credits: { consumed: DecimalString; remaining: DecimalString };
  /** Present only when the backend runs with RAG_DEBUG=true. */
  debug: RetrievalDebug | null;
}
