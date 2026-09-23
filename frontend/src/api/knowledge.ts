import type { KnowledgeSearchRequest, KnowledgeSearchResponse } from "../types/knowledge";
import { request } from "./client";

export function searchKnowledge(body: KnowledgeSearchRequest): Promise<KnowledgeSearchResponse> {
  return request<KnowledgeSearchResponse>("/knowledge/search", {
    method: "POST",
    body: JSON.stringify(body),
  });
}
