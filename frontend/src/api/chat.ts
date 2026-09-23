import type { ChatRequest, ChatResponse } from "../types/chat";
import { request } from "./client";

export function sendChatMessage(body: ChatRequest): Promise<ChatResponse> {
  return request<ChatResponse>("/chat", { method: "POST", body: JSON.stringify(body) });
}
