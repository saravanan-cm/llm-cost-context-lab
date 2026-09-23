import { useCallback, useEffect, useState } from "react";
import { sendChatMessage } from "../api/chat";
import { fetchUsage } from "../api/usage";
import MessageInput from "../components/MessageInput";
import MessageList from "../components/MessageList";
import UsageIndicator from "../components/UsageIndicator";
import type { ChatMessage, CostBreakdown } from "../types/chat";
import type { UsageSummary } from "../types/usage";

export default function ChatPage() {
  const [conversationId] = useState(() => crypto.randomUUID());
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [pending, setPending] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [usage, setUsage] = useState<UsageSummary | null>(null);
  const [lastCost, setLastCost] = useState<CostBreakdown | null>(null);

  const refreshUsage = useCallback(async () => {
    try {
      setUsage(await fetchUsage());
    } catch {
      // Usage display is non-critical; keep the last known value.
    }
  }, []);

  useEffect(() => {
    void refreshUsage();
  }, [refreshUsage]);

  async function handleSend(text: string) {
    setError(null);
    setMessages((prev) => [...prev, { id: crypto.randomUUID(), role: "user", content: text }]);
    setPending(true);
    try {
      const reply = await sendChatMessage({ conversation_id: conversationId, message: text });
      setMessages((prev) => [
        ...prev,
        {
          id: crypto.randomUUID(),
          role: "assistant",
          content: reply.assistant_message,
          answerType: reply.answer_type,
          sources: reply.sources,
          debug: reply.debug,
        },
      ]);
      setLastCost(reply.cost);
      await refreshUsage();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unknown error");
    } finally {
      setPending(false);
    }
  }

  return (
    <main className="chat">
      <header className="chat__header">
        <h1>LLM Cost & Context Lab</h1>
        <UsageIndicator usage={usage} lastCost={lastCost} />
      </header>
      <MessageList messages={messages} />
      {pending && <p className="status">Assistant is thinking...</p>}
      {error && <p className="error">{error}</p>}
      <MessageInput onSend={handleSend} disabled={pending} />
    </main>
  );
}
