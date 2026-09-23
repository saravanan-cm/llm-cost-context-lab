import type { ChatMessage } from "../types/chat";
import DebugPanel from "./DebugPanel";
import SourceList from "./SourceList";

interface Props {
  messages: ChatMessage[];
}

export default function MessageList({ messages }: Props) {
  if (messages.length === 0) {
    return <p className="empty">Start the conversation by sending a message.</p>;
  }
  return (
    <ul className="messages">
      {messages.map((m) => (
        <li
          key={m.id}
          className={`message message--${m.role}${m.answerType === "no_context" ? " message--no-context" : ""}`}
        >
          <span className="message__role">{m.role === "user" ? "You" : "Assistant"}</span>
          <p>{m.content}</p>
          {m.sources && m.sources.length > 0 && <SourceList sources={m.sources} />}
          {m.debug && <DebugPanel debug={m.debug} />}
        </li>
      ))}
    </ul>
  );
}
