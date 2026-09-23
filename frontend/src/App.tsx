import { useEffect, useState } from "react";
import ChatPage from "./pages/ChatPage";
import KnowledgeSearchPage from "./pages/KnowledgeSearchPage";

// The knowledge search page is a development tool; it is not reachable in production builds.
const DEV_TOOLS = import.meta.env.DEV;
const KNOWLEDGE_HASH = "#/dev/knowledge";

function useHash(): string {
  const [hash, setHash] = useState(window.location.hash);
  useEffect(() => {
    const onChange = () => setHash(window.location.hash);
    window.addEventListener("hashchange", onChange);
    return () => window.removeEventListener("hashchange", onChange);
  }, []);
  return hash;
}

export default function App() {
  const hash = useHash();
  const showKnowledge = DEV_TOOLS && hash === KNOWLEDGE_HASH;

  return (
    <>
      {DEV_TOOLS && (
        <nav className="devnav">
          <a href="#/" aria-current={!showKnowledge}>
            Chat
          </a>
          <a href={KNOWLEDGE_HASH} aria-current={showKnowledge}>
            Knowledge search (dev)
          </a>
        </nav>
      )}
      {showKnowledge ? <KnowledgeSearchPage /> : <ChatPage />}
    </>
  );
}
