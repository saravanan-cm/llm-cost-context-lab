import type { Source } from "../types/chat";

interface Props {
  sources: Source[];
}

function sourceLabel(source: string): string {
  return source.charAt(0).toUpperCase() + source.slice(1);
}

/** Numbered to match the [n] citations the model is asked to use. */
export default function SourceList({ sources }: Props) {
  return (
    <div className="sources">
      <span className="sources__label">Sources</span>
      <ol>
        {sources.map((s) => (
          <li key={s.url}>
            <a href={s.url} target="_blank" rel="noreferrer">
              {s.title}
            </a>{" "}
            — {sourceLabel(s.source)}
          </li>
        ))}
      </ol>
    </div>
  );
}
