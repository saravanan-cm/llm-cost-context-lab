"""Text cleanup applied to every document before chunking."""

import dataclasses
import re

from app.knowledge.models import Document

# Sections that add little retrieval value (citations, link lists).
EXCLUDED_SECTIONS = frozenset(
    {
        "references",
        "external links",
        "see also",
        "further reading",
        "notes",
        "notes and references",
        "bibliography",
        "sources",
        "citations",
        "footnotes",
        "works cited",
    }
)

_HEADING = re.compile(r"^(={2,6})\s*(.+?)\s*\1$")
# TextExtracts renders math as "{\displaystyle ...}"; it is noise for embeddings.
_DISPLAYSTYLE = re.compile(r"\{\\displaystyle[^{}]*(?:\{[^{}]*\}[^{}]*)*\}")
_SPACES = re.compile(r"[ \t ]+")
_BLANK_LINES = re.compile(r"\n{3,}")


class TextCleaner:
    def __init__(self, excluded_sections: frozenset[str] = EXCLUDED_SECTIONS) -> None:
        self._excluded = excluded_sections

    def clean(self, document: Document) -> Document:
        return dataclasses.replace(document, text=self.clean_text(document.text))

    def clean_text(self, text: str) -> str:
        lines: list[str] = []
        skip_below: int | None = None  # heading level of an excluded section being skipped

        for raw in text.splitlines():
            line = raw.strip()
            heading = _HEADING.match(line)
            if heading:
                level, name = len(heading.group(1)), heading.group(2).strip()
                if skip_below is not None and level > skip_below:
                    continue
                skip_below = None
                if name.lower() in self._excluded:
                    skip_below = level
                    continue
                # Keep headings as their own paragraph; they give chunks useful context.
                lines.extend(["", name, ""])
                continue
            if skip_below is not None:
                continue
            line = _SPACES.sub(" ", _DISPLAYSTYLE.sub("", line)).strip()
            lines.append(line)

        return _BLANK_LINES.sub("\n\n", "\n".join(lines)).strip()
