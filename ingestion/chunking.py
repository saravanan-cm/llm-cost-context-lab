"""Text-aware chunking.

Strategy: split into paragraphs; split paragraphs longer than ``chunk_size`` into sentences
(and over-long sentences into words); then greedily pack those pieces into chunks of at most
``chunk_size`` characters. Each new chunk starts with the trailing pieces of the previous chunk
up to ``chunk_overlap`` characters, so context spanning a boundary is not lost.

Sizes are in characters, not tokens. That is simple, deterministic and model-independent.
"""

import re
from dataclasses import dataclass

from app.knowledge.models import Chunk, Document, make_chunk_id

_PARAGRAPH_BREAK = re.compile(r"\n\s*\n")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")


@dataclass(frozen=True)
class _Piece:
    text: str
    starts_paragraph: bool


class TextChunker:
    def __init__(self, chunk_size: int, chunk_overlap: int) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if not 0 <= chunk_overlap < chunk_size:
            raise ValueError("chunk_overlap must be >= 0 and smaller than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    @property
    def config_signature(self) -> str:
        return f"chars:{self.chunk_size}:{self.chunk_overlap}"

    def chunk(self, document: Document) -> list[Chunk]:
        texts = self.split_text(document.text)
        return [
            Chunk(
                chunk_id=make_chunk_id(document.document_id, index),
                document_id=document.document_id,
                chunk_index=index,
                text=text,
                source=document.source,
                article_title=document.title,
                source_url=document.source_url,
                metadata={
                    **document.metadata,
                    "language": document.language,
                    "fetched_at": document.fetched_at.isoformat(),
                    "chunk_count": len(texts),
                },
            )
            for index, text in enumerate(texts)
        ]

    def split_text(self, text: str) -> list[str]:
        chunks: list[str] = []
        current: list[_Piece] = []

        for piece in self._pieces(text):
            if current and _length(current + [piece]) > self.chunk_size:
                chunks.append(_join(current))
                current = self._overlap_tail(current)
                while current and _length(current + [piece]) > self.chunk_size:
                    current.pop(0)
            current.append(piece)

        if current:
            chunks.append(_join(current))
        return chunks

    def _pieces(self, text: str) -> list[_Piece]:
        pieces: list[_Piece] = []
        for paragraph in _PARAGRAPH_BREAK.split(text):
            paragraph = " ".join(paragraph.split())
            if not paragraph:
                continue
            parts = [paragraph] if len(paragraph) <= self.chunk_size else self._split_long(paragraph)
            pieces.extend(_Piece(part, starts_paragraph=i == 0) for i, part in enumerate(parts))
        return pieces

    def _split_long(self, paragraph: str) -> list[str]:
        parts: list[str] = []
        for sentence in _SENTENCE_END.split(paragraph):
            if len(sentence) <= self.chunk_size:
                parts.append(sentence)
            else:
                parts.extend(self._split_words(sentence))
        return parts

    def _split_words(self, sentence: str) -> list[str]:
        parts: list[str] = []
        current = ""
        for word in sentence.split(" "):
            while len(word) > self.chunk_size:  # pathological: a single huge "word"
                if current:
                    parts.append(current)
                    current = ""
                parts.append(word[: self.chunk_size])
                word = word[self.chunk_size :]
            candidate = f"{current} {word}" if current else word
            if len(candidate) > self.chunk_size:
                parts.append(current)
                current = word
            else:
                current = candidate
        if current:
            parts.append(current)
        return parts

    def _overlap_tail(self, pieces: list[_Piece]) -> list[_Piece]:
        tail: list[_Piece] = []
        for piece in reversed(pieces):
            if _length([piece, *tail]) > self.chunk_overlap:
                break
            tail.insert(0, piece)
        return tail


def _join(pieces: list[_Piece]) -> str:
    out = pieces[0].text
    for piece in pieces[1:]:
        out += ("\n\n" if piece.starts_paragraph else " ") + piece.text
    return out


def _length(pieces: list[_Piece]) -> int:
    return len(_join(pieces)) if pieces else 0
