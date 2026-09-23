from typing import Protocol

from app.knowledge.models import Document


class KnowledgeSource(Protocol):
    """A source of documents, e.g. Wikipedia, AWS docs, internal docs."""

    name: str

    def fetch(self, reference: str) -> Document:
        """Fetch one document by a source-specific reference (a title, URL, path, ...).

        Raises ``SourceUnavailableError``, ``DocumentNotFoundError``.
        """
        ...
