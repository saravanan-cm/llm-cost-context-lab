class IngestionError(Exception):
    """A per-document failure. The pipeline records it and continues with other documents."""


class SourceUnavailableError(IngestionError):
    """The knowledge source could not be reached or returned an error."""


class DocumentNotFoundError(IngestionError):
    """The requested document does not exist (or is not a usable article)."""


class EmptyDocumentError(IngestionError):
    """The document has no usable text after cleaning."""
