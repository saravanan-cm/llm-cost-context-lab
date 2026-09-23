"""Logging setup.

Structured fields are passed with ``logger.info("event", extra={...})`` and rendered as
``key=value`` pairs (default) or as JSON lines (``LOG_JSON=true``).
"""

import json
import logging
from datetime import UTC, datetime

from app.core.request_context import request_id_var

LOG_FORMAT = "%(asctime)s %(levelname)s [%(name)s] %(message)s"

_RESERVED = set(logging.LogRecord("", 0, "", 0, "", (), None).__dict__) | {
    "message",
    "asctime",
    "color_message",
    "taskName",
}


def _extras(record: logging.LogRecord) -> dict[str, object]:
    return {k: v for k, v in record.__dict__.items() if k not in _RESERVED}


class KeyValueFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        base = super().format(record)
        fields = " ".join(f"{k}={v}" for k, v in _extras(record).items())
        return f"{base} {fields}" if fields else base


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.fromtimestamp(record.created, UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            **_extras(record),
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


class RequestIdFilter(logging.Filter):
    """Adds ``request_id`` to records logged while an HTTP request is being handled."""

    def filter(self, record: logging.LogRecord) -> bool:
        request_id = request_id_var.get()
        if request_id is not None and not hasattr(record, "request_id"):
            record.request_id = request_id
        return True


def configure_logging(level: str, json_format: bool = False) -> None:
    handler = logging.StreamHandler()
    handler.addFilter(RequestIdFilter())
    handler.setFormatter(JsonFormatter() if json_format else KeyValueFormatter(LOG_FORMAT))
    logging.basicConfig(level=level.upper(), handlers=[handler], force=True)
