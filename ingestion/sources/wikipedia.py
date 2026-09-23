"""Wikipedia source using the official MediaWiki Action API (TextExtracts).

``prop=extracts&explaintext=1`` returns the article as plain text with ``== Heading ==``
section markers, with no HTML, infoboxes, navboxes or citations, so no HTML scraping is needed.
"""

import logging
import time
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any
from urllib.parse import quote

import httpx

from app.knowledge.models import Document
from ingestion.errors import DocumentNotFoundError, SourceUnavailableError

logger = logging.getLogger(__name__)

SOURCE_NAME = "wikipedia"
_RETRY_STATUSES = frozenset({429, 503})
_MAX_BACKOFF_SECONDS = 60.0


class WikipediaSource:
    name = SOURCE_NAME

    def __init__(
        self,
        *,
        language: str,
        user_agent: str,
        timeout_seconds: float = 20.0,
        min_interval_seconds: float = 1.0,
        max_attempts: int = 4,
        client: httpx.Client | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.language = language
        self.api_url = f"https://{language}.wikipedia.org/w/api.php"
        self._client = client or httpx.Client(
            headers={"User-Agent": user_agent},
            timeout=timeout_seconds,
            transport=httpx.HTTPTransport(retries=2),
        )
        self._min_interval = min_interval_seconds
        self._max_attempts = max_attempts
        self._sleep = sleep
        self._last_request = float("-inf")

    def fetch(self, reference: str) -> Document:
        params = {
            "action": "query",
            "format": "json",
            "formatversion": "2",
            "prop": "extracts|info|pageprops",
            "explaintext": "1",
            "exsectionformat": "wiki",
            "inprop": "url",
            "ppprop": "disambiguation",
            "redirects": "1",
            "titles": reference,
        }
        try:
            response = self._get_with_backoff(params)
            response.raise_for_status()
            payload = response.json()
        except httpx.HTTPStatusError as exc:
            raise SourceUnavailableError(
                f"Wikipedia API returned HTTP {exc.response.status_code} for '{reference}'"
            ) from exc
        except httpx.HTTPError as exc:
            raise SourceUnavailableError(f"Could not reach Wikipedia API: {type(exc).__name__}") from exc
        except ValueError as exc:
            raise SourceUnavailableError("Wikipedia API returned invalid JSON") from exc
        return parse_page(payload, requested_title=reference, language=self.language)

    def _get_with_backoff(self, params: dict[str, str]) -> httpx.Response:
        """GET, retrying rate-limit/unavailable responses and honouring ``Retry-After``.

        Requests are also spaced by ``min_interval_seconds`` (Wikimedia API etiquette).
        """
        for attempt in range(self._max_attempts):
            wait = self._min_interval - (time.monotonic() - self._last_request)
            if wait > 0:
                self._sleep(wait)
            self._last_request = time.monotonic()
            response = self._client.get(self.api_url, params=params)
            if response.status_code not in _RETRY_STATUSES or attempt == self._max_attempts - 1:
                return response
            delay = _retry_after(response) or 2.0 * 2**attempt
            logger.info(
                "wikipedia_backoff",
                extra={"status": response.status_code, "attempt": attempt + 1, "delay_s": delay},
            )
            self._sleep(min(delay, _MAX_BACKOFF_SECONDS))
        raise AssertionError("unreachable")

    def close(self) -> None:
        self._client.close()


def _retry_after(response: httpx.Response) -> float | None:
    try:
        return float(response.headers["Retry-After"])
    except (KeyError, ValueError):
        return None


def parse_page(
    payload: dict[str, Any],
    *,
    requested_title: str,
    language: str,
    fetched_at: datetime | None = None,
) -> Document:
    """Convert a MediaWiki ``action=query`` response (formatversion=2) into a Document."""
    if "error" in payload:
        info = payload["error"].get("info", "unknown error")
        raise SourceUnavailableError(f"Wikipedia API error: {info}")

    pages = payload.get("query", {}).get("pages") or []
    if not pages:
        raise DocumentNotFoundError(f"Wikipedia returned no page for '{requested_title}'")
    page = pages[0]
    if page.get("missing") or page.get("invalid"):
        raise DocumentNotFoundError(f"Wikipedia article not found: '{requested_title}'")
    if "disambiguation" in (page.get("pageprops") or {}):
        raise DocumentNotFoundError(
            f"'{requested_title}' is a disambiguation page; use a more specific title"
        )

    page_id = page["pageid"]
    title = page["title"]
    source_url = page.get("fullurl") or (
        f"https://{language}.wikipedia.org/wiki/{quote(title.replace(' ', '_'))}"
    )
    return Document(
        document_id=f"{SOURCE_NAME}:{language}:{page_id}",
        text=page.get("extract") or "",
        title=title,
        source=SOURCE_NAME,
        source_url=source_url,
        language=language,
        fetched_at=fetched_at or datetime.now(UTC),
        metadata={
            "page_id": page_id,
            "revision_id": page.get("lastrevid"),
            "requested_title": requested_title,
        },
    )
