"""Wikipedia response -> Document, using recorded-style payloads and httpx.MockTransport."""

import httpx
import pytest

from ingestion.errors import DocumentNotFoundError, SourceUnavailableError
from ingestion.sources.wikipedia import WikipediaSource, parse_page
from tests.fakes import FETCHED_AT

KAFKA_PAYLOAD = {
    "batchcomplete": True,
    "query": {
        "redirects": [{"from": "Kafka (software)", "to": "Apache Kafka"}],
        "pages": [
            {
                "pageid": 41321945,
                "ns": 0,
                "title": "Apache Kafka",
                "extract": "Apache Kafka is a distributed event store.\n\n== History ==\nKafka was developed at LinkedIn.",
                "fullurl": "https://en.wikipedia.org/wiki/Apache_Kafka",
                "lastrevid": 123,
            }
        ],
    },
}


def test_parse_page_maps_to_document():
    doc = parse_page(KAFKA_PAYLOAD, requested_title="Kafka (software)", language="en", fetched_at=FETCHED_AT)

    assert doc.document_id == "wikipedia:en:41321945"
    assert doc.title == "Apache Kafka"
    assert doc.source == "wikipedia"
    assert doc.source_url == "https://en.wikipedia.org/wiki/Apache_Kafka"
    assert doc.language == "en"
    assert doc.fetched_at == FETCHED_AT
    assert doc.text.startswith("Apache Kafka is a distributed event store.")
    assert doc.metadata == {"page_id": 41321945, "revision_id": 123, "requested_title": "Kafka (software)"}


def test_parse_page_builds_url_when_missing():
    payload = {"query": {"pages": [{"pageid": 1, "title": "Docker (software)", "extract": "x"}]}}

    doc = parse_page(payload, requested_title="Docker (software)", language="de")

    assert doc.source_url == "https://de.wikipedia.org/wiki/Docker_%28software%29"


@pytest.mark.parametrize(
    "page",
    [
        {"ns": 0, "title": "Nope", "missing": True},
        {"title": "<bad>", "invalid": True, "invalidreason": "bad title"},
    ],
)
def test_missing_or_invalid_article(page):
    with pytest.raises(DocumentNotFoundError, match="not found"):
        parse_page({"query": {"pages": [page]}}, requested_title="Nope", language="en")


def test_disambiguation_page_rejected():
    payload = {"query": {"pages": [{"pageid": 1, "title": "Python", "extract": "...", "pageprops": {"disambiguation": ""}}]}}

    with pytest.raises(DocumentNotFoundError, match="disambiguation"):
        parse_page(payload, requested_title="Python", language="en")


def test_api_error_payload():
    with pytest.raises(SourceUnavailableError, match="maxlag"):
        parse_page({"error": {"code": "maxlag", "info": "maxlag exceeded"}}, requested_title="X", language="en")


def _source(handler, sleeps=None) -> WikipediaSource:
    client = httpx.Client(transport=httpx.MockTransport(handler))
    sleep = sleeps.append if sleeps is not None else (lambda seconds: None)
    return WikipediaSource(language="en", user_agent="test", client=client, sleep=sleep, min_interval_seconds=0)


def test_fetch_uses_action_api_with_plain_text_extracts():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(request.url.params)
        seen["host"] = request.url.host
        return httpx.Response(200, json=KAFKA_PAYLOAD)

    doc = _source(handler).fetch("Kafka (software)")

    assert doc.title == "Apache Kafka"
    assert seen["host"] == "en.wikipedia.org"
    assert seen["action"] == "query" and seen["explaintext"] == "1" and seen["redirects"] == "1"
    assert seen["titles"] == "Kafka (software)"


def test_fetch_http_error_is_source_unavailable():
    with pytest.raises(SourceUnavailableError, match="HTTP 404"):
        _source(lambda request: httpx.Response(404)).fetch("Apache Kafka")


def test_rate_limit_is_retried_honouring_retry_after():
    responses = iter([httpx.Response(429, headers={"Retry-After": "7"}), httpx.Response(200, json=KAFKA_PAYLOAD)])
    sleeps: list[float] = []

    doc = _source(lambda request: next(responses), sleeps).fetch("Apache Kafka")

    assert doc.title == "Apache Kafka"
    assert sleeps == [7.0]


def test_persistent_rate_limit_gives_up_with_clear_error():
    sleeps: list[float] = []

    with pytest.raises(SourceUnavailableError, match="HTTP 429"):
        _source(lambda request: httpx.Response(429), sleeps).fetch("Apache Kafka")
    assert sleeps == [2.0, 4.0, 8.0]  # exponential backoff between 4 attempts


def test_fetch_network_error_is_source_unavailable():
    def handler(request):
        raise httpx.ConnectError("boom")

    with pytest.raises(SourceUnavailableError, match="Could not reach"):
        _source(handler).fetch("Apache Kafka")
