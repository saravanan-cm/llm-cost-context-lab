"""OpenAI implementation of ``LLMProvider`` using the Responses API."""

import logging
from typing import Any

import openai
from pydantic import SecretStr

from app.core.errors import (
    LLMConfigurationError,
    LLMProviderError,
    LLMRateLimitedError,
    LLMResponseError,
    LLMTimeoutError,
)
from app.services.llm.base import LLMResult, LLMUsage

logger = logging.getLogger(__name__)


class OpenAIProvider:
    name = "openai"

    def __init__(
        self,
        *,
        api_key: SecretStr | None,
        model: str,
        instructions: str,
        timeout_seconds: float,
        max_retries: int,
        client: Any | None = None,
    ) -> None:
        self.model = model
        self._api_key = api_key
        self._instructions = instructions
        self._timeout = timeout_seconds
        self._max_retries = max_retries
        self._client = client

    def _get_client(self) -> Any:
        # Created lazily so the app starts without a key and only fails when a request is made.
        if self._client is None:
            if self._api_key is None or not self._api_key.get_secret_value().strip():
                raise LLMConfigurationError(
                    "The LLM provider is not configured: OPENAI_API_KEY is missing."
                )
            self._client = openai.OpenAI(
                api_key=self._api_key.get_secret_value(),
                timeout=self._timeout,
                max_retries=self._max_retries,
            )
        return self._client

    def generate(
        self, message: str, *, max_output_tokens: int, instructions: str | None = None
    ) -> LLMResult:
        client = self._get_client()
        try:
            response = client.responses.create(
                model=self.model,
                instructions=instructions if instructions is not None else self._instructions,
                input=message,
                max_output_tokens=max_output_tokens,
            )
        except openai.APITimeoutError as exc:
            raise LLMTimeoutError() from exc
        except openai.AuthenticationError as exc:
            logger.error("openai_auth_failed", extra={"status": exc.status_code})
            raise LLMConfigurationError("The LLM provider rejected the configured credentials.") from exc
        except openai.NotFoundError as exc:
            logger.error("openai_model_not_found", extra={"model": self.model})
            raise LLMConfigurationError("The configured LLM model is not available.") from exc
        except openai.RateLimitError as exc:
            raise LLMRateLimitedError() from exc
        except openai.APIStatusError as exc:
            logger.error("openai_api_error", extra={"status": exc.status_code, "type": type(exc).__name__})
            raise LLMProviderError() from exc
        except openai.APIConnectionError as exc:
            logger.error("openai_connection_error", extra={"type": type(exc).__name__})
            raise LLMProviderError("Could not reach the LLM provider.") from exc

        return self._to_result(response)

    def _to_result(self, response: Any) -> LLMResult:
        usage = getattr(response, "usage", None)
        text = getattr(response, "output_text", None)
        if usage is None or not isinstance(text, str) or not text:
            logger.error(
                "openai_unexpected_response",
                extra={"has_usage": usage is not None, "status": getattr(response, "status", None)},
            )
            raise LLMResponseError()
        try:
            llm_usage = LLMUsage(
                input_tokens=int(usage.input_tokens),
                output_tokens=int(usage.output_tokens),
                total_tokens=int(usage.total_tokens),
            )
        except (AttributeError, TypeError, ValueError) as exc:
            raise LLMResponseError() from exc

        metadata = {}
        if status := getattr(response, "status", None):
            metadata["status"] = str(status)
        return LLMResult(
            text=text,
            model=str(getattr(response, "model", None) or self.model),
            usage=llm_usage,
            provider=self.name,
            request_id=getattr(response, "id", None),
            metadata=metadata,
        )
