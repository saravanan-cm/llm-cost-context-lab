"""Provider-agnostic LLM interface.

The rest of the application depends only on ``LLMProvider`` and ``LLMResult``; vendor
SDKs are confined to their provider module.
"""

from dataclasses import dataclass, field
from typing import Protocol


@dataclass(frozen=True)
class LLMUsage:
    """Token counts as reported by the provider (not estimated)."""

    input_tokens: int
    output_tokens: int
    total_tokens: int


@dataclass(frozen=True)
class LLMResult:
    text: str
    model: str
    usage: LLMUsage
    provider: str
    request_id: str | None = None
    metadata: dict[str, str] = field(default_factory=dict)


class LLMProvider(Protocol):
    name: str
    model: str
    """The configured model. Used for pricing estimates before a call is made."""

    def generate(
        self, message: str, *, max_output_tokens: int, instructions: str | None = None
    ) -> LLMResult:
        """Return the assistant reply for ``message``.

        ``instructions`` (system prompt) overrides the provider's configured default.

        Raises ``LLMConfigurationError``, ``LLMTimeoutError``, ``LLMRateLimitedError``,
        ``LLMProviderError`` or ``LLMResponseError`` on failure.
        """
        ...
