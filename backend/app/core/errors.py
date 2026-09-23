"""Application errors.

``message`` is always safe to return to clients. Internal details (provider errors,
stack traces) are logged where the error is raised and never put in ``message``.
"""


class AppError(Exception):
    status_code: int = 500
    code: str = "internal_error"
    default_message: str = "An unexpected error occurred."

    def __init__(self, message: str | None = None) -> None:
        self.message = message or self.default_message
        super().__init__(self.message)


class LLMConfigurationError(AppError):
    status_code = 503
    code = "llm_not_configured"
    default_message = "The LLM provider is not configured correctly."


class LLMTimeoutError(AppError):
    status_code = 504
    code = "llm_timeout"
    default_message = "The LLM provider timed out. Please try again."


class LLMRateLimitedError(AppError):
    status_code = 503
    code = "llm_rate_limited"
    default_message = "The LLM provider is currently rate limited. Please try again shortly."


class LLMProviderError(AppError):
    status_code = 502
    code = "llm_provider_error"
    default_message = "The LLM provider returned an error."


class LLMResponseError(AppError):
    status_code = 502
    code = "llm_invalid_response"
    default_message = "The LLM provider returned an unexpected response."


class PricingNotConfiguredError(AppError):
    status_code = 500
    code = "pricing_not_configured"
    default_message = "Pricing is not configured for the selected model."


class InvalidQuestionError(AppError):
    status_code = 422
    code = "invalid_question"
    default_message = "The question is empty."


class InsufficientCreditsError(AppError):
    status_code = 402
    code = "insufficient_credits"
    default_message = "Insufficient credits for this request."


class UserNotFoundError(AppError):
    status_code = 404
    code = "user_not_found"
    default_message = "User not found."


class EmbeddingConfigurationError(AppError):
    status_code = 503
    code = "embedding_not_configured"
    default_message = "The embedding provider is not configured correctly."


class EmbeddingProviderError(AppError):
    status_code = 502
    code = "embedding_provider_error"
    default_message = "The embedding provider returned an error."


class EmbeddingDimensionError(AppError):
    status_code = 500
    code = "embedding_dimension_mismatch"
    default_message = "The embedding provider returned vectors of an unexpected dimension."


class VectorStoreUnavailableError(AppError):
    status_code = 503
    code = "vector_store_unavailable"
    default_message = "The knowledge base is currently unavailable."


class VectorStoreConfigurationError(AppError):
    status_code = 500
    code = "vector_store_misconfigured"
    default_message = "The knowledge base is not configured correctly."


class KnowledgeBaseNotReadyError(AppError):
    status_code = 503
    code = "knowledge_base_not_ready"
    default_message = "The knowledge base has not been built yet. Run the ingestion pipeline first."
