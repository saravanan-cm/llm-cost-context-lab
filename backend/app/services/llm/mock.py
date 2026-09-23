from app.services.llm.base import LLMResult, LLMUsage


class MockLLMProvider:
    """Offline stand-in (LLM_PROVIDER=mock) for local development without an API key.

    Token counts are rough approximations; real providers report actual usage.
    """

    name = "mock"
    model = "mock-echo"

    def generate(
        self, message: str, *, max_output_tokens: int, instructions: str | None = None
    ) -> LLMResult:
        if instructions is None:
            text = f"[mock] You said: {message}"
        else:
            text = f"[mock] Received a {len(message)}-character prompt with custom instructions."
        input_tokens = max(1, (len(message) + len(instructions or "")) // 4)
        output_tokens = min(max_output_tokens, max(1, len(text) // 4))
        return LLMResult(
            text=text,
            model=self.model,
            usage=LLMUsage(input_tokens, output_tokens, input_tokens + output_tokens),
            provider=self.name,
        )
