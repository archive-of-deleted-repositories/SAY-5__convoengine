"""Explicit provider selection and paid-mode configuration validation."""

from convoengine.config import Settings
from convoengine.orchestrator.llm_client import ClaudeProvider, FakeLLMProvider, LLMProvider


def build_provider(settings: Settings) -> LLMProvider:
    if settings.llm_provider == "fake":
        return FakeLLMProvider()
    if settings.llm_provider == "claude":
        key = settings.service_api_key.get_secret_value()
        if not 32 <= len(key) <= 512 or not key.isascii() or any(char.isspace() for char in key):
            raise ValueError(
                "Claude mode requires CONVOENGINE_SERVICE_API_KEY (32-512 non-space ASCII characters)."
            )
        return ClaudeProvider(
            api_key=settings.claude_api_key.get_secret_value(), model=settings.claude_model
        )
    raise ValueError("CONVOENGINE_LLM_PROVIDER must be fake or claude.")
