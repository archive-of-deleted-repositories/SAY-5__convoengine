"""Configuration loaded from environment variables."""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration.

    Override via env vars prefixed with CONVOENGINE_.
    """

    model_config = SettingsConfigDict(env_prefix="CONVOENGINE_", env_file=".env", extra="ignore")

    db_url: str = "postgresql+psycopg://convo:convo@localhost:5432/convoengine"

    # Confidence thresholds. Below floor returns a template response; below very_low
    # immediately escalates regardless of the rolling window.
    confidence_floor: float = 0.6
    very_low_floor: float = 0.3
    low_conf_window: int = 3

    # Per-channel overrides (chat is typically more forgiving than email).
    chat_confidence_floor: float = 0.55
    email_confidence_floor: float = 0.65

    # Operator queue webhook.
    operator_url: str = "http://localhost:9000/queue"
    operator_hmac_secret: str = "dev-shared-secret"
    operator_timeout_s: float = 5.0

    # LLM provider selection: "fake" | "claude".
    llm_provider: str = "fake"

    # Pagination caps.
    max_summary_turns: int = 8


def get_settings() -> Settings:
    """Return a fresh Settings instance. Cheap enough not to memoize."""

    return Settings()
