"""Provider selection must fail closed before database initialization."""

from __future__ import annotations

import os
import subprocess
import sys
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool

from convoengine import api
from convoengine.config import Settings
from convoengine.orchestrator.llm_client import ClaudeProvider, FakeLLMProvider


@pytest.mark.parametrize("provider", ["claude", "unknown"])
def test_invalid_provider_configuration_rejected_before_database(
    monkeypatch: pytest.MonkeyPatch, provider: str
) -> None:
    monkeypatch.setattr(
        api, "get_settings", lambda: Settings(_env_file=None, llm_provider=provider)
    )
    database = Mock()
    monkeypatch.setattr(api, "make_engine", database)
    with pytest.raises(ValueError):
        api.build_app(db_url="sqlite://")
    database.assert_not_called()


def test_paid_ingest_rejects_missing_service_auth_before_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = Settings(
        _env_file=None,
        llm_provider="claude",
        claude_api_key="synthetic-provider-key",
        claude_model="synthetic-model",
        service_api_key="s" * 32,
    )
    monkeypatch.setattr(api, "get_settings", lambda: settings)
    monkeypatch.setattr(
        api,
        "make_engine",
        lambda _: create_engine(
            "sqlite://", poolclass=StaticPool, connect_args={"check_same_thread": False}
        ),
    )
    provider = FakeLLMProvider(name="claude")
    with TestClient(api.build_app(provider=provider, db_url="sqlite://")) as client:
        response = client.post(
            "/v1/ingest/email", json={"sender": "synthetic@example.test", "body": "Hello"}
        )
    assert response.status_code == 401
    assert provider.calls == []


def test_invalid_module_configuration_is_not_hidden_by_empty_fallback_app() -> None:
    env = {**os.environ, "CONVOENGINE_LLM_PROVIDER": "unknown", "CONVOENGINE_DB_URL": "sqlite://"}
    result = subprocess.run(
        [sys.executable, "-c", "import convoengine.api"],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0


def test_injected_claude_cannot_bypass_paid_configuration(monkeypatch: pytest.MonkeyPatch) -> None:
    database = Mock()
    monkeypatch.setattr(api, "make_engine", database)
    with pytest.raises(ValueError):
        api.build_app(
            provider=ClaudeProvider("synthetic-provider-key", "synthetic-model"),
            settings=Settings(_env_file=None, llm_provider="fake"),
            db_url="sqlite://",
        )
    database.assert_not_called()


def test_paid_admission_keeps_headroom_in_the_default_database_pool() -> None:
    with pytest.raises(ValueError):
        Settings(_env_file=None, paid_max_concurrency=9)
