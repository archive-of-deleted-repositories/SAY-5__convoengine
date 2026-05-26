"""TestClient-based smoke tests for the FastAPI surface."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from convoengine.api import build_app
from convoengine.orchestrator.conversation import ConversationState
from convoengine.orchestrator.llm_client import FakeLLMProvider, ScriptedResponse


def _scripted(
    text: str, conf: float, state: ConversationState, action: str = "reply"
) -> ScriptedResponse:
    return ScriptedResponse(
        raw_json=(
            f'{{"schema_version":1,"action":"{action}","response_text":"{text}",'
            f'"confidence":{conf},"suggested_state":"{state.value}"}}'
        )
    )


@pytest.fixture
def client() -> Iterator[TestClient]:
    provider = FakeLLMProvider()
    provider.script_default(_scripted("Sure I can help.", 0.9, ConversationState.ANSWERING))
    # Use a per-test temp file so the API's session_factory and the engine see
    # the same SQLite database. ":memory:" is connection-scoped and won't work.
    fd, path = tempfile.mkstemp(suffix=".sqlite")
    os.close(fd)
    try:
        app = build_app(provider=provider, db_url=f"sqlite:///{path}")
        with TestClient(app) as c:
            yield c
    finally:
        if os.path.exists(path):
            os.unlink(path)


def test_healthz_returns_ok(client: TestClient) -> None:
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_ingest_email_returns_reply(client: TestClient) -> None:
    r = client.post(
        "/v1/ingest/email",
        json={"sender": "alice@example.com", "body": "Hi"},
    )
    assert r.status_code == 200
    payload = r.json()
    assert payload["decision"] == "use_llm"
    assert payload["reply"] == "Sure I can help."
    assert payload["conversation_id"].startswith("conv-")


def test_ingest_chat_then_get_conversation(client: TestClient) -> None:
    r = client.post(
        "/v1/ingest/chat",
        json={"user_id": "u-1", "user_email": "alice@example.com", "text": "Hi"},
    )
    assert r.status_code == 200
    cid = r.json()["conversation_id"]

    detail = client.get(f"/v1/conversations/{cid}")
    assert detail.status_code == 200
    body = detail.json()
    assert body["state"] == ConversationState.ANSWERING.value
    assert len(body["turns"]) == 2


def test_operator_take_over_rejected_when_not_escalated(client: TestClient) -> None:
    r = client.post(
        "/v1/ingest/email",
        json={"sender": "alice@example.com", "body": "Hi"},
    )
    cid = r.json()["conversation_id"]
    take = client.post(
        f"/v1/operator/take_over/{cid}",
        json={"body": "Hi, this is Pat.", "operator_id": "op-1"},
    )
    assert take.status_code == 409
