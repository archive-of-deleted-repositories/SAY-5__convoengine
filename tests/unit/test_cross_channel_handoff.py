"""Cross-channel handoff: an email conversation continues on chat with full context."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from convoengine.api import build_app
from convoengine.orchestrator.conversation import ConversationState
from convoengine.orchestrator.llm_client import FakeLLMProvider, ScriptedResponse


def _scripted(action: str, text: str, conf: float, state: ConversationState) -> ScriptedResponse:
    return ScriptedResponse(
        raw_json=(
            f'{{"schema_version":1,"action":"{action}","response_text":"{text}",'
            f'"confidence":{conf},"suggested_state":"{state.value}"}}'
        )
    )


@pytest.fixture
def client_with_provider() -> Iterator[tuple[TestClient, FakeLLMProvider]]:
    provider = FakeLLMProvider()
    # GREETING -> CLARIFYING via user_message-style reply with clarify action.
    provider.script_state(
        ConversationState.GREETING,
        _scripted("clarify", "Could you tell me more?", 0.8, ConversationState.CLARIFYING),
    )
    # CLARIFYING -> ANSWERING when user clarifies.
    provider.script_state(
        ConversationState.CLARIFYING,
        _scripted("reply", "Got it. Here's the answer.", 0.9, ConversationState.ANSWERING),
    )
    fd, path = tempfile.mkstemp(suffix=".sqlite")
    os.close(fd)
    try:
        app = build_app(provider=provider, db_url=f"sqlite:///{path}")
        with TestClient(app) as c:
            yield c, provider
    finally:
        if os.path.exists(path):
            os.unlink(path)


def test_email_then_chat_continues_same_conversation_in_clarifying_state(
    client_with_provider: tuple[TestClient, FakeLLMProvider],
) -> None:
    client, _provider = client_with_provider

    # Turn 1: email in. After this, GREETING -> CLARIFYING via the "clarify" action.
    r1 = client.post("/v1/ingest/email", json={"sender": "alice@example.com", "body": "Hi"})
    assert r1.status_code == 200
    conv_id = r1.json()["conversation_id"]

    snapshot1 = client.get(f"/v1/conversations/{conv_id}").json()
    assert snapshot1["state"] == ConversationState.CLARIFYING.value
    assert len(snapshot1["turns"]) == 2  # in + out

    # Turn 2: same user opens a chat. user_id is new, but user_email is on file.
    r2 = client.post(
        "/v1/ingest/chat",
        json={
            "user_id": "u-99",
            "user_email": "alice@example.com",
            "text": "Sure, here is the detail.",
        },
    )
    assert r2.status_code == 200
    # Critical assertion: same conversation_id.
    assert r2.json()["conversation_id"] == conv_id

    snapshot2 = client.get(f"/v1/conversations/{conv_id}").json()
    # We were in CLARIFYING; CLARIFYING-state script returns "reply", which
    # maps to user_clear_question -> ANSWERING.
    assert snapshot2["state"] == ConversationState.ANSWERING.value
    # Now should have 4 turns total: email-in, email-out, chat-in, chat-out.
    assert len(snapshot2["turns"]) == 4
    channels = {t["channel"] for t in snapshot2["turns"]}
    assert channels == {"email", "chat"}


def test_chat_without_email_on_file_creates_separate_conversation(
    client_with_provider: tuple[TestClient, FakeLLMProvider],
) -> None:
    client, _provider = client_with_provider

    r1 = client.post("/v1/ingest/email", json={"sender": "alice@example.com", "body": "Hi"})
    conv_id_email = r1.json()["conversation_id"]

    # No user_email provided; the chat user_id is brand new.
    r2 = client.post(
        "/v1/ingest/chat",
        json={"user_id": "u-stranger", "text": "Hi there"},
    )
    conv_id_chat = r2.json()["conversation_id"]
    assert conv_id_chat != conv_id_email


def test_link_identities_endpoint_attaches_chat_user_to_email_conversation(
    client_with_provider: tuple[TestClient, FakeLLMProvider],
) -> None:
    client, _provider = client_with_provider

    r1 = client.post("/v1/ingest/email", json={"sender": "alice@example.com", "body": "Hi"})
    conv_id = r1.json()["conversation_id"]

    link = client.post(
        "/v1/identities/link",
        json={"conversation_id": conv_id, "channel": "chat", "identity": "u-77"},
    )
    assert link.status_code == 200
    assert link.json() == {"conversation_id": conv_id, "linked": True}

    # Subsequent chat by u-77 with no email-on-file should now hit the same conv.
    r2 = client.post(
        "/v1/ingest/chat",
        json={"user_id": "u-77", "text": "follow up"},
    )
    assert r2.json()["conversation_id"] == conv_id


def test_subsequent_chat_uses_first_chat_conversation(
    client_with_provider: tuple[TestClient, FakeLLMProvider],
) -> None:
    client, _provider = client_with_provider

    # Establish via email + chat handoff.
    r1 = client.post("/v1/ingest/email", json={"sender": "alice@example.com", "body": "Hi"})
    conv_id = r1.json()["conversation_id"]
    client.post(
        "/v1/ingest/chat",
        json={
            "user_id": "u-99",
            "user_email": "alice@example.com",
            "text": "Sure, here is the detail.",
        },
    )
    # Now a chat-only follow-up should land on the same conversation.
    r3 = client.post(
        "/v1/ingest/chat",
        json={"user_id": "u-99", "text": "And one more thing"},
    )
    assert r3.json()["conversation_id"] == conv_id
