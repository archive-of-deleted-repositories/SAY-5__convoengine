"""End-to-end operator handoff: low-conf trigger -> queue post -> operator takeover."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from convoengine.api import build_app
from convoengine.operator.escalation import (
    HTTPSender,
    OperatorClient,
    compute_hmac,
    verify_signature,
)
from convoengine.orchestrator.conversation import ConversationState
from convoengine.orchestrator.llm_client import FakeLLMProvider, ScriptedResponse


class CapturingSender:
    """In-memory operator-queue receiver that records and verifies HMAC."""

    def __init__(self, secret: str) -> None:
        self._secret = secret
        self.received: list[dict] = []

    async def post(self, url: str, headers: dict[str, str], body: bytes) -> tuple[int, str]:
        sig = headers.get("X-ConvoEngine-Signature", "")
        if not verify_signature(self._secret, body, sig):
            return 401, "invalid signature"
        self.received.append(json.loads(body))
        return 202, "ok"


def _scripted(action: str, text: str, conf: float, state: ConversationState) -> ScriptedResponse:
    return ScriptedResponse(
        raw_json=(
            f'{{"schema_version":1,"action":"{action}","response_text":"{text}",'
            f'"confidence":{conf},"suggested_state":"{state.value}"}}'
        )
    )


@pytest.fixture
def setup() -> Iterator[tuple[TestClient, CapturingSender]]:
    secret = "operator-shared-secret"
    sender = CapturingSender(secret)

    provider = FakeLLMProvider()
    provider.script_default(_scripted("clarify", "uh?", 0.1, ConversationState.CLARIFYING))

    fd, path = tempfile.mkstemp(suffix=".sqlite")
    os.close(fd)
    try:
        app = build_app(provider=provider, db_url=f"sqlite:///{path}")
        # Patch the operator client with our capturing sender + matching secret.
        from convoengine.api import AppState

        st: AppState = app.state.convo
        st.engine.operator_client = OperatorClient(
            url="http://operator/queue", secret=secret, sender=sender
        )
        with TestClient(app) as c:
            yield c, sender
    finally:
        if os.path.exists(path):
            os.unlink(path)


def test_low_conf_trigger_posts_signed_payload_to_operator_queue(
    setup: tuple[TestClient, CapturingSender],
) -> None:
    client, sender = setup
    r = client.post("/v1/ingest/email", json={"sender": "alice@example.com", "body": "Help"})
    assert r.status_code == 200
    assert r.json()["decision"] == "escalate"
    assert len(sender.received) == 1
    payload = sender.received[0]
    assert payload["current_state"] == "escalated"
    assert payload["trigger"].startswith("confidence")
    assert len(payload["last_turns"]) >= 1
    # The signature was already verified inside CapturingSender.


def test_operator_take_over_transitions_to_operator_active(
    setup: tuple[TestClient, CapturingSender],
) -> None:
    client, _ = setup
    r = client.post("/v1/ingest/email", json={"sender": "alice@example.com", "body": "Help"})
    conv_id = r.json()["conversation_id"]

    take = client.post(
        f"/v1/operator/take_over/{conv_id}",
        json={"body": "Hi, I'll help from here.", "operator_id": "op-1"},
    )
    assert take.status_code == 200
    assert take.json()["state"] == ConversationState.OPERATOR_ACTIVE.value


def test_after_operator_takeover_subsequent_user_messages_do_not_autoreply(
    setup: tuple[TestClient, CapturingSender],
) -> None:
    client, _ = setup
    r = client.post("/v1/ingest/email", json={"sender": "alice@example.com", "body": "Help"})
    conv_id = r.json()["conversation_id"]
    client.post(
        f"/v1/operator/take_over/{conv_id}",
        json={"body": "Hi.", "operator_id": "op-1"},
    )

    r2 = client.post(
        "/v1/ingest/email",
        json={"sender": "alice@example.com", "body": "Followup question"},
    )
    # Engine returns reply=None when OPERATOR_ACTIVE.
    assert r2.status_code == 200
    assert r2.json()["reply"] is None

    snapshot = client.get(f"/v1/conversations/{conv_id}").json()
    assert snapshot["state"] == ConversationState.OPERATOR_ACTIVE.value
    # Find the operator turn we recorded.
    assert any(t["channel"] == "operator" and t["body"] == "Hi." for t in snapshot["turns"])


def test_pinned_hmac_reference_for_operator_interop() -> None:
    """Pinned HMAC reference so operator-side webhook receivers can interop.

    Reproduce with:
        echo -n '{"a":1,"b":2}' | openssl dgst -sha256 -hmac "shared-secret"
    """
    secret = "shared-secret"
    body = b'{"a":1,"b":2}'
    expected = compute_hmac(secret, body)
    assert expected == "40d50073c3d041e5c354fc1f955d8a221bd747d3d727839b79b5ea2d0532a32a"
    assert verify_signature(secret, body, expected)


def test_sample_payload_can_be_round_tripped() -> None:
    secret = "k"
    body = b'{"x":1}'
    sig = compute_hmac(secret, body)
    assert verify_signature(secret, body, sig)
    assert not verify_signature(secret, body, "deadbeef")


_ = HTTPSender  # silence unused import
