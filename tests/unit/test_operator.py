"""Unit tests for the operator-queue escalation client."""

from __future__ import annotations

import json

import pytest

from convoengine.operator.escalation import (
    OperatorClient,
    RecordingSender,
    TurnSummary,
    build_summary,
    compute_hmac,
    verify_signature,
)
from convoengine.orchestrator.conversation import ConversationState


def _sample_summary() -> object:
    return build_summary(
        conversation_id="conv-test",
        current_state=ConversationState.ESCALATED,
        trigger="confidence below floor for 3 consecutive turns",
        last_turns=[
            TurnSummary(
                channel="email",
                direction="in",
                body="hello?",
                state_before=ConversationState.GREETING,
                state_after=ConversationState.CLARIFYING,
            ),
            TurnSummary(
                channel="email",
                direction="out",
                body="Could you clarify?",
                state_before=ConversationState.CLARIFYING,
                state_after=ConversationState.CLARIFYING,
                confidence=0.4,
            ),
        ],
        suggested_actions=["ask account email", "share refund policy"],
    )


@pytest.mark.asyncio
async def test_operator_client_signs_and_sends_payload() -> None:
    sender = RecordingSender()
    client = OperatorClient(url="http://operator/queue", secret="s3cr3t", sender=sender)
    status, _ = await client.submit(_sample_summary())  # type: ignore[arg-type]

    assert status == 202
    assert len(sender.sent) == 1
    url, headers, body = sender.sent[0]
    assert url == "http://operator/queue"
    assert headers["Content-Type"] == "application/json"
    assert "X-ConvoEngine-Signature" in headers
    parsed = json.loads(body)
    assert parsed["conversation_id"] == "conv-test"
    assert parsed["current_state"] == "escalated"
    assert len(parsed["last_turns"]) == 2


@pytest.mark.asyncio
async def test_operator_client_signature_verifies_round_trip() -> None:
    sender = RecordingSender()
    client = OperatorClient(url="http://operator/queue", secret="s3cr3t", sender=sender)
    await client.submit(_sample_summary())  # type: ignore[arg-type]
    _, headers, body = sender.sent[0]
    assert verify_signature("s3cr3t", body, headers["X-ConvoEngine-Signature"])
    assert not verify_signature("wrong", body, headers["X-ConvoEngine-Signature"])


def test_compute_hmac_is_deterministic() -> None:
    body = b'{"x":1}'
    assert compute_hmac("k", body) == compute_hmac("k", body)
    assert compute_hmac("k", body) != compute_hmac("other", body)
