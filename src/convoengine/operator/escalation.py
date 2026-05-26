"""POST a structured conversation summary to the operator queue."""

from __future__ import annotations

import hashlib
import hmac
import json
from dataclasses import dataclass
from typing import Any, Protocol

import httpx
from pydantic import BaseModel, ConfigDict, Field

from convoengine.orchestrator.conversation import ConversationState


class TurnSummary(BaseModel):
    """A single turn entry in the escalation summary."""

    model_config = ConfigDict(extra="forbid")

    channel: str
    direction: str
    body: str
    state_before: ConversationState
    state_after: ConversationState
    confidence: float | None = None


class EscalationPayload(BaseModel):
    """The body POSTed to the operator queue."""

    model_config = ConfigDict(extra="forbid")

    conversation_id: str
    current_state: ConversationState
    trigger: str = Field(description="Human-readable reason this escalation fired.")
    last_turns: list[TurnSummary]
    suggested_actions: list[str] = Field(default_factory=list)


def build_summary(
    *,
    conversation_id: str,
    current_state: ConversationState,
    trigger: str,
    last_turns: list[TurnSummary],
    suggested_actions: list[str] | None = None,
) -> EscalationPayload:
    return EscalationPayload(
        conversation_id=conversation_id,
        current_state=current_state,
        trigger=trigger,
        last_turns=last_turns,
        suggested_actions=suggested_actions or [],
    )


def compute_hmac(secret: str, body: bytes) -> str:
    """Compute the X-ConvoEngine-Signature header value: sha256 HMAC, hex."""
    return hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()


class HTTPSender(Protocol):
    """Pluggable transport so tests can swap in an in-process recorder."""

    async def post(self, url: str, headers: dict[str, str], body: bytes) -> tuple[int, str]: ...


class HttpxSender:
    """Default sender backed by httpx.AsyncClient."""

    def __init__(self, timeout: float = 5.0) -> None:
        self._timeout = timeout

    async def post(
        self, url: str, headers: dict[str, str], body: bytes
    ) -> tuple[int, str]:  # pragma: no cover - exercised in integration tests
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            r = await client.post(url, content=body, headers=headers)
            return r.status_code, r.text


@dataclass
class RecordingSender:
    """In-process sender used by unit tests."""

    sent: list[tuple[str, dict[str, str], bytes]]
    response: tuple[int, str] = (202, "ok")

    def __init__(self, response: tuple[int, str] = (202, "ok")) -> None:
        self.sent = []
        self.response = response

    async def post(self, url: str, headers: dict[str, str], body: bytes) -> tuple[int, str]:
        self.sent.append((url, headers, body))
        return self.response


class OperatorClient:
    """High-level client that signs and sends an EscalationPayload."""

    def __init__(self, url: str, secret: str, sender: HTTPSender) -> None:
        self._url = url
        self._secret = secret
        self._sender = sender

    async def submit(self, payload: EscalationPayload) -> tuple[int, str]:
        body = payload.model_dump_json().encode()
        sig = compute_hmac(self._secret, body)
        headers = {
            "Content-Type": "application/json",
            "X-ConvoEngine-Signature": sig,
        }
        return await self._sender.post(self._url, headers, body)


def verify_signature(secret: str, body: bytes, signature: str) -> bool:
    """Constant-time signature verification for operator-side webhook receivers."""
    expected = compute_hmac(secret, body)
    return hmac.compare_digest(expected, signature)


def payload_from_dict(data: dict[str, Any]) -> EscalationPayload:
    """Round-trip a serialized payload."""
    return EscalationPayload.model_validate(json.loads(json.dumps(data)))
