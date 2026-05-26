"""LLM client abstractions and providers."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field, field_validator

from convoengine.orchestrator.conversation import ConversationState

SCHEMA_VERSION = 1


class LLMResponse(BaseModel):
    """Structured-output schema that every LLM turn must satisfy."""

    model_config = ConfigDict(extra="forbid")

    schema_version: int = Field(default=SCHEMA_VERSION)
    action: str = Field(description="Short controller tag: reply | clarify | escalate | close.")
    response_text: str
    confidence: float = Field(ge=0.0, le=1.0)
    suggested_state: ConversationState

    @field_validator("action")
    @classmethod
    def _action_in_alphabet(cls, v: str) -> str:
        allowed = {"reply", "clarify", "escalate", "close"}
        if v not in allowed:
            raise ValueError(f"action {v!r} not in {sorted(allowed)}")
        return v


@dataclass(frozen=True)
class LLMRequest:
    """The shape sent to a provider."""

    state: ConversationState
    history: tuple[str, ...]
    last_message: str
    channel: str
    strict: bool = False  # set by the guardrails retry path

    def hash_key(self) -> str:
        h = hashlib.sha256()
        h.update(self.state.value.encode())
        h.update(self.last_message.encode())
        return h.hexdigest()[:16]


class LLMProvider(Protocol):
    """Provider interface. Returns a raw JSON-encoded string."""

    name: str

    async def complete(self, request: LLMRequest) -> str: ...


@dataclass
class ScriptedResponse:
    """A single scripted reply for the fake provider."""

    raw_json: str
    valid: bool = True


@dataclass
class FakeLLMProvider:
    """Deterministic provider used by tests and bench.

    Look-up order:
      1. Exact (state, hash) match.
      2. State default.
      3. Global default.
    """

    name: str = "fake"
    by_key: dict[tuple[ConversationState, str], ScriptedResponse] = field(default_factory=dict)
    by_state: dict[ConversationState, ScriptedResponse] = field(default_factory=dict)
    default: ScriptedResponse = field(
        default_factory=lambda: ScriptedResponse(
            raw_json=(
                '{"schema_version":1,"action":"reply",'
                '"response_text":"Thanks. Could you share more detail?",'
                '"confidence":0.82,"suggested_state":"clarifying"}'
            )
        )
    )
    calls: list[LLMRequest] = field(default_factory=list)

    def script(
        self, state: ConversationState, last_message: str, response: ScriptedResponse
    ) -> None:
        req = LLMRequest(state=state, history=(), last_message=last_message, channel="email")
        self.by_key[(state, req.hash_key())] = response

    def script_state(self, state: ConversationState, response: ScriptedResponse) -> None:
        self.by_state[state] = response

    def script_default(self, response: ScriptedResponse) -> None:
        self.default = response

    async def complete(self, request: LLMRequest) -> str:
        self.calls.append(request)
        key = (request.state, request.hash_key())
        if key in self.by_key:
            return self.by_key[key].raw_json
        if request.state in self.by_state:
            return self.by_state[request.state].raw_json
        return self.default.raw_json


class ClaudeProvider:
    """Stub for a real Claude provider. Never wired into CI."""

    name = "claude"

    def __init__(self, api_key: str | None) -> None:
        self._api_key = api_key

    async def complete(self, request: LLMRequest) -> str:  # pragma: no cover - not used in tests
        if not self._api_key:
            raise RuntimeError("ClaudeProvider requires an API key")
        raise NotImplementedError("ClaudeProvider is a stub. Wire to anthropic SDK when needed.")
