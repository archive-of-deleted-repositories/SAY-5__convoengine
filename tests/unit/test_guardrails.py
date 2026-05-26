"""Tests for the guardrails layer."""

from __future__ import annotations

import pytest

from convoengine.orchestrator.conversation import ConversationState
from convoengine.orchestrator.guardrails import (
    GuardrailError,
    call_with_guardrails,
    parse_response,
)
from convoengine.orchestrator.llm_client import LLMRequest

VALID_PAYLOAD = (
    '{"schema_version":1,"action":"reply","response_text":"ok",'
    '"confidence":0.9,"suggested_state":"answering"}'
)


def test_parse_response_accepts_valid_json() -> None:
    parsed = parse_response(VALID_PAYLOAD)
    assert parsed.action == "reply"
    assert parsed.confidence == 0.9
    assert parsed.suggested_state is ConversationState.ANSWERING


def test_parse_response_rejects_malformed_json() -> None:
    with pytest.raises(GuardrailError):
        parse_response("not json {")


def test_parse_response_rejects_unknown_action() -> None:
    bad = (
        '{"schema_version":1,"action":"sing","response_text":"x",'
        '"confidence":0.9,"suggested_state":"answering"}'
    )
    with pytest.raises(GuardrailError):
        parse_response(bad)


def test_parse_response_rejects_confidence_out_of_range() -> None:
    bad = (
        '{"schema_version":1,"action":"reply","response_text":"x",'
        '"confidence":1.7,"suggested_state":"answering"}'
    )
    with pytest.raises(GuardrailError):
        parse_response(bad)


@pytest.mark.asyncio
async def test_guardrails_pass_through_valid_first_call() -> None:
    async def provider(_req: LLMRequest) -> str:
        return VALID_PAYLOAD

    outcome = await call_with_guardrails(
        provider,
        LLMRequest(
            state=ConversationState.GREETING,
            history=(),
            last_message="hi",
            channel="email",
        ),
    )
    assert outcome.retried is False
    assert outcome.response.action == "reply"


@pytest.mark.asyncio
async def test_guardrails_retry_once_on_malformed_then_succeed() -> None:
    state = {"calls": 0}

    async def provider(req: LLMRequest) -> str:
        state["calls"] += 1
        if state["calls"] == 1:
            assert req.strict is False
            return "{this is not json"
        assert req.strict is True
        return VALID_PAYLOAD

    outcome = await call_with_guardrails(
        provider,
        LLMRequest(
            state=ConversationState.GREETING,
            history=(),
            last_message="hi",
            channel="email",
        ),
    )
    assert outcome.retried is True
    assert outcome.response.action == "reply"
    assert state["calls"] == 2


@pytest.mark.asyncio
async def test_guardrails_raise_when_retry_also_fails() -> None:
    async def provider(_req: LLMRequest) -> str:
        return "still not json"

    with pytest.raises(GuardrailError):
        await call_with_guardrails(
            provider,
            LLMRequest(
                state=ConversationState.GREETING,
                history=(),
                last_message="hi",
                channel="email",
            ),
        )
