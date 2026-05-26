"""End-to-end orchestrator tests against in-memory SQLite + FakeLLMProvider."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from sqlalchemy.orm import Session

from convoengine.channels._base import ChannelMessage, Direction
from convoengine.operator.escalation import OperatorClient, RecordingSender
from convoengine.orchestrator.conversation import ConversationState
from convoengine.orchestrator.engine import Engine
from convoengine.orchestrator.fallback import FallbackConfig, FallbackDecision
from convoengine.orchestrator.llm_client import FakeLLMProvider, ScriptedResponse
from convoengine.repo import list_turns
from convoengine.store import (
    Conversation,
    create_all,
    make_engine,
    make_session_factory,
)


@pytest.fixture
def session() -> Iterator[Session]:
    eng = make_engine("sqlite:///:memory:")
    create_all(eng)
    factory = make_session_factory(eng)
    s = factory()
    try:
        yield s
    finally:
        s.close()
        eng.dispose()


def _scripted(
    action: str, text: str, confidence: float, state: ConversationState
) -> ScriptedResponse:
    return ScriptedResponse(
        raw_json=(
            f'{{"schema_version":1,"action":"{action}","response_text":"{text}",'
            f'"confidence":{confidence},"suggested_state":"{state.value}"}}'
        )
    )


def _msg(text: str, channel: str = "email", sender: str = "alice@example.com") -> ChannelMessage:
    return ChannelMessage(channel=channel, direction=Direction.IN, sender=sender, body=text)


@pytest.mark.asyncio
async def test_engine_happy_path_use_llm_persists_two_turns(session: Session) -> None:
    provider = FakeLLMProvider()
    provider.script_default(_scripted("reply", "Sure.", 0.9, ConversationState.ANSWERING))
    engine = Engine(provider=provider, fallback_config=FallbackConfig())

    result = await engine.handle_message(session, _msg("Hi, can you help?"))
    session.commit()

    assert result.decision is FallbackDecision.USE_LLM
    assert result.response is not None
    assert result.response.body == "Sure."
    turns = list_turns(session, result.conversation_id)
    assert len(turns) == 2
    assert turns[0].direction == "in"
    assert turns[1].direction == "out"
    conv = session.get(Conversation, result.conversation_id)
    assert conv is not None
    assert conv.state == ConversationState.ANSWERING.value


@pytest.mark.asyncio
async def test_engine_low_confidence_emits_template(session: Session) -> None:
    provider = FakeLLMProvider()
    provider.script_default(_scripted("clarify", "uh", 0.4, ConversationState.CLARIFYING))
    engine = Engine(provider=provider, fallback_config=FallbackConfig())

    result = await engine.handle_message(session, _msg("ambiguous"))
    session.commit()

    assert result.decision is FallbackDecision.USE_TEMPLATE
    assert result.response is not None
    assert "rephrase" in result.response.body or "rephrase" in result.response.body.lower()


@pytest.mark.asyncio
async def test_engine_three_low_conf_turns_escalate_with_operator_post(session: Session) -> None:
    provider = FakeLLMProvider()
    provider.script_default(_scripted("clarify", "uh", 0.4, ConversationState.CLARIFYING))
    sender = RecordingSender()
    op = OperatorClient("http://operator/queue", "shh", sender)
    engine = Engine(provider=provider, fallback_config=FallbackConfig(), operator_client=op)

    r1 = await engine.handle_message(session, _msg("hello?"))
    session.commit()
    r2 = await engine.handle_message(session, _msg("anyone there?"))
    session.commit()
    r3 = await engine.handle_message(session, _msg("hi?"))
    session.commit()

    assert r1.decision is FallbackDecision.USE_TEMPLATE
    assert r2.decision is FallbackDecision.USE_TEMPLATE
    assert r3.decision is FallbackDecision.ESCALATE
    assert len(sender.sent) == 1
    conv = session.get(Conversation, r1.conversation_id)
    assert conv is not None
    assert conv.state == ConversationState.ESCALATED.value


@pytest.mark.asyncio
async def test_engine_single_very_low_confidence_immediately_escalates(session: Session) -> None:
    provider = FakeLLMProvider()
    provider.script_default(_scripted("clarify", "??", 0.1, ConversationState.CLARIFYING))
    sender = RecordingSender()
    op = OperatorClient("http://operator/queue", "shh", sender)
    engine = Engine(provider=provider, fallback_config=FallbackConfig(), operator_client=op)

    result = await engine.handle_message(session, _msg("can you help"))
    session.commit()

    assert result.decision is FallbackDecision.ESCALATE
    assert result.escalated_payload is not None
    assert result.escalated_payload.current_state is ConversationState.ESCALATED
    assert len(sender.sent) == 1


@pytest.mark.asyncio
async def test_engine_guardrails_failure_falls_back_to_template(session: Session) -> None:
    provider = FakeLLMProvider()
    provider.script_default(ScriptedResponse(raw_json="not json", valid=False))
    engine = Engine(provider=provider, fallback_config=FallbackConfig())

    result = await engine.handle_message(session, _msg("anything"))
    session.commit()

    assert result.decision is FallbackDecision.USE_TEMPLATE
    turns = list_turns(session, result.conversation_id)
    assert any(t.flags.get("fallback") == "template" for t in turns)
