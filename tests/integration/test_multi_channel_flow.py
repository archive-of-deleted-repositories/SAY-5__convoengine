"""End-to-end integration test: 6-turn conversation spanning email and chat.

Scenario:
    1. Email inbound: user asks something
    2. Email outbound: high-confidence reply
    3. Email inbound: low-confidence ambiguous follow-up -> template fallback
    4. Chat inbound (same user, email-on-file): orchestrator continues same
       conversation
    5. Chat outbound: high-confidence answer
    6. Email inbound: a "thanks, close" message -> close

We assert at the end that the conversation has all turns persisted, both
channels are recorded, and final state is CLOSED.
"""

from __future__ import annotations

import pytest

from convoengine.channels._base import ChannelMessage, Direction
from convoengine.operator.escalation import OperatorClient, RecordingSender
from convoengine.orchestrator.conversation import ConversationState
from convoengine.orchestrator.engine import Engine
from convoengine.orchestrator.fallback import FallbackConfig
from convoengine.orchestrator.llm_client import FakeLLMProvider, ScriptedResponse
from convoengine.repo import list_turns

pytestmark = pytest.mark.integration


def _scripted(action: str, text: str, conf: float, state: ConversationState) -> ScriptedResponse:
    return ScriptedResponse(
        raw_json=(
            f'{{"schema_version":1,"action":"{action}","response_text":"{text}",'
            f'"confidence":{conf},"suggested_state":"{state.value}"}}'
        )
    )


@pytest.mark.asyncio
async def test_six_turn_conversation_across_email_and_chat(session_factory) -> None:  # type: ignore[no-untyped-def]
    provider = FakeLLMProvider()
    # We script per-state defaults to drive the flow deterministically.
    provider.script_state(
        ConversationState.GREETING,
        _scripted("reply", "Happy to help. What's the order id?", 0.9, ConversationState.ANSWERING),
    )
    provider.script_state(
        ConversationState.ANSWERING,
        _scripted("clarify", "hmm", 0.4, ConversationState.CLARIFYING),
    )
    provider.script_state(
        ConversationState.CLARIFYING,
        _scripted("reply", "Your refund posted yesterday.", 0.9, ConversationState.ANSWERING),
    )

    op = OperatorClient("http://operator/queue", "shh", RecordingSender())
    engine = Engine(provider=provider, fallback_config=FallbackConfig(), operator_client=op)

    s = session_factory()
    try:
        # Turn 1: email in, expects high-confidence reply (GREETING -> ANSWERING).
        msg1 = ChannelMessage(
            channel="email",
            direction=Direction.IN,
            sender="alice@example.com",
            body="Hi, can you check on a refund?",
        )
        r1 = await engine.handle_message(s, msg1)
        s.commit()
        assert r1.decision.value == "use_llm"
        conv_id = r1.conversation_id

        # Turn 2: email in, low-confidence -> template fallback (still ANSWERING).
        msg2 = ChannelMessage(
            channel="email",
            direction=Direction.IN,
            sender="alice@example.com",
            body="huh?",
        )
        r2 = await engine.handle_message(s, msg2)
        s.commit()
        assert r2.decision.value == "use_template"

        # Turn 3: switch to chat. Same user, email-on-file. Conversation
        # continues from current state, not a new one. We deliberately script
        # the ANSWERING-state response to be low-confidence so we land on
        # template, then for the next turn we'll move to CLARIFYING.
        msg3 = ChannelMessage(
            channel="chat",
            direction=Direction.IN,
            sender="u-42",
            body="i mean the order from last week",
        )
        r3 = await engine.handle_message(s, msg3, extra_identities=[("email", "alice@example.com")])
        s.commit()
        assert r3.conversation_id == conv_id
        # The ANSWERING-state script returns confidence 0.4 -> template.
        assert r3.decision.value == "use_template"

        # Move the conversation explicitly to CLARIFYING for the next inbound.
        from convoengine.repo import set_state
        from convoengine.store import Conversation

        conv = s.get(Conversation, conv_id)
        assert conv is not None
        set_state(s, conv, ConversationState.CLARIFYING)
        s.commit()

        # Turn 4: chat in, CLARIFYING-state script returns high-confidence reply.
        msg4 = ChannelMessage(
            channel="chat",
            direction=Direction.IN,
            sender="u-42",
            body="order #12345",
        )
        r4 = await engine.handle_message(s, msg4)
        s.commit()
        assert r4.decision.value == "use_llm"

        # Turn 5: email in, ANSWERING state again. Switch script to high-conf reply.
        provider.script_state(
            ConversationState.ANSWERING,
            _scripted("reply", "Anything else?", 0.85, ConversationState.ANSWERING),
        )
        msg5 = ChannelMessage(
            channel="email",
            direction=Direction.IN,
            sender="alice@example.com",
            body="thanks!",
        )
        r5 = await engine.handle_message(s, msg5)
        s.commit()
        assert r5.decision.value == "use_llm"

        # Turn 6: explicit close.
        provider.script_state(
            ConversationState.ANSWERING,
            _scripted("close", "You're welcome.", 0.95, ConversationState.CLOSED),
        )
        msg6 = ChannelMessage(
            channel="email",
            direction=Direction.IN,
            sender="alice@example.com",
            body="all set, bye",
        )
        r6 = await engine.handle_message(s, msg6)
        s.commit()
        assert r6.decision.value == "use_llm"

        # Final assertions: conversation has many turns across both channels.
        all_turns = list_turns(s, conv_id)
        channels_seen = {t.channel for t in all_turns}
        assert {"email", "chat"} <= channels_seen
        assert len(all_turns) >= 12  # 6 inbound + at least 6 outbound

        conv = s.get(Conversation, conv_id)
        assert conv is not None
        assert conv.state == ConversationState.CLOSED.value
    finally:
        s.close()
