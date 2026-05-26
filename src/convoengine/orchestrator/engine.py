"""Orchestrator engine: turn a ChannelMessage into a persisted response.

Pulls together:
    channels.ChannelMessage -> repo.get_or_create_conversation
    -> llm_client.LLMRequest -> guardrails.call_with_guardrails
    -> fallback.decide -> persist turn -> operator escalation (if needed)
    -> outbound ChannelMessage

The engine is hermetic-testable: it takes a session, an LLM provider, an
optional OperatorClient, and a FallbackConfig. No globals.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from sqlalchemy.orm import Session

from convoengine.channels._base import ChannelMessage, Direction
from convoengine.operator.escalation import (
    EscalationPayload,
    OperatorClient,
    TurnSummary,
    build_summary,
)
from convoengine.orchestrator.conversation import (
    ConversationState,
    StateMachine,
    TransitionError,
)
from convoengine.orchestrator.fallback import (
    TEMPLATE_RESPONSE_TEXT,
    FallbackConfig,
    FallbackDecision,
    FallbackInput,
    decide,
)
from convoengine.orchestrator.guardrails import GuardrailError, call_with_guardrails
from convoengine.orchestrator.llm_client import LLMProvider, LLMRequest
from convoengine.repo import (
    bump_low_conf_streak,
    get_or_create_conversation,
    list_turns,
    record_turn,
    set_state,
)
from convoengine.store import Conversation

log = logging.getLogger(__name__)


@dataclass
class HandleResult:
    """What handle_message returns to callers (the API or tests)."""

    conversation_id: str
    inbound_turn_id: str
    outbound_turn_id: str | None
    decision: FallbackDecision
    response: ChannelMessage | None
    escalated_payload: EscalationPayload | None = None


@dataclass
class Engine:
    """Per-app orchestrator. Construct once and reuse."""

    provider: LLMProvider
    fallback_config: FallbackConfig
    operator_client: OperatorClient | None = None
    state_machine: StateMachine = None  # type: ignore[assignment]
    max_summary_turns: int = 8

    def __post_init__(self) -> None:
        if self.state_machine is None:
            self.state_machine = StateMachine()

    async def handle_message(
        self,
        session: Session,
        inbound: ChannelMessage,
        *,
        extra_identities: list[tuple[str, str]] | None = None,
    ) -> HandleResult:
        """Process one inbound message.

        Resolves identity, runs guardrails, applies fallback, persists turns, and
        if needed escalates to the operator queue.
        """
        if inbound.direction is not Direction.IN:
            raise ValueError("handle_message expects an inbound ChannelMessage")

        conv, _created = get_or_create_conversation(
            session,
            channel=inbound.channel,
            identity=inbound.sender,
            extra_identities=extra_identities,
        )

        state_before = ConversationState(conv.state)

        # Operator-active conversations bypass the LLM: the message is recorded
        # but no autonomous reply is generated.
        if state_before is ConversationState.OPERATOR_ACTIVE:
            inbound_turn = record_turn(
                session,
                conversation_id=conv.id,
                channel=inbound.channel,
                direction="in",
                body=inbound.body,
                state_before=state_before,
                state_after=state_before,
                flags={"reason": "operator_active_no_autoreply"},
            )
            session.flush()
            return HandleResult(
                conversation_id=conv.id,
                inbound_turn_id=inbound_turn.id,
                outbound_turn_id=None,
                decision=FallbackDecision.USE_LLM,  # placeholder; no action taken
                response=None,
            )

        request = LLMRequest(
            state=state_before,
            history=tuple(t.body for t in list_turns(session, conv.id, limit=8)),
            last_message=inbound.body,
            channel=inbound.channel,
        )

        try:
            outcome = await call_with_guardrails(self.provider.complete, request)
        except GuardrailError as exc:
            log.warning("guardrails failed: %s", exc)
            # Treat as a low-confidence template response.
            inbound_turn = record_turn(
                session,
                conversation_id=conv.id,
                channel=inbound.channel,
                direction="in",
                body=inbound.body,
                state_before=state_before,
                state_after=state_before,
                flags={"guardrails_failed": True},
            )
            outbound = self._template_response(inbound)
            outbound_turn = record_turn(
                session,
                conversation_id=conv.id,
                channel=inbound.channel,
                direction="out",
                body=outbound.body,
                state_before=state_before,
                state_after=state_before,
                flags={"fallback": "template", "reason": "guardrails_failed"},
            )
            session.flush()
            return HandleResult(
                conversation_id=conv.id,
                inbound_turn_id=inbound_turn.id,
                outbound_turn_id=outbound_turn.id,
                decision=FallbackDecision.USE_TEMPLATE,
                response=outbound.with_conversation_id(conv.id),
            )

        signal = FallbackInput(
            response=outcome.response,
            consecutive_low=conv.low_conf_streak,
        )
        verdict = decide(self.fallback_config, signal)

        # Record inbound turn.
        inbound_turn = record_turn(
            session,
            conversation_id=conv.id,
            channel=inbound.channel,
            direction="in",
            body=inbound.body,
            state_before=state_before,
            state_after=state_before,
            flags={"guardrails_retried": outcome.retried},
            payload={"llm_action": outcome.response.action},
        )

        if verdict.decision is FallbackDecision.USE_LLM:
            new_state = _apply_event(
                self.state_machine,
                state_before,
                event=_event_for_action(outcome.response.action),
                fallback_state=outcome.response.suggested_state,
            )
            outbound = ChannelMessage(
                channel=inbound.channel,
                direction=Direction.OUT,
                sender=f"convoengine@{inbound.channel}",
                body=outcome.response.response_text,
            )
            outbound_turn = record_turn(
                session,
                conversation_id=conv.id,
                channel=inbound.channel,
                direction="out",
                body=outbound.body,
                state_before=state_before,
                state_after=new_state,
                confidence=outcome.response.confidence,
                flags={"fallback": "use_llm"},
                payload={"action": outcome.response.action},
            )
            set_state(session, conv, new_state)
            bump_low_conf_streak(conv, low=False)
            session.flush()
            return HandleResult(
                conversation_id=conv.id,
                inbound_turn_id=inbound_turn.id,
                outbound_turn_id=outbound_turn.id,
                decision=verdict.decision,
                response=outbound.with_conversation_id(conv.id),
            )

        if verdict.decision is FallbackDecision.USE_TEMPLATE:
            outbound = self._template_response(inbound)
            outbound_turn = record_turn(
                session,
                conversation_id=conv.id,
                channel=inbound.channel,
                direction="out",
                body=outbound.body,
                state_before=state_before,
                state_after=state_before,
                confidence=outcome.response.confidence,
                flags={
                    "fallback": "template",
                    "reason": verdict.reason,
                    "consecutive_low": verdict.new_consecutive_low,
                },
            )
            conv.low_conf_streak = verdict.new_consecutive_low
            session.flush()
            return HandleResult(
                conversation_id=conv.id,
                inbound_turn_id=inbound_turn.id,
                outbound_turn_id=outbound_turn.id,
                decision=verdict.decision,
                response=outbound.with_conversation_id(conv.id),
            )

        # ESCALATE branch.
        new_state = _apply_event(
            self.state_machine,
            state_before,
            event="escalate",
            fallback_state=ConversationState.ESCALATED,
        )
        outbound = ChannelMessage(
            channel=inbound.channel,
            direction=Direction.OUT,
            sender=f"convoengine@{inbound.channel}",
            body=(
                "I'm pulling in a teammate who can help with this. "
                "They'll follow up here shortly."
            ),
        )
        outbound_turn = record_turn(
            session,
            conversation_id=conv.id,
            channel=inbound.channel,
            direction="out",
            body=outbound.body,
            state_before=state_before,
            state_after=new_state,
            confidence=outcome.response.confidence,
            flags={"fallback": "escalate", "reason": verdict.reason},
        )
        set_state(session, conv, new_state)
        conv.low_conf_streak = 0
        session.flush()

        summary = self._build_summary(session, conv, trigger=verdict.reason)
        if self.operator_client is not None:
            await self.operator_client.submit(summary)

        return HandleResult(
            conversation_id=conv.id,
            inbound_turn_id=inbound_turn.id,
            outbound_turn_id=outbound_turn.id,
            decision=verdict.decision,
            response=outbound.with_conversation_id(conv.id),
            escalated_payload=summary,
        )

    def _template_response(self, inbound: ChannelMessage) -> ChannelMessage:
        return ChannelMessage(
            channel=inbound.channel,
            direction=Direction.OUT,
            sender=f"convoengine@{inbound.channel}",
            body=TEMPLATE_RESPONSE_TEXT,
        )

    def _build_summary(
        self, session: Session, conv: Conversation, *, trigger: str
    ) -> EscalationPayload:
        all_turns = list_turns(session, conv.id)
        tail = all_turns[-self.max_summary_turns :]
        turn_summaries = [
            TurnSummary(
                channel=t.channel,
                direction=t.direction,
                body=t.body,
                state_before=ConversationState(t.state_before),
                state_after=ConversationState(t.state_after),
                confidence=t.confidence,
            )
            for t in tail
        ]
        return build_summary(
            conversation_id=conv.id,
            current_state=ConversationState(conv.state),
            trigger=trigger,
            last_turns=turn_summaries,
            suggested_actions=["review last 3 turns", "respond with refund policy"],
        )


def _event_for_action(action: str) -> str:
    """Translate an LLM action tag into a state-machine event."""
    return {
        "reply": "user_clear_question",
        "clarify": "user_message",
        "escalate": "escalate",
        "close": "close",
    }.get(action, "user_message")


def _apply_event(
    sm: StateMachine,
    state: ConversationState,
    *,
    event: str,
    fallback_state: ConversationState,
) -> ConversationState:
    """Apply an event; if the transition is not in the table, snap to the suggested state."""
    try:
        return sm.apply(state, event)
    except TransitionError:
        # Suggested state must still be reachable by some event; if not, stay put.
        if sm.can(state, "user_message"):
            return sm.apply(state, "user_message")
        return fallback_state
