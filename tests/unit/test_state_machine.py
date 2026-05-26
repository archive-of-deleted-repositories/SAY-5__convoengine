"""Unit tests for the pure state machine."""

from __future__ import annotations

import pytest

from convoengine.orchestrator.conversation import (
    TRANSITIONS,
    ConversationState,
    StateMachine,
    TransitionError,
)


def test_state_machine_applies_documented_transitions() -> None:
    sm = StateMachine()
    assert sm.apply(ConversationState.GREETING, "user_message") is ConversationState.CLARIFYING
    assert (
        sm.apply(ConversationState.CLARIFYING, "user_clear_question") is ConversationState.ANSWERING
    )
    assert sm.apply(ConversationState.ANSWERING, "user_follow_up") is ConversationState.CLARIFYING


def test_state_machine_rejects_unknown_event() -> None:
    sm = StateMachine()
    with pytest.raises(TransitionError):
        sm.apply(ConversationState.GREETING, "operator_resolve")


def test_state_machine_blocks_transition_from_terminal_state() -> None:
    sm = StateMachine()
    with pytest.raises(TransitionError):
        sm.apply(ConversationState.CLOSED, "user_message")


def test_state_machine_can_predicate_matches_apply() -> None:
    sm = StateMachine()
    for t in TRANSITIONS:
        assert sm.can(t.src, t.event)
    assert not sm.can(ConversationState.GREETING, "operator_resolve")


def test_allowed_events_lists_every_outgoing_event() -> None:
    sm = StateMachine()
    events = sm.allowed_events(ConversationState.GREETING)
    assert {"user_message", "user_clear_question", "low_confidence", "escalate", "close"} <= events


def test_low_confidence_loops_without_state_change() -> None:
    sm = StateMachine()
    for state in (
        ConversationState.GREETING,
        ConversationState.CLARIFYING,
        ConversationState.ANSWERING,
    ):
        assert sm.apply(state, "low_confidence") is state
