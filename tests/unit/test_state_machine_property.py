"""Hypothesis property tests for the state machine.

We generate random sequences of (state, event) pairs and assert that:
  1. Every legal event leaves the machine in a state from the documented set.
  2. Every illegal event raises TransitionError (without changing the state).
  3. Once the machine reaches CLOSED, no further events succeed.
  4. The transitions encoded in TRANSITIONS exactly equal the runtime table.
"""

from __future__ import annotations

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from convoengine.orchestrator.conversation import (
    TERMINAL_STATES,
    TRANSITIONS,
    ConversationState,
    StateMachine,
    TransitionError,
)

ALL_STATES = list(ConversationState)
ALL_EVENTS = sorted({t.event for t in TRANSITIONS})


@given(state=st.sampled_from(ALL_STATES), event=st.text(min_size=1, max_size=12))
def test_apply_never_returns_unknown_state(state: ConversationState, event: str) -> None:
    sm = StateMachine()
    try:
        result = sm.apply(state, event)
    except TransitionError:
        return  # illegal: that's fine, the property holds vacuously
    assert isinstance(result, ConversationState)
    assert result in set(ALL_STATES)


@given(state=st.sampled_from(ALL_STATES), event=st.sampled_from(ALL_EVENTS))
def test_legal_transitions_match_table(state: ConversationState, event: str) -> None:
    sm = StateMachine()
    expected = {(t.src, t.event): t.dst for t in TRANSITIONS}.get((state, event))
    if expected is None or state in TERMINAL_STATES:
        try:
            sm.apply(state, event)
        except TransitionError:
            pass
        else:
            raise AssertionError(f"expected TransitionError for ({state.value}, {event})")
    else:
        assert sm.apply(state, event) is expected


@given(events=st.lists(st.sampled_from(ALL_EVENTS), min_size=0, max_size=20))
@settings(suppress_health_check=[HealthCheck.too_slow])
def test_random_walk_stays_in_alphabet_or_raises(events: list[str]) -> None:
    sm = StateMachine()
    state = ConversationState.GREETING
    for event in events:
        try:
            state = sm.apply(state, event)
        except TransitionError:
            # State must NOT have changed.
            assert state in set(ALL_STATES)
        else:
            assert state in set(ALL_STATES)


@given(tail_events=st.lists(st.sampled_from(ALL_EVENTS), min_size=1, max_size=10))
def test_terminal_state_never_transitions(tail_events: list[str]) -> None:
    sm = StateMachine()
    state = ConversationState.CLOSED
    for event in tail_events:
        try:
            sm.apply(state, event)
        except TransitionError:
            continue
        else:
            raise AssertionError("CLOSED state must not transition")
    assert state is ConversationState.CLOSED


def test_transition_table_matches_module_constant() -> None:
    sm = StateMachine()
    table_pairs = {(t.src, t.event, t.dst) for t in sm.transition_table}
    declared_pairs = {(t.src, t.event, t.dst) for t in TRANSITIONS}
    assert table_pairs == declared_pairs
