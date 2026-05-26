"""Per-conversation state machine.

Transitions are encoded as data so they can be property-tested and rendered
into documentation. The machine is pure: it takes the current state plus an
event and returns the next state. Persistence happens elsewhere.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable


class ConversationState(str, Enum):
    """Possible per-conversation states."""

    GREETING = "greeting"
    CLARIFYING = "clarifying"
    ANSWERING = "answering"
    ESCALATED = "escalated"
    OPERATOR_ACTIVE = "operator_active"
    CLOSED = "closed"


class TransitionError(ValueError):
    """Raised when a transition is not allowed from the current state."""


@dataclass(frozen=True)
class Transition:
    """A single rule in the state-transition table."""

    src: ConversationState
    event: str
    dst: ConversationState


# The full set of allowed transitions. Encoded as data so we can both run the
# machine and render it for docs and property tests.
TRANSITIONS: tuple[Transition, ...] = (
    Transition(ConversationState.GREETING, "user_message", ConversationState.CLARIFYING),
    Transition(ConversationState.GREETING, "user_clear_question", ConversationState.ANSWERING),
    Transition(ConversationState.GREETING, "low_confidence", ConversationState.GREETING),
    Transition(ConversationState.GREETING, "escalate", ConversationState.ESCALATED),
    Transition(ConversationState.GREETING, "close", ConversationState.CLOSED),
    Transition(ConversationState.CLARIFYING, "user_message", ConversationState.CLARIFYING),
    Transition(ConversationState.CLARIFYING, "user_clear_question", ConversationState.ANSWERING),
    Transition(ConversationState.CLARIFYING, "low_confidence", ConversationState.CLARIFYING),
    Transition(ConversationState.CLARIFYING, "escalate", ConversationState.ESCALATED),
    Transition(ConversationState.CLARIFYING, "close", ConversationState.CLOSED),
    Transition(ConversationState.ANSWERING, "user_message", ConversationState.ANSWERING),
    Transition(ConversationState.ANSWERING, "user_clear_question", ConversationState.ANSWERING),
    Transition(ConversationState.ANSWERING, "user_follow_up", ConversationState.CLARIFYING),
    Transition(ConversationState.ANSWERING, "low_confidence", ConversationState.ANSWERING),
    Transition(ConversationState.ANSWERING, "escalate", ConversationState.ESCALATED),
    Transition(ConversationState.ANSWERING, "close", ConversationState.CLOSED),
    Transition(ConversationState.ESCALATED, "operator_takeover", ConversationState.OPERATOR_ACTIVE),
    Transition(ConversationState.ESCALATED, "close", ConversationState.CLOSED),
    Transition(ConversationState.OPERATOR_ACTIVE, "user_message", ConversationState.OPERATOR_ACTIVE),
    Transition(ConversationState.OPERATOR_ACTIVE, "operator_message", ConversationState.OPERATOR_ACTIVE),
    Transition(ConversationState.OPERATOR_ACTIVE, "operator_resolve", ConversationState.ANSWERING),
    Transition(ConversationState.OPERATOR_ACTIVE, "close", ConversationState.CLOSED),
)


# Terminal states cannot transition out except to themselves.
TERMINAL_STATES: frozenset[ConversationState] = frozenset({ConversationState.CLOSED})


class StateMachine:
    """Pure state machine over (state, event) -> state."""

    def __init__(self, transitions: Iterable[Transition] = TRANSITIONS) -> None:
        self._table: dict[tuple[ConversationState, str], ConversationState] = {
            (t.src, t.event): t.dst for t in transitions
        }

    def allowed_events(self, state: ConversationState) -> frozenset[str]:
        return frozenset(event for (src, event) in self._table if src is state)

    def can(self, state: ConversationState, event: str) -> bool:
        return (state, event) in self._table

    def apply(self, state: ConversationState, event: str) -> ConversationState:
        if state in TERMINAL_STATES:
            raise TransitionError(f"state {state.value!r} is terminal, cannot apply {event!r}")
        try:
            return self._table[(state, event)]
        except KeyError as exc:
            raise TransitionError(
                f"no transition from {state.value!r} on event {event!r}"
            ) from exc

    @property
    def transition_table(self) -> tuple[Transition, ...]:
        return tuple(
            Transition(src, event, dst) for (src, event), dst in self._table.items()
        )
