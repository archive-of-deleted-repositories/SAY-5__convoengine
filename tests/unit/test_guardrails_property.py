"""Hypothesis property tests for the guardrails layer.

Every random LLM response either:
  - parses into LLMResponse (and the fields are within the declared range), OR
  - raises GuardrailError

There is no third outcome. Silent acceptance of garbage is the bug class
we are guarding against.
"""

from __future__ import annotations

import json

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from convoengine.orchestrator.conversation import ConversationState
from convoengine.orchestrator.guardrails import GuardrailError, parse_response

VALID_ACTIONS = ["reply", "clarify", "escalate", "close"]


def _build_valid(action: str, conf: float, state: ConversationState, text: str) -> str:
    return json.dumps(
        {
            "schema_version": 1,
            "action": action,
            "response_text": text,
            "confidence": conf,
            "suggested_state": state.value,
        }
    )


@given(
    action=st.sampled_from(VALID_ACTIONS),
    conf=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    state=st.sampled_from(list(ConversationState)),
    text=st.text(min_size=0, max_size=200),
)
def test_well_formed_payloads_always_parse(
    action: str, conf: float, state: ConversationState, text: str
) -> None:
    parsed = parse_response(_build_valid(action, conf, state, text))
    assert parsed.action == action
    assert 0.0 <= parsed.confidence <= 1.0
    assert parsed.suggested_state is state


@given(raw=st.text(min_size=0, max_size=200))
@settings(suppress_health_check=[HealthCheck.too_slow])
def test_arbitrary_text_either_parses_or_raises(raw: str) -> None:
    try:
        parsed = parse_response(raw)
    except GuardrailError:
        return  # bad input: the only other allowed outcome
    # If we got here, the payload happened to be valid JSON matching the schema.
    assert parsed.action in VALID_ACTIONS
    assert 0.0 <= parsed.confidence <= 1.0


@given(
    action=st.text(min_size=1, max_size=20).filter(lambda s: s not in VALID_ACTIONS),
    conf=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
)
def test_invalid_action_always_raises(action: str, conf: float) -> None:
    raw = json.dumps(
        {
            "schema_version": 1,
            "action": action,
            "response_text": "x",
            "confidence": conf,
            "suggested_state": "answering",
        }
    )
    try:
        parse_response(raw)
    except GuardrailError:
        return
    raise AssertionError(f"expected GuardrailError for action {action!r}")


@given(conf=st.floats(allow_nan=False).filter(lambda c: c < 0.0 or c > 1.0))
def test_confidence_outside_range_always_raises(conf: float) -> None:
    raw = json.dumps(
        {
            "schema_version": 1,
            "action": "reply",
            "response_text": "x",
            "confidence": conf,
            "suggested_state": "answering",
        }
    )
    try:
        parse_response(raw)
    except GuardrailError:
        return
    raise AssertionError(f"expected GuardrailError for confidence {conf}")
