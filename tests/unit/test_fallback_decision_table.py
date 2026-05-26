"""Full decision-table coverage for the fallback dispatcher.

We enumerate every interesting combination of:
  - single-turn confidence relative to floor and very-low-floor
  - rolling-window streak length relative to the window size

and assert exactly one branch fires for each.
"""

from __future__ import annotations

import itertools

import pytest

from convoengine.orchestrator.conversation import ConversationState
from convoengine.orchestrator.fallback import (
    FallbackConfig,
    FallbackDecision,
    FallbackInput,
    decide,
)
from convoengine.orchestrator.llm_client import LLMResponse


def _resp(confidence: float) -> LLMResponse:
    return LLMResponse(
        action="reply",
        response_text="x",
        confidence=confidence,
        suggested_state=ConversationState.ANSWERING,
    )


CONFIG = FallbackConfig(floor=0.6, very_low_floor=0.3, window=3)

# We pick representative confidence values, one in each "band":
#   very-low: 0.1
#   low:      0.45  (floor > x >= very_low_floor)
#   medium:   0.7   (>= floor)
#   high:     0.95
CONFIDENCE_VALUES: tuple[float, ...] = (0.0, 0.1, 0.29, 0.3, 0.45, 0.59, 0.6, 0.7, 0.95, 1.0)
STREAK_VALUES: tuple[int, ...] = (0, 1, 2, 3, 5)


def _expected_decision(confidence: float, streak: int) -> FallbackDecision:
    if confidence < CONFIG.very_low_floor:
        return FallbackDecision.ESCALATE
    if confidence < CONFIG.floor:
        new_streak = streak + 1
        if new_streak >= CONFIG.window:
            return FallbackDecision.ESCALATE
        return FallbackDecision.USE_TEMPLATE
    return FallbackDecision.USE_LLM


@pytest.mark.parametrize(
    "confidence,streak",
    list(itertools.product(CONFIDENCE_VALUES, STREAK_VALUES)),
)
def test_decision_table_complete_enumeration(confidence: float, streak: int) -> None:
    result = decide(CONFIG, FallbackInput(response=_resp(confidence), consecutive_low=streak))
    expected = _expected_decision(confidence, streak)
    assert result.decision is expected, (
        f"confidence={confidence}, streak={streak}: "
        f"expected {expected.value}, got {result.decision.value}"
    )


def test_decision_table_cardinality() -> None:
    cells = list(itertools.product(CONFIDENCE_VALUES, STREAK_VALUES))
    assert len(cells) == len(CONFIDENCE_VALUES) * len(STREAK_VALUES)
    # Sanity: more than one cell hits each branch.
    decisions = [
        decide(CONFIG, FallbackInput(response=_resp(c), consecutive_low=s)).decision
        for c, s in cells
    ]
    assert decisions.count(FallbackDecision.USE_LLM) >= 1
    assert decisions.count(FallbackDecision.USE_TEMPLATE) >= 1
    assert decisions.count(FallbackDecision.ESCALATE) >= 1


def test_decision_table_size_is_at_least_50() -> None:
    """Documented size of the decision-table sweep."""
    assert len(CONFIDENCE_VALUES) * len(STREAK_VALUES) >= 50
