"""Tests for the confidence-threshold fallback dispatcher."""

from __future__ import annotations

import pytest

from convoengine.orchestrator.conversation import ConversationState
from convoengine.orchestrator.fallback import (
    FallbackConfig,
    FallbackDecision,
    FallbackInput,
    decide,
)
from convoengine.orchestrator.llm_client import LLMResponse


def _resp(confidence: float, action: str = "reply") -> LLMResponse:
    return LLMResponse(
        action=action,
        response_text="x",
        confidence=confidence,
        suggested_state=ConversationState.ANSWERING,
    )


@pytest.mark.parametrize(
    "confidence,consecutive_low,expected_decision",
    [
        (0.95, 0, FallbackDecision.USE_LLM),
        (0.61, 0, FallbackDecision.USE_LLM),
        (0.59, 0, FallbackDecision.USE_TEMPLATE),
        (0.59, 1, FallbackDecision.USE_TEMPLATE),
        (0.59, 2, FallbackDecision.ESCALATE),
        (0.29, 0, FallbackDecision.ESCALATE),
        (0.05, 0, FallbackDecision.ESCALATE),
    ],
)
def test_decide_matches_decision_table(
    confidence: float, consecutive_low: int, expected_decision: FallbackDecision
) -> None:
    config = FallbackConfig()
    signal = FallbackInput(response=_resp(confidence), consecutive_low=consecutive_low)
    result = decide(config, signal)
    assert result.decision is expected_decision


def test_streak_resets_when_confidence_high() -> None:
    config = FallbackConfig()
    out = decide(config, FallbackInput(response=_resp(0.9), consecutive_low=2))
    assert out.new_consecutive_low == 0


def test_streak_increments_when_confidence_low() -> None:
    config = FallbackConfig()
    out = decide(config, FallbackInput(response=_resp(0.4), consecutive_low=1))
    assert out.new_consecutive_low == 2
