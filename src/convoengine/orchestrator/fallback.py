"""Confidence-threshold fallback dispatcher."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from convoengine.orchestrator.llm_client import LLMResponse


class FallbackDecision(str, Enum):
    """What to do with a turn after the confidence check."""

    USE_LLM = "use_llm"
    USE_TEMPLATE = "use_template"
    ESCALATE = "escalate"


@dataclass(frozen=True)
class FallbackConfig:
    """Per-channel-tunable thresholds and rolling-window size."""

    floor: float = 0.6
    very_low_floor: float = 0.3
    window: int = 3


@dataclass(frozen=True)
class FallbackInput:
    """The signal we use to dispatch the fallback decision."""

    response: LLMResponse
    consecutive_low: int


@dataclass(frozen=True)
class FallbackOutput:
    """The decision and rationale."""

    decision: FallbackDecision
    reason: str
    new_consecutive_low: int


TEMPLATE_RESPONSE_TEXT = (
    "I'm not fully sure I understand. Could you rephrase, "
    "or let me know if you'd like to speak with someone?"
)


def decide(config: FallbackConfig, signal: FallbackInput) -> FallbackOutput:
    """Pure dispatcher.

    Rules, in order:
      1. confidence < very_low_floor  -> ESCALATE (single-turn dropout)
      2. confidence < floor and consecutive_low+1 >= window -> ESCALATE
      3. confidence < floor                                  -> USE_TEMPLATE
      4. otherwise                                           -> USE_LLM
    """
    conf = signal.response.confidence

    if conf < config.very_low_floor:
        return FallbackOutput(
            decision=FallbackDecision.ESCALATE,
            reason=f"confidence {conf:.2f} below very-low floor {config.very_low_floor:.2f}",
            new_consecutive_low=signal.consecutive_low + 1,
        )

    if conf < config.floor:
        new_streak = signal.consecutive_low + 1
        if new_streak >= config.window:
            return FallbackOutput(
                decision=FallbackDecision.ESCALATE,
                reason=(
                    f"confidence {conf:.2f} below floor {config.floor:.2f} for "
                    f"{new_streak} consecutive turns (window={config.window})"
                ),
                new_consecutive_low=new_streak,
            )
        return FallbackOutput(
            decision=FallbackDecision.USE_TEMPLATE,
            reason=f"confidence {conf:.2f} below floor {config.floor:.2f}, single-turn template",
            new_consecutive_low=new_streak,
        )

    return FallbackOutput(
        decision=FallbackDecision.USE_LLM,
        reason="confidence above floor",
        new_consecutive_low=0,
    )
