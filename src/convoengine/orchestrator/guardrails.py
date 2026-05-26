"""Structured-output guardrails: validate JSON, retry with stricter prompt once."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Awaitable, Callable

from pydantic import ValidationError

from convoengine.orchestrator.llm_client import LLMRequest, LLMResponse


class GuardrailError(RuntimeError):
    """Raised when the LLM response cannot be validated even after retry."""


@dataclass
class GuardrailOutcome:
    """The result of running an LLM call through the guardrails."""

    response: LLMResponse
    retried: bool


CompleteFn = Callable[[LLMRequest], Awaitable[str]]


def parse_response(raw: str) -> LLMResponse:
    """Parse and validate a raw response. Raises GuardrailError on failure."""
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise GuardrailError(f"response was not valid JSON: {exc}") from exc
    try:
        return LLMResponse.model_validate(data)
    except ValidationError as exc:
        raise GuardrailError(f"response did not match schema: {exc}") from exc


async def call_with_guardrails(complete: CompleteFn, request: LLMRequest) -> GuardrailOutcome:
    """Run complete(); on validation failure, retry once with strict=True.

    The provider implementation is expected to interpret strict=True as "add
    a tightened prompt instructing strict JSON output".
    """
    try:
        first = await complete(request)
        return GuardrailOutcome(response=parse_response(first), retried=False)
    except GuardrailError:
        retry_request = LLMRequest(
            state=request.state,
            history=request.history,
            last_message=request.last_message,
            channel=request.channel,
            strict=True,
        )
        second = await complete(retry_request)
        try:
            return GuardrailOutcome(response=parse_response(second), retried=True)
        except GuardrailError as exc:
            raise GuardrailError(f"guardrails failed after retry: {exc}") from exc
