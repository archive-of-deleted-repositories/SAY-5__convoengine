"""Real Claude wire contract through synthetic HTTPX transports only."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest

from convoengine.orchestrator.conversation import ConversationState
from convoengine.orchestrator.llm_client import ClaudeProvider, LLMProviderError, LLMRequest

KEY = "synthetic-provider-key"
VALID = {
    "schema_version": 1,
    "action": "reply",
    "response_text": "Synthetic answer.",
    "confidence": 0.9,
    "suggested_state": "answering",
}


def envelope(verdict: Any = None) -> dict[str, Any]:
    return {
        "type": "message",
        "role": "assistant",
        "stop_reason": "end_turn",
        "content": [{"type": "text", "text": json.dumps(VALID if verdict is None else verdict)}],
    }


def request(text: str = "Synthetic message.") -> LLMRequest:
    return LLMRequest(
        state=ConversationState.GREETING,
        history=("Previous synthetic turn.",),
        last_message=text,
        channel="chat",
    )


def provider(handler: Any, **kwargs: Any) -> ClaudeProvider:
    return ClaudeProvider(KEY, "synthetic-model", transport=httpx.MockTransport(handler), **kwargs)


@pytest.mark.asyncio
async def test_messages_payload_and_response_contract() -> None:
    calls = []

    def handle(req: httpx.Request) -> httpx.Response:
        calls.append(req)
        assert str(req.url) == "https://api.anthropic.com/v1/messages"
        assert req.method == "POST"
        assert req.headers["x-api-key"] == KEY
        assert req.headers["anthropic-version"] == "2023-06-01"
        assert req.headers["accept-encoding"] == "identity"
        payload = json.loads(req.content)
        assert payload["model"] == "synthetic-model"
        assert payload["max_tokens"] == 512
        assert "tools" not in payload
        assert KEY not in req.content.decode()
        assert json.loads(payload["messages"][0]["content"]) == {
            "state": "greeting",
            "history": ["Previous synthetic turn."],
            "last_message": "Synthetic message.",
            "channel": "chat",
        }
        schema = payload["output_config"]["format"]["schema"]
        assert schema["additionalProperties"] is False
        assert set(schema["required"]) == set(VALID)
        return httpx.Response(200, json=envelope())

    assert json.loads(await provider(handle).complete(request())) == VALID
    assert len(calls) == 1


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema_version", 2),
        ("schema_version", True),
        ("schema_version", 1.0),
        ("schema_version", "1"),
        ("confidence", True),
        ("confidence", "0.9"),
        ("confidence", float("nan")),
        ("confidence", float("inf")),
        ("confidence", -0.1),
        ("confidence", 1.1),
        ("response_text", ""),
        ("response_text", 42),
        ("response_text", "x" * 8001),
        ("action", "tool_call"),
        ("suggested_state", "unknown"),
        ("unexpected", "field"),
    ],
)
@pytest.mark.asyncio
async def test_strict_response_fields_are_not_coerced_or_defaulted(field: str, value: Any) -> None:
    verdict = {**VALID, field: value}
    with pytest.raises(LLMProviderError, match="invalid structured"):
        await provider(lambda _: httpx.Response(200, json=envelope(verdict))).complete(request())


@pytest.mark.parametrize("missing", list(VALID))
@pytest.mark.asyncio
async def test_every_response_field_is_required(missing: str) -> None:
    verdict = dict(VALID)
    del verdict[missing]
    with pytest.raises(LLMProviderError):
        await provider(lambda _: httpx.Response(200, json=envelope(verdict))).complete(request())


@pytest.mark.parametrize(
    "changes",
    [
        {"type": "error"},
        {"role": "user"},
        {"stop_reason": "max_tokens"},
        {"stop_reason": "refusal"},
        {"content": []},
        {"content": [{"type": "tool_use", "name": "something"}]},
        {"content": [{"type": "text", "text": 42}]},
    ],
)
@pytest.mark.asyncio
async def test_incomplete_or_nontext_envelope_is_rejected(changes: dict[str, Any]) -> None:
    with pytest.raises(LLMProviderError):
        await provider(lambda _: httpx.Response(200, json={**envelope(), **changes})).complete(
            request()
        )


@pytest.mark.asyncio
async def test_duplicate_json_keys_are_rejected() -> None:
    response = envelope()
    response["content"][0]["text"] = json.dumps(VALID).replace(
        '"confidence": 0.9', '"confidence": 0.1, "confidence": 0.9'
    )
    with pytest.raises(LLMProviderError):
        await provider(lambda _: httpx.Response(200, json=response)).complete(request())


@pytest.mark.parametrize("status", [301, 307, 400, 401, 429, 500])
@pytest.mark.asyncio
async def test_error_responses_are_not_retried_or_echoed(status: int) -> None:
    calls = []

    def handle(req: httpx.Request) -> httpx.Response:
        calls.append(req)
        return httpx.Response(status, text=KEY, headers={"location": "https://untrusted.test/"})

    with pytest.raises(LLMProviderError) as error:
        await provider(handle).complete(request())
    assert KEY not in str(error.value)
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_transport_error_does_not_echo_details() -> None:
    def handle(_: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError(KEY)

    with pytest.raises(LLMProviderError, match="request failed") as error:
        await provider(handle).complete(request())
    assert KEY not in str(error.value)


class Chunks(httpx.AsyncByteStream):
    def __init__(self, chunks: list[bytes], delay: float = 0) -> None:
        self.chunks = chunks
        self.delay = delay
        self.closed = False
        self.cancelled = False

    async def __aiter__(self) -> AsyncIterator[bytes]:
        try:
            for chunk in self.chunks:
                await asyncio.sleep(self.delay)
                yield chunk
        except asyncio.CancelledError:
            self.cancelled = True
            raise

    async def aclose(self) -> None:
        self.closed = True


@pytest.mark.parametrize("header", ["65537", "9" * 5000, "not-a-size"])
@pytest.mark.asyncio
async def test_declared_oversized_or_invalid_response_is_rejected(header: str) -> None:
    stream = Chunks([b"{}"])
    with pytest.raises(LLMProviderError):
        await provider(
            lambda _: httpx.Response(200, headers={"content-length": header}, stream=stream)
        ).complete(request())
    assert stream.closed


@pytest.mark.asyncio
async def test_actual_stream_size_is_bounded_despite_small_header() -> None:
    stream = Chunks([b" " * 32768] * 3)
    with pytest.raises(LLMProviderError, match="size limit"):
        await provider(
            lambda _: httpx.Response(200, headers={"content-length": "1"}, stream=stream)
        ).complete(request())
    assert stream.closed


@pytest.mark.asyncio
async def test_total_deadline_cancels_a_trickling_stream_and_closes_it() -> None:
    stream = Chunks([b" "] * 1000, delay=0.01)
    started = time.monotonic()
    with pytest.raises(LLMProviderError, match="timed out"):
        await provider(lambda _: httpx.Response(200, stream=stream), timeout_seconds=0.04).complete(
            request()
        )
    assert time.monotonic() - started < 0.5
    assert stream.closed and stream.cancelled


@pytest.mark.parametrize("text", ["x" * (512 * 1024), "🙂" * 140000, "invalid\ud800text"])
@pytest.mark.asyncio
async def test_request_byte_limit_rejects_before_transport(text: str) -> None:
    calls = []

    def handle(req: httpx.Request) -> httpx.Response:
        calls.append(req)
        return httpx.Response(200, json=envelope())

    with pytest.raises(LLMProviderError):
        await provider(handle).complete(request(text))
    assert calls == []


@pytest.mark.parametrize("timeout", [0, -1, float("inf"), float("nan")])
def test_invalid_deadlines_are_rejected(timeout: float) -> None:
    with pytest.raises(ValueError, match="timeout"):
        ClaudeProvider(KEY, "synthetic-model", timeout_seconds=timeout)
