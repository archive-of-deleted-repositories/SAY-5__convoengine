"""Paid service boundaries use synthetic transports and isolated memory SQLite."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from typing import Any

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, func, select
from sqlalchemy.pool import StaticPool

from convoengine import api
from convoengine.config import Settings
from convoengine.orchestrator.llm_client import ClaudeProvider
from convoengine.store import Conversation, ConversationIdentity, Turn

SERVICE_KEY = "synthetic-service-key-32-characters-long"
HEADERS = {"Authorization": f"Bearer {SERVICE_KEY}"}
VERDICT = '{"schema_version":1,"action":"reply","response_text":"Synthetic answer.","confidence":0.9,"suggested_state":"answering"}'


def valid_response() -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "type": "message",
            "role": "assistant",
            "stop_reason": "end_turn",
            "content": [{"type": "text", "text": VERDICT}],
        },
    )


def paid_app(monkeypatch: pytest.MonkeyPatch, handler: Callable[..., Any] | None = None) -> FastAPI:
    database = create_engine(
        "sqlite://", poolclass=StaticPool, connect_args={"check_same_thread": False}
    )
    monkeypatch.setattr(api, "make_engine", lambda _: database)
    settings = Settings(
        _env_file=None,
        llm_provider="claude",
        claude_api_key="synthetic-provider-key",
        claude_model="synthetic-model",
        service_api_key=SERVICE_KEY,
        paid_max_concurrency=1,
    )
    provider = ClaudeProvider(
        "synthetic-provider-key",
        "synthetic-model",
        transport=httpx.MockTransport(handler or (lambda _: valid_response())),
    )
    return api.build_app(provider=provider, settings=settings, db_url="sqlite://")


def mounted(app: FastAPI) -> FastAPI:
    parent = FastAPI()
    parent.mount("/prefix", app)
    return parent


@pytest.mark.parametrize("placement", ["direct", "mounted", "root_path"])
def test_authentication_uses_the_route_relative_path(
    monkeypatch: pytest.MonkeyPatch, placement: str
) -> None:
    calls = []
    app = paid_app(monkeypatch, lambda req: (calls.append(req), valid_response())[1])
    target = mounted(app) if placement == "mounted" else app
    prefix = "/prefix" if placement != "direct" else ""
    with TestClient(target, root_path=prefix if placement == "root_path" else "") as client:
        response = client.post(
            prefix + "/v1/ingest/email", json={"sender": "synthetic@example.test", "body": "Hi"}
        )
        assert response.status_code == 401
        assert client.get(prefix + "/v1/conversations/missing").status_code == 401
    assert calls == []


@pytest.mark.parametrize(
    "method,path",
    [
        ("POST", "/v1/ingest/email"),
        ("POST", "/v1/ingest/chat"),
        ("GET", "/v1/conversations/missing"),
        ("POST", "/v1/identities/link"),
        ("POST", "/v1/operator/take_over/missing"),
    ],
)
@pytest.mark.parametrize(
    "headers",
    [
        {},
        {"Authorization": "Bearer wrong"},
        {"Cookie": f"service_key={SERVICE_KEY}"},
        [("Authorization", f"Bearer {SERVICE_KEY}"), ("Authorization", f"Bearer {SERVICE_KEY}")],
    ],
)
def test_every_v1_route_requires_one_valid_header_before_body_or_db(
    monkeypatch: pytest.MonkeyPatch, method: str, path: str, headers: Any
) -> None:
    app = paid_app(monkeypatch)
    with TestClient(app) as client:

        def forbidden_session() -> None:
            raise AssertionError("rejected request reached DB")

        monkeypatch.setattr(app.state.convo, "session_factory", forbidden_session)
        response = client.request(
            method, path + f"?api_key={SERVICE_KEY}", headers=headers, content=b"invalid JSON"
        )
        assert response.status_code == 401
        assert response.headers["www-authenticate"] == "Bearer"
        assert SERVICE_KEY not in response.text


def test_valid_paid_turn_is_persisted_and_health_remains_public(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []
    app = paid_app(monkeypatch, lambda req: (calls.append(req), valid_response())[1])
    with TestClient(app) as client:
        assert client.get("/healthz").status_code == 200
        response = client.post(
            "/v1/ingest/chat", headers=HEADERS, json={"user_id": "synthetic", "text": "Hi"}
        )
        assert response.status_code == 200
        assert response.json()["reply"] == "Synthetic answer."
        detail = client.get(
            f'/v1/conversations/{response.json()["conversation_id"]}', headers=HEADERS
        )
        assert detail.status_code == 200
        assert [turn["body"] for turn in detail.json()["turns"]] == ["Hi", "Synthetic answer."]
        assert len(calls) == 1


@pytest.mark.parametrize(
    "current,suggested", [("closed", "answering"), ("escalated", "operator_active")]
)
def test_model_suggestion_cannot_bypass_disallowed_state_transitions(
    monkeypatch: pytest.MonkeyPatch, current: str, suggested: str
) -> None:
    verdict = {**json.loads(VERDICT), "suggested_state": suggested}
    app = paid_app(
        monkeypatch,
        lambda _: httpx.Response(
            200,
            json={
                "type": "message",
                "role": "assistant",
                "stop_reason": "end_turn",
                "content": [{"type": "text", "text": json.dumps(verdict)}],
            },
        ),
    )
    with TestClient(app) as client:
        with app.state.convo.session_factory() as session:
            conv = Conversation(state=current)
            session.add(conv)
            session.flush()
            session.add(
                ConversationIdentity(
                    conversation_id=conv.id, channel="email", identity="synthetic@example.test"
                )
            )
            conversation_id = conv.id
            session.commit()
        response = client.post(
            "/v1/ingest/email",
            headers=HEADERS,
            json={"sender": "synthetic@example.test", "body": "Hi"},
        )
        assert response.status_code == 200
        detail = client.get(f"/v1/conversations/{conversation_id}", headers=HEADERS)
        assert detail.json()["state"] == current
        assert {turn["state_after"] for turn in detail.json()["turns"]} == {current}


@pytest.mark.parametrize("kind", ["http_error", "invalid_verdict"])
def test_provider_errors_return_generic503_without_fallback_or_persistence(
    monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    calls = []

    def fail(req: httpx.Request) -> httpx.Response:
        calls.append(req)
        if kind == "http_error":
            return httpx.Response(401, text="synthetic-provider-key")
        return httpx.Response(
            200,
            json={
                "type": "message",
                "role": "assistant",
                "stop_reason": "end_turn",
                "content": [{"type": "text", "text": "synthetic-provider-key"}],
            },
        )

    app = paid_app(monkeypatch, fail)
    with TestClient(app) as client:
        response = client.post(
            "/v1/ingest/email",
            headers=HEADERS,
            json={"sender": "synthetic@example.test", "body": "Hi"},
        )
        assert response.status_code == 503
        assert response.json() == {"detail": "Claude service unavailable."}
        assert len(calls) == 1
        with app.state.convo.session_factory() as session:
            for model in (Conversation, ConversationIdentity, Turn):
                assert session.scalar(select(func.count()).select_from(model)) == 0


@pytest.mark.parametrize("prefix", ["", "/prefix"])
@pytest.mark.asyncio
async def test_oversized_stream_rejected_before_database_even_when_mounted(
    monkeypatch: pytest.MonkeyPatch, prefix: str
) -> None:
    app = paid_app(monkeypatch)

    def forbidden_session() -> None:
        raise AssertionError("oversized request reached DB")

    monkeypatch.setattr(app.state.convo, "session_factory", forbidden_session)

    async def body() -> Any:
        for _ in range(3):
            yield b"x" * (256 * 1024)

    target = mounted(app) if prefix else app
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=target), base_url="http://synthetic.test"
    ) as client:
        response = await client.post(prefix + "/v1/ingest/email", headers=HEADERS, content=body())
    assert response.status_code == 413


@pytest.mark.parametrize("prefix", ["", "/prefix"])
@pytest.mark.parametrize("finish", ["success", "cancel", "error"])
@pytest.mark.asyncio
async def test_ingest_admission_is_nonqueued_and_released_on_every_exit(
    monkeypatch: pytest.MonkeyPatch, prefix: str, finish: str
) -> None:
    started = asyncio.Event()
    release = asyncio.Event()
    calls = []

    async def handle(req: httpx.Request) -> httpx.Response:
        calls.append(req)
        if len(calls) == 1:
            started.set()
            await release.wait()
            if finish == "error":
                return httpx.Response(500)
        return valid_response()

    app = paid_app(monkeypatch, handle)
    target = mounted(app) if prefix else app
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=target), base_url="http://synthetic.test"
    ) as client:
        url = prefix + "/v1/ingest/email"
        first = asyncio.create_task(
            client.post(url, headers=HEADERS, json={"sender": "first@example.test", "body": "Hi"})
        )
        await asyncio.wait_for(started.wait(), timeout=1)
        try:
            second = await client.post(
                url, headers=HEADERS, json={"sender": "second@example.test", "body": "Hi"}
            )
            assert second.status_code == 429
            assert len(calls) == 1
        finally:
            if finish == "cancel":
                first.cancel()
            else:
                release.set()
        if finish == "cancel":
            with pytest.raises(asyncio.CancelledError):
                await first
        else:
            assert (await first).status_code == (503 if finish == "error" else 200)
        if finish != "success":
            with app.state.convo.session_factory() as session:
                assert session.scalar(select(func.count()).select_from(Conversation)) == 0
        third = await client.post(
            url, headers=HEADERS, json={"sender": "third@example.test", "body": "Hi"}
        )
        assert third.status_code == 200
        assert len(calls) == 2


@pytest.mark.asyncio
async def test_slow_request_body_times_out_and_releases_admission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from convoengine import security

    monkeypatch.setattr(security, "BODY_READ_TIMEOUT_SECONDS", 0.02)
    app = paid_app(monkeypatch)

    async def trickle() -> Any:
        for _ in range(100):
            yield b" "
            await asyncio.sleep(0.01)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://synthetic.test"
    ) as client:
        timed_out = await client.post("/v1/ingest/email", headers=HEADERS, content=trickle())
        assert timed_out.status_code == 408
        accepted = await client.post(
            "/v1/ingest/email",
            headers=HEADERS,
            json={"sender": "synthetic@example.test", "body": "Hi"},
        )
        assert accepted.status_code == 200


@pytest.mark.parametrize("prefix", ["", "/prefix"])
@pytest.mark.asyncio
async def test_all_authenticated_database_routes_share_admission(
    monkeypatch: pytest.MonkeyPatch, prefix: str
) -> None:
    started = asyncio.Event()
    release = asyncio.Event()

    async def handle(_: httpx.Request) -> httpx.Response:
        started.set()
        await release.wait()
        return valid_response()

    app = paid_app(monkeypatch, handle)
    target = mounted(app) if prefix else app
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=target), base_url="http://synthetic.test"
    ) as client:
        first = asyncio.create_task(
            client.post(
                prefix + "/v1/ingest/email",
                headers=HEADERS,
                json={"sender": "synthetic@example.test", "body": "Hi"},
            )
        )
        await asyncio.wait_for(started.wait(), timeout=1)
        try:
            for method, path, payload in [
                ("GET", "/v1/conversations/missing", None),
                (
                    "POST",
                    "/v1/identities/link",
                    {
                        "conversation_id": "missing",
                        "channel": "email",
                        "identity": "synthetic@example.test",
                    },
                ),
                (
                    "POST",
                    "/v1/operator/take_over/missing",
                    {"body": "Hi", "operator_id": "synthetic"},
                ),
            ]:
                response = await client.request(
                    method, prefix + path, headers=HEADERS, json=payload
                )
                assert response.status_code == 429
            assert (await client.get(prefix + "/healthz")).status_code == 200
        finally:
            release.set()
            await first
