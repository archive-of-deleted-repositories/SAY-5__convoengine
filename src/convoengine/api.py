"""FastAPI app exposing ingest endpoints and a conversation read API."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.orm import Session, sessionmaker

from convoengine.channels._base import ChannelMessage, Direction
from convoengine.channels.chat import ChatWebhookPayload
from convoengine.config import Settings, get_settings
from convoengine.operator.escalation import HttpxSender, OperatorClient
from convoengine.orchestrator.conversation import ConversationState
from convoengine.orchestrator.engine import Engine
from convoengine.orchestrator.fallback import FallbackConfig
from convoengine.orchestrator.llm_client import FakeLLMProvider, LLMProvider
from convoengine.repo import list_turns, set_state
from convoengine.store import (
    Conversation,
    create_all,
    make_engine,
    make_session_factory,
)


class EmailIngest(BaseModel):
    """Payload accepted on POST /v1/ingest/email."""

    model_config = ConfigDict(extra="ignore")
    sender: str
    body: str


class ChatIngest(ChatWebhookPayload):
    """Alias so the OpenAPI docs read cleanly."""


class OperatorReply(BaseModel):
    model_config = ConfigDict(extra="forbid")
    body: str = Field(min_length=1)
    operator_id: str = Field(min_length=1)


class IdentityLink(BaseModel):
    model_config = ConfigDict(extra="forbid")
    conversation_id: str = Field(min_length=1)
    channel: str = Field(min_length=1)
    identity: str = Field(min_length=1)


class TurnView(BaseModel):
    id: str
    channel: str
    direction: str
    body: str
    state_before: str
    state_after: str
    confidence: float | None
    flags: dict[str, Any]


class ConversationView(BaseModel):
    id: str
    state: str
    low_conf_streak: int
    turns: list[TurnView]


@dataclass
class AppState:
    """Resources owned by the FastAPI app."""

    settings: Settings
    session_factory: sessionmaker[Session]
    engine: Engine


def build_app(*, provider: LLMProvider | None = None, db_url: str | None = None) -> FastAPI:
    """Construct a FastAPI app instance.

    Both `provider` and `db_url` are injectable so tests can pass in a Fake
    provider and a per-test database.
    """
    settings = get_settings()
    actual_db_url = db_url or settings.db_url
    eng = make_engine(actual_db_url)
    create_all(eng)
    factory = make_session_factory(eng)

    llm: LLMProvider = provider if provider is not None else FakeLLMProvider()
    operator_client = OperatorClient(
        url=settings.operator_url,
        secret=settings.operator_hmac_secret,
        sender=HttpxSender(timeout=settings.operator_timeout_s),
    )
    fallback_config = FallbackConfig(
        floor=settings.confidence_floor,
        very_low_floor=settings.very_low_floor,
        window=settings.low_conf_window,
    )
    orchestrator = Engine(
        provider=llm,
        fallback_config=fallback_config,
        operator_client=operator_client,
        max_summary_turns=settings.max_summary_turns,
    )

    state = AppState(settings=settings, session_factory=factory, engine=orchestrator)
    app = FastAPI(title="ConvoEngine", version="0.1.0")
    app.state.convo = state
    _attach_routes(app)
    return app


def _get_state(app_state: Any = None) -> AppState:
    # FastAPI injects via Depends() but we read from app.state.
    # The Depends() seam is set up in routes.
    raise RuntimeError("get_state should be overridden via Depends; see _attach_routes")


@contextmanager
def _session_for(state: AppState) -> Iterator[Session]:
    s = state.session_factory()
    try:
        yield s
        s.commit()
    except Exception:
        s.rollback()
        raise
    finally:
        s.close()


def _attach_routes(app: FastAPI) -> None:
    def state_dep() -> AppState:
        return app.state.convo  # type: ignore[no-any-return]

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/v1/ingest/email")
    async def ingest_email(
        payload: EmailIngest, state: AppState = Depends(state_dep)
    ) -> dict[str, Any]:
        msg = ChannelMessage(
            channel="email",
            direction=Direction.IN,
            sender=payload.sender,
            body=payload.body,
        )
        with _session_for(state) as session:
            result = await state.engine.handle_message(session, msg)
        return {
            "conversation_id": result.conversation_id,
            "decision": result.decision.value,
            "reply": result.response.body if result.response else None,
        }

    @app.post("/v1/ingest/chat")
    async def ingest_chat(
        payload: ChatIngest, state: AppState = Depends(state_dep)
    ) -> dict[str, Any]:
        msg = ChannelMessage(
            channel="chat",
            direction=Direction.IN,
            sender=payload.user_id,
            body=payload.text,
            attachments=tuple(payload.attachments),
        )
        extras: list[tuple[str, str]] = []
        if payload.user_email:
            extras.append(("email", payload.user_email))
        with _session_for(state) as session:
            result = await state.engine.handle_message(session, msg, extra_identities=extras)
        return {
            "conversation_id": result.conversation_id,
            "decision": result.decision.value,
            "reply": result.response.body if result.response else None,
        }

    @app.get("/v1/conversations/{conversation_id}")
    def get_conversation(
        conversation_id: str, state: AppState = Depends(state_dep)
    ) -> ConversationView:
        with _session_for(state) as session:
            conv = session.get(Conversation, conversation_id)
            if conv is None:
                raise HTTPException(status_code=404, detail="conversation not found")
            turns = list_turns(session, conv.id)
            return ConversationView(
                id=conv.id,
                state=conv.state,
                low_conf_streak=conv.low_conf_streak,
                turns=[
                    TurnView(
                        id=t.id,
                        channel=t.channel,
                        direction=t.direction,
                        body=t.body,
                        state_before=t.state_before,
                        state_after=t.state_after,
                        confidence=t.confidence,
                        flags=t.flags,
                    )
                    for t in turns
                ],
            )

    @app.post("/v1/identities/link")
    def link_identities(
        payload: IdentityLink, state: AppState = Depends(state_dep)
    ) -> dict[str, Any]:
        """Link a (channel, identity) pair to an existing conversation.

        Used to register, e.g., that a chat `user_id` belongs to the same
        person as an email address that already has a conversation. After
        this call, future inbound messages on the new channel will join the
        existing conversation.
        """
        from convoengine.repo import link_identity

        with _session_for(state) as session:
            conv = session.get(Conversation, payload.conversation_id)
            if conv is None:
                raise HTTPException(status_code=404, detail="conversation not found")
            link_identity(
                session,
                conv.id,
                channel=payload.channel,
                identity=payload.identity,
            )
            return {"conversation_id": conv.id, "linked": True}

    @app.post("/v1/operator/take_over/{conversation_id}")
    def operator_take_over(
        conversation_id: str,
        payload: OperatorReply,
        state: AppState = Depends(state_dep),
    ) -> dict[str, Any]:
        with _session_for(state) as session:
            conv = session.get(Conversation, conversation_id)
            if conv is None:
                raise HTTPException(status_code=404, detail="conversation not found")
            if conv.state not in {
                ConversationState.ESCALATED.value,
                ConversationState.OPERATOR_ACTIVE.value,
            }:
                raise HTTPException(
                    status_code=409,
                    detail=f"cannot take over from state {conv.state!r}",
                )
            from convoengine.repo import record_turn

            state_before = ConversationState(conv.state)
            record_turn(
                session,
                conversation_id=conv.id,
                channel="operator",
                direction="out",
                body=payload.body,
                state_before=state_before,
                state_after=ConversationState.OPERATOR_ACTIVE,
                flags={"operator_id": payload.operator_id},
            )
            set_state(session, conv, ConversationState.OPERATOR_ACTIVE)
            return {"conversation_id": conv.id, "state": conv.state}


# Convenience module-level instance for `uvicorn convoengine.api:app`.
try:
    app = build_app()
except Exception:  # pragma: no cover - graceful degrade when env is incomplete
    app = FastAPI(title="ConvoEngine (uninitialized)")
