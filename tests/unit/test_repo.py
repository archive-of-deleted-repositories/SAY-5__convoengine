"""Repository / persistence tests against an in-memory SQLite engine."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from sqlalchemy.orm import Session

from convoengine.orchestrator.conversation import ConversationState
from convoengine.repo import (
    find_conversation_by_identity,
    get_or_create_conversation,
    link_identity,
    list_turns,
    record_turn,
)
from convoengine.store import create_all, make_engine, make_session_factory


@pytest.fixture
def session() -> Iterator[Session]:
    engine = make_engine("sqlite:///:memory:")
    create_all(engine)
    factory = make_session_factory(engine)
    s = factory()
    try:
        yield s
        s.commit()
    finally:
        s.close()
        engine.dispose()


def test_get_or_create_creates_new_conversation(session: Session) -> None:
    conv, created = get_or_create_conversation(
        session, channel="email", identity="alice@example.com"
    )
    assert created is True
    assert conv.state == ConversationState.GREETING.value
    assert conv.id.startswith("conv-")


def test_get_or_create_reuses_when_identity_matches(session: Session) -> None:
    first, _ = get_or_create_conversation(session, channel="email", identity="alice@example.com")
    session.commit()
    second, created = get_or_create_conversation(
        session, channel="email", identity="alice@example.com"
    )
    assert created is False
    assert second.id == first.id


def test_link_identity_is_idempotent(session: Session) -> None:
    conv, _ = get_or_create_conversation(session, channel="email", identity="alice@example.com")
    link_identity(session, conv.id, channel="chat", identity="u-42")
    link_identity(session, conv.id, channel="chat", identity="u-42")
    session.commit()
    found = find_conversation_by_identity(session, "chat", "u-42")
    assert found is not None
    assert found.id == conv.id


def test_record_turn_writes_payload(session: Session) -> None:
    conv, _ = get_or_create_conversation(session, channel="email", identity="alice@example.com")
    record_turn(
        session,
        conversation_id=conv.id,
        channel="email",
        direction="in",
        body="hello",
        state_before=ConversationState.GREETING,
        state_after=ConversationState.CLARIFYING,
        confidence=None,
        flags={"intent": "greet"},
        payload={"raw": {"a": 1}},
    )
    session.commit()
    rows = list_turns(session, conv.id)
    assert len(rows) == 1
    assert rows[0].body == "hello"
    assert rows[0].flags == {"intent": "greet"}
    assert rows[0].state_after == ConversationState.CLARIFYING.value


def test_cross_channel_lookup_returns_same_conversation(session: Session) -> None:
    # Email identity creates the conversation.
    conv, _ = get_or_create_conversation(session, channel="email", identity="alice@example.com")
    session.commit()
    # A chat ingest knows the user's email-on-file and asks for either.
    found, created = get_or_create_conversation(
        session,
        channel="chat",
        identity="u-42",
        extra_identities=[("email", "alice@example.com")],
    )
    assert created is False
    assert found.id == conv.id
    session.commit()
    # And now the chat user_id is linked too.
    via_chat = find_conversation_by_identity(session, "chat", "u-42")
    assert via_chat is not None
    assert via_chat.id == conv.id
