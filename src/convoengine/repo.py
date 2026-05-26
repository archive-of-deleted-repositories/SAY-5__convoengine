"""Repository helpers: identity resolution, conversation lookup, turn writes."""

from __future__ import annotations

from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from convoengine.orchestrator.conversation import ConversationState
from convoengine.store import Conversation, ConversationIdentity, Turn


def find_conversation_by_identity(
    session: Session, channel: str, identity: str
) -> Conversation | None:
    """Locate an existing conversation given a channel-scoped identity."""
    stmt = (
        select(Conversation)
        .join(ConversationIdentity, ConversationIdentity.conversation_id == Conversation.id)
        .where(ConversationIdentity.channel == channel)
        .where(ConversationIdentity.identity == identity)
        .limit(1)
    )
    return session.execute(stmt).scalar_one_or_none()


def find_conversation_by_any_identity(
    session: Session, identities: list[tuple[str, str]]
) -> Conversation | None:
    """Return the first conversation whose identity matches any (channel, identity) pair."""
    for channel, identity in identities:
        conv = find_conversation_by_identity(session, channel, identity)
        if conv is not None:
            return conv
    return None


def get_or_create_conversation(
    session: Session,
    *,
    channel: str,
    identity: str,
    extra_identities: list[tuple[str, str]] | None = None,
) -> tuple[Conversation, bool]:
    """Return (conversation, created). Will reuse a conversation if any provided identity matches."""
    search_keys = [(channel, identity)]
    if extra_identities:
        search_keys.extend(extra_identities)

    existing = find_conversation_by_any_identity(session, search_keys)
    if existing is not None:
        # Make sure this channel/identity is linked.
        link_identity(session, existing.id, channel=channel, identity=identity)
        if extra_identities:
            for ch, ident in extra_identities:
                link_identity(session, existing.id, channel=ch, identity=ident)
        return existing, False

    conv = Conversation(state=ConversationState.GREETING.value)
    session.add(conv)
    session.flush()  # populate id
    link_identity(session, conv.id, channel=channel, identity=identity)
    if extra_identities:
        for ch, ident in extra_identities:
            link_identity(session, conv.id, channel=ch, identity=ident)
    return conv, True


def link_identity(session: Session, conversation_id: str, *, channel: str, identity: str) -> None:
    """Idempotently link an identity to a conversation."""
    stmt = select(ConversationIdentity).where(
        ConversationIdentity.conversation_id == conversation_id,
        ConversationIdentity.channel == channel,
        ConversationIdentity.identity == identity,
    )
    if session.execute(stmt).scalar_one_or_none() is not None:
        return
    session.add(
        ConversationIdentity(conversation_id=conversation_id, channel=channel, identity=identity)
    )


def record_turn(
    session: Session,
    *,
    conversation_id: str,
    channel: str,
    direction: str,
    body: str,
    state_before: ConversationState,
    state_after: ConversationState,
    confidence: float | None = None,
    flags: dict[str, Any] | None = None,
    payload: dict[str, Any] | None = None,
) -> Turn:
    """Append a turn row."""
    turn = Turn(
        conversation_id=conversation_id,
        channel=channel,
        direction=direction,
        body=body,
        state_before=state_before.value,
        state_after=state_after.value,
        confidence=confidence,
        flags=flags or {},
        payload=payload or {},
    )
    session.add(turn)
    return turn


def list_turns(session: Session, conversation_id: str, limit: int | None = None) -> list[Turn]:
    stmt = select(Turn).where(Turn.conversation_id == conversation_id).order_by(Turn.created_at)
    if limit is not None:
        stmt = stmt.limit(limit)
    return list(session.execute(stmt).scalars())


def set_state(session: Session, conversation: Conversation, state: ConversationState) -> None:
    conversation.state = state.value


def bump_low_conf_streak(conversation: Conversation, low: bool) -> None:
    if low:
        conversation.low_conf_streak += 1
    else:
        conversation.low_conf_streak = 0
