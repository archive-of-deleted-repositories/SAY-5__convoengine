"""SQLAlchemy models + session helpers."""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import (
    JSON,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    create_engine,
)
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    Session,
    mapped_column,
    relationship,
    sessionmaker,
)


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


class Base(DeclarativeBase):
    """Declarative base."""


class Conversation(Base):
    """One conversation, possibly spanning multiple channels."""

    __tablename__ = "conversations"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: _new_id("conv"))
    state: Mapped[str] = mapped_column(String(32), default="greeting", nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow, nullable=False
    )
    low_conf_streak: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    identities: Mapped[list[ConversationIdentity]] = relationship(
        back_populates="conversation", cascade="all, delete-orphan"
    )
    turns: Mapped[list[Turn]] = relationship(
        back_populates="conversation",
        cascade="all, delete-orphan",
        order_by="Turn.created_at",
    )


class ConversationIdentity(Base):
    """A channel-specific identity (email address or chat user_id) linked to a conversation."""

    __tablename__ = "conversation_identities"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: _new_id("ident"))
    conversation_id: Mapped[str] = mapped_column(
        ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False, index=True
    )
    channel: Mapped[str] = mapped_column(String(16), nullable=False)
    identity: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, nullable=False
    )

    conversation: Mapped[Conversation] = relationship(back_populates="identities")


class Turn(Base):
    """A single inbound or outbound turn within a conversation."""

    __tablename__ = "turns"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: _new_id("turn"))
    conversation_id: Mapped[str] = mapped_column(
        ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False, index=True
    )
    channel: Mapped[str] = mapped_column(String(16), nullable=False)
    direction: Mapped[str] = mapped_column(String(8), nullable=False)
    body: Mapped[str] = mapped_column(String, nullable=False)
    state_before: Mapped[str] = mapped_column(String(32), nullable=False)
    state_after: Mapped[str] = mapped_column(String(32), nullable=False)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    flags: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    payload: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, nullable=False
    )

    conversation: Mapped[Conversation] = relationship(back_populates="turns")


def make_engine(url: str, echo: bool = False) -> Any:
    """Build a SQLAlchemy engine. Module-level for easy reuse."""

    return create_engine(url, echo=echo, future=True)


def make_session_factory(engine: Any) -> sessionmaker[Session]:
    return sessionmaker(bind=engine, autoflush=False, expire_on_commit=False, future=True)


def create_all(engine: Any) -> None:
    """Create all tables. Used by tests; production uses Alembic."""
    Base.metadata.create_all(engine)


@contextmanager
def session_scope(factory: sessionmaker[Session]) -> Iterator[Session]:
    """Commit-on-success, rollback-on-error session context."""
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
