"""Channel protocol and shared message shape."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field


class Direction(str, Enum):
    """Direction of a turn relative to the engine."""

    IN = "in"
    OUT = "out"


class ChannelMessage(BaseModel):
    """Uniform internal shape carried across all channels."""

    model_config = ConfigDict(frozen=True)

    conversation_id: str | None = Field(
        default=None,
        description="Set by the orchestrator after identity resolution. None on raw ingest.",
    )
    channel: str = Field(description="email | chat")
    direction: Direction
    sender: str = Field(description="Sender identity. Email address for email, user_id for chat.")
    body: str
    attachments: tuple[str, ...] = Field(default_factory=tuple)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))

    def with_conversation_id(self, conversation_id: str) -> ChannelMessage:
        """Return a new message with the conversation id stamped on it."""

        return self.model_copy(update={"conversation_id": conversation_id})


@runtime_checkable
class Channel(Protocol):
    """A duplex channel adapter.

    Implementations must be hermetic-testable; the email and chat adapters both
    accept a backing store/transport that can be swapped for an in-memory mock
    in unit tests.
    """

    name: str

    async def poll(self) -> list[ChannelMessage]:
        """Return any new inbound messages and remove them from the source."""
        ...

    async def send(self, message: ChannelMessage) -> None:
        """Deliver an outbound message."""
        ...
