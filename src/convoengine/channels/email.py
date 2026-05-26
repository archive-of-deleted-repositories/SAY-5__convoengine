"""Email channel: IMAP-flavored polling against a pluggable store."""

from __future__ import annotations

from collections import deque
from typing import Protocol

from convoengine.channels._base import ChannelMessage, Direction


class EmailStore(Protocol):
    """Backing store for the email adapter. Real IMAP in prod, in-memory in tests."""

    async def fetch_new(self) -> list[ChannelMessage]:
        """Return all unread messages and mark them as read."""
        ...

    async def deliver(self, message: ChannelMessage) -> None:
        """Send an outbound message."""
        ...


class MockEmailStore:
    """In-memory IMAP stand-in for tests."""

    def __init__(self) -> None:
        self._inbox: deque[ChannelMessage] = deque()
        self.sent: list[ChannelMessage] = []

    def push_inbound(self, sender: str, body: str) -> None:
        """Simulate an incoming email."""
        self._inbox.append(
            ChannelMessage(
                channel="email",
                direction=Direction.IN,
                sender=sender,
                body=body,
            )
        )

    async def fetch_new(self) -> list[ChannelMessage]:
        out: list[ChannelMessage] = []
        while self._inbox:
            out.append(self._inbox.popleft())
        return out

    async def deliver(self, message: ChannelMessage) -> None:
        self.sent.append(message)


class EmailChannel:
    """Channel adapter for email."""

    name = "email"

    def __init__(self, store: EmailStore) -> None:
        self._store = store

    async def poll(self) -> list[ChannelMessage]:
        return await self._store.fetch_new()

    async def send(self, message: ChannelMessage) -> None:
        if message.channel != "email":
            raise ValueError(f"EmailChannel cannot send a {message.channel!r} message")
        if message.direction is not Direction.OUT:
            raise ValueError("EmailChannel.send requires direction=out")
        await self._store.deliver(message)
