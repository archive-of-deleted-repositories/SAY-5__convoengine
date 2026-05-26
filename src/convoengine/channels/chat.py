"""Chat channel: HTTP webhook on the way in, callback URL on the way out."""

from __future__ import annotations

from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from convoengine.channels._base import ChannelMessage, Direction


class ChatWebhookPayload(BaseModel):
    """Inbound payload from a chat platform's outgoing-webhook integration."""

    model_config = ConfigDict(extra="ignore")

    user_id: str = Field(description="Stable user identifier from the chat platform.")
    user_email: str | None = Field(
        default=None,
        description="Optional email-on-file used for cross-channel identity linkage.",
    )
    text: str
    attachments: list[str] = Field(default_factory=list)


class ChatTransport(Protocol):
    """Outbound transport for chat replies."""

    async def deliver(self, message: ChannelMessage) -> None: ...


class InMemoryChatTransport:
    """Test/dev transport; just stores what was sent."""

    def __init__(self) -> None:
        self.sent: list[ChannelMessage] = []

    async def deliver(self, message: ChannelMessage) -> None:
        self.sent.append(message)


class ChatChannel:
    """Channel adapter for chat. Polling is a no-op; ingest is push-driven."""

    name = "chat"

    def __init__(self, transport: ChatTransport) -> None:
        self._transport = transport
        self._pending: list[ChannelMessage] = []

    def ingest(self, payload: ChatWebhookPayload) -> ChannelMessage:
        """Synchronously translate a webhook payload into a ChannelMessage and queue it."""
        message = ChannelMessage(
            channel="chat",
            direction=Direction.IN,
            sender=payload.user_id,
            body=payload.text,
            attachments=tuple(payload.attachments),
        )
        self._pending.append(message)
        return message

    async def poll(self) -> list[ChannelMessage]:
        out, self._pending = self._pending, []
        return out

    async def send(self, message: ChannelMessage) -> None:
        if message.channel != "chat":
            raise ValueError(f"ChatChannel cannot send a {message.channel!r} message")
        if message.direction is not Direction.OUT:
            raise ValueError("ChatChannel.send requires direction=out")
        await self._transport.deliver(message)
