"""Channel adapter unit tests."""

from __future__ import annotations

import pytest

from convoengine.channels import (
    ChannelMessage,
    ChatChannel,
    ChatWebhookPayload,
    Direction,
    EmailChannel,
    MockEmailStore,
)
from convoengine.channels.chat import InMemoryChatTransport


@pytest.mark.asyncio
async def test_email_channel_polls_inbound_and_marks_them_consumed() -> None:
    store = MockEmailStore()
    store.push_inbound("alice@example.com", "hello")
    store.push_inbound("bob@example.com", "hi there")
    channel = EmailChannel(store)

    first = await channel.poll()
    second = await channel.poll()

    assert len(first) == 2
    assert second == []
    assert {m.sender for m in first} == {"alice@example.com", "bob@example.com"}
    assert all(m.direction is Direction.IN for m in first)


@pytest.mark.asyncio
async def test_email_channel_send_records_message() -> None:
    store = MockEmailStore()
    channel = EmailChannel(store)
    msg = ChannelMessage(
        channel="email",
        direction=Direction.OUT,
        sender="bot@convoengine.test",
        body="Welcome.",
    )

    await channel.send(msg)

    assert store.sent == [msg]


@pytest.mark.asyncio
async def test_email_channel_rejects_wrong_channel_or_direction() -> None:
    store = MockEmailStore()
    channel = EmailChannel(store)
    wrong_channel = ChannelMessage(
        channel="chat", direction=Direction.OUT, sender="bot", body="x"
    )
    wrong_dir = ChannelMessage(
        channel="email", direction=Direction.IN, sender="x@y.z", body="x"
    )

    with pytest.raises(ValueError):
        await channel.send(wrong_channel)
    with pytest.raises(ValueError):
        await channel.send(wrong_dir)


@pytest.mark.asyncio
async def test_chat_channel_ingest_then_poll_returns_message() -> None:
    transport = InMemoryChatTransport()
    chat = ChatChannel(transport)
    payload = ChatWebhookPayload(
        user_id="u-42",
        user_email="alice@example.com",
        text="What is your refund policy?",
    )

    chat.ingest(payload)
    drained = await chat.poll()

    assert len(drained) == 1
    assert drained[0].sender == "u-42"
    assert drained[0].body == "What is your refund policy?"
    assert drained[0].channel == "chat"
    assert drained[0].direction is Direction.IN

    # second poll is empty
    assert await chat.poll() == []


@pytest.mark.asyncio
async def test_chat_channel_send_goes_through_transport() -> None:
    transport = InMemoryChatTransport()
    chat = ChatChannel(transport)
    msg = ChannelMessage(
        channel="chat", direction=Direction.OUT, sender="bot", body="ok"
    )

    await chat.send(msg)

    assert transport.sent == [msg]


def test_channel_message_with_conversation_id_is_immutable_copy() -> None:
    msg = ChannelMessage(
        channel="email", direction=Direction.IN, sender="x@y.z", body="hi"
    )
    stamped = msg.with_conversation_id("conv-1")
    assert msg.conversation_id is None
    assert stamped.conversation_id == "conv-1"
    assert stamped.sender == "x@y.z"
