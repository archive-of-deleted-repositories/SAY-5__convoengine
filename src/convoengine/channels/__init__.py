"""Channel adapters: email and chat."""

from convoengine.channels._base import Channel, ChannelMessage, Direction
from convoengine.channels.chat import ChatChannel, ChatWebhookPayload
from convoengine.channels.email import EmailChannel, EmailStore, MockEmailStore

__all__ = [
    "Channel",
    "ChannelMessage",
    "ChatChannel",
    "ChatWebhookPayload",
    "Direction",
    "EmailChannel",
    "EmailStore",
    "MockEmailStore",
]
