# Channel linkage

A `conversation_id` is the unit of state. A single user can reach the engine
on email and chat simultaneously and we want a single conversation to span
both channels.

## Identities

Each channel has its own notion of "who sent this":

- email: the sender's email address
- chat:  the chat platform's stable `user_id`

We persist these in the `conversation_identities` table as
`(conversation_id, channel, identity)` rows. A conversation has many identities.

## Lookup

When a new inbound message arrives, the orchestrator calls
`get_or_create_conversation(channel, identity, extra_identities=...)`.

- The primary `(channel, identity)` pair is always checked.
- `extra_identities` is a list of `(channel, identity)` pairs the caller
  already knows about for the same user. The chat webhook sends the user's
  email-on-file as an extra identity, which is what lets the orchestrator
  find an existing email-started conversation.

If any of the provided identities already point at a conversation, that
conversation is reused and every identity in the call is linked to it. If
none match, a new conversation is created and every identity is linked to
that fresh row.

## Concretely

```
1. Email in from alice@example.com
   - no conversation_identity matches  -> create conv-A
   - link (email, alice@example.com)   to conv-A

2. Chat in from user_id=u-42, user_email=alice@example.com
   - lookup (chat, u-42)               -> miss
   - lookup (email, alice@example.com) -> hit -> conv-A
   - link (chat, u-42)                 to conv-A

3. Chat in from user_id=u-42
   - lookup (chat, u-42)               -> conv-A
```

The whole machinery is six functions in `src/convoengine/repo.py`. No
graph traversal, no JOIN through n tables; just an indexed lookup per
identity.
