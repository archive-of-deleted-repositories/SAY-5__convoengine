# Operator handoff

When a conversation escalates (either by a single very-low-confidence turn
or by three consecutive low-confidence turns), ConvoEngine:

1. Records the inbound turn and emits an explicit "I'm pulling in a teammate"
   outbound message.
2. Transitions state to `escalated`.
3. POSTs a signed structured summary to the configured operator queue.
4. Stops generating autonomous replies on subsequent inbound messages.

An operator can take over the conversation explicitly via the
`POST /v1/operator/take_over/{conversation_id}` endpoint. This:

- Records the operator's first message as a `channel=operator, direction=out`
  turn.
- Transitions state to `operator_active`.

After takeover, further inbound messages are recorded but never trigger the
LLM (`reply: null` in the response). They are visible to the operator in
the conversation history.

## Escalation payload

```json
{
  "conversation_id": "conv-9a3b...",
  "current_state": "escalated",
  "trigger": "confidence 0.10 below very-low floor 0.30",
  "last_turns": [
    {
      "channel": "email",
      "direction": "in",
      "body": "Help",
      "state_before": "greeting",
      "state_after": "greeting",
      "confidence": null
    },
    {
      "channel": "email",
      "direction": "out",
      "body": "I'm pulling in a teammate ...",
      "state_before": "greeting",
      "state_after": "escalated",
      "confidence": 0.1
    }
  ],
  "suggested_actions": ["review last 3 turns", "respond with refund policy"]
}
```

## HMAC signing

The body is signed with `X-ConvoEngine-Signature: <hex-sha256-hmac>`.

```
signature = sha256_hmac(secret, body_bytes)
```

### Pinned reference

```
secret = "shared-secret"
body   = '{"a":1,"b":2}'
signature = 40d50073c3d041e5c354fc1f955d8a221bd747d3d727839b79b5ea2d0532a32a
```

Reproduce with:

```bash
echo -n '{"a":1,"b":2}' | openssl dgst -sha256 -hmac "shared-secret"
```

This pin appears as a unit test
(`tests/unit/test_operator_handoff.py::test_pinned_hmac_reference_for_operator_interop`)
so an operator-side webhook receiver can sanity-check its implementation
against the same value.

## Operator endpoint shape

```
POST /v1/operator/take_over/{conversation_id}
Content-Type: application/json

{ "body": "Hi, I can help.", "operator_id": "op-1" }

200 OK
{ "conversation_id": "conv-...", "state": "operator_active" }

409 Conflict   # not in escalated/operator_active state
```
