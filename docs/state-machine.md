# State machine

```
greeting ─ user_message ────────────────► clarifying
greeting ─ user_clear_question ─────────► answering
greeting ─ low_confidence ──────────────► greeting
greeting ─ escalate ────────────────────► escalated
greeting ─ close ───────────────────────► closed

clarifying ─ user_message ──────────────► clarifying
clarifying ─ user_clear_question ───────► answering
clarifying ─ low_confidence ────────────► clarifying
clarifying ─ escalate ──────────────────► escalated
clarifying ─ close ─────────────────────► closed

answering ─ user_message ───────────────► answering
answering ─ user_follow_up ─────────────► clarifying
answering ─ low_confidence ─────────────► answering
answering ─ escalate ───────────────────► escalated
answering ─ close ──────────────────────► closed

escalated ─ operator_takeover ──────────► operator_active
escalated ─ close ──────────────────────► closed

operator_active ─ user_message ─────────► operator_active
operator_active ─ operator_message ─────► operator_active
operator_active ─ operator_resolve ─────► answering
operator_active ─ close ────────────────► closed

closed ─ (terminal)
```

## Invariants

1. `closed` is terminal. No transition leaves it.
2. `low_confidence` is a no-op on `greeting`, `clarifying`, and `answering`.
   The state is unchanged; only the rolling-window counter advances.
3. Every conversation begins in `greeting`.
4. Reaching `escalated` posts a structured summary to the operator queue.
5. A call to `POST /v1/operator/take_over/{id}` transitions `escalated` to
   `operator_active`.

## Encoded as data

The transition table lives in `src/convoengine/orchestrator/conversation.py`
as a `TRANSITIONS` tuple of `Transition(src, event, dst)` rows. Hypothesis
property tests (`tests/unit/test_state_machine_property.py`, added in v1)
generate random event sequences against this table and assert no illegal
transition can ever occur.
