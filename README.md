# ConvoEngine

Multi-channel conversational AI backend. A Python FastAPI service that
orchestrates LLM turns across email and chat for the same user, persists
per-conversation state to PostgreSQL, validates every model response against a
Pydantic schema, and falls back to template responses or operator escalation
when model confidence drops below a configurable threshold.

## How this differs

ConvoEngine sits next to but is distinct from typical "LLM wrapper" projects on
three axes:

1. **Multi-channel.** A single conversation can span email (polled inbox) and
   chat (HTTP webhook). The orchestrator looks up an existing
   `conversation_id` by the sender's identity and continues from the previous
   state regardless of which channel the new turn arrived on.

2. **Per-conversation state machine.** Every conversation lives in one of
   `greeting`, `clarifying`, `answering`, `escalated`, `operator_active`, or
   `closed`. Transitions are encoded as data, persisted to Postgres, and
   property-tested with Hypothesis.

3. **Confidence-threshold fallback.** Every LLM call returns a structured
   `{action, response_text, confidence, suggested_state}` payload. Below the
   threshold we emit a template response. Three consecutive low-confidence
   turns (or a single very-low one) flips the conversation to `escalated` and
   posts a structured summary to an operator queue.

## Stack

- Python 3.11+, FastAPI, Pydantic v2
- SQLAlchemy 2 + Alembic + PostgreSQL
- httpx for outbound, structlog for logging
- FakeLLMProvider for hermetic tests, Claude provider stub for live use
- pytest + Hypothesis + testcontainers
- Multi-stage Dockerfile, docker-compose for local

## Quickstart

```bash
make install
make lint typecheck test
docker compose up -d postgres
make test-integration
```

## Layout

```
src/convoengine/
  api.py                  FastAPI app
  channels/
    _base.py              Channel protocol, ChannelMessage
    email.py              IMAP-flavored poller (mock store in tests)
    chat.py               HTTP webhook handler
  orchestrator/
    conversation.py       per-conversation state machine
    llm_client.py         FakeLLMProvider + ClaudeProvider stub
    guardrails.py         structured-output validation + retry
    fallback.py           confidence-threshold dispatch
  store.py                SQLAlchemy models + migrations
  operator/
    escalation.py         POST to operator queue
  config.py
```

## Layered roadmap

| Layer | Commit |
|-------|--------|
| base  | scaffold + 8 build stages |
| v1    | hypothesis property tests + decision-table coverage |
| v2    | per-conversation latency bench + bench-regress gate |
| v3    | cross-channel handoff with context carry-over |
| v4    | live operator handoff with conversation summary |

## License

MIT. See [LICENSE](LICENSE).
