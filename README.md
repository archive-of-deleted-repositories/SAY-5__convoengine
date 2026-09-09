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
- FakeLLMProvider for hermetic tests, explicitly configured Claude Messages provider
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
    llm_client.py         FakeLLMProvider + bounded ClaudeProvider
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

## Optional Claude service mode

The default `CONVOENGINE_LLM_PROVIDER=fake` remains deterministic and needs no API credentials. To use Claude, configure all of:

| Variable | Required value |
|---|---|
| `CONVOENGINE_LLM_PROVIDER` | `claude` |
| `CONVOENGINE_CLAUDE_MODEL` | An explicit Claude model that supports structured JSON outputs |
| `CONVOENGINE_CLAUDE_API_KEY` | The provider API key |
| `CONVOENGINE_SERVICE_API_KEY` | A random service key, 32–512 non-space ASCII characters |
| `CONVOENGINE_DB_URL` | The configured database connection URL |

Then run `uvicorn convoengine.api:app`. Provider configuration is validated before opening the database. The module app opens its database during server startup; configuration/startup errors propagate instead of creating an empty fallback app. Programmatic `build_app` callers must select Claude mode even when injecting a Claude provider.

Every `/v1` request in Claude mode requires `Authorization: Bearer <service-key>`, including conversation reads, identity linkage, and operator commands. This applies to mounted applications and reverse-proxy `root_path` prefixes. The key authenticates a trusted service authorized to manage all conversations; end-user ownership and identity verification belong in that service. `/healthz` remains public. Keep both keys on the server and out of browser code.

All authenticated `/v1` requests share four concurrent slots per process by default, configurable with `CONVOENGINE_PAID_MAX_CONCURRENCY` from 1 to 8. Extra requests receive HTTP 429 before body/database processing, without queued work. The slot is held until completion, failure, or cancellation. Every authenticated `/v1` body is capped at 512 KiB and must finish arriving within 5 seconds. These are per-process limits; multiple workers multiply the available slots. Existing synchronous database operations and operator dispatch have their own timing characteristics.

Each Claude completion sends at most 512 KiB, requests at most 512 output tokens, reads at most 64 KiB of an uncompressed response, and has an 8-second total network deadline that also terminates trickling responses. Requests use the fixed Anthropic Messages endpoint without inherited proxies or redirects. Only a completed text response matching the strict local schema is accepted. Model-suggested states cannot bypass the controller's allowed transitions or claim operator control.

Claude transport, status, truncation, refusal, size, or schema errors return a generic HTTP 503 and roll back the turn, including a newly created conversation. There is one paid request per turn and no automatic paid retry or switch to fake output. Valid low-confidence responses still follow the existing template/operator policy. Fake-provider guardrail retries remain available for deterministic tests.

Tests use synthetic HTTPX transports and in-memory SQLite. They cover the actual Claude payload, strict response validation, request/response bounds, total deadlines, mounted authentication, nonqueued admission, cancellation/rollback, and state-transition authority. No live provider is needed. See [Claude structured outputs](https://platform.claude.com/docs/en/build-with-claude/structured-outputs) for model support and the wire format.

## License

MIT. See [LICENSE](LICENSE).
