# Confidence thresholds and fallback

Every LLM call returns a structured payload:

```json
{
  "schema_version": 1,
  "action": "reply | clarify | escalate | close",
  "response_text": "...",
  "confidence": 0.0,
  "suggested_state": "greeting | clarifying | answering | escalated | closed"
}
```

The fallback dispatcher classifies the turn into one of three branches.

## Decision table

| `confidence < very_low_floor` | `confidence < floor` and streak+1 >= window | `confidence < floor` | decision |
|---|---|---|---|
| yes | -                                       | -    | `ESCALATE` (immediately, post to operator queue) |
| no  | yes                                     | yes  | `ESCALATE` (rolling-window trigger)              |
| no  | no                                      | yes  | `USE_TEMPLATE` (single-turn template reply)      |
| no  | no                                      | no   | `USE_LLM` (use the LLM's response)               |

## Defaults

| Knob | Value | Env var |
|---|---|---|
| `floor` | 0.6 | `CONVOENGINE_CONFIDENCE_FLOOR` |
| `very_low_floor` | 0.3 | `CONVOENGINE_VERY_LOW_FLOOR` |
| `window` | 3 | `CONVOENGINE_LOW_CONF_WINDOW` |

Per-channel overrides exist so chat (which is typically noisier) can be more
forgiving than email:

| Channel | Floor |
|---|---|
| chat  | 0.55 (`CONVOENGINE_CHAT_CONFIDENCE_FLOOR`) |
| email | 0.65 (`CONVOENGINE_EMAIL_CONFIDENCE_FLOOR`) |

## Template body

The single-turn template fallback uses one short, neutral string. It is the
public-facing acknowledgement that asks the user to rephrase or escalate.
The text is defined in `convoengine.orchestrator.fallback.TEMPLATE_RESPONSE_TEXT`.
