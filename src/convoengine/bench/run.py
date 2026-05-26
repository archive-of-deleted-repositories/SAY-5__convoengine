"""Per-conversation latency benchmark.

Spins up N simulated conversations (split across email and chat), drives K
turns through each one via the orchestrator using the FakeLLMProvider, and
prints per-turn and end-to-end latency statistics. SQLite-backed (a temp
file) so it is hermetic.

Usage:
    python -m convoengine.bench.run --conversations 100 --turns 10
    python -m convoengine.bench.run --conversations 10 --turns 5 --json bench.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import tempfile
import time
from collections.abc import Iterable
from dataclasses import asdict, dataclass

from sqlalchemy.orm import Session, sessionmaker

from convoengine.channels._base import ChannelMessage, Direction
from convoengine.orchestrator.conversation import ConversationState
from convoengine.orchestrator.engine import Engine
from convoengine.orchestrator.fallback import FallbackConfig
from convoengine.orchestrator.llm_client import FakeLLMProvider, ScriptedResponse
from convoengine.store import create_all, make_engine, make_session_factory


@dataclass
class BenchResult:
    """Latency and throughput numbers from a bench run."""

    conversations: int
    turns: int
    total_turns: int
    total_seconds: float
    turns_per_second: float
    p50_turn_ms: float
    p95_turn_ms: float
    p99_turn_ms: float
    p50_conv_ms: float
    p95_conv_ms: float
    p99_conv_ms: float


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = max(0, min(len(s) - 1, int(round(pct / 100.0 * (len(s) - 1)))))
    return s[idx]


def _scripted(action: str, text: str, conf: float, state: ConversationState) -> ScriptedResponse:
    return ScriptedResponse(
        raw_json=(
            f'{{"schema_version":1,"action":"{action}","response_text":"{text}",'
            f'"confidence":{conf},"suggested_state":"{state.value}"}}'
        )
    )


def _build_provider() -> FakeLLMProvider:
    provider = FakeLLMProvider()
    provider.script_state(
        ConversationState.GREETING,
        _scripted("reply", "ok", 0.9, ConversationState.ANSWERING),
    )
    provider.script_state(
        ConversationState.CLARIFYING,
        _scripted("reply", "got it", 0.9, ConversationState.ANSWERING),
    )
    provider.script_state(
        ConversationState.ANSWERING,
        _scripted("reply", "anything else?", 0.9, ConversationState.ANSWERING),
    )
    return provider


async def _run_one_conversation(
    engine: Engine,
    session_factory: sessionmaker[Session],
    user_idx: int,
    turns: int,
) -> tuple[float, list[float]]:
    """Drive `turns` messages through one conversation. Returns (total_s, per_turn_s)."""
    per_turn: list[float] = []
    channel = "chat" if user_idx % 2 == 0 else "email"
    sender = f"u-{user_idx}" if channel == "chat" else f"u-{user_idx}@example.com"
    overall_start = time.perf_counter()
    for t in range(turns):
        session = session_factory()
        try:
            msg = ChannelMessage(
                channel=channel,
                direction=Direction.IN,
                sender=sender,
                body=f"turn {t}",
            )
            turn_start = time.perf_counter()
            await engine.handle_message(session, msg)
            session.commit()
            per_turn.append(time.perf_counter() - turn_start)
        finally:
            session.close()
    return time.perf_counter() - overall_start, per_turn


async def run_bench(conversations: int, turns: int, db_path: str | None = None) -> BenchResult:
    cleanup_path = False
    if db_path is None:
        fd, db_path = tempfile.mkstemp(suffix=".sqlite")
        os.close(fd)
        cleanup_path = True

    db_url = f"sqlite:///{db_path}"
    engine_db = make_engine(db_url)
    create_all(engine_db)
    factory = make_session_factory(engine_db)
    provider = _build_provider()
    orchestrator = Engine(
        provider=provider,
        fallback_config=FallbackConfig(),
        operator_client=None,
    )

    start = time.perf_counter()
    tasks: Iterable[asyncio.Future[tuple[float, list[float]]]] = [
        asyncio.create_task(_run_one_conversation(orchestrator, factory, i, turns))
        for i in range(conversations)
    ]
    results = await asyncio.gather(*tasks)
    total_seconds = time.perf_counter() - start

    all_turn_ms = [t * 1000.0 for _, per in results for t in per]
    conv_ms = [c * 1000.0 for c, _ in results]
    total_turns = conversations * turns

    out = BenchResult(
        conversations=conversations,
        turns=turns,
        total_turns=total_turns,
        total_seconds=round(total_seconds, 4),
        turns_per_second=round(total_turns / max(total_seconds, 1e-9), 2),
        p50_turn_ms=round(statistics.median(all_turn_ms), 3),
        p95_turn_ms=round(_percentile(all_turn_ms, 95), 3),
        p99_turn_ms=round(_percentile(all_turn_ms, 99), 3),
        p50_conv_ms=round(statistics.median(conv_ms), 3),
        p95_conv_ms=round(_percentile(conv_ms, 95), 3),
        p99_conv_ms=round(_percentile(conv_ms, 99), 3),
    )

    engine_db.dispose()
    if cleanup_path and os.path.exists(db_path):
        os.unlink(db_path)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="ConvoEngine latency bench")
    parser.add_argument("--conversations", type=int, default=100)
    parser.add_argument("--turns", type=int, default=10)
    parser.add_argument("--json", type=str, default=None, help="write JSON result to path")
    args = parser.parse_args()

    result = asyncio.run(run_bench(args.conversations, args.turns))
    payload = asdict(result)
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(payload, fh, indent=2)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
