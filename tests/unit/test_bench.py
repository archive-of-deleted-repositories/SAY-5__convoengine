"""Smoke tests for the bench module: it runs and produces sensible numbers."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile

import pytest

from convoengine.bench.run import run_bench


@pytest.mark.asyncio
async def test_run_bench_returns_filled_result() -> None:
    result = await run_bench(conversations=4, turns=3)
    assert result.total_turns == 12
    assert result.turns_per_second > 0
    # Sanity bounds: anything is too slow if a turn takes more than a second.
    assert result.p99_turn_ms < 1000
    assert result.p50_turn_ms <= result.p95_turn_ms <= result.p99_turn_ms


@pytest.mark.asyncio
async def test_run_bench_emits_per_conversation_metrics() -> None:
    result = await run_bench(conversations=3, turns=3)
    assert result.p50_conv_ms > 0
    assert result.p95_conv_ms >= result.p50_conv_ms


def test_bench_cli_runs_and_writes_json() -> None:
    fd, path = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    try:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "convoengine.bench.run",
                "--conversations",
                "2",
                "--turns",
                "2",
                "--json",
                path,
            ],
            check=True,
        )
        with open(path) as fh:
            data = json.load(fh)
        assert data["total_turns"] == 4
        assert "p95_turn_ms" in data
    finally:
        if os.path.exists(path):
            os.unlink(path)
