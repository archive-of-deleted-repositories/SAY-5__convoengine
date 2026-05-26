"""Compare a fresh bench run against a stored baseline and gate on drift.

Usage:
    python -m convoengine.bench.regress --gate 0.30 [--baseline bench/baseline.json]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from dataclasses import asdict

from convoengine.bench.run import run_bench

DEFAULT_BASELINE = os.path.join(os.path.dirname(__file__), "baseline.json")


def _check_metric(name: str, baseline: float, current: float, gate: float) -> tuple[bool, str]:
    """For latency metrics, higher is worse. For throughput, lower is worse.

    Returns (ok, message). gate is a fraction (0.30 = 30% drift tolerance).
    """
    if baseline <= 0:
        return True, f"{name}: no baseline"
    if name == "turns_per_second":
        # Drop in throughput.
        delta = (current - baseline) / baseline
        ok = delta >= -gate
        return ok, f"{name}: baseline={baseline:.2f} current={current:.2f} delta={delta:+.2%}"
    # Latency: higher is worse.
    delta = (current - baseline) / baseline
    ok = delta <= gate
    return ok, f"{name}: baseline={baseline:.2f} current={current:.2f} delta={delta:+.2%}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate", type=float, default=0.30)
    parser.add_argument("--baseline", type=str, default=DEFAULT_BASELINE)
    parser.add_argument("--conversations", type=int, default=10)  # CI smoke default
    parser.add_argument("--turns", type=int, default=10)
    parser.add_argument("--write-baseline", action="store_true")
    args = parser.parse_args()

    result = asyncio.run(run_bench(args.conversations, args.turns))
    current = asdict(result)
    print(json.dumps(current, indent=2))

    if args.write_baseline:
        with open(args.baseline, "w") as fh:
            json.dump(current, fh, indent=2)
        print(f"baseline written: {args.baseline}")
        return

    if not os.path.exists(args.baseline):
        print(f"no baseline at {args.baseline}; run with --write-baseline first")
        sys.exit(0)
    with open(args.baseline) as fh:
        baseline = json.load(fh)

    metrics = [
        "turns_per_second",
        "p50_turn_ms",
        "p95_turn_ms",
        "p99_turn_ms",
        "p50_conv_ms",
        "p95_conv_ms",
        "p99_conv_ms",
    ]

    all_ok = True
    for metric in metrics:
        ok, msg = _check_metric(metric, baseline.get(metric, 0.0), current[metric], args.gate)
        marker = "OK" if ok else "REGRESS"
        print(f"  [{marker}] {msg}")
        all_ok = all_ok and ok

    if not all_ok:
        print(f"\nregression detected at gate {args.gate:.0%}")
        sys.exit(1)
    print(f"\nno regression at gate {args.gate:.0%}")


if __name__ == "__main__":
    main()
