#!/usr/bin/env python3
"""Reproducible benchmark of the NTQR alarm's ~O(Q^3) cost.

Makes the manuscript's alarm-timing claim reproducible: times
``alarm_misaligned`` on a real trio of judges at several corpus sizes Q and
writes ``output/data/alarm_timings.csv``. WARNING: Q=100 takes ~3 minutes
(the enumeration is cubic) — that is the finding, not a bug.

Usage:
    uv run python scripts/bench_alarm.py --qs 20 50 100
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

from ntqr_allotment.experts import generate_population, sample_items
from ntqr_allotment.ntqr_eval import alarm_misaligned
from ntqr_allotment.pipeline import votes_for

_ROOT = Path(__file__).resolve().parent.parent


def benchmark(qs: list[int], *, seed: int = 0) -> list[tuple[int, float, bool]]:
    """Return (Q, seconds, misaligned) for each corpus size in ``qs``."""
    pop = generate_population(12, seed=seed)
    rows: list[tuple[int, float, bool]] = []
    for q in qs:
        items = sample_items(q, prevalence_a=0.5, seed=seed + 1)
        votes = votes_for(pop[:3], items, seed=seed + 29)
        start = time.perf_counter()
        flagged = alarm_misaligned(votes, max_q=q)
        rows.append((q, time.perf_counter() - start, flagged))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--qs", type=int, nargs="+", default=[20, 50, 100])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rows = benchmark(args.qs, seed=args.seed)
    out = _ROOT / "output" / "data" / "alarm_timings.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["Q", "seconds", "misaligned"])
        for q, secs, flagged in rows:
            w.writerow([q, f"{secs:.3f}", flagged])
    print(str(out))
    for q, secs, flagged in rows:
        print(f"Q={q:>4}  {secs:8.3f}s  misaligned={flagged}")


if __name__ == "__main__":
    main()
