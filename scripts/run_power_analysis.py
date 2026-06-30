#!/usr/bin/env python3
"""Thin orchestrator: run the power study on the real sweep output and write the table.

Reads ``output/data/sweep_results.json`` (per-seed ``eie_error`` rows), computes the
representative-vs-ideological contrast and every pairwise strategy matrix per panel
size, and writes ``output/data/power_analysis.csv``. All numbers are recomputed from
the sweep JSON -- nothing is hardcoded. Prints the written path for manifest collection.
"""

from __future__ import annotations

from pathlib import Path

from ntqr_allotment.power_study import analyze, write_power_table

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SWEEP_JSON = _REPO_ROOT / "output" / "data" / "sweep_results.json"
_OUT_CSV = _REPO_ROOT / "output" / "data" / "power_analysis.csv"


def main() -> None:
    results = analyze(_SWEEP_JSON, seed=0, n_perm=5000)
    written = write_power_table(results, _OUT_CSV)
    print(written)
    # Compact human summary (stderr-free; stdout first line is the artifact path).
    for r in results:
        seeds = "n/a" if r.seeds_for_80 is None else str(r.seeds_for_80)
        print(
            f"  {r.contrast}: d={r.observed_d:+.3f} perm_p={r.perm_p:.3f} "
            f"MDE80={r.mde_80:.3f} "
            f"seeds_for_80={seeds} [{r.verdict}]"
        )


if __name__ == "__main__":
    main()
