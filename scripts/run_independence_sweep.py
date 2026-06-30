"""Thin orchestrator: run the error-correlation tolerance sweep, write a CSV.

Builds a small default :class:`IndependenceGrid`, runs the sweep, aggregates by
(rho, strategy, panel_size), and writes a tidy CSV to
``output/data/independence_sweep.csv``. Prints the output path to stdout
(manifest convention). All computation lives in
:mod:`ntqr_allotment.independence_sweep`; this script only handles I/O.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from ntqr_allotment.independence_sweep import (
    IndependenceGrid,
    aggregate_independence,
    run_independence_sweep,
)

# panel_size is intentionally fixed at the trio (3): the exact NTQR solver
# evaluates a trio, and the trio is this study's correlation-tolerance unit.
# Sweeping panel_size here would inject size-invariant duplicate trio cells into
# the error~correlation regression (the ideological draw's first three experts
# are identical at size 3 and 6), double-counting points and faking the slope's
# effective N. We therefore vary rho and strategy at fixed trio size, and add
# more seeds for a tighter aggregate.
_DEFAULT_GRID = IndependenceGrid(
    rhos=(0.0, 0.3, 0.6, 0.9),
    strategies=("representative_sortition", "ideological_selection"),
    panel_sizes=(3,),
    seeds=(0, 1, 2, 3, 4, 5),
    n_experts=24,
    n_items=120,
)

_HEADER = [
    "rho",
    "strategy",
    "panel_size",
    "n_experts",
    "n_items",
    "n",
    "eie_mean",
    "eie_std",
    "eie_ci95",
    "corr_mean",
]


def write_csv(output_path: Path, grid: IndependenceGrid) -> Path:
    """Run the sweep for ``grid`` and write aggregates to ``output_path``."""
    rows = run_independence_sweep(grid)
    aggregates = aggregate_independence(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(_HEADER)
        for item in aggregates:
            writer.writerow(
                [
                    item.rho,
                    item.strategy,
                    item.panel_size,
                    item.n_experts,
                    item.n_items,
                    item.n,
                    item.eie_mean,
                    item.eie_std,
                    item.eie_ci95,
                    item.corr_mean,
                ]
            )
    return output_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the error-correlation tolerance sweep and write a CSV."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="CSV output path. Defaults to output/data/independence_sweep.csv.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    output_path = args.output or root / "output" / "data" / "independence_sweep.csv"
    written = write_csv(output_path, _DEFAULT_GRID)
    print(written.resolve())


if __name__ == "__main__":
    main()
