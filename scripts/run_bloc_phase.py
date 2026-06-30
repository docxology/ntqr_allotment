#!/usr/bin/env python3
"""Run the bloc-confound phase sweep (thin orchestrator).

Drives ``ntqr_allotment.bloc_confound.run_bloc_phase`` across
strategy x bloc_correlation x bias_std x mean_expertise x panel_size x seed and
writes per-cell rows plus a strategy x bloc_correlation aggregate. This is the
"extend" half of the confirm-then-extend study: it installs the composition->error
-correlation channel the baseline sweep omits and measures where the four
strategies separate.

It runs TWO grids:
  * the MAIN grid keys the shared confound on ``ideology`` -- the axis
    representative sortition balances -- and sweeps the full coupling ladder.
  * a NEGATIVE CONTROL keys the confound on ``expertise_tier`` (an axis sortition
    does NOT balance) at the coupling endpoints, to show the representative
    robustness is CONDITIONAL on balancing the axis the confound rides on, not a
    blanket property of sortition.

It also reports a per-regime robustness check (does ideological > representative
hold in every regime at high coupling, or is the pooled headline a Simpson
artifact?) and the degenerate-trio rate by strategy (so the averaged subsample is
auditable).

Outputs (under ``output/data/``):
  * ``bloc_phase.csv``          -- one row per cell (both grids; ``axis`` column)
  * ``bloc_phase_summary.json`` -- aggregates + headline + negative_control +
    robustness + degeneracy
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from pathlib import Path

from ntqr_allotment.bloc_confound import (
    BlocPhaseGrid,
    run_bloc_phase,
    run_concentration_sweep,
)

_HERE = Path(__file__).resolve()
_PROJECT_ROOT = _HERE.parents[1]
_DATA_DIR = _PROJECT_ROOT / "output" / "data"

_BLOC_CORRELATIONS = (0.0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9)
_STRATEGIES = (
    "expertise_threshold",
    "representative_sortition",
    "random_selection",
    "ideological_selection",
)
_FIELDS = [
    "axis",
    "strategy",
    "bloc_correlation",
    "bias_std",
    "mean_expertise",
    "panel_size",
    "n_items",
    "seed",
    "n_trios",
    "eie_error",
    "mv_error",
    "mean_abs_pair_corr",
]


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else float("nan")


def _ci95(values: list[float]) -> float:
    n = len(values)
    if n <= 1:
        return 0.0
    mean = _mean(values)
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    return float(math.sqrt(var) / math.sqrt(n) * 1.96)


def _valid(rows: list) -> list:
    return [r for r in rows if r.n_trios > 0 and math.isfinite(r.eie_error)]


def _aggregate_cells(results: list) -> list[dict[str, object]]:
    by_key: dict[tuple[str, float], list[float]] = defaultdict(list)
    corr_by_key: dict[tuple[str, float], list[float]] = defaultdict(list)
    for r in _valid(results):
        by_key[(r.strategy, r.bloc_correlation)].append(r.eie_error)
    for r in results:
        if math.isfinite(r.mean_abs_pair_corr):
            corr_by_key[(r.strategy, r.bloc_correlation)].append(r.mean_abs_pair_corr)
    rhos = sorted({r.bloc_correlation for r in results})
    cells: list[dict[str, object]] = []
    for strategy in _STRATEGIES:
        for rho in rhos:
            vals = by_key.get((strategy, rho), [])
            corr = corr_by_key.get((strategy, rho), [])
            cells.append(
                {
                    "strategy": strategy,
                    "bloc_correlation": rho,
                    "n": len(vals),
                    "eie_mean": _mean(vals),
                    "eie_ci95": _ci95(vals),
                    "corr_mean": _mean(corr),
                }
            )
    return cells


def _cell_mean(cells: list[dict[str, object]], strategy: str, rho: float) -> float:
    for c in cells:
        if c["strategy"] == strategy and abs(float(c["bloc_correlation"]) - rho) < 1e-9:
            return float(c["eie_mean"])
    return float("nan")


def _cell_mean_by(cells: list[dict[str, object]], key: str, value: float) -> float:
    for c in cells:
        if abs(float(c[key]) - value) < 1e-9:
            return float(c["eie_mean"])
    return float("nan")


def _headline(cells: list[dict[str, object]], rho_lo: float, rho_hi: float) -> dict[str, object]:
    def sep(rho: float) -> float:
        return _cell_mean(cells, "ideological_selection", rho) - _cell_mean(
            cells, "representative_sortition", rho
        )

    return {
        "rho_lo": rho_lo,
        "rho_hi": rho_hi,
        "sep_lo": sep(rho_lo),
        "sep_hi": sep(rho_hi),
        "representative_lo": _cell_mean(cells, "representative_sortition", rho_lo),
        "representative_hi": _cell_mean(cells, "representative_sortition", rho_hi),
        "ideological_lo": _cell_mean(cells, "ideological_selection", rho_lo),
        "ideological_hi": _cell_mean(cells, "ideological_selection", rho_hi),
        "random_lo": _cell_mean(cells, "random_selection", rho_lo),
        "random_hi": _cell_mean(cells, "random_selection", rho_hi),
    }


def _robustness(results: list, rho_hi: float) -> dict[str, object]:
    """Per-regime paired check at high coupling: ideological vs representative.

    Matches each (bias_std, mean_expertise, panel_size, seed) cell across the two
    strategies and tests whether the ideological-minus-representative gap is
    positive per regime -- guarding the pooled headline against a Simpson reversal.
    """
    def keyed(strategy: str) -> dict[tuple, float]:
        out: dict[tuple, float] = {}
        for r in _valid(results):
            if r.strategy == strategy and abs(r.bloc_correlation - rho_hi) < 1e-9:
                out[(r.bias_std, r.mean_expertise, r.panel_size, r.seed)] = r.eie_error
        return out

    ideo = keyed("ideological_selection")
    rep = keyed("representative_sortition")
    diffs = [ideo[k] - rep[k] for k in ideo.keys() & rep.keys()]
    n = len(diffs)
    positive = sum(1 for d in diffs if d > 0)
    mean_diff = _mean(diffs)
    return {
        "rho_hi": rho_hi,
        "n_paired": n,
        "n_ideo_gt_rep": positive,
        "frac_ideo_gt_rep": (positive / n) if n else float("nan"),
        "paired_mean_diff": mean_diff,
        "paired_ci95": _ci95(diffs),
    }


def _degeneracy(results: list, rho_hi: float) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for strategy in _STRATEGIES:
        rows = [r for r in results if r.strategy == strategy and abs(r.bloc_correlation - rho_hi) < 1e-9]
        if not rows:
            continue
        degenerate = sum(1 for r in rows if r.n_trios == 0)
        out[strategy] = {
            "n_cells": len(rows),
            "degenerate_frac": degenerate / len(rows),
            "mean_n_trios": _mean([float(r.n_trios) for r in rows]),
        }
    return out


def _write_rows(writer: csv.DictWriter, results: list) -> None:
    for r in results:
        writer.writerow(
            {
                "axis": r.axis,
                "strategy": r.strategy,
                "bloc_correlation": r.bloc_correlation,
                "bias_std": r.bias_std,
                "mean_expertise": r.mean_expertise,
                "panel_size": r.panel_size,
                "n_items": r.n_items,
                "seed": r.seed,
                "n_trios": r.n_trios,
                "eie_error": r.eie_error,
                "mv_error": r.mv_error,
                "mean_abs_pair_corr": r.mean_abs_pair_corr,
            }
        )


def _aggregate_concentration(results: list) -> list[dict[str, object]]:
    """Aggregate concentration-dial trials by concentration level."""
    by_c: dict[float, list[float]] = defaultdict(list)
    corr_by_c: dict[float, list[float]] = defaultdict(list)
    for r in results:
        if r.n_trios > 0 and math.isfinite(r.eie_error):
            by_c[r.concentration].append(r.eie_error)
        if math.isfinite(r.mean_abs_pair_corr):
            corr_by_c[r.concentration].append(r.mean_abs_pair_corr)
    cells = []
    for c in sorted(by_c.keys() | corr_by_c.keys()):
        vals = by_c.get(c, [])
        cells.append(
            {
                "concentration": c,
                "n": len(vals),
                "eie_mean": _mean(vals),
                "eie_ci95": _ci95(vals),
                "corr_mean": _mean(corr_by_c.get(c, [])),
            }
        )
    return cells


def _monotone_fraction(cells: list[dict[str, object]]) -> float:
    """Fraction of adjacent concentration steps where mean EIE error increases."""
    ordered = sorted(cells, key=lambda c: float(c["concentration"]))
    means = [float(c["eie_mean"]) for c in ordered if math.isfinite(float(c["eie_mean"]))]
    if len(means) < 2:
        return float("nan")
    ups = sum(1 for a, b in zip(means, means[1:]) if b >= a)
    return ups / (len(means) - 1)


def _write_concentration_csv(path: Path, results: list) -> None:
    fields = [
        "concentration", "bloc_correlation", "bias_std", "mean_expertise",
        "panel_size", "n_items", "seed", "n_trios", "eie_error", "mv_error", "mean_abs_pair_corr",
    ]
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for r in results:
            writer.writerow({f: getattr(r, f) for f in fields})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    parser.add_argument("--seeds", type=int, default=24)
    parser.add_argument("--smoke", action="store_true", help="tiny grid for CI/dev")
    args = parser.parse_args()

    if args.smoke:
        common = dict(
            strategies=_STRATEGIES,
            bias_stds=(0.3,),
            mean_expertises=(0.72,),
            panel_sizes=(6,),
            seeds=tuple(range(3)),
            n_trios=4,
        )
        main_grid = BlocPhaseGrid(bloc_correlations=(0.0, 0.9), axis="ideology", **common)
        control_grid = BlocPhaseGrid(bloc_correlations=(0.0, 0.9), axis="expertise_tier", **common)
        concentrations = (0.0, 1.0)
        conc_kwargs = dict(bias_stds=(0.3,), mean_expertises=(0.72,), seeds=tuple(range(3)), n_trios=4)
    else:
        common = dict(
            strategies=_STRATEGIES,
            bias_stds=(0.1, 0.3, 0.5),
            mean_expertises=(0.65, 0.75),
            panel_sizes=(3, 6),
            seeds=tuple(range(args.seeds)),
            n_experts=96,
            n_items=300,
            n_trios=6,
        )
        main_grid = BlocPhaseGrid(bloc_correlations=_BLOC_CORRELATIONS, axis="ideology", **common)
        # Negative control: orthogonal axis, swept across the FULL coupling ladder
        # over a focused regime set so it is a companion curve, not two endpoints.
        control_grid = BlocPhaseGrid(
            strategies=_STRATEGIES,
            bloc_correlations=_BLOC_CORRELATIONS,
            bias_stds=(0.1, 0.3, 0.5),
            mean_expertises=(0.70,),
            panel_sizes=(6,),
            seeds=tuple(range(args.seeds)),
            n_experts=96,
            n_items=300,
            n_trios=6,
            axis="expertise_tier",
        )
        concentrations = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
        conc_kwargs = dict(
            bias_stds=(0.1, 0.3, 0.5),
            mean_expertises=(0.65, 0.75),
            seeds=tuple(range(args.seeds)),
            n_trios=6,
        )

    rho_lo, rho_hi = main_grid.bloc_correlations[0], main_grid.bloc_correlations[-1]

    n_main, n_ctrl = len(main_grid.tasks()), len(control_grid.tasks())
    print(f"[bloc-phase] MAIN grid (axis=ideology): {n_main} cells on {args.workers} workers ...", flush=True)
    main_results = run_bloc_phase(main_grid, workers=args.workers)
    print(f"[bloc-phase] CONTROL grid (axis=expertise_tier, full rho): {n_ctrl} cells ...", flush=True)
    control_results = run_bloc_phase(control_grid, workers=args.workers)
    print(f"[bloc-phase] CONCENTRATION dial at rho={rho_hi}: {len(concentrations)} levels ...", flush=True)
    conc_results = run_concentration_sweep(
        concentrations=concentrations,
        bloc_correlation=rho_hi,
        panel_size=6,
        n_experts=96,
        n_items=300,
        workers=args.workers,
        **conc_kwargs,
    )

    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = _DATA_DIR / "bloc_phase.csv"
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=_FIELDS)
        writer.writeheader()
        _write_rows(writer, main_results)
        _write_rows(writer, control_results)

    _write_concentration_csv(_DATA_DIR / "bloc_concentration.csv", conc_results)

    main_cells = _aggregate_cells(main_results)
    control_cells = _aggregate_cells(control_results)
    conc_cells = _aggregate_concentration(conc_results)
    summary: dict[str, object] = {
        "strategies": list(_STRATEGIES),
        "cells": main_cells,
        "headline": _headline(main_cells, rho_lo, rho_hi),
        "negative_control": {
            "axis": "expertise_tier",
            "cells": control_cells,
            "headline": _headline(control_cells, rho_lo, rho_hi),
        },
        "robustness": _robustness(main_results, rho_hi),
        "degeneracy": _degeneracy(main_results, rho_hi),
        "concentration": {
            "bloc_correlation": rho_hi,
            "cells": conc_cells,
            "eie_balanced": _cell_mean_by(conc_cells, "concentration", 0.0),
            "eie_concentrated": _cell_mean_by(conc_cells, "concentration", 1.0),
            "monotone_increasing_fraction": _monotone_fraction(conc_cells),
        },
    }
    json_path = _DATA_DIR / "bloc_phase_summary.json"
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True))

    h = summary["headline"]
    nc = summary["negative_control"]["headline"]
    rb = summary["robustness"]
    print(f"[bloc-phase] wrote {csv_path} ({len(main_results) + len(control_results)} rows)", flush=True)
    print(f"[bloc-phase] wrote {json_path}", flush=True)
    print(
        f"[bloc-phase] MAIN sep(ideo-rep): rho={rho_lo} -> {h['sep_lo']:+.4f} | "
        f"rho={rho_hi} -> {h['sep_hi']:+.4f}",
        flush=True,
    )
    print(
        f"[bloc-phase] CONTROL (orthogonal axis) rep rho0->{nc['representative_lo']:.4f} "
        f"rho0.9->{nc['representative_hi']:.4f}  sep_hi={nc['sep_hi']:+.4f}",
        flush=True,
    )
    print(
        f"[bloc-phase] robustness: ideo>rep in {rb['n_ideo_gt_rep']}/{rb['n_paired']} regimes "
        f"(paired mean diff {rb['paired_mean_diff']:+.4f} +/- {rb['paired_ci95']:.4f})",
        flush=True,
    )
    cc = summary["concentration"]
    print(
        f"[bloc-phase] concentration dial @ rho={rho_hi}: balanced {cc['eie_balanced']:.4f} -> "
        f"concentrated {cc['eie_concentrated']:.4f} (monotone-up fraction {cc['monotone_increasing_fraction']:.2f})",
        flush=True,
    )


if __name__ == "__main__":
    main()
