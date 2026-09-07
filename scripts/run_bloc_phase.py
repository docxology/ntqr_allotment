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
import os
from pathlib import Path

from ntqr_allotment.bloc_confound import (
    BlocPhaseGrid,
    PHASE_STRATEGIES,
    build_bloc_phase_summary,
    run_bloc_phase,
    run_concentration_sweep,
)

_HERE = Path(__file__).resolve()
_PROJECT_ROOT = _HERE.parents[1]
_DATA_DIR = _PROJECT_ROOT / "output" / "data"

_BLOC_CORRELATIONS = (0.0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9)
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
            strategies=PHASE_STRATEGIES,
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
            strategies=PHASE_STRATEGIES,
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
            strategies=PHASE_STRATEGIES,
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

    summary = build_bloc_phase_summary(main_results, control_results, conc_results, rho_lo, rho_hi)
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
