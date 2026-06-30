#!/usr/bin/env python3
"""Thin orchestrator: render the four EXTENDED figures for the new tracks.

Emits, into ``output/figures/``:
  - ``error_vs_correlation.png`` — the centerpiece tolerance scatter (EIE error
    vs NTQR-measured realized correlation), from ``output/data/independence_sweep.csv``.
  - ``strategy_correlation.png`` — mean realized correlation per formation rule.
  - ``alarm_power.png``          — alarm firing power vs panel size (small-Q,
    computed live via :func:`ntqr_allotment.ensemble.alarm_power_curve`).
  - ``fairness_maximin.png``     — per-candidate maximin selection probabilities.

All inputs are real (the committed sweep CSV plus small deterministic live runs);
no figure number is hand-authored. Paths are printed to stdout for manifest
collection. Deterministic (seeded; ``MPLBACKEND=Agg`` is set inside figures.py).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from ntqr_allotment.ensemble import alarm_power_curve
from ntqr_allotment.experts import generate_population, sample_items
from ntqr_allotment.fairness import maximin_fairness
from ntqr_allotment.figures import (
    plot_alarm_power,
    plot_cross_family_contrast,
    plot_cross_family_multiseed,
    plot_error_vs_correlation,
    plot_fairness_maximin,
    plot_method_pipeline_schematic,
    plot_postdoc_age_bias_heatmap,
    plot_postdoc_empirical_alignment,
    plot_postdoc_strategy_ranking,
    plot_power_design_diagnosis,
    plot_power_vs_n,
    plot_strategy_correlation,
    plot_track_ranking_inversion,
    plot_trio_conditioning,
)
from ntqr_allotment.power_analysis import analytic_power
from ntqr_allotment.power_study import analyze as analyze_power

DEGENERATE = -1.0


def _load_independence_aggregates(csv_path: Path) -> list[dict[str, object]]:
    """Read the independence sweep CSV into non-degenerate aggregate row dicts."""
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Missing {csv_path}. Regenerate with "
            "`uv run python scripts/run_independence_sweep.py`."
        )
    rows: list[dict[str, object]] = []
    with csv_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            eie_mean = float(row["eie_mean"])
            if eie_mean == DEGENERATE or int(row["n"]) <= 0:
                continue
            rows.append(
                {
                    "rho": float(row["rho"]),
                    "strategy": row["strategy"],
                    "panel_size": int(row["panel_size"]),
                    "n": int(row["n"]),
                    "eie_mean": eie_mean,
                    "eie_ci95": float(row["eie_ci95"]),
                    "corr_mean": float(row["corr_mean"]),
                }
            )
    if not rows:
        raise ValueError(f"No non-degenerate rows in {csv_path}.")
    return rows


def main() -> None:
    from ntqr_allotment.determinism import ensure_deterministic_hashing

    ensure_deterministic_hashing()  # maximin_fairness forms representative panels
    root = Path(__file__).resolve().parents[1]
    figure_dir = root / "output" / "figures"
    aggregates = _load_independence_aggregates(
        root / "output" / "data" / "independence_sweep.csv"
    )

    # Small deterministic live runs for the consistency-power and fairness panels.
    population = generate_population(18, seed=0)
    items = sample_items(18, prevalence_a=0.5, seed=7)
    # safety (1,1): the tight spec under which the alarm actually fires on this
    # synthetic population (at the default (2,2) it stays flat at 0 here).
    curve = alarm_power_curve(
        population, items, sizes=(3, 5, 7), seeds=(0, 1, 2), safety=(1, 1), max_q=20
    )
    fairness = maximin_fairness(population, panel_size=3, seed=0, panel_count=20)

    paths = [
        plot_error_vs_correlation(aggregates, figure_dir / "error_vs_correlation.png"),
        plot_strategy_correlation(aggregates, figure_dir / "strategy_correlation.png"),
        plot_alarm_power(curve, figure_dir / "alarm_power.png"),
        plot_fairness_maximin(fairness, figure_dir / "fairness_maximin.png"),
    ]

    # Power design-diagnosis panel: observed effect vs MDE per contrast, recomputed
    # from the real sweep JSON (deterministic; seeded permutations).
    sweep_json = root / "output" / "data" / "sweep_results.json"
    if sweep_json.exists():
        power_rows = analyze_power(sweep_json, seed=0, n_perm=5000)
        paths.append(
            plot_power_design_diagnosis(
                power_rows, figure_dir / "power_design_diagnosis.png"
            )
        )

    # Size-penalty mechanism panel: per-trio error-correlation vs panel size,
    # from the trio-conditioning diagnostic. Shows the realized correlation does
    # NOT grow with size (so "size hurts" is not an error-independence breakdown).
    trio_json = root / "output" / "data" / "trio_conditioning.json"
    if trio_json.exists():
        paths.append(
            plot_trio_conditioning(
                json.loads(trio_json.read_text()),
                figure_dir / "trio_conditioning.png",
            )
        )

    # Cross-family decorrelation panel: rendered from the (live) cross-family
    # artifact when present. Reading the same JSON twice is byte-deterministic.
    cross_json = root / "output" / "data" / "cross_family_results.json"
    if cross_json.exists():
        data = json.loads(cross_json.read_text())
        contrast = {
            "mean_abs_same_family": data["mean_abs_same_family"],
            "mean_abs_cross_family": data["mean_abs_cross_family"],
            "delta_cross_minus_same": data["delta_cross_minus_same"],
            "label": data.get("provenance_label", "live empirical, n-limited"),
            "n_items": data.get("n_items"),
            "n_same_pairs": data.get("n_same_pairs"),
            "n_cross_pairs": data.get("n_cross_pairs"),
            "nonzero_pairs": data.get(
                "nonzero_pairs",
                sum(
                    1
                    for value in data.get("pair_abs_corr", {}).values()
                    if abs(float(value)) > 0.0
                ),
            ),
            "total_pairs": data.get("total_pairs", len(data.get("pair_abs_corr", {}))),
        }
        paths.append(
            plot_cross_family_contrast(contrast, figure_dir / "cross_family_contrast.png")
        )

    # Multi-seed cross-family deltas (sign-stability across independent runs).
    ms_json = root / "output" / "data" / "cross_family_multiseed.json"
    if ms_json.exists():
        deltas = json.loads(ms_json.read_text())["deltas"]
        paths.append(
            plot_cross_family_multiseed(deltas, figure_dir / "cross_family_multiseed.png")
        )

    postdoc_json = root / "output" / "data" / "postdoc_panel_results.json"
    postdoc_alignment_json = root / "output" / "data" / "postdoc_panel_alignment.json"
    if postdoc_json.exists() and postdoc_alignment_json.exists():
        postdoc_payload = json.loads(postdoc_json.read_text())
        postdoc_alignment = json.loads(postdoc_alignment_json.read_text())
        paths.extend(
            [
                plot_postdoc_strategy_ranking(
                    postdoc_payload,
                    figure_dir / "postdoc_strategy_ranking.png",
                ),
                plot_postdoc_age_bias_heatmap(
                    postdoc_payload,
                    figure_dir / "postdoc_age_bias_heatmap.png",
                ),
                plot_postdoc_empirical_alignment(
                    postdoc_alignment,
                    figure_dir / "postdoc_empirical_alignment.png",
                ),
            ]
        )

    # Method schematic + cross-track ranking inversion. Both read the manuscript
    # variable tokens so every panel number matches the rendered prose exactly
    # (the deterministic pipeline regenerates the tokens each run, so steady-state
    # reads are drift-free). Skipped if the tokens have not been emitted yet.
    variables_json = root / "output" / "manuscript_variables.json"
    if variables_json.exists():
        variables = json.loads(variables_json.read_text())
        paths.append(
            plot_method_pipeline_schematic(
                {
                    "n_experts": int(variables["N_EXPERTS"]),
                    "n_items": int(variables["N_ITEMS"]),
                    "n_trios": int(variables["N_TRIOS"]),
                },
                figure_dir / "method_pipeline_schematic.png",
            )
        )
        # Matched three-seat grain on both tracks: synthetic POWER_*_SIZE3 vs live
        # POSTDOC_*_EIE. Ranks are compared, magnitudes are NOT pooled.
        synthetic_size3 = [
            ["expertise_threshold", float(variables["POWER_EXPERTISE_THRESHOLD_SIZE3"])],
            ["random_selection", float(variables["POWER_RANDOM_SELECTION_SIZE3"])],
            ["representative_sortition", float(variables["POWER_REPRESENTATIVE_SORTITION_SIZE3"])],
            ["ideological_selection", float(variables["POWER_IDEOLOGICAL_SELECTION_SIZE3"])],
        ]
        live_three_seat = [
            ["representative_sortition", float(variables["POSTDOC_REPRESENTATIVE_EIE"])],
            ["ideological_selection", float(variables["POSTDOC_SAME_BIAS_EIE"])],
            ["random_selection", float(variables["POSTDOC_RANDOM_EIE"])],
            ["expertise_threshold", float(variables["POSTDOC_EXPERTISE_EIE"])],
        ]
        paths.append(
            plot_track_ranking_inversion(
                synthetic_size3,
                live_three_seat,
                figure_dir / "track_ranking_inversion.png",
            )
        )

    # Statistical-power curves: power vs n for a few effect sizes (analytic, exact).
    ns = [5, 10, 20, 40, 80, 160, 320]
    power_curves = [
        (f"d={d}", ns, [analytic_power(d, n) for n in ns])
        for d in (0.2, 0.5, 0.8)
    ]
    paths.append(plot_power_vs_n(power_curves, figure_dir / "power_vs_n.png"))

    for path in paths:
        print(path.resolve())


if __name__ == "__main__":
    main()
