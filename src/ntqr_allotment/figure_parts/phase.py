"""Bloc-confound phase diagram: where the strategies separate.

Two panels from ``bloc_phase_summary.json`` cells:

* left  -- mean EIE recovery error vs within-bloc error coupling, one line per
  strategy. At zero coupling the three composition strategies collapse together
  (the reproduced baseline result); as coupling rises they fan out.
* right -- the NTQR-measured realized error correlation vs coupling, the
  mechanism: representative sortition stays decorrelated by balancing blocs while
  single-bloc ideological selection concentrates the shared confound.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from ntqr_allotment.figure_parts._common import (
    LEGEND_FONT_SIZE,
    SOURCE_NOTE_FONT_SIZE,
    _apply_axis_style,
    _humanize_strategy_name,
    _set_claim_title,
    _strategy_color,
)

# Plot strategies in a fixed, legible order (best-baseline first).
_ORDER = (
    "expertise_threshold",
    "representative_sortition",
    "random_selection",
    "ideological_selection",
)


def _cell(row: object, field: str) -> object:
    if isinstance(row, Mapping):
        return row.get(field)
    return getattr(row, field, None)


def _series(cells: Sequence[object], strategy: str, value_field: str) -> tuple[list[float], list[float], list[float]]:
    rows = [c for c in cells if _cell(c, "strategy") == strategy]
    rows = [r for r in rows if _cell(r, "n") and int(_cell(r, "n")) > 0]
    rows.sort(key=lambda r: float(_cell(r, "bloc_correlation")))
    xs = [float(_cell(r, "bloc_correlation")) for r in rows]
    ys = [float(_cell(r, value_field)) for r in rows]
    ci = [float(_cell(r, "eie_ci95") or 0.0) for r in rows]
    return xs, ys, ci


def plot_bloc_phase_diagram(
    cells: Sequence[object],
    output_path: Path,
    *,
    source_note: str | None = None,
) -> Path:
    """Render the two-panel phase diagram. ``cells`` are summary cell dicts."""
    if not cells:
        raise ValueError("bloc phase diagram: expected at least one summary cell")

    strategies = [s for s in _ORDER if any(_cell(c, "strategy") == s for c in cells)]
    if not strategies:
        raise ValueError("bloc phase diagram: no known strategies in cells")

    fig, (ax_err, ax_corr) = plt.subplots(1, 2, figsize=(15.0, 6.2))

    for strategy in strategies:
        color = _strategy_color(strategy)
        label = _humanize_strategy_name(strategy)
        xs, ys, ci = _series(cells, strategy, "eie_mean")
        if xs:
            ax_err.plot(xs, ys, marker="o", color=color, label=label, linewidth=2.1, markersize=6)
            lo = [y - c for y, c in zip(ys, ci)]
            hi = [y + c for y, c in zip(ys, ci)]
            ax_err.fill_between(xs, lo, hi, color=color, alpha=0.13, linewidth=0)
        cxs, cys, _ = _series(cells, strategy, "corr_mean")
        if cxs:
            ax_corr.plot(cxs, cys, marker="s", color=color, label=label, linewidth=2.1, markersize=6)

    ax_err.set_xlabel("Within-bloc error coupling  ρ  (shared-confound strength)")
    ax_err.set_ylabel("Mean oracle-referenced EIE recovery error")
    _set_claim_title(ax_err, "Strategies fan out as errors become bloc-correlated")
    _apply_axis_style(ax_err, grid_axis="both")
    ax_err.legend(fontsize=LEGEND_FONT_SIZE, frameon=False, loc="upper left")

    ax_corr.set_xlabel("Within-bloc error coupling  ρ  (shared-confound strength)")
    ax_corr.set_ylabel("NTQR-measured realized trio error correlation")
    _set_claim_title(ax_corr, "Representativeness suppresses the shared confound")
    _apply_axis_style(ax_corr, grid_axis="both")
    ax_corr.legend(fontsize=LEGEND_FONT_SIZE, frameon=False, loc="upper left")

    note = source_note or (
        "Source: output/data/bloc_phase_summary.json — mean over bias-spread x stringency x "
        "panel-size x seed regimes. Left: recovery error; right: the mechanism (measured correlation)."
    )
    # Reserve bottom space ourselves and save directly: the shared _save_figure
    # re-runs tight_layout(), which would collide the figure-level note with the
    # two x-axis labels.
    fig.tight_layout(rect=(0.0, 0.06, 1.0, 1.0))
    fig.text(0.5, 0.015, note, ha="center", va="bottom", fontsize=SOURCE_NOTE_FONT_SIZE, color="#4f5d66")
    resolved = Path(output_path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.savefig(resolved, dpi=200, bbox_inches="tight")
    finally:
        plt.close(fig)
    return resolved


def plot_concentration_dial(
    cells: Sequence[object],
    output_path: Path,
    *,
    source_note: str | None = None,
) -> Path:
    """Recovery error and measured correlation vs the representativeness dial.

    ``cells`` are the concentration summary cells (concentration, eie_mean,
    eie_ci95, corr_mean). One panel, twin y-axes: EIE recovery error (left) and
    NTQR-measured trio correlation (right) against the single-bloc concentration
    dial at fixed coupling. The Herfindahl index runs 1/B (balanced) to 1
    (single-bloc) monotonically with the dial.
    """
    if not cells:
        raise ValueError("concentration dial: expected at least one summary cell")
    rows = [c for c in cells if _cell(c, "n") and int(_cell(c, "n")) > 0]
    rows.sort(key=lambda c: float(_cell(c, "concentration")))
    xs = [float(_cell(c, "concentration")) for c in rows]
    err = [float(_cell(c, "eie_mean")) for c in rows]
    ci = [float(_cell(c, "eie_ci95") or 0.0) for c in rows]
    corr = [float(_cell(c, "corr_mean")) for c in rows]

    fig, ax = plt.subplots(figsize=(9.4, 6.0))
    err_color = _strategy_color("ideological_selection")
    corr_color = _strategy_color("random_selection")
    ax.plot(xs, err, marker="o", color=err_color, linewidth=2.3, markersize=7, label="EIE recovery error")
    ax.fill_between(
        xs, [e - c for e, c in zip(err, ci)], [e + c for e, c in zip(err, ci)],
        color=err_color, alpha=0.13, linewidth=0,
    )
    ax.set_xlabel("Single-bloc concentration  c   (Herfindahl index  H: 1/B at c=0  ->  1 at c=1)")
    ax.set_ylabel("Mean oracle-referenced EIE recovery error", color=err_color)
    ax.tick_params(axis="y", labelcolor=err_color)
    _apply_axis_style(ax, grid_axis="x")
    _set_claim_title(ax, "Recovery error rises with single-bloc concentration")

    ax2 = ax.twinx()
    ax2.plot(xs, corr, marker="s", color=corr_color, linewidth=2.0, markersize=6, label="measured trio correlation")
    ax2.set_ylabel("NTQR-measured realized trio error correlation", color=corr_color)
    ax2.tick_params(axis="y", labelcolor=corr_color)

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=LEGEND_FONT_SIZE, frameon=False, loc="upper left")

    note = source_note or (
        "Source: output/data/bloc_phase_summary.json (concentration block) — mean over regimes/seeds at fixed coupling."
    )
    fig.tight_layout(rect=(0.0, 0.05, 1.0, 1.0))
    fig.text(0.5, 0.015, note, ha="center", va="bottom", fontsize=SOURCE_NOTE_FONT_SIZE, color="#4f5d66")
    resolved = Path(output_path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.savefig(resolved, dpi=200, bbox_inches="tight")
    finally:
        plt.close(fig)
    return resolved


__all__ = ["plot_bloc_phase_diagram", "plot_concentration_dial"]
