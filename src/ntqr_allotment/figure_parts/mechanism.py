"""Mechanism figure: why does enlarging the panel hurt recovery?

Two panels, from ``output/data/trio_conditioning.json``:

* **A — realized error-correlation does not grow with panel size.** One line per
  strategy of mean ``|pairwise error-correlation|`` vs panel size; all stay near
  the small baseline the EIE solver assumes (zero). This is the load-bearing
  refutation of the "larger panels pull in more error-correlated trios" story.
* **B — correlation *does* predict per-trio error, within a strategy.** Bars of the
  within-strategy Pearson(per-trio error, ``|corr|``). Correlation is a relevant
  axis (strongly so for competence-first), yet panel A shows it does not grow with
  size, so the size penalty cannot be a correlation effect -- it is an aggregation
  effect of the trio-only solver.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from ntqr_allotment.figure_parts._common import (
    _apply_axis_style,
    _humanize_strategy_name,
    _set_claim_title,
    _strategy_color,
    plt,
    save_figure,
)
from ntqr_allotment.trio_conditioning import pearson

_STRATEGY_ORDER = (
    "expertise_threshold",
    "random_selection",
    "representative_sortition",
    "ideological_selection",
)


def plot_trio_conditioning(data: dict, output_path: Path) -> Path:
    """Render the two-panel size-penalty mechanism figure from the diagnostic JSON."""
    records = data.get("records")
    if not records:
        raise ValueError("trio_conditioning data has no 'records' to plot")

    sizes = sorted({int(r["panel_size"]) for r in records})
    corr_by: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    err_by: dict[str, list[float]] = defaultdict(list)
    corr_flat_by: dict[str, list[float]] = defaultdict(list)
    for r in records:
        strat = str(r["strategy"])
        corr_by[strat][int(r["panel_size"])].append(float(r["mean_abs_corr"]))
        err_by[strat].append(float(r["eie_error"]))
        corr_flat_by[strat].append(float(r["mean_abs_corr"]))

    strategies = [s for s in _STRATEGY_ORDER if s in corr_by]

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(12.0, 4.6))

    # Panel A: mean |error-correlation| vs panel size.
    for strat in strategies:
        ys = [sum(corr_by[strat][s]) / len(corr_by[strat][s]) for s in sizes]
        ax_a.plot(
            sizes,
            ys,
            marker="o",
            color=_strategy_color(strat),
            label=_humanize_strategy_name(strat),
            linewidth=2.0,
        )
    ax_a.axhline(0.0, linestyle="--", color="#888888", linewidth=1.0)
    ax_a.annotate(
        "EIE solver assumes 0",
        xy=(sizes[0], 0.0),
        xytext=(sizes[0], 0.0),
        fontsize=8,
        color="#555555",
        va="bottom",
    )
    ax_a.set_xticks(sizes)
    ax_a.set_xlabel("panel size (members)")
    ax_a.set_ylabel("mean |pairwise error-correlation|")
    ax_a.set_ylim(bottom=-0.002)
    _apply_axis_style(ax_a)
    ax_a.legend(fontsize=7, loc="upper left")
    _set_claim_title(ax_a, "A. Error-correlation stays flat")

    # Panel B: within-strategy Pearson(error, |corr|).
    rs = [pearson(err_by[s], corr_flat_by[s]) for s in strategies]
    colors = [_strategy_color(s) for s in strategies]
    ypos = range(len(strategies))
    ax_b.barh(list(ypos), rs, color=colors)
    ax_b.set_yticks(list(ypos))
    ax_b.set_yticklabels([_humanize_strategy_name(s) for s in strategies], fontsize=8)
    ax_b.axvline(0.0, color="#333333", linewidth=0.8)
    for y, r in zip(ypos, rs):
        ax_b.annotate(f"{r:+.2f}", xy=(r, y), xytext=(3 if r >= 0 else -3, 0),
                      textcoords="offset points", va="center",
                      ha="left" if r >= 0 else "right", fontsize=8)
    ax_b.set_xlabel("within-strategy Pearson( per-trio error , |error-correlation| )")
    ax_b.set_xlim(-0.2, 1.0)
    _apply_axis_style(ax_b)
    _set_claim_title(ax_b, "B. Correlation predicts error (yet stays flat)")

    fig.tight_layout()
    return save_figure(fig, output_path)


__all__ = ["plot_trio_conditioning"]
