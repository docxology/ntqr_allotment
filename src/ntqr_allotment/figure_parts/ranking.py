from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from ntqr_allotment.figure_parts._common import (
    ANNOTATION_FONT_SIZE,
    AXIS_LABEL_FONT_SIZE,
    SOURCE_NOTE_FONT_SIZE,
    TICK_LABEL_FONT_SIZE,
    _add_source_note,
    _apply_axis_style,
    _coerce_non_empty_string,
    _coerce_non_negative_float,
    _format_regime_label,
    _humanize_strategy_name,
    _set_claim_title,
    _strategy_color,
    _validated_effect_rows,
    _validated_ranking_rows,
    plt,
    save_figure,
)

# Display labels that read sensibly across *both* tracks. The synthetic pipeline
# key ``ideological_selection`` is surfaced as "same-bias selection" in the live
# postdoctoral panel, so the cross-track figure names it for the mechanism.
_TRACK_LABELS = {
    "expertise_threshold": "expertise threshold",
    "random_selection": "random selection",
    "representative_sortition": "representative sortition",
    "ideological_selection": "single-bloc (same-bias)",
}


def plot_strategy_ranking(
    ranking: Sequence[Sequence[object]],
    output_path: Path,
) -> Path:
    rows = _validated_ranking_rows(ranking)
    ordered_rows = sorted(rows, key=lambda row: (row[1], row[0]))

    labels = [
        f"{rank}. {_humanize_strategy_name(strategy)}"
        for rank, (strategy, _, _) in enumerate(ordered_rows, start=1)
    ]
    means = [mean for _, mean, _ in ordered_rows]
    ci95_values = [ci95 for _, _, ci95 in ordered_rows]
    colors = [_strategy_color(strategy) for strategy, _, _ in ordered_rows]

    fig, ax = plt.subplots(figsize=(9.5, max(4.8, 1.5 + len(ordered_rows) * 0.9)))
    y_positions = np.arange(len(ordered_rows))
    ax.barh(y_positions, means, xerr=ci95_values, capsize=5, color=colors, alpha=0.9)
    label_x = max(mean + ci95 for mean, ci95 in zip(means, ci95_values)) * 1.08
    for y, mean, ci95 in zip(y_positions, means, ci95_values):
        ax.text(
            label_x,
            y,
            f"{mean:.3f} +/- {ci95:.3f}",
            va="center",
            fontsize=ANNOTATION_FONT_SIZE,
        )
    ax.set_yticks(y_positions, labels=labels)
    ax.invert_yaxis()
    ax.set_xlabel("Weighted mean EIE error")
    _set_claim_title(ax, "Formation strategy sets the no-answer-key error ceiling")
    ax.set_xlim(left=0.0, right=label_x * 1.22)
    _apply_axis_style(ax, grid_axis="x")
    return save_figure(fig, output_path)


def plot_rep_vs_ideo_effect(
    effects: Sequence[Mapping[str, object]],
    output_path: Path,
) -> Path:
    rows = _validated_effect_rows(effects)
    ordered_rows = sorted(rows, key=lambda row: row[0])

    x_positions = np.arange(len(ordered_rows))
    labels = [
        _format_regime_label(regime_key, index=index)
        for index, (regime_key, _, _) in enumerate(ordered_rows)
    ]
    effect_values = [effect for _, effect, _ in ordered_rows]
    ci95_values = [ci95 for _, _, ci95 in ordered_rows]

    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    ax.errorbar(
        x_positions,
        effect_values,
        yerr=ci95_values,
        fmt="o",
        capsize=5,
        color="#E45756",
        ecolor="#4D4D4D",
        markersize=7,
        linewidth=2,
    )
    ax.axhline(0.0, color="#222222", linestyle="--", linewidth=1)
    ax.set_xticks(
        x_positions,
        labels=labels,
        rotation=25,
        ha="right",
        fontsize=TICK_LABEL_FONT_SIZE,
    )
    ax.set_ylabel("Ideological - representative EIE error")
    _set_claim_title(ax, "Representative-vs-ideological contrast is regime-specific")
    _apply_axis_style(ax, grid_axis="y")
    return save_figure(fig, output_path)


def _validated_track_rows(
    rows: Sequence[Sequence[object]],
    track_label: str,
) -> list[tuple[str, float]]:
    """Validate one track's ``[strategy, eie_error]`` rows for the inversion plot."""

    if not rows:
        raise ValueError(f"track ranking inversion: {track_label} track has no rows")
    validated: list[tuple[str, float]] = []
    seen: set[str] = set()
    for index, row in enumerate(rows):
        context = f"track ranking inversion {track_label} row {index}"
        if len(row) != 2:
            raise ValueError(f"{context}: expected [strategy, eie_error]")
        strategy = _coerce_non_empty_string(row[0], "strategy", context)
        if strategy in seen:
            raise ValueError(f"{context}: duplicate strategy {strategy!r}")
        seen.add(strategy)
        value = _coerce_non_negative_float(row[1], "eie_error", context)
        validated.append((strategy, value))
    return validated


def plot_track_ranking_inversion(
    synthetic: Sequence[Sequence[object]],
    live: Sequence[Sequence[object]],
    output_path: Path,
) -> Path:
    """Render the cross-track strategy-ranking inversion as a slope chart.

    ``synthetic`` and ``live`` are each ``[strategy, eie_error]`` rows at a
    *matched* panel grain (three-seat). The figure compares the **rank order**
    each track assigns to the four formation strategies; magnitudes are NOT
    pooled across the two tracks. The slope lines make the inversion immediate:
    the rule that is best blind on synthetic data is worst under the live model,
    and vice versa.
    """

    syn_rows = _validated_track_rows(synthetic, "synthetic")
    live_rows = _validated_track_rows(live, "live")
    if {s for s, _ in syn_rows} != {s for s, _ in live_rows}:
        raise ValueError(
            "track ranking inversion: synthetic and live tracks must rank the same strategies"
        )

    syn_order = sorted(syn_rows, key=lambda r: (r[1], r[0]))
    live_order = sorted(live_rows, key=lambda r: (r[1], r[0]))
    n = len(syn_order)
    rank_y = {strategy: float(n - rank) for rank, (strategy, _) in enumerate(syn_order)}
    left_pos = {strategy: rank_y[strategy] for strategy in rank_y}
    right_pos = {
        strategy: float(n - rank) for rank, (strategy, _) in enumerate(live_order)
    }
    left_val = dict(syn_order)
    right_val = dict(live_order)

    left_x, right_x = 0.0, 1.0
    fig, ax = plt.subplots(figsize=(11.0, 6.2))

    for strategy in left_pos:
        color = _strategy_color(strategy)
        ax.plot(
            [left_x, right_x],
            [left_pos[strategy], right_pos[strategy]],
            color=color,
            linewidth=2.6,
            alpha=0.85,
            zorder=1,
        )
        for x, pos, value in (
            (left_x, left_pos[strategy], left_val[strategy]),
            (right_x, right_pos[strategy], right_val[strategy]),
        ):
            ax.scatter([x], [pos], s=240, color=color, zorder=2, edgecolor="white", linewidth=1.5)
            ax.annotate(
                f"{value:.3f}",
                (x, pos),
                textcoords="offset points",
                xytext=(0, 13),
                ha="center",
                fontsize=ANNOTATION_FONT_SIZE,
                color="#2c3e50",
                fontweight="bold",
            )
        ax.annotate(
            _TRACK_LABELS.get(strategy, _humanize_strategy_name(strategy)),
            (left_x, left_pos[strategy]),
            textcoords="offset points",
            xytext=(-14, 0),
            ha="right",
            va="center",
            fontsize=TICK_LABEL_FONT_SIZE,
            color=color,
            fontweight="bold",
        )
        ax.annotate(
            _TRACK_LABELS.get(strategy, _humanize_strategy_name(strategy)),
            (right_x, right_pos[strategy]),
            textcoords="offset points",
            xytext=(14, 0),
            ha="left",
            va="center",
            fontsize=TICK_LABEL_FONT_SIZE,
            color=color,
            fontweight="bold",
        )

    ax.set_xlim(-0.62, 1.62)
    ax.set_ylim(0.4, float(n) + 0.8)
    ax.set_xticks([left_x, right_x])
    ax.set_xticklabels(
        [
            "Synthetic\n(3-seat EIE,\nactive profile)",
            "Live Gemma\n(3-seat EIE,\nsingle model)",
        ],
        fontsize=AXIS_LABEL_FONT_SIZE,
        fontweight="bold",
    )
    ax.set_yticks([float(rank) for rank in range(1, n + 1)])
    ax.set_yticklabels(
        ["rank 4 (worst)", "rank 3", "rank 2", "rank 1 (best)"][-n:],
        fontsize=TICK_LABEL_FONT_SIZE,
    )
    ax.tick_params(length=0)
    for spine in ("top", "right", "left", "bottom"):
        ax.spines[spine].set_visible(False)

    _set_claim_title(ax, "Strategy ranking inverts between the synthetic and live tracks")
    _add_source_note(
        fig,
        "Matched three-seat grain: synthetic POWER_*_SIZE3 (output/data/sweep_aggregated.csv) "
        "vs live POSTDOC_*_EIE (output/data/postdoc_panel_results.json). Ranks compared, "
        "magnitudes NOT pooled; live track is one local gemma3:4b model, descriptive and n-limited.",
        y=0.015,
        fontsize=SOURCE_NOTE_FONT_SIZE,
    )
    return save_figure(fig, output_path)
