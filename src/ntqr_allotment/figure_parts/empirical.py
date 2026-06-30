from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from ntqr_allotment.statistics_analysis import bootstrap_ci, exact_sign_test

from ntqr_allotment.figure_parts._common import (
    ANNOTATION_FONT_SIZE,
    HEATMAP_CELL_FONT_SIZE,
    GRID_ALPHA,
    GRID_STYLE,
    LEGEND_FONT_SIZE,
    STRATEGY_COLORS,
    _apply_axis_style,
    _coerce_float,
    _coerce_non_empty_string,
    _coerce_non_negative_float,
    _row_field,
    _set_claim_title,
    plt,
    save_figure,
)

POSTDOC_LABELS = {
    "representative_sortition": "representative sortition",
    "random_selection": "random selection",
    "ideological_selection": "same-bias selection",
    "expertise_threshold": "expertise threshold",
}
POSTDOC_ORDER = (
    "expertise_threshold",
    "representative_sortition",
    "random_selection",
    "ideological_selection",
)


def plot_cross_family_contrast(
    contrast: object,
    output_path: Path,
) -> Path:
    context = "cross-family contrast"
    same = _coerce_non_negative_float(
        _row_field(contrast, "mean_abs_same_family"), "mean_abs_same_family", context
    )
    cross = _coerce_non_negative_float(
        _row_field(contrast, "mean_abs_cross_family"), "mean_abs_cross_family", context
    )
    delta = _coerce_float(
        _row_field(contrast, "delta_cross_minus_same"), "delta_cross_minus_same", context
    )
    label = _coerce_non_empty_string(_row_field(contrast, "label"), "label", context)
    n_same = _row_field(contrast, "n_same_pairs")
    n_cross = _row_field(contrast, "n_cross_pairs")
    n_items = _row_field(contrast, "n_items")
    nonzero_pairs = _row_field(contrast, "nonzero_pairs")
    total_pairs = _row_field(contrast, "total_pairs")

    fig, ax = plt.subplots(figsize=(8.6, 5.6))
    bars = ax.bar(
        ["same-family", "cross-family"],
        [same, cross],
        color=["#E45756", "#4C78A8"],
        width=0.6,
    )
    for rect, value in zip(bars, (same, cross)):
        ax.text(
            rect.get_x() + rect.get_width() / 2,
            rect.get_height(),
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=ANNOTATION_FONT_SIZE,
        )
    ax.set_ylabel("Mean |pairwise error correlation|")
    _set_claim_title(ax, f"Cross-family pairs are less correlated here (delta={delta:+.3f})")
    notes: list[str] = []
    if n_items is not None:
        notes.append(f"items={int(n_items)}")
    if n_same is not None and n_cross is not None:
        notes.append(f"pairs: same={int(n_same)}, cross={int(n_cross)}")
    if nonzero_pairs is not None and total_pairs is not None:
        notes.append(f"nonzero={int(nonzero_pairs)}/{int(total_pairs)}")
    notes.append("single-run test: none")
    ax.text(
        0.5,
        -0.18,
        " | ".join(notes),
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=ANNOTATION_FONT_SIZE,
    )
    ax.text(
        0.5,
        1.02,
        label,
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=ANNOTATION_FONT_SIZE,
        color="#4f5d66",
    )
    ax.grid(True, axis="y", linestyle=GRID_STYLE, alpha=GRID_ALPHA)
    _apply_axis_style(ax)
    return save_figure(fig, output_path)


def plot_cross_family_multiseed(
    deltas: Sequence[object],
    output_path: Path,
) -> Path:
    values = [
        _coerce_float(d, "delta", "cross-family multi-seed") for d in deltas
    ]
    if not values:
        raise ValueError("cross-family multi-seed: expected at least one delta")
    positions = list(range(1, len(values) + 1))
    mean = sum(values) / len(values)
    ci_low, ci_high = bootstrap_ci(values, n_boot=10000, alpha=0.05, seed=0)
    sign = exact_sign_test(values)
    colors = ["#4C78A8" if v < 0 else "#E45756" for v in values]
    fig, ax = plt.subplots(figsize=(9.2, 5.8))
    ax.axhline(0.0, color="#888888", linewidth=1.0, linestyle="--")
    ax.bar(positions, values, color=colors, width=0.6)
    ax.axhspan(ci_low, ci_high, color="#4C78A8", alpha=0.14, label="mean 95% CI")
    ax.axhline(mean, color="#E45756", linewidth=1.5, label=f"mean {mean:.3f}")
    for x, value in zip(positions, values):
        va = "top" if value < 0 else "bottom"
        ax.text(x, value, f"{value:.3f}", ha="center", va=va, fontsize=ANNOTATION_FONT_SIZE)
    ax.set_xticks(positions)
    ax.set_xlabel("run (independent corpus)")
    ax.set_ylabel("cross - same |error correlation|")
    _set_claim_title(
        ax,
        "Cross-family deltas by run "
        f"(negative {sign.negative}/{sign.n_nonzero}, sign p={sign.p_value:.3f})"
    )
    ax.grid(True, axis="y", linestyle=GRID_STYLE, alpha=GRID_ALPHA)
    _apply_axis_style(ax)
    ax.legend(fontsize=LEGEND_FONT_SIZE, frameon=False)
    return save_figure(fig, output_path)


def plot_postdoc_strategy_ranking(
    payload_or_aggregates: object,
    output_path: Path,
) -> Path:
    aggregates = _postdoc_aggregates(payload_or_aggregates)
    if not aggregates:
        raise ValueError("postdoc strategy ranking: expected aggregate rows")
    grouped: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in aggregates:
        grouped.setdefault((str(row["track"]), str(row["strategy"])), []).append(row)
    ranking_rows: list[tuple[str, str, float, float, float]] = []
    for (track, strategy), rows in grouped.items():
        means = [float(row["eie_mean"]) for row in rows]
        lows = [float(row.get("eie_ci_low", row["eie_mean"])) for row in rows]
        highs = [float(row.get("eie_ci_high", row["eie_mean"])) for row in rows]
        ranking_rows.append(
            (
                track,
                strategy,
                float(np.mean(means)),
                float(np.mean(lows)),
                float(np.mean(highs)),
            )
        )

    present_strategies = _ordered_postdoc_strategies(row[1] for row in ranking_rows)
    y_lookup = {strategy: idx for idx, strategy in enumerate(present_strategies)}
    track_offsets = {"analytical": -0.16, "live": 0.16}
    track_markers = {"analytical": "s", "live": "o"}
    track_labels = {"analytical": "analytical vote model", "live": "live Gemma"}

    fig, ax = plt.subplots(figsize=(11.8, 6.8))
    for track in ("analytical", "live"):
        rows = [row for row in ranking_rows if row[0] == track]
        if not rows:
            continue
        xs = [row[2] for row in rows]
        ys = [y_lookup[row[1]] + track_offsets.get(track, 0.0) for row in rows]
        xerr = [
            [max(0.0, row[2] - row[3]) for row in rows],
            [max(0.0, row[4] - row[2]) for row in rows],
        ]
        colors = [STRATEGY_COLORS.get(row[1], "#4C78A8") for row in rows]
        ax.errorbar(
            xs,
            ys,
            xerr=xerr,
            fmt=track_markers.get(track, "o"),
            markersize=8.5,
            linewidth=0,
            elinewidth=2.1,
            capsize=4,
            color="#222222",
            ecolor="#3a4650",
            label=track_labels.get(track, track),
        )
        ax.scatter(
            xs,
            ys,
            s=72,
            marker=track_markers.get(track, "o"),
            c=colors,
            edgecolor="#172026",
            linewidth=0.7,
            zorder=3,
        )
        for x, y_value, row in zip(xs, ys, rows):
            ax.text(
                x + 0.008,
                y_value,
                f"{row[2]:.3f}",
                va="center",
                fontsize=ANNOTATION_FONT_SIZE,
            )

    ax.set_yticks(
        np.arange(len(present_strategies)),
        [POSTDOC_LABELS.get(strategy, strategy) for strategy in present_strategies],
    )
    ax.invert_yaxis()
    ax.set_xlabel("Oracle-referenced EIE error (lower is better)")
    _set_claim_title(
        ax,
        "Gemma postdoc ranking compares sampling strategies within one model",
    )
    ax.text(
        0.99,
        -0.13,
        "Intervals are descriptive over seeds and panel sizes; live rows use one local gemma3:4b model.",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=ANNOTATION_FONT_SIZE,
        color="#4f5d66",
    )
    _apply_axis_style(ax, grid_axis="x", legend=True)
    return save_figure(fig, output_path)


def plot_postdoc_age_bias_heatmap(
    payload_or_aggregates: object,
    output_path: Path,
) -> Path:
    aggregates = [row for row in _postdoc_aggregates(payload_or_aggregates) if row.get("track") == "live"]
    if not aggregates:
        raise ValueError("postdoc age-bias heatmap: expected live aggregate rows")
    strategies = _ordered_postdoc_strategies(str(row["strategy"]) for row in aggregates)
    panel_sizes = sorted({int(row["panel_size"]) for row in aggregates})
    matrix = np.full((len(strategies), len(panel_sizes)), np.nan)
    labels: dict[tuple[int, int], str] = {}
    for row in aggregates:
        i = strategies.index(str(row["strategy"]))
        j = panel_sizes.index(int(row["panel_size"]))
        value = float(row["age_disparity_mean"])
        low = float(row.get("age_disparity_ci_low", value))
        high = float(row.get("age_disparity_ci_high", value))
        matrix[i, j] = value
        labels[(i, j)] = f"{value:+.2f}\n[{low:+.2f},{high:+.2f}]"
    finite = matrix[np.isfinite(matrix)]
    limit = max(0.05, float(np.max(np.abs(finite))) if finite.size else 0.05)

    fig, ax = plt.subplots(figsize=(9.8, 6.9))
    im = ax.imshow(matrix, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
    ax.set_xticks(range(len(panel_sizes)), [str(size) for size in panel_sizes])
    ax.set_yticks(range(len(strategies)), [POSTDOC_LABELS.get(s, s) for s in strategies])
    ax.set_xlabel("Panel size")
    ax.set_ylabel("Sampling strategy")
    _set_claim_title(ax, "Live Gemma age-disparity by sampling strategy and size")
    for i, strategy in enumerate(strategies):
        for j, panel_size in enumerate(panel_sizes):
            value = matrix[i, j]
            if np.isfinite(value):
                ax.text(
                    j,
                    i,
                    labels[(i, j)],
                    ha="center",
                    va="center",
                    fontsize=HEATMAP_CELL_FONT_SIZE,
                    color="#172026",
                )
    cbar = fig.colorbar(im, ax=ax, shrink=0.86)
    cbar.set_label("Older-minus-younger recommendation rate")
    ax.text(
        0.5,
        -0.16,
        "Positive cells favor older synthetic applicants; negative cells favor younger ones.",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=ANNOTATION_FONT_SIZE,
        color="#4f5d66",
    )
    _apply_axis_style(ax)
    return save_figure(fig, output_path)


def plot_postdoc_empirical_alignment(
    alignment_payload: object,
    output_path: Path,
) -> Path:
    if not isinstance(alignment_payload, dict):
        raise ValueError("postdoc alignment: expected payload mapping")
    cells = alignment_payload.get("cells")
    if not isinstance(cells, list) or not cells:
        raise ValueError("postdoc alignment: expected cells")
    strategies = _ordered_postdoc_strategies(str(cell["strategy"]) for cell in cells)
    panel_sizes = sorted({int(cell["panel_size"]) for cell in cells})
    matrix = np.full((len(strategies), len(panel_sizes)), np.nan)
    labels: dict[tuple[int, int], str] = {}
    for cell in cells:
        i = strategies.index(str(cell["strategy"]))
        j = panel_sizes.index(int(cell["panel_size"]))
        unresolved = bool(cell.get("unresolved"))
        agrees = bool(cell.get("sign_agrees"))
        matrix[i, j] = 0.0 if unresolved else 1.0 if agrees else -1.0
        status = "unresolved" if unresolved else "aligned" if agrees else "mismatch"
        analytical_sign = str(cell.get("analytical_sign", cell.get("predicted_sign", "?")))
        live_sign = str(cell.get("live_sign", cell.get("observed_sign", "?")))
        labels[(i, j)] = f"{status}\nA {analytical_sign}\nG {live_sign}"

    fig, ax = plt.subplots(figsize=(9.6, 6.8))
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=-1.0, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(panel_sizes)), [str(size) for size in panel_sizes])
    ax.set_yticks(range(len(strategies)), [POSTDOC_LABELS.get(s, s) for s in strategies])
    ax.set_xlabel("Panel size")
    ax.set_ylabel("Sampling strategy")
    _set_claim_title(ax, "Analytical and Gemma signs are compared cell by cell")
    for i in range(len(strategies)):
        for j in range(len(panel_sizes)):
            ax.text(
                j,
                i,
                labels.get((i, j), "missing"),
                ha="center",
                va="center",
                fontsize=HEATMAP_CELL_FONT_SIZE,
                color="#172026",
            )
    cbar = fig.colorbar(im, ax=ax, shrink=0.86, ticks=[-1, 0, 1])
    cbar.ax.set_yticklabels(["mismatch", "unresolved", "aligned"])
    ax.text(
        0.5,
        -0.16,
        str(alignment_payload.get("caveat", "single-model live companion; descriptive only")),
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=ANNOTATION_FONT_SIZE,
        color="#4f5d66",
    )
    _apply_axis_style(ax)
    return save_figure(fig, output_path)


def _postdoc_aggregates(payload_or_aggregates: object) -> list[dict[str, object]]:
    if isinstance(payload_or_aggregates, dict):
        aggregates = payload_or_aggregates.get("aggregates")
    else:
        aggregates = payload_or_aggregates
    if not isinstance(aggregates, Sequence) or isinstance(aggregates, (str, bytes)):
        raise ValueError("postdoc aggregates: expected a sequence")
    rows: list[dict[str, object]] = []
    for row in aggregates:
        if not isinstance(row, dict):
            raise ValueError("postdoc aggregates: each row must be a mapping")
        rows.append(row)
    return rows


def _ordered_postdoc_strategies(strategies: object) -> list[str]:
    present = {str(strategy) for strategy in strategies}
    ordered = [strategy for strategy in POSTDOC_ORDER if strategy in present]
    ordered.extend(sorted(present.difference(ordered)))
    return ordered
