from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Sequence

import numpy as np

from ntqr_allotment.figure_parts._common import (
    ANNOTATION_FONT_SIZE,
    _apply_axis_style,
    _humanize_strategy_name,
    _independence_strategies,
    _is_degenerate_eie_error,
    _mean,
    _set_claim_title,
    _strategy_color,
    _validated_independence_rows,
    plt,
    save_figure,
)


def plot_error_vs_correlation(
    aggregates: Sequence[object],
    output_path: Path,
) -> Path:
    rows = _validated_independence_rows(aggregates)
    grouped: dict[float, list[tuple[float, float, float]]] = defaultdict(list)
    for rho, corr_mean, eie_mean, eie_ci95 in rows:
        if _is_degenerate_eie_error(eie_mean):
            continue
        grouped[rho].append((corr_mean, eie_mean, eie_ci95))

    if not grouped:
        raise ValueError(
            "error vs correlation: no valid cells remain after filtering degenerate "
            "eie_mean values (-1.0 and non-finite)"
        )

    fig, ax = plt.subplots(figsize=(9.3, 5.8))
    for rho in sorted(grouped):
        points = sorted(grouped[rho], key=lambda point: point[0])
        corr_values = [corr for corr, _, _ in points]
        eie_values = [eie for _, eie, _ in points]
        ci95_values = [ci95 for _, _, ci95 in points]
        ax.errorbar(
            corr_values,
            eie_values,
            yerr=ci95_values,
            marker="o",
            linewidth=2.2,
            capsize=5,
            label=f"rho={rho:.2f}",
            markersize=7,
        )
        ax.text(
            corr_values[-1],
            eie_values[-1],
            f" rho={rho:.2f}",
            fontsize=ANNOTATION_FONT_SIZE,
            va="center",
        )

    ax.set_xlabel("NTQR-measured realized pairwise error correlation")
    ax.set_ylabel("Oracle-referenced EIE error")
    _set_claim_title(ax, "Injected correlation is measured; recovery slope is unresolved")
    _apply_axis_style(ax, grid_axis="both", legend=True)
    return save_figure(fig, output_path)


def plot_strategy_correlation(
    aggregates: Sequence[object],
    output_path: Path,
) -> Path:
    rows = _validated_independence_rows(aggregates)
    strategies = _independence_strategies(aggregates)
    grouped: dict[str, list[float]] = defaultdict(list)
    for (_, corr_mean, _, _), strategy in zip(rows, strategies):
        grouped[strategy].append(corr_mean)

    means = {strategy: _mean(values) for strategy, values in grouped.items()}
    ordered = sorted(means.items(), key=lambda item: (item[1], item[0]))
    labels = [_humanize_strategy_name(strategy) for strategy, _ in ordered]
    values = [mean for _, mean in ordered]

    fig, ax = plt.subplots(figsize=(9.2, max(4.8, 1.4 + len(ordered) * 0.85)))
    y_positions = np.arange(len(ordered))
    colors = [_strategy_color(strategy) for strategy, _ in ordered]
    ax.barh(y_positions, values, color=colors, alpha=0.9)
    label_x = max(values) * 1.04 if values else 0.0
    for y, value in zip(y_positions, values):
        ax.text(label_x, y, f"{value:.3f}", va="center", fontsize=ANNOTATION_FONT_SIZE)
    ax.set_yticks(y_positions, labels=labels)
    ax.invert_yaxis()
    ax.set_xlabel("Mean realized error correlation")
    _set_claim_title(ax, "Single-bloc selection induces the highest trio correlation")
    ax.set_xlim(right=label_x * 1.18 if label_x else 1.0)
    _apply_axis_style(ax, grid_axis="x")
    return save_figure(fig, output_path)
