from __future__ import annotations

from pathlib import Path

import numpy as np

from ntqr_allotment.figure_parts._common import (
    ANNOTATION_FONT_SIZE,
    _apply_axis_style,
    _set_claim_title,
    _validated_realised_probabilities,
    plt,
    save_figure,
)


def plot_fairness_maximin(
    report_or_probs: object,
    output_path: Path,
) -> Path:
    probabilities = _validated_realised_probabilities(report_or_probs)
    ordered = sorted(probabilities.items(), key=lambda item: (item[1], item[0]))
    labels = [candidate for candidate, _ in ordered]
    values = [prob for _, prob in ordered]
    min_prob = min(values)
    mean_prob = float(np.mean(np.asarray(values, dtype=float)))
    max_prob = max(values)

    fig, ax = plt.subplots(figsize=(10, max(4.8, 1.4 + len(ordered) * 0.35)))
    x_positions = np.arange(len(ordered))
    ax.bar(x_positions, values, color="#4C78A8")
    ax.axhline(min_prob, color="#E45756", linestyle="-", linewidth=1.5, label="min (maximin)")
    ax.axhline(mean_prob, color="#54A24B", linestyle="--", linewidth=1.5, label="mean")
    ax.axhline(max_prob, color="#F58518", linestyle=":", linewidth=1.5, label="max")
    ax.set_xticks(x_positions, labels=labels, rotation=45, ha="right")
    ax.set_ylabel("Realised selection probability")
    _set_claim_title(ax, "Representative lottery fairness is a selection-probability floor")
    for x, value in zip(x_positions, values):
        ax.text(x, value, f"{value:.2f}", ha="center", va="bottom", fontsize=ANNOTATION_FONT_SIZE)
    _apply_axis_style(ax, grid_axis="y", legend=True)
    return save_figure(fig, output_path)
