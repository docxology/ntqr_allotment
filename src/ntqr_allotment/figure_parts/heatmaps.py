from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

from ._common import (
    AXIS_LABEL_FONT_SIZE,
    HEATMAP_CELL_FONT_SIZE,
    SOURCE_NOTE_FONT_SIZE,
    SUBTITLE_FONT_SIZE,
    TICK_LABEL_FONT_SIZE,
    TITLE_FONT_SIZE,
    _add_source_note,
    _apply_axis_style,
    _coerce_float,
    _coerce_non_empty_string,
    _coerce_non_negative_float,
    _coerce_positive_int,
    _humanize_strategy_name,
    save_figure,
)


def plot_rep_vs_ideo_heatmap(
    cells: Sequence[Mapping[str, object]],
    output_path: Path,
) -> Path:
    rows = _validated_rep_cells(cells)
    panels = sorted({row["panel_size"] for row in rows})
    expertises = sorted({row["mean_expertise"] for row in rows})
    biases = sorted({row["bias_std"] for row in rows})
    grouped = _mean_group(rows, ("panel_size", "mean_expertise", "bias_std"), "effect")
    resolved = _any_group(rows, ("panel_size", "mean_expertise", "bias_std"), "excludes_zero")

    ncols = min(2, len(panels))
    nrows = math.ceil(len(panels) / ncols)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(7.4 * ncols, 5.4 * nrows),
        squeeze=False,
        constrained_layout=True,
    )
    values = [abs(value) for value in grouped.values()]
    vmax = max(values) if values else 1.0
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    image = None
    for ax, panel_size in zip(axes.ravel(), panels, strict=False):
        matrix = _matrix(
            expertises,
            biases,
            lambda expertise, bias: grouped.get((panel_size, expertise, bias)),
        )
        image = ax.imshow(matrix, origin="lower", cmap="RdBu_r", norm=norm, aspect="auto")
        _label_heatmap_axes(ax, biases, expertises)
        ax.set_title(f"Panel size {panel_size}", fontsize=SUBTITLE_FONT_SIZE, fontweight="bold")
        for y_index, expertise in enumerate(expertises):
            for x_index, bias in enumerate(biases):
                value = grouped.get((panel_size, expertise, bias))
                if value is None:
                    continue
                marker = "*" if resolved.get((panel_size, expertise, bias), False) else ""
                ax.text(
                    x_index,
                    y_index,
                    f"{value:+.2f}{marker}",
                    ha="center",
                    va="center",
                    fontsize=HEATMAP_CELL_FONT_SIZE,
                    color=_text_color(value, vmax),
                )
        _apply_axis_style(ax)
    for ax in axes.ravel()[len(panels) :]:
        ax.axis("off")
    fig.suptitle(
        "Bias, expertise, and panel size shape the representative-vs-same-bias gap",
        fontsize=TITLE_FONT_SIZE,
        fontweight="bold",
    )
    if image is not None:
        colorbar = fig.colorbar(
            image,
            ax=axes.ravel().tolist(),
            shrink=0.82,
            pad=0.02,
            label="EIE error delta",
        )
        colorbar.set_label("Ideological - representative EIE", fontsize=AXIS_LABEL_FONT_SIZE)
        colorbar.ax.tick_params(labelsize=TICK_LABEL_FONT_SIZE)
    _add_source_note(
        fig,
        "* marks cells whose descriptive 95% interval excludes zero; positive means same-bias selection has higher EIE.",
        y=-0.014,
    )
    return save_figure(fig, output_path)


def plot_pre_post_ntqr_heatmap(
    cells: Sequence[Mapping[str, object]],
    output_path: Path,
) -> Path:
    rows = _validated_pre_post_cells(cells)
    panels = sorted({row["panel_size"] for row in rows})
    strategies = sorted({row["strategy"] for row in rows})
    expertises = sorted({row["mean_expertise"] for row in rows})
    biases = sorted({row["bias_std"] for row in rows})
    grouped = _mean_group(
        rows,
        ("strategy", "panel_size", "mean_expertise", "bias_std"),
        "eie_minus_mv",
    )

    fig, axes = plt.subplots(
        len(panels),
        len(strategies),
        figsize=(5.3 * len(strategies), 4.35 * len(panels)),
        squeeze=False,
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    values = [abs(value) for value in grouped.values()]
    vmax = max(values) if values else 1.0
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    image = None
    for row_index, panel_size in enumerate(panels):
        for col_index, strategy in enumerate(strategies):
            ax = axes[row_index][col_index]
            matrix = _matrix(
                expertises,
                biases,
                lambda expertise, bias: grouped.get(
                    (strategy, panel_size, expertise, bias)
                ),
            )
            image = ax.imshow(matrix, origin="lower", cmap="PuOr_r", norm=norm, aspect="auto")
            _label_heatmap_axes(ax, biases, expertises)
            if row_index == 0:
                ax.set_title(
                    _humanize_strategy_name(strategy),
                    fontsize=SUBTITLE_FONT_SIZE,
                    fontweight="bold",
                )
            if col_index == 0:
                ax.set_ylabel(f"panel {panel_size}\nmean expertise")
            else:
                ax.set_ylabel("")
            for y_index, expertise in enumerate(expertises):
                for x_index, bias in enumerate(biases):
                    value = grouped.get((strategy, panel_size, expertise, bias))
                    if value is None:
                        continue
                    ax.text(
                        x_index,
                        y_index,
                        f"{value:+.2f}",
                        ha="center",
                        va="center",
                        fontsize=HEATMAP_CELL_FONT_SIZE,
                        color=_text_color(value, vmax),
                    )
            _apply_axis_style(ax)
    fig.suptitle(
        "Pre/post NTQR contrasts remain regime-specific across sampling designs",
        fontsize=TITLE_FONT_SIZE,
        fontweight="bold",
    )
    if image is not None:
        colorbar = fig.colorbar(
            image,
            ax=axes.ravel().tolist(),
            shrink=0.82,
            pad=0.02,
            label="EIE - majority-vote error",
        )
        colorbar.set_label("EIE - majority-vote error", fontsize=AXIS_LABEL_FONT_SIZE)
        colorbar.ax.tick_params(labelsize=TICK_LABEL_FONT_SIZE)
    _add_source_note(
        fig,
        "Negative cells mean EIE is lower than the oracle-referenced majority-vote baseline for that regime.",
        y=-0.014,
    )
    return save_figure(fig, output_path)


def plot_theory_alignment_heatmap(
    prediction_payload: Mapping[str, object],
    output_path: Path,
) -> Path:
    raw_cells = prediction_payload.get("rep_vs_ideo_cells")
    if not isinstance(raw_cells, Sequence):
        raise ValueError("theory alignment: rep_vs_ideo_cells must be a sequence")
    rows = _validated_rep_cells(raw_cells)
    panels = sorted({row["panel_size"] for row in rows})
    biases = sorted({row["bias_std"] for row in rows})
    grouped: dict[tuple[int, float], list[bool]] = defaultdict(list)
    for row in rows:
        grouped[(row["panel_size"], row["bias_std"])].append(
            str(row["prediction_status"]).startswith("aligned")
        )

    matrix = np.full((len(panels), len(biases)), np.nan)
    labels: dict[tuple[int, float], str] = {}
    for y_index, panel_size in enumerate(panels):
        for x_index, bias in enumerate(biases):
            values = grouped.get((panel_size, bias), [])
            if not values:
                continue
            aligned = sum(values)
            matrix[y_index, x_index] = aligned / len(values)
            labels[(panel_size, bias)] = f"{aligned}/{len(values)}"

    fig, ax = plt.subplots(figsize=(9.8, 5.9))
    image = ax.imshow(matrix, origin="lower", cmap="YlGnBu", vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(np.arange(len(biases)), labels=[f"{bias:.2f}" for bias in biases])
    ax.set_yticks(np.arange(len(panels)), labels=[str(panel) for panel in panels])
    ax.set_xlabel("Bias spread")
    ax.set_ylabel("Panel size")
    ax.set_title(
        "Analytical direction checks are summarized by observed alignment",
        fontsize=TITLE_FONT_SIZE,
        fontweight="bold",
    )
    for y_index, panel_size in enumerate(panels):
        for x_index, bias in enumerate(biases):
            label = labels.get((panel_size, bias))
            if label:
                ax.text(
                    x_index,
                    y_index,
                    label,
                    ha="center",
                    va="center",
                    fontsize=HEATMAP_CELL_FONT_SIZE,
                )
    colorbar = fig.colorbar(image, ax=ax, label="Fraction aligned across expertise levels")
    colorbar.set_label("Fraction aligned across expertise levels", fontsize=AXIS_LABEL_FONT_SIZE)
    colorbar.ax.tick_params(labelsize=TICK_LABEL_FONT_SIZE)
    ax.text(
        0.5,
        -0.2,
        _alignment_note(prediction_payload),
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=SOURCE_NOTE_FONT_SIZE,
        color="#4f5d66",
    )
    _apply_axis_style(ax)
    return save_figure(fig, output_path)


def _validated_rep_cells(
    cells: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    if not cells:
        raise ValueError("rep-vs-ideo heatmap: expected at least one cell")
    rows: list[dict[str, object]] = []
    for index, cell in enumerate(cells):
        context = f"rep-vs-ideo heatmap cell {index}"
        status = str(cell.get("prediction_status", ""))
        rows.append(
            {
                "panel_size": _coerce_positive_int(cell.get("panel_size"), "panel_size", context),
                "mean_expertise": _coerce_float(cell.get("mean_expertise"), "mean_expertise", context),
                "bias_std": _coerce_non_negative_float(cell.get("bias_std"), "bias_std", context),
                "effect": _coerce_float(cell.get("effect"), "effect", context),
                "ci95": _coerce_non_negative_float(cell.get("ci95"), "ci95", context),
                "excludes_zero": bool(cell.get("excludes_zero", False)),
                "prediction_status": status,
            }
        )
    return rows


def _validated_pre_post_cells(
    cells: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    if not cells:
        raise ValueError("pre/post heatmap: expected at least one cell")
    rows: list[dict[str, object]] = []
    for index, cell in enumerate(cells):
        context = f"pre/post heatmap cell {index}"
        rows.append(
            {
                "strategy": _coerce_non_empty_string(cell.get("strategy"), "strategy", context),
                "panel_size": _coerce_positive_int(cell.get("panel_size"), "panel_size", context),
                "mean_expertise": _coerce_float(cell.get("mean_expertise"), "mean_expertise", context),
                "bias_std": _coerce_non_negative_float(cell.get("bias_std"), "bias_std", context),
                "eie_minus_mv": _coerce_float(cell.get("eie_minus_mv"), "eie_minus_mv", context),
            }
        )
    return rows


def _mean_group(
    rows: Sequence[Mapping[str, object]],
    keys: tuple[str, ...],
    field: str,
) -> dict[tuple[object, ...], float]:
    grouped: dict[tuple[object, ...], list[float]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in keys)].append(float(row[field]))
    return {key: float(np.mean(values)) for key, values in grouped.items()}


def _any_group(
    rows: Sequence[Mapping[str, object]],
    keys: tuple[str, ...],
    field: str,
) -> dict[tuple[object, ...], bool]:
    grouped: dict[tuple[object, ...], list[bool]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in keys)].append(bool(row[field]))
    return {key: any(values) for key, values in grouped.items()}


def _matrix(
    y_values: Sequence[float],
    x_values: Sequence[float],
    value_for: object,
) -> np.ndarray:
    matrix = np.full((len(y_values), len(x_values)), np.nan)
    for y_index, y_value in enumerate(y_values):
        for x_index, x_value in enumerate(x_values):
            value = value_for(y_value, x_value)
            if value is not None:
                matrix[y_index, x_index] = float(value)
    return matrix


def _label_heatmap_axes(
    ax: plt.Axes,
    biases: Sequence[float],
    expertises: Sequence[float],
) -> None:
    ax.set_xticks(np.arange(len(biases)), labels=[f"{bias:.2f}" for bias in biases])
    ax.set_yticks(
        np.arange(len(expertises)),
        labels=[f"{expertise:.2f}" for expertise in expertises],
    )
    ax.set_xlabel("Bias spread")
    ax.set_ylabel("Mean expertise")
    ax.tick_params(labelsize=TICK_LABEL_FONT_SIZE)


def _text_color(value: float, vmax: float) -> str:
    return "white" if abs(value) > vmax * 0.55 else "#172026"


def _alignment_note(payload: Mapping[str, object]) -> str:
    checks = payload.get("monotone_checks")
    if not isinstance(checks, Sequence):
        return "Directional prediction: ideological-minus-representative EIE should be positive."
    parts = []
    for check in checks:
        if not isinstance(check, Mapping) or check.get("status") != "tested":
            continue
        parts.append(
            f"{check.get('axis')}: {check.get('n_aligned')}/{check.get('n_comparisons')} aligned"
        )
    return "; ".join(parts) if parts else "Directional prediction status summarized by cell counts."


__all__ = [
    "plot_pre_post_ntqr_heatmap",
    "plot_rep_vs_ideo_heatmap",
    "plot_theory_alignment_heatmap",
]
