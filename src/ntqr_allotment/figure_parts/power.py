from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from ntqr_allotment.figure_parts._common import (
    ANNOTATION_FONT_SIZE,
    DEFAULT_ALARM_MEASURED_POINTS,
    GRID_ALPHA,
    GRID_STYLE,
    LEGEND_FONT_SIZE,
    TICK_LABEL_FONT_SIZE,
    _apply_axis_style,
    _coerce_float,
    _coerce_non_empty_string,
    _coerce_non_negative_float,
    _coerce_positive_int,
    _ci95,
    _is_degenerate_eie_error,
    _mean,
    _normalized_alarm_points,
    _power_curve_group_key,
    _row_field,
    _set_claim_title,
    _strategy_color,
    _validated_alarm_power_rows,
    plt,
    save_figure,
)

#: Trio->six-seat changes smaller than this (~the per-point CI scale) are labelled
#: "flat" rather than a noise-driven direction. MUST match the same tolerance in
#: ``manuscript_variables._size_direction`` so the figure label and the table
#: Direction column agree.
_FLAT_DIRECTION_TOL = 0.01


def _contrast_label(row: object, context: str) -> str:
    group_a = _row_field(row, "group_a")
    group_b = _row_field(row, "group_b")
    panel_size = _row_field(row, "panel_size")
    if group_a is not None and group_b is not None and panel_size is not None:
        return (
            f"{_coerce_non_empty_string(group_a, 'group_a', context).replace('_', ' ')}"
            " vs "
            f"{_coerce_non_empty_string(group_b, 'group_b', context).replace('_', ' ')}"
            f" (p={int(panel_size)})"
        )
    contrast = _coerce_non_empty_string(_row_field(row, "contrast"), "contrast", context)
    return contrast.replace("pair_", "").replace("__vs__", " vs ").replace("_", " ")


def _optional_float(row: object, field: str) -> float | None:
    value = _row_field(row, field)
    if value in (None, ""):
        return None
    return float(value)


def plot_power_curve(
    rows: Sequence[Mapping[str, object]],
    output_path: Path,
) -> Path:
    if not rows:
        raise ValueError("power curve: expected at least one sweep row")

    grouped: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for index, row in enumerate(rows):
        context = f"power curve row {index}"
        eie_error = _coerce_float(
            row.get("eie_error"),
            "eie_error",
            context,
            allow_nonfinite=True,
        )
        if _is_degenerate_eie_error(eie_error):
            continue
        group_key = _power_curve_group_key(row, context)
        panel_size = _coerce_positive_int(row.get("panel_size"), "panel_size", context)
        grouped[group_key][panel_size].append(eie_error)

    if not grouped:
        raise ValueError(
            "power curve: no valid rows remain after filtering degenerate eie_error values "
            "(-1.0 and non-finite)"
        )

    fig, ax = plt.subplots(figsize=(9.8, 5.8))
    for group_key in sorted(grouped):
        panel_groups = grouped[group_key]
        panel_sizes = sorted(panel_groups)
        means = [_mean(panel_groups[panel_size]) for panel_size in panel_sizes]
        ci95_values = [_ci95(panel_groups[panel_size]) for panel_size in panel_sizes]
        color = _strategy_color(group_key.replace(" ", "_"))
        ax.errorbar(
            panel_sizes,
            means,
            yerr=ci95_values,
            marker="o",
            linewidth=2.3,
            capsize=5,
            label=group_key,
            color=color,
            markersize=7,
        )
        if len(means) >= 2:
            # Label the trio -> six-seat step (the two smallest sizes), matching the
            # manuscript table's Size-3/Size-6 Direction column, with a flat band so a
            # sub-CI change (e.g. expertise threshold ~0.001) is not over-read as a
            # direction. Using first->last instead would contradict the table on
            # non-monotone curves (random rises 3->6 then returns by 12).
            step_delta = means[1] - means[0]
            if abs(step_delta) < _FLAT_DIRECTION_TOL:
                direction = "flat"
            else:
                direction = "rises" if step_delta > 0 else "falls"
            ax.text(
                panel_sizes[-1] + 0.05,
                means[-1],
                f"{group_key} {direction}",
                fontsize=ANNOTATION_FONT_SIZE,
                va="center",
                color=color,
            )

    ax.set_xlabel("Panel / ensemble size")
    ax.set_ylabel("Mean EIE error")
    _set_claim_title(ax, "Panel size is strategy-conditional, not uniformly beneficial")
    _apply_axis_style(ax, grid_axis="both", legend=True)
    return save_figure(fig, output_path)


def plot_alarm_cost_curve(
    output_path: Path,
    measured_points: Sequence[tuple[int, float]] = DEFAULT_ALARM_MEASURED_POINTS,
) -> Path:
    points = _normalized_alarm_points(measured_points)

    q_values = np.asarray([q for q, _ in points], dtype=float)
    seconds = np.asarray([runtime for _, runtime in points], dtype=float)
    scale = float(np.mean(seconds / (q_values**3)))
    curve_x = np.linspace(q_values.min(), q_values.max(), 200)
    curve_y = scale * (curve_x**3)

    fig, ax = plt.subplots(figsize=(9, 5.6))
    ax.plot(curve_x, curve_y, color="#4C78A8", linewidth=2.4, label=r"Mean-scaled $O(Q^3)$")
    ax.plot(q_values, seconds, color="#F58518", marker="o", linewidth=2.0, label="Measured")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Corpus size Q")
    ax.set_ylabel("Runtime (seconds)")
    _set_claim_title(ax, "Alarm enumeration is a small-corpus scaling limit")
    ax.grid(True, which="both", linestyle=GRID_STYLE, alpha=GRID_ALPHA)
    _apply_axis_style(ax, legend=True)
    return save_figure(fig, output_path)


def plot_alarm_power(
    curve: Sequence[Sequence[object]],
    output_path: Path,
) -> Path:
    points = _validated_alarm_power_rows(curve)
    ordered = sorted(points, key=lambda point: point[0])
    sizes = [size for size, _ in ordered]
    powers = [power for _, power in ordered]

    fig, ax = plt.subplots(figsize=(9, 5.6))
    ax.plot(sizes, powers, color="#4C78A8", marker="o", linewidth=2.3, markersize=7)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("Panel / ensemble size")
    ax.set_ylabel("Alarm firing rate")
    _set_claim_title(ax, "The small-Q consistency alarm is saturated here")
    _apply_axis_style(ax, grid_axis="both")
    return save_figure(fig, output_path)


def plot_power_design_diagnosis(
    power_rows: Sequence[object],
    output_path: Path,
) -> Path:
    if not power_rows:
        raise ValueError("power design diagnosis: expected at least one contrast row")

    labels: list[str] = []
    observed: list[float] = []
    mdes: list[float] = []
    annotations: list[str] = []
    for index, row in enumerate(power_rows):
        context = f"power row {index}"
        labels.append(_contrast_label(row, context))
        obs = abs(_coerce_float(_row_field(row, "observed_d"), "observed_d", context))
        mde = _coerce_non_negative_float(_row_field(row, "mde_80"), "mde_80", context)
        observed.append(obs)
        mdes.append(mde)
        perm_p = _optional_float(row, "perm_p")
        seeds_for_80 = _row_field(row, "seeds_for_80")
        verdict = _row_field(row, "verdict")
        annotation_parts = [f"|d|={obs:.2f}", f"MDE={mde:.2f}"]
        if perm_p is not None:
            annotation_parts.append(f"p={perm_p:.3f}")
        if seeds_for_80 not in (None, ""):
            annotation_parts.append(f"n80={seeds_for_80}")
        if verdict not in (None, ""):
            annotation_parts.append(str(verdict))
        annotations.append(" | ".join(annotation_parts))

    positions = list(range(len(labels)))
    colors = ["#54A24B" if obs >= mde else "#E45756" for obs, mde in zip(observed, mdes)]
    fig, ax = plt.subplots(figsize=(13.5, max(5.2, 0.72 * len(labels) + 1.4)))
    ax.barh(positions, observed, color=colors, height=0.6, label="observed |d|")
    ax.scatter(mdes, positions, color="black", marker="D", zorder=3, label="MDE @ 80% power")
    label_x = max([*observed, *mdes]) * 1.08 if observed or mdes else 1.0
    for y, annotation in zip(positions, annotations):
        ax.text(label_x, y, annotation, va="center", fontsize=ANNOTATION_FONT_SIZE)
    ax.set_yticks(positions)
    ax.set_yticklabels(labels, fontsize=TICK_LABEL_FONT_SIZE)
    ax.set_xlabel("Standardized effect size (|d|)")
    _set_claim_title(ax, "Power diagnosis separates observed effects from design limits")
    ax.set_xlim(right=label_x * 1.85)
    _apply_axis_style(ax, grid_axis="x")
    ax.legend(loc="lower right", fontsize=LEGEND_FONT_SIZE, frameon=False)
    return save_figure(fig, output_path)


def plot_power_vs_n(
    curves: Sequence[Sequence[object]],
    output_path: Path,
) -> Path:
    if not curves:
        raise ValueError("power vs n: expected at least one (label, ns, powers) curve")
    fig, ax = plt.subplots(figsize=(9.2, 5.7))
    for index, entry in enumerate(curves):
        if len(entry) != 3:
            raise ValueError(f"power vs n: curve {index} must be (label, ns, powers)")
        label, ns, powers = entry
        ns_values = list(ns)
        power_values = list(powers)
        ax.plot(ns_values, power_values, marker="o", linewidth=2.3, markersize=6.5, label=str(label))
        if ns_values and power_values:
            ax.text(
                ns_values[-1],
                power_values[-1],
                f" {label}",
                fontsize=ANNOTATION_FONT_SIZE,
                va="center",
            )
    ax.axhline(0.8, color="#888888", linewidth=1.2, linestyle="--", label="80% power")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("Per-group observations (n)")
    ax.set_ylabel("Analytic power")
    _set_claim_title(ax, "Small effects require many more per-group observations")
    _apply_axis_style(ax, grid_axis="both", legend=True)
    return save_figure(fig, output_path)
