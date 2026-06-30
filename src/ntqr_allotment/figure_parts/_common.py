from __future__ import annotations

import math
import textwrap
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

DEGENERATE_EIE_ERROR = -1.0
DEFAULT_ALARM_MEASURED_POINTS: tuple[tuple[int, float], ...] = (
    (20, 0.6),
    (50, 8.0),
    (100, 89.7),
)
TITLE_FONT_SIZE = 17.0
SUBTITLE_FONT_SIZE = 14.5
AXIS_LABEL_FONT_SIZE = 13.0
TICK_LABEL_FONT_SIZE = 11.5
LEGEND_FONT_SIZE = 11.0
ANNOTATION_FONT_SIZE = 11.0
HEATMAP_CELL_FONT_SIZE = 10.5
SOURCE_NOTE_FONT_SIZE = 10.5
SOURCE_NOTE_WRAP_WIDTH = 118
GRID_ALPHA = 0.28
GRID_STYLE = "--"

STRATEGY_COLORS: dict[str, str] = {
    "expertise_threshold": "#4C78A8",
    "random_selection": "#72B7B2",
    "representative_sortition": "#54A24B",
    "ideological_selection": "#E45756",
}


def _row_field(row: object, field: str) -> object:
    if isinstance(row, Mapping):
        return row.get(field)
    return getattr(row, field, None)


def _independence_field(row: object, field: str) -> object:
    if isinstance(row, Mapping):
        return row.get(field)
    return getattr(row, field, None)


def _validated_independence_rows(
    aggregates: Sequence[object],
) -> list[tuple[float, float, float, float]]:
    if not aggregates:
        raise ValueError("independence aggregate: expected at least one aggregate row")

    validated_rows: list[tuple[float, float, float, float]] = []
    for index, row in enumerate(aggregates):
        context = f"independence aggregate row {index}"
        rho = _coerce_float(_independence_field(row, "rho"), "rho", context)
        corr_mean = _coerce_float(_independence_field(row, "corr_mean"), "corr_mean", context)
        eie_mean = _coerce_float(
            _independence_field(row, "eie_mean"),
            "eie_mean",
            context,
            allow_nonfinite=True,
        )
        eie_ci95 = _coerce_non_negative_float(
            _independence_field(row, "eie_ci95"),
            "eie_ci95",
            context,
        )
        validated_rows.append((rho, corr_mean, eie_mean, eie_ci95))
    return validated_rows


def _independence_strategies(aggregates: Sequence[object]) -> list[str]:
    strategies: list[str] = []
    for index, row in enumerate(aggregates):
        context = f"independence aggregate row {index}"
        strategies.append(
            _coerce_non_empty_string(_independence_field(row, "strategy"), "strategy", context)
        )
    return strategies


def _validated_alarm_power_rows(
    curve: Sequence[Sequence[object]],
) -> list[tuple[int, float]]:
    if not curve:
        raise ValueError("alarm power curve: expected at least one (panel_size, power) row")

    validated_rows: list[tuple[int, float]] = []
    for index, point in enumerate(curve):
        context = f"alarm power curve point {index}"
        if not isinstance(point, Sequence) or isinstance(point, (str, bytes)) or len(point) != 2:
            raise ValueError(f"{context}: expected (panel_size, power)")
        panel_size = _coerce_positive_int(point[0], "panel_size", context)
        power = _coerce_float(point[1], "power", context)
        if not 0.0 <= power <= 1.0:
            raise ValueError(f"{context}: power must be in [0, 1]")
        validated_rows.append((panel_size, power))
    return validated_rows


def _validated_realised_probabilities(report_or_probs: object) -> dict[str, float]:
    probabilities = _independence_field(report_or_probs, "realised_probabilities")
    if probabilities is None:
        probabilities = report_or_probs
    if not isinstance(probabilities, Mapping):
        raise ValueError("fairness maximin: expected a probability mapping or FairnessReport")
    if not probabilities:
        raise ValueError("fairness maximin: probability mapping must be non-empty")

    validated: dict[str, float] = {}
    for candidate, prob in probabilities.items():
        context = f"fairness maximin candidate {candidate!r}"
        candidate_key = _coerce_non_empty_string(str(candidate), "candidate", context)
        validated[candidate_key] = _coerce_non_negative_float(prob, "probability", context)
    return validated


def _validated_ranking_rows(
    ranking: Sequence[Sequence[object]],
) -> list[tuple[str, float, float]]:
    if not ranking:
        raise ValueError("strategy ranking: expected at least one ranking row")

    validated_rows: list[tuple[str, float, float]] = []
    for index, row in enumerate(ranking):
        if len(row) != 3:
            raise ValueError(f"strategy ranking row {index}: expected [strategy, mean, ci95]")
        context = f"strategy ranking row {index}"
        strategy = _coerce_non_empty_string(row[0], "strategy", context)
        mean = _coerce_float(row[1], "mean", context)
        ci95 = _coerce_non_negative_float(row[2], "ci95", context)
        validated_rows.append((strategy, mean, ci95))
    return validated_rows


def _validated_effect_rows(
    effects: Sequence[Mapping[str, object]],
) -> list[tuple[tuple[float, ...], float, float]]:
    if not effects:
        raise ValueError("rep_vs_ideo effect: expected at least one effect row")

    validated_rows: list[tuple[tuple[float, ...], float, float]] = []
    for index, row in enumerate(effects):
        regime_key_raw = row.get("regime_key")
        if not isinstance(regime_key_raw, Sequence) or isinstance(regime_key_raw, (str, bytes)):
            raise ValueError(f"rep_vs_ideo effect row {index}: regime_key must be a sequence")
        if len(regime_key_raw) < 2:
            raise ValueError(
                f"rep_vs_ideo effect row {index}: regime_key must include panel_size and mean_expertise"
            )
        regime_key = tuple(
            _coerce_float(value, f"regime_key[{key_index}]", f"rep_vs_ideo effect row {index}")
            for key_index, value in enumerate(regime_key_raw)
        )
        effect = _coerce_float(row.get("effect"), "effect", f"rep_vs_ideo effect row {index}")
        ci95 = _coerce_non_negative_float(
            row.get("ci95"),
            "ci95",
            f"rep_vs_ideo effect row {index}",
        )
        validated_rows.append((regime_key, effect, ci95))
    return validated_rows


def _normalized_alarm_points(
    measured_points: Sequence[tuple[int, float]],
) -> list[tuple[int, float]]:
    if not measured_points:
        raise ValueError("alarm cost curve: expected at least one measured point")

    grouped: dict[int, list[float]] = defaultdict(list)
    for index, point in enumerate(measured_points):
        if len(point) != 2:
            raise ValueError(f"alarm cost curve point {index}: expected (Q, seconds)")
        q = _coerce_positive_int(point[0], "Q", f"alarm cost curve point {index}")
        seconds = _coerce_positive_float(point[1], "seconds", f"alarm cost curve point {index}")
        grouped[q].append(seconds)

    return [(q, _mean(grouped[q])) for q in sorted(grouped)]


def _format_regime_label(regime_key: Sequence[float], index: int) -> str:
    panel_size = int(regime_key[0])
    mean_expertise = regime_key[1]
    return f"R{index + 1}: p={panel_size}, e={mean_expertise:.2f}"


def _power_curve_group_key(row: Mapping[str, object], context: str) -> str:
    strategy = row.get("strategy")
    if strategy is not None:
        return _humanize_strategy_name(
            _coerce_non_empty_string(strategy, "strategy", context)
        )
    mean_expertise = _coerce_float(row.get("mean_expertise"), "mean_expertise", context)
    return f"mean_expertise={mean_expertise:.2f}"


def _humanize_strategy_name(strategy: str) -> str:
    return strategy.replace("_", " ")


def _strategy_color(strategy: str, fallback: str = "#4C78A8") -> str:
    return STRATEGY_COLORS.get(strategy, fallback)


def _apply_axis_style(
    ax: plt.Axes,
    *,
    grid_axis: str | None = None,
    legend: bool = False,
) -> None:
    ax.tick_params(labelsize=TICK_LABEL_FONT_SIZE)
    ax.xaxis.label.set_size(AXIS_LABEL_FONT_SIZE)
    ax.yaxis.label.set_size(AXIS_LABEL_FONT_SIZE)
    ax.title.set_size(TITLE_FONT_SIZE)
    if grid_axis is not None:
        ax.grid(True, axis=grid_axis, linestyle=GRID_STYLE, alpha=GRID_ALPHA)
    if legend:
        ax.legend(fontsize=LEGEND_FONT_SIZE, frameon=False)


def _set_claim_title(ax: plt.Axes, title: str) -> None:
    ax.set_title(title, fontsize=TITLE_FONT_SIZE, fontweight="bold", pad=10)


def _add_source_note(
    fig: plt.Figure,
    note: str,
    *,
    y: float = 0.02,
    fontsize: float = SOURCE_NOTE_FONT_SIZE,
) -> None:
    wrapped_note = _wrap_source_note(note)
    fig.text(
        0.5,
        y,
        wrapped_note,
        ha="center",
        va="bottom",
        fontsize=fontsize,
        color="#4f5d66",
        linespacing=1.18,
    )


def _wrap_source_note(note: str) -> str:
    lines: list[str] = []
    for paragraph in note.splitlines():
        lines.extend(
            textwrap.wrap(
                paragraph,
                width=SOURCE_NOTE_WRAP_WIDTH,
                break_long_words=False,
                break_on_hyphens=False,
            )
            or [""]
        )
    return "\n".join(lines)


def save_figure(fig: plt.Figure, output_path: Path) -> Path:
    from ntqr_allotment import figures

    return figures._save_figure(fig, output_path)


def _save_figure(fig: plt.Figure, output_path: Path) -> Path:
    resolved_path = Path(output_path)
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        if not fig.get_constrained_layout():
            fig.tight_layout()
        fig.savefig(resolved_path, dpi=200, bbox_inches="tight")
    finally:
        plt.close(fig)
    return resolved_path


def _coerce_non_empty_string(value: object, field: str, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context}: {field} must be a non-empty string")
    return value


def _coerce_float(
    value: object,
    field: str,
    context: str,
    *,
    allow_nonfinite: bool = False,
) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context}: {field} must be numeric") from exc
    if not allow_nonfinite and not math.isfinite(number):
        raise ValueError(f"{context}: {field} must be finite")
    return number


def _coerce_non_negative_float(value: object, field: str, context: str) -> float:
    number = _coerce_float(value, field, context)
    if number < 0.0:
        raise ValueError(f"{context}: {field} must be non-negative")
    return number


def _coerce_positive_float(value: object, field: str, context: str) -> float:
    number = _coerce_float(value, field, context)
    if number <= 0.0:
        raise ValueError(f"{context}: {field} must be positive")
    return number


def _coerce_positive_int(value: object, field: str, context: str) -> int:
    number = _coerce_float(value, field, context)
    if number <= 0.0 or not float(number).is_integer():
        raise ValueError(f"{context}: {field} must be a positive integer")
    return int(number)


def _is_degenerate_eie_error(value: float) -> bool:
    return not math.isfinite(value) or value == DEGENERATE_EIE_ERROR


def _mean(values: Sequence[float]) -> float:
    return float(np.mean(np.asarray(values, dtype=float)))


def _ci95(values: Sequence[float]) -> float:
    if len(values) <= 1:
        return 0.0
    array = np.asarray(values, dtype=float)
    std = float(np.std(array, ddof=1))
    return float(1.96 * std / math.sqrt(len(array)))
