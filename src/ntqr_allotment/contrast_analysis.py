from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Mapping

REPRESENTATIVE_STRATEGY = "representative_sortition"
IDEOLOGICAL_STRATEGY = "ideological_selection"

PREDICTION_RATIONALES = {
    "rep_vs_ideo_sign": (
        "Single-bloc ideological selection should not outperform a representative "
        "draw when ideological bias is the manipulated dependence source."
    ),
    "bias_direction": (
        "Larger bias spread should widen the ideological-minus-representative "
        "EIE contrast when other regime coordinates are held fixed."
    ),
    "expertise_direction": (
        "Higher mean expertise should lower representative-sortition EIE error "
        "when other regime coordinates are held fixed."
    ),
    "pre_post_boundary": (
        "EIE-minus-majority-vote is descriptive: MV is the supervised/oracle "
        "baseline and EIE is the no-answer-key NTQR recovery metric."
    ),
}


def write_analytical_predictions(
    aggregate_csv: Path,
    output_path: Path,
    *,
    metadata: Mapping[str, object] | None = None,
) -> Path:
    payload = build_analytical_predictions(aggregate_csv, metadata=metadata)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output_path


def build_analytical_predictions(
    aggregate_csv: Path,
    *,
    metadata: Mapping[str, object] | None = None,
) -> dict[str, object]:
    rows = _read_aggregates(aggregate_csv)
    rep_cells = _rep_vs_ideo_cells(rows)
    pre_post_cells = _pre_post_cells(rows)
    checks = _monotone_checks(rows, rep_cells)
    return {
        "schema_version": 1,
        "source": aggregate_csv.name,
        "metadata": dict(metadata or {}),
        "prediction_rationales": PREDICTION_RATIONALES,
        "rep_vs_ideo_cells": rep_cells,
        "pre_post_cells": pre_post_cells,
        "monotone_checks": checks,
    }


def _read_aggregates(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = [_coerced_row(row) for row in reader]
    if not rows:
        raise ValueError(f"{path}: expected at least one aggregate row")
    return rows


def _coerced_row(row: Mapping[str, str]) -> dict[str, object]:
    return {
        "profile_name": row.get("profile_name", ""),
        "config_hash": row.get("config_hash", ""),
        "strategy": _string(row, "strategy"),
        "panel_size": _int(row, "panel_size"),
        "mean_expertise": _float(row, "mean_expertise"),
        "expertise_heterogeneity": _float(row, "expertise_heterogeneity"),
        "bias_std": _float(row, "bias_std"),
        "n_items": _int(row, "n_items"),
        "prevalence_a": _float(row, "prevalence_a"),
        "n_experts": _int(row, "n_experts"),
        "n": _int(row, "n"),
        "eie_mean": _float(row, "eie_mean"),
        "eie_ci95": _non_negative_float(row, "eie_ci95"),
        "mv_mean": _float(row, "mv_mean"),
        "mv_ci95": _non_negative_float(row, "mv_ci95"),
    }


def _rep_vs_ideo_cells(rows: Iterable[Mapping[str, object]]) -> list[dict[str, object]]:
    by_regime: dict[tuple[object, ...], dict[str, Mapping[str, object]]] = defaultdict(dict)
    for row in rows:
        by_regime[_regime_key(row)][str(row["strategy"])] = row

    cells: list[dict[str, object]] = []
    for regime_key in sorted(by_regime):
        strategies = by_regime[regime_key]
        rep = strategies.get(REPRESENTATIVE_STRATEGY)
        ideo = strategies.get(IDEOLOGICAL_STRATEGY)
        if rep is None or ideo is None:
            continue
        effect = float(ideo["eie_mean"]) - float(rep["eie_mean"])
        ci95 = math.sqrt(float(rep["eie_ci95"]) ** 2 + float(ideo["eie_ci95"]) ** 2)
        excludes_zero = abs(effect) > ci95
        cells.append(
            {
                **_regime_dict(rep),
                "rep_mean_eie": round(float(rep["eie_mean"]), 6),
                "ideo_mean_eie": round(float(ideo["eie_mean"]), 6),
                "effect": round(effect, 6),
                "ci95": round(ci95, 6),
                "effect_ci_low": round(effect - ci95, 6),
                "effect_ci_high": round(effect + ci95, 6),
                "excludes_zero": excludes_zero,
                "predicted_sign": "positive",
                "observed_sign": _sign_label(effect),
                "prediction_status": _prediction_status(effect, ci95),
            }
        )
    if not cells:
        raise ValueError("analytical predictions: no representative-vs-ideological cells")
    return cells


def _pre_post_cells(rows: Iterable[Mapping[str, object]]) -> list[dict[str, object]]:
    cells: list[dict[str, object]] = []
    for row in rows:
        delta = float(row["eie_mean"]) - float(row["mv_mean"])
        ci95 = math.sqrt(float(row["eie_ci95"]) ** 2 + float(row["mv_ci95"]) ** 2)
        cells.append(
            {
                **_regime_dict(row),
                "strategy": row["strategy"],
                "mv_mean": round(float(row["mv_mean"]), 6),
                "eie_mean": round(float(row["eie_mean"]), 6),
                "eie_minus_mv": round(delta, 6),
                "ci95": round(ci95, 6),
                "prediction_status": "descriptive_pre_post",
            }
        )
    return cells


def _monotone_checks(
    rows: Iterable[Mapping[str, object]],
    rep_cells: list[Mapping[str, object]],
) -> list[dict[str, object]]:
    checks: list[dict[str, object]] = []
    checks.append(
        _axis_check(
            rows=rep_cells,
            axis="bias_std",
            value_field="effect",
            group_fields=("panel_size", "mean_expertise", "n_items", "prevalence_a"),
            direction="nondecreasing",
            prediction="higher bias widens ideological-minus-representative EIE",
        )
    )
    representative_rows = [row for row in rows if row["strategy"] == REPRESENTATIVE_STRATEGY]
    checks.append(
        _axis_check(
            rows=representative_rows,
            axis="mean_expertise",
            value_field="eie_mean",
            group_fields=("panel_size", "bias_std", "n_items", "prevalence_a"),
            direction="nonincreasing",
            prediction="higher expertise lowers representative-sortition EIE",
        )
    )
    checks.append(
        {
            "axis": "panel_size",
            "direction": "descriptive",
            "prediction": (
                "larger panels reduce sampling variance but need not monotonically "
                "lower EIE under correlated bias"
            ),
            "n_comparisons": 0,
            "n_aligned": 0,
            "alignment_rate": None,
            "status": "not_asserted",
        }
    )
    return checks


def _axis_check(
    *,
    rows: Iterable[Mapping[str, object]],
    axis: str,
    value_field: str,
    group_fields: tuple[str, ...],
    direction: str,
    prediction: str,
) -> dict[str, object]:
    grouped: dict[tuple[object, ...], list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[field] for field in group_fields)].append(row)

    comparisons = 0
    aligned = 0
    examples: list[dict[str, object]] = []
    for key, group_rows in sorted(grouped.items()):
        ordered = sorted(group_rows, key=lambda row: float(row[axis]))
        if len(ordered) < 2:
            continue
        low = ordered[0]
        high = ordered[-1]
        low_value = float(low[value_field])
        high_value = float(high[value_field])
        ok = high_value >= low_value if direction == "nondecreasing" else high_value <= low_value
        comparisons += 1
        aligned += int(ok)
        if len(examples) < 8:
            examples.append(
                {
                    "group": dict(zip(group_fields, key, strict=True)),
                    "low_axis": low[axis],
                    "high_axis": high[axis],
                    "low_value": round(low_value, 6),
                    "high_value": round(high_value, 6),
                    "aligned": ok,
                }
            )
    return {
        "axis": axis,
        "direction": direction,
        "prediction": prediction,
        "n_comparisons": comparisons,
        "n_aligned": aligned,
        "alignment_rate": round(aligned / comparisons, 6) if comparisons else None,
        "status": "tested" if comparisons else "unavailable",
        "examples": examples,
    }


def _regime_key(row: Mapping[str, object]) -> tuple[object, ...]:
    return (
        row["panel_size"],
        row["mean_expertise"],
        row["expertise_heterogeneity"],
        row["bias_std"],
        row["n_items"],
        row["prevalence_a"],
        row["n_experts"],
    )


def _regime_dict(row: Mapping[str, object]) -> dict[str, object]:
    return {
        "panel_size": row["panel_size"],
        "mean_expertise": row["mean_expertise"],
        "expertise_heterogeneity": row["expertise_heterogeneity"],
        "bias_std": row["bias_std"],
        "n_items": row["n_items"],
        "prevalence_a": row["prevalence_a"],
        "n_experts": row["n_experts"],
        "n": row["n"],
    }


def _prediction_status(effect: float, ci95: float) -> str:
    if effect > 0.0 and abs(effect) > ci95:
        return "aligned_resolved"
    if effect > 0.0:
        return "aligned_uncertain"
    if abs(effect) <= ci95:
        return "zero_uncertain"
    return "misaligned_resolved"


def _sign_label(value: float) -> str:
    if value > 0.0:
        return "positive"
    if value < 0.0:
        return "negative"
    return "zero"


def _string(row: Mapping[str, str], field: str) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value:
        raise ValueError(f"aggregate row missing {field}")
    return value


def _float(row: Mapping[str, str], field: str) -> float:
    try:
        value = float(row[field])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"aggregate row {field} must be numeric") from exc
    if not math.isfinite(value):
        raise ValueError(f"aggregate row {field} must be finite")
    return value


def _non_negative_float(row: Mapping[str, str], field: str) -> float:
    value = _float(row, field)
    if value < 0.0:
        raise ValueError(f"aggregate row {field} must be non-negative")
    return value


def _int(row: Mapping[str, str], field: str) -> int:
    value = _float(row, field)
    if not float(value).is_integer():
        raise ValueError(f"aggregate row {field} must be an integer")
    return int(value)


__all__ = [
    "PREDICTION_RATIONALES",
    "build_analytical_predictions",
    "write_analytical_predictions",
]
