from __future__ import annotations

import csv
import json
from pathlib import Path

from ntqr_allotment.figures import (
    plot_alarm_cost_curve,
    plot_bloc_phase_diagram,
    plot_concentration_dial,
    plot_pre_post_ntqr_heatmap,
    plot_power_curve,
    plot_rep_vs_ideo_effect,
    plot_rep_vs_ideo_heatmap,
    plot_strategy_ranking,
    plot_theory_alignment_heatmap,
)


def _load_alarm_points(csv_path: Path) -> list[tuple[int, float]] | None:
    """Real (Q, seconds) points from the shipped benchmark CSV, if present."""
    if not csv_path.exists():
        return None
    points: list[tuple[int, float]] = []
    with csv_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            points.append((int(row["Q"]), float(row["seconds"])))
    return points or None


def _load_sweep_results(data_path: Path) -> dict[str, object]:
    if not data_path.exists():
        raise FileNotFoundError(
            f"Missing required data file: {data_path}. Regenerate it with "
            "`uv run python scripts/run_sweep.py`."
        )

    payload = json.loads(data_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"sweep results payload must be a JSON object: {data_path}")
    return payload


def _require_list(payload: dict[str, object], key: str) -> list[object]:
    value = payload.get(key)
    if not isinstance(value, list):
        raise ValueError(f"sweep results payload missing list field: {key}")
    return value


def _load_prediction_payload(data_path: Path) -> dict[str, object]:
    if not data_path.exists():
        raise FileNotFoundError(
            f"Missing required data file: {data_path}. Regenerate it with "
            "`uv run python scripts/run_analytical_predictions.py`."
        )
    payload = json.loads(data_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"analytical prediction payload must be a JSON object: {data_path}")
    return payload


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    data_path = root / "output" / "data" / "sweep_results.json"
    prediction_path = root / "output" / "data" / "analytical_predictions.json"
    figure_dir = root / "output" / "figures"
    payload = _load_sweep_results(data_path)
    predictions = _load_prediction_payload(prediction_path)
    alarm_points = _load_alarm_points(root / "output" / "data" / "alarm_timings.csv")
    bloc_summary_path = root / "output" / "data" / "bloc_phase_summary.json"
    if not bloc_summary_path.exists():
        raise FileNotFoundError(
            f"Missing required data file: {bloc_summary_path}. Regenerate it with "
            "`uv run python scripts/run_bloc_phase.py`."
        )
    bloc_summary = json.loads(bloc_summary_path.read_text(encoding="utf-8"))
    bloc_cells = _require_list(bloc_summary, "cells")
    concentration = bloc_summary.get("concentration")
    if not isinstance(concentration, dict) or "cells" not in concentration:
        raise ValueError(
            f"{bloc_summary_path} missing 'concentration' block. Regenerate with "
            "`uv run python scripts/run_bloc_phase.py`."
        )
    concentration_cells = concentration["cells"]

    alarm_kwargs = {"measured_points": alarm_points} if alarm_points else {}
    paths = [
        plot_strategy_ranking(
            _require_list(payload, "ranking"),
            figure_dir / "strategy_ranking.png",
        ),
        plot_power_curve(
            _require_list(payload, "rows"),
            figure_dir / "power_curve.png",
        ),
        plot_rep_vs_ideo_effect(
            _require_list(payload, "rep_vs_ideo"),
            figure_dir / "rep_vs_ideo_effect.png",
        ),
        plot_rep_vs_ideo_heatmap(
            _require_list(predictions, "rep_vs_ideo_cells"),
            figure_dir / "rep_vs_ideo_heatmap.png",
        ),
        plot_pre_post_ntqr_heatmap(
            _require_list(predictions, "pre_post_cells"),
            figure_dir / "pre_post_ntqr_heatmap.png",
        ),
        plot_theory_alignment_heatmap(
            predictions,
            figure_dir / "theory_vs_observed_alignment.png",
        ),
        plot_alarm_cost_curve(figure_dir / "alarm_cost_curve.png", **alarm_kwargs),
        plot_bloc_phase_diagram(
            bloc_cells,
            figure_dir / "bloc_phase_diagram.png",
        ),
        plot_concentration_dial(
            concentration_cells,
            figure_dir / "bloc_concentration_dial.png",
        ),
    ]

    for path in paths:
        print(path.resolve())


if __name__ == "__main__":
    main()
