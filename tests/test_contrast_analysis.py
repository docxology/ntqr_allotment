from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from ntqr_allotment.contrast_analysis import (
    build_analytical_predictions,
    write_analytical_predictions,
)


HEADER = [
    "profile_name",
    "config_hash",
    "strategy",
    "panel_size",
    "mean_expertise",
    "expertise_heterogeneity",
    "bias_std",
    "n_items",
    "prevalence_a",
    "n_experts",
    "n",
    "eie_mean",
    "eie_std",
    "eie_ci95",
    "mv_mean",
    "mv_std",
    "mv_ci95",
]


def _write_csv(path: Path) -> Path:
    rows = []
    for panel_size in (3, 6):
        for expertise in (0.68, 0.74):
            for bias in (0.1, 0.3):
                rep_eie = 0.4 - expertise / 2 + bias / 10 + panel_size / 100
                ideo_eie = rep_eie + 0.05 + bias / 4
                rows.append(
                    [
                        "manuscript_contrast",
                        "abc123",
                        "representative_sortition",
                        panel_size,
                        expertise,
                        0.08,
                        bias,
                        240,
                        0.5,
                        96,
                        12,
                        rep_eie,
                        0.01,
                        0.02,
                        rep_eie + 0.01,
                        0.01,
                        0.02,
                    ]
                )
                rows.append(
                    [
                        "manuscript_contrast",
                        "abc123",
                        "ideological_selection",
                        panel_size,
                        expertise,
                        0.08,
                        bias,
                        240,
                        0.5,
                        96,
                        12,
                        ideo_eie,
                        0.01,
                        0.02,
                        ideo_eie - 0.02,
                        0.01,
                        0.02,
                    ]
                )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(HEADER)
        writer.writerows(rows)
    return path


def test_build_analytical_predictions_emits_cell_and_alignment_schema(tmp_path: Path) -> None:
    payload = build_analytical_predictions(
        _write_csv(tmp_path / "sweep_aggregated.csv"),
        metadata={"profile_name": "manuscript_contrast"},
    )

    assert payload["schema_version"] == 1
    assert payload["metadata"]["profile_name"] == "manuscript_contrast"
    assert len(payload["rep_vs_ideo_cells"]) == 8
    assert len(payload["pre_post_cells"]) == 16
    assert {cell["predicted_sign"] for cell in payload["rep_vs_ideo_cells"]} == {
        "positive"
    }
    assert all(
        str(cell["prediction_status"]).startswith("aligned")
        for cell in payload["rep_vs_ideo_cells"]
    )
    checks = {check["axis"]: check for check in payload["monotone_checks"]}
    assert checks["bias_std"]["status"] == "tested"
    assert checks["mean_expertise"]["status"] == "tested"
    assert checks["panel_size"]["status"] == "not_asserted"


def test_write_analytical_predictions_round_trips_json(tmp_path: Path) -> None:
    output = write_analytical_predictions(
        _write_csv(tmp_path / "sweep_aggregated.csv"),
        tmp_path / "analytical_predictions.json",
        metadata={"profile_name": "manuscript_contrast"},
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["source"].endswith("sweep_aggregated.csv")
    assert payload["rep_vs_ideo_cells"][0]["effect"] > 0


def test_build_analytical_predictions_rejects_missing_aggregate_file(
    tmp_path: Path,
) -> None:
    with pytest.raises(FileNotFoundError):
        build_analytical_predictions(tmp_path / "missing.csv")
