from __future__ import annotations

import csv
import re
from pathlib import Path

import pytest

import json

from ntqr_allotment.manuscript_variables import (
    INDEPENDENCE_TOKEN_NAMES,
    compute_alarm_tokens,
    compute_bloc_phase_tokens,
    compute_cross_family_tokens,
    compute_extension_tokens,
    compute_independence_tokens,
    compute_postdoc_panel_tokens,
    compute_power_tokens,
    compute_threshold_tokens,
    compute_tokens,
    generate_variables,
    main,
)

TOKEN_PATTERN = re.compile(r"\{\{([A-Z0-9_]+)\}\}")

INDEPENDENCE_HEADER = [
    "rho",
    "strategy",
    "panel_size",
    "n_experts",
    "n_items",
    "n",
    "eie_mean",
    "eie_std",
    "eie_ci95",
    "corr_mean",
]


def _write_independence_csv(path: Path, rows: list[list[str]]) -> Path:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(INDEPENDENCE_HEADER)
        writer.writerows(rows)
    return path

CSV_HEADER = [
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


def _write_csv(path: Path, rows: list[list[str]]) -> Path:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(CSV_HEADER)
        writer.writerows(rows)
    return path


def _bloc_summary_fixture() -> dict:
    strategies = [
        ("expertise_threshold", 0.04, 0.09, 0.01, 0.02),
        ("representative_sortition", 0.15, 0.17, 0.01, 0.02),
        ("random_selection", 0.16, 0.20, 0.01, 0.06),
        ("ideological_selection", 0.16, 0.27, 0.01, 0.13),
    ]

    def cells(scale: float = 1.0) -> list[dict]:
        out = []
        for name, lo, hi, clo, chi in strategies:
            for rho, eie, corr in ((0.0, lo, clo), (0.9, hi * scale, chi)):
                out.append(
                    {"strategy": name, "bloc_correlation": rho, "n": 200,
                     "eie_mean": eie, "eie_ci95": 0.01, "corr_mean": corr}
                )
        return out

    return {
        "strategies": [s[0] for s in strategies],
        "cells": cells(),
        "headline": {"rho_lo": 0.0, "rho_hi": 0.9, "sep_lo": 0.0, "sep_hi": 0.10,
                     "representative_lo": 0.15, "representative_hi": 0.17,
                     "ideological_lo": 0.16, "ideological_hi": 0.27,
                     "random_lo": 0.16, "random_hi": 0.20},
        "robustness": {"rho_hi": 0.9, "n_paired": 288, "n_ideo_gt_rep": 270,
                       "frac_ideo_gt_rep": 0.9375, "paired_mean_diff": 0.10, "paired_ci95": 0.02},
        "negative_control": {
            "axis": "expertise_tier",
            "cells": cells(),
            "headline": {"rho_lo": 0.0, "rho_hi": 0.9, "sep_lo": 0.0, "sep_hi": 0.05,
                         "representative_lo": 0.15, "representative_hi": 0.19,
                         "ideological_lo": 0.16, "ideological_hi": 0.21,
                         "random_lo": 0.16, "random_hi": 0.20},
        },
        "concentration": {
            "bloc_correlation": 0.9,
            "cells": [
                {"concentration": 0.0, "n": 100, "eie_mean": 0.16, "eie_ci95": 0.01, "corr_mean": 0.02},
                {"concentration": 1.0, "n": 100, "eie_mean": 0.25, "eie_ci95": 0.01, "corr_mean": 0.13},
            ],
            "eie_balanced": 0.16,
            "eie_concentrated": 0.25,
            "monotone_increasing_fraction": 1.0,
        },
    }


def test_compute_bloc_phase_tokens_surfaces_phase_and_negative_control(tmp_path: Path) -> None:
    summary_path = tmp_path / "bloc_phase_summary.json"
    summary_path.write_text(json.dumps(_bloc_summary_fixture()))
    tokens = compute_bloc_phase_tokens(summary_path)
    assert tokens["BLOC_RHO_HI"] == "0.90"
    assert tokens["BLOC_N_RHO_LEVELS"] == "2"
    assert tokens["BLOC_REPRESENTATIVE_SORTITION_HI"] == "0.170"
    assert tokens["BLOC_IDEOLOGICAL_SELECTION_HI"] == "0.270"
    assert tokens["BLOC_CORR_IDEOLOGICAL_SELECTION_HI"] == "0.130"
    assert tokens["BLOC_SEP_HI"] == "0.100"
    assert tokens["BLOC_ROBUST_FRAC"] == "270/288"
    # Negative control: representative degrades under the orthogonal axis
    # (BLOC_CTRL_REP_HI reads from the control headline, where rep climbs to 0.19).
    assert tokens["BLOC_CTRL_AXIS"] == "expertise_tier"
    assert tokens["BLOC_CTRL_REP_HI"] == "0.190"
    # Concentration dial.
    assert tokens["BLOC_DIAL_BALANCED"] == "0.160"
    assert tokens["BLOC_DIAL_CONCENTRATED"] == "0.250"
    assert tokens["BLOC_DIAL_N_LEVELS"] == "2"


def test_compute_bloc_phase_tokens_raises_for_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        compute_bloc_phase_tokens(tmp_path / "absent.json")


def test_compute_tokens_ranks_strategies_by_weighted_mean_and_effect(
    tmp_path: Path,
) -> None:
    csv_path = _write_csv(
        tmp_path / "fixture_weighted.csv",
        [
            [
                "representative_sortition",
                "3",
                "0.72",
                "0.08",
                "0.3",
                "150",
                "0.5",
                "48",
                "10",
                "0.400",
                "0.0",
                "0.100",
                "0.0",
                "0.0",
                "0.0",
            ],
            [
                "representative_sortition",
                "6",
                "0.72",
                "0.08",
                "0.3",
                "150",
                "0.5",
                "48",
                "90",
                "0.100",
                "0.0",
                "0.020",
                "0.0",
                "0.0",
                "0.0",
            ],
            [
                "ideological_selection",
                "3",
                "0.72",
                "0.08",
                "0.3",
                "150",
                "0.5",
                "48",
                "90",
                "0.200",
                "0.0",
                "0.030",
                "0.0",
                "0.0",
                "0.0",
            ],
            [
                "ideological_selection",
                "6",
                "0.72",
                "0.08",
                "0.3",
                "150",
                "0.5",
                "48",
                "10",
                "0.300",
                "0.0",
                "0.040",
                "0.0",
                "0.0",
                "0.0",
            ],
        ],
    )

    tokens = compute_tokens(csv_path)

    assert tokens["RANK_BEST_STRATEGY"] == "representative sortition"
    assert tokens["RANK_WORST_STRATEGY"] == "ideological selection"
    assert tokens["RANK_REPRESENTATIVE_SORTITION_EIE"] == "0.130"
    assert tokens["RANK_IDEOLOGICAL_SELECTION_EIE"] == "0.210"
    assert tokens["REP_VS_IDEO_P6_EFFECT"] == "0.200"
    assert tokens["REP_VS_IDEO_P6_VERDICT"] == "supported"
def test_compute_tokens_excludes_degenerate_rows(tmp_path: Path) -> None:
    csv_path = _write_csv(
        tmp_path / "fixture_degenerate.csv",
        [
            [
                "representative_sortition",
                "3",
                "0.72",
                "0.08",
                "0.3",
                "150",
                "0.5",
                "48",
                "10",
                "0.400",
                "0.0",
                "0.100",
                "0.0",
                "0.0",
                "0.0",
            ],
            [
                "representative_sortition",
                "6",
                "0.72",
                "0.08",
                "0.3",
                "150",
                "0.5",
                "48",
                "90",
                "0.100",
                "0.0",
                "0.020",
                "0.0",
                "0.0",
                "0.0",
            ],
            [
                "representative_sortition",
                "6",
                "0.72",
                "0.08",
                "0.3",
                "150",
                "0.5",
                "48",
                "1000",
                "-1.0",
                "0.0",
                "9.999",
                "0.0",
                "0.0",
                "0.0",
            ],
            [
                "ideological_selection",
                "3",
                "0.72",
                "0.08",
                "0.3",
                "150",
                "0.5",
                "48",
                "90",
                "0.200",
                "0.0",
                "0.030",
                "0.0",
                "0.0",
                "0.0",
            ],
            [
                "ideological_selection",
                "6",
                "0.72",
                "0.08",
                "0.3",
                "150",
                "0.5",
                "48",
                "10",
                "0.300",
                "0.0",
                "0.040",
                "0.0",
                "0.0",
                "0.0",
            ],
        ],
    )

    tokens = compute_tokens(csv_path)

    assert tokens["POWER_REPRESENTATIVE_SORTITION_SIZE6"] == "0.100"
    assert tokens["RANK_REPRESENTATIVE_SORTITION_EIE"] == "0.130"
def test_compute_tokens_raises_for_missing_files_and_columns(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing.csv"
    with pytest.raises(FileNotFoundError):
        compute_tokens(missing_path)

    invalid_csv = tmp_path / "missing_column.csv"
    invalid_csv.write_text(
        "strategy,panel_size,eie_ci95,n\nrepresentative_sortition,3,0.1,10\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="eie_mean"):
        compute_tokens(invalid_csv)


def _producible_tokens(repo_root: Path) -> set[str]:
    """Every token the render actually injects — the canonical honesty backing set.

    Mirrors ``generate_variables`` exactly (sweep + alarm + independence + extension),
    so the orphan gate checks prose against precisely what the render hydrates.
    """
    return set(generate_variables(repo_root))


def _scan_tokens(text: str) -> set[str]:
    return set(TOKEN_PATTERN.findall(text))


def test_no_orphan_token_in_manuscript() -> None:
    """Honesty gate: every {{TOKEN}} in prose must be backed by a producer.

    Producible-but-unused tokens are fine; orphan prose tokens are not.
    """
    repo_root = Path(__file__).resolve().parents[1]
    manuscript_tokens: set[str] = set()
    for manuscript_path in sorted((repo_root / "manuscript").glob("*.md")):
        manuscript_tokens |= _scan_tokens(manuscript_path.read_text(encoding="utf-8"))

    producible = _producible_tokens(repo_root)
    orphans = manuscript_tokens - producible
    assert orphans == set(), f"orphan tokens not backed by any producer: {sorted(orphans)}"


def test_no_orphan_gate_catches_a_fake_token() -> None:
    """Negative control: a synthetic {{FAKE_TOKEN}} must register as an orphan."""
    repo_root = Path(__file__).resolve().parents[1]
    producible = _producible_tokens(repo_root)

    synthetic_prose = "Recovery error was {{RANK_BEST_STRATEGY}} and {{FAKE_TOKEN}}."
    scanned = _scan_tokens(synthetic_prose)
    orphans = scanned - producible

    assert "FAKE_TOKEN" in orphans
    assert "RANK_BEST_STRATEGY" not in orphans


def test_results_source_ledger_is_not_blanket_sweep_only() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    text = (repo_root / "manuscript" / "03_results.md").read_text(encoding="utf-8")

    assert "All numbers below are emitted from the aggregated sweep" not in text
    assert "postdoc_panel_results.json" in text
    assert "postdoc_panel_alignment.json" in text
    assert "alarm_timings.csv" in text


def test_postdoc_figures_are_standalone_blocks() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    text = (repo_root / "manuscript" / "03_results.md").read_text(encoding="utf-8")
    figure_lines = [
        line for line in text.splitlines() if "#fig:postdoc" in line
    ]

    assert len(figure_lines) == 3
    assert all(line.startswith("![") for line in figure_lines)


def test_manuscript_avoids_live_source_overreach() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    discussion = (repo_root / "manuscript" / "04_discussion.md").read_text(
        encoding="utf-8"
    )

    assert "source of every number here" not in discussion
    assert "separate live artifacts" in discussion


def test_postdoc_live_caveat_is_single_model_not_family_comparison() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    text = (repo_root / "manuscript" / "03_results.md").read_text(encoding="utf-8")
    rendered = (repo_root / "output" / "manuscript" / "03_results.md").read_text(
        encoding="utf-8"
    )

    assert "sign-stability across re-runs is **untested**" not in text
    assert "not a model-family comparison" in " ".join(text.split())
    assert "not a model-family comparison" in " ".join(rendered.split())
    assert "not a human-review validation" in text


def test_independence_token_name_constant_matches_producer() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    produced = set(
        compute_independence_tokens(repo_root / "output" / "data" / "independence_sweep.csv")
    )
    assert produced == set(INDEPENDENCE_TOKEN_NAMES)


def test_compute_independence_tokens_values_computed_by_hand(tmp_path: Path) -> None:
    """Expected values derived BY HAND from a tiny fixture, not from the producer.

    Two rho levels, two cells each (one degenerate cell dropped). By hand:
      usable rows (corr, eie):
        rho=0.0: (0.10, 0.40), (0.20, 0.60)  -> mean corr 0.15, mean eie 0.50
        rho=0.8: (0.50, 0.30), (0.70, 0.50)  -> mean corr 0.60, mean eie 0.40
      OLS slope of eie on corr over all four points:
        xs=[0.10,0.20,0.50,0.70] mean_x=0.375
        ys=[0.40,0.60,0.30,0.50] mean_y=0.45
        Sxy = (-0.275)(-0.05)+(-0.175)(0.15)+(0.125)(-0.15)+(0.325)(0.05)
            = 0.01375 - 0.02625 - 0.01875 + 0.01625 = -0.015
        Sxx = 0.075625+0.030625+0.015625+0.105625 = 0.2275
        slope = -0.015 / 0.2275 = -0.065934...  -> "-0.066"
    """
    csv_path = _write_independence_csv(
        tmp_path / "indep.csv",
        [
            ["0.0", "representative_sortition", "3", "24", "120", "3", "0.40", "0.0", "0.0", "0.10"],
            ["0.0", "ideological_selection", "3", "24", "120", "3", "0.60", "0.0", "0.0", "0.20"],
            # degenerate cell: dropped (eie_mean == -1.0)
            ["0.0", "random_selection", "3", "24", "120", "0", "-1.0", "0.0", "0.0", "0.05"],
            ["0.8", "representative_sortition", "3", "24", "120", "3", "0.30", "0.0", "0.0", "0.50"],
            ["0.8", "ideological_selection", "3", "24", "120", "3", "0.50", "0.0", "0.0", "0.70"],
        ],
    )

    tokens = compute_independence_tokens(csv_path)

    assert tokens["CORR_AT_RHO0"] == "0.150"
    assert tokens["CORR_AT_RHO_HIGH"] == "0.600"
    assert tokens["EIE_AT_RHO0"] == "0.500"
    assert tokens["EIE_AT_RHO_HIGH"] == "0.400"
    assert tokens["TOLERANCE_SLOPE"] == "-0.066"
    # Grid dims are read (modal) from the n_experts/n_items columns, not hardcoded.
    assert tokens["INDEP_N_EXPERTS"] == "24"
    assert tokens["INDEP_N_ITEMS"] == "120"


def test_compute_independence_tokens_different_fixture_different_value(tmp_path: Path) -> None:
    """A DIFFERENT fixture must yield a DIFFERENT token value (positive slope here).

    By hand, two points (corr, eie): (0.10, 0.20), (0.40, 0.80).
      slope = (0.80-0.20)/(0.40-0.10) = 0.60/0.30 = 2.000 -> "2.000"
      CORR_AT_RHO0 = 0.10, CORR_AT_RHO_HIGH = 0.40
    """
    csv_path = _write_independence_csv(
        tmp_path / "indep2.csv",
        [
            ["0.0", "representative_sortition", "3", "24", "120", "3", "0.20", "0.0", "0.0", "0.10"],
            ["0.9", "representative_sortition", "3", "24", "120", "3", "0.80", "0.0", "0.0", "0.40"],
        ],
    )

    tokens = compute_independence_tokens(csv_path)

    assert tokens["TOLERANCE_SLOPE"] == "2.000"
    assert tokens["CORR_AT_RHO0"] == "0.100"
    assert tokens["CORR_AT_RHO_HIGH"] == "0.400"


def test_compute_independence_tokens_raises_missing_file_and_column(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        compute_independence_tokens(tmp_path / "absent.csv")

    no_corr = tmp_path / "no_corr.csv"
    no_corr.write_text("rho,eie_mean,n\n0.0,0.4,3\n", encoding="utf-8")
    with pytest.raises(ValueError, match="corr_mean"):
        compute_independence_tokens(no_corr)


def test_compute_independence_tokens_raises_when_all_degenerate(tmp_path: Path) -> None:
    csv_path = _write_independence_csv(
        tmp_path / "all_degen.csv",
        [
            ["0.0", "representative_sortition", "3", "24", "120", "0", "-1.0", "0.0", "0.0", "0.05"],
            ["0.9", "representative_sortition", "3", "24", "120", "0", "-1.0", "0.0", "0.0", "0.12"],
        ],
    )
    with pytest.raises(ValueError, match="No non-degenerate independence rows"):
        compute_independence_tokens(csv_path)


def test_compute_independence_tokens_raises_on_zero_correlation_variance(tmp_path: Path) -> None:
    csv_path = _write_independence_csv(
        tmp_path / "flat_corr.csv",
        [
            ["0.0", "representative_sortition", "3", "24", "120", "3", "0.40", "0.0", "0.0", "0.30"],
            ["0.9", "representative_sortition", "3", "24", "120", "3", "0.80", "0.0", "0.0", "0.30"],
        ],
    )
    with pytest.raises(ValueError, match="positive variance"):
        compute_independence_tokens(csv_path)


def test_compute_independence_tokens_raises_on_bad_float(tmp_path: Path) -> None:
    csv_path = _write_independence_csv(
        tmp_path / "bad_float.csv",
        [
            ["0.0", "representative_sortition", "3", "24", "120", "3", "not_a_float", "0.0", "0.0", "0.30"],
        ],
    )
    with pytest.raises(ValueError, match="eie_mean"):
        compute_independence_tokens(csv_path)


def test_compute_tokens_raises_on_empty_strategy(tmp_path: Path) -> None:
    csv_path = _write_csv(
        tmp_path / "empty_strategy.csv",
        [
            ["", "3", "0.72", "0.08", "0.3", "150", "0.5", "48", "10", "0.4", "0.0", "0.1", "0.0", "0.0", "0.0"],
        ],
    )
    with pytest.raises(ValueError, match="strategy"):
        compute_tokens(csv_path)


def test_compute_tokens_raises_on_bad_int(tmp_path: Path) -> None:
    csv_path = _write_csv(
        tmp_path / "bad_int.csv",
        [
            ["representative_sortition", "x", "0.72", "0.08", "0.3", "150", "0.5", "48", "10", "0.4", "0.0", "0.1", "0.0", "0.0", "0.0"],
        ],
    )
    with pytest.raises(ValueError, match="panel_size"):
        compute_tokens(csv_path)


def test_compute_tokens_raises_when_all_rows_degenerate(tmp_path: Path) -> None:
    csv_path = _write_csv(
        tmp_path / "all_sentinel.csv",
        [
            ["representative_sortition", "3", "0.72", "0.08", "0.3", "150", "0.5", "48", "10", "-1.0", "0.0", "0.1", "0.0", "0.0", "0.0"],
        ],
    )
    with pytest.raises(ValueError, match="No non-degenerate rows"):
        compute_tokens(csv_path)


def test_compute_tokens_raises_on_zero_sample_weight(tmp_path: Path) -> None:
    """A finite eie row with n=0 reaches the weighted-mean zero-weight guard."""
    csv_path = _write_csv(
        tmp_path / "zero_weight.csv",
        [
            ["representative_sortition", "3", "0.72", "0.08", "0.3", "150", "0.5", "48", "0", "0.4", "0.0", "0.1", "0.0", "0.0", "0.0"],
            ["representative_sortition", "6", "0.72", "0.08", "0.3", "150", "0.5", "48", "0", "0.1", "0.0", "0.02", "0.0", "0.0", "0.0"],
            ["ideological_selection", "3", "0.72", "0.08", "0.3", "150", "0.5", "48", "0", "0.2", "0.0", "0.03", "0.0", "0.0", "0.0"],
            ["ideological_selection", "6", "0.72", "0.08", "0.3", "150", "0.5", "48", "0", "0.3", "0.0", "0.04", "0.0", "0.0", "0.0"],
        ],
    )
    with pytest.raises(ValueError, match="no positive sample weight"):
        compute_tokens(csv_path)


def test_compute_tokens_contract_mismatch_raises(tmp_path: Path) -> None:
    """Only one strategy present: REP_VS_IDEO requires both -> require_rows fails.

    This exercises the missing-rows detail path before the contract check, proving
    the strict contract guards an incomplete strategy set.
    """
    csv_path = _write_csv(
        tmp_path / "one_strategy.csv",
        [
            ["representative_sortition", "3", "0.72", "0.08", "0.3", "150", "0.5", "48", "10", "0.4", "0.0", "0.1", "0.0", "0.0", "0.0"],
            ["representative_sortition", "6", "0.72", "0.08", "0.3", "150", "0.5", "48", "10", "0.1", "0.0", "0.02", "0.0", "0.0", "0.0"],
        ],
    )
    with pytest.raises(ValueError, match="ideological_selection"):
        compute_tokens(csv_path)


def test_grid_tokens_fall_back_when_columns_absent(tmp_path: Path) -> None:
    """The base CSV_HEADER lacks n_items/n_experts columns; grid tokens fall back."""
    csv_path = _write_csv(
        tmp_path / "no_grid_cols.csv",
        [
            ["representative_sortition", "3", "0.72", "0.08", "0.3", "150", "0.5", "48", "10", "0.4", "0.0", "0.1", "0.0", "0.0", "0.0"],
            ["representative_sortition", "6", "0.72", "0.08", "0.3", "150", "0.5", "48", "90", "0.1", "0.0", "0.02", "0.0", "0.0", "0.0"],
            ["ideological_selection", "3", "0.72", "0.08", "0.3", "150", "0.5", "48", "90", "0.2", "0.0", "0.03", "0.0", "0.0", "0.0"],
            ["ideological_selection", "6", "0.72", "0.08", "0.3", "150", "0.5", "48", "10", "0.3", "0.0", "0.04", "0.0", "0.0", "0.0"],
        ],
    )
    tokens = compute_tokens(csv_path)
    # CSV_HEADER has no n_items/n_experts columns -> modal fallback to GRID defaults.
    assert tokens["N_EXPERTS"] == "48"
    assert tokens["N_ITEMS"] == "150"
    # N_SEEDS is the max per-cell n in the CSV.
    assert tokens["N_SEEDS"] == "90"


def test_main_prints_real_tokens(capsys: pytest.CaptureFixture[str]) -> None:
    main()
    captured = capsys.readouterr()
    assert "RANK_BEST_STRATEGY" in captured.out


# --- Session 6: extension token producers (power / cross-family / threshold) ---

_REPO = Path(__file__).resolve().parents[1]
_DATA = _REPO / "output" / "data"


def test_compute_power_tokens_emits_design_budget_keys() -> None:
    tokens = compute_power_tokens(_DATA / "sweep_results.json")
    assert {
        "POWER_TOTAL_CONTRASTS",
        "POWER_WELL_POWERED_COUNT",
        "POWER_UNDERPOWERED_COUNT",
        "POWER_MDE80",
        "POWER_MIN_SEEDS_FOR_80",
        "POWER_MAX_SEEDS_FOR_80",
    } <= set(tokens)
    # well + underpowered == total (recomputed, not hardcoded)
    assert int(tokens["POWER_WELL_POWERED_COUNT"]) + int(
        tokens["POWER_UNDERPOWERED_COUNT"]
    ) == int(tokens["POWER_TOTAL_CONTRASTS"])


def test_compute_cross_family_tokens_from_live_artifact() -> None:
    tokens = compute_cross_family_tokens(_DATA / "cross_family_results.json")
    assert tokens["CROSS_FAMILY_LABEL"]
    assert "CROSS_FAMILY_DELTA" in tokens
    assert int(tokens["CROSS_FAMILY_N_SAME"]) >= 1
    assert int(tokens["CROSS_FAMILY_N_CROSS"]) >= 1
    assert int(tokens["CROSS_FAMILY_N_ITEMS"]) > 0
    assert int(tokens["CROSS_FAMILY_NUM_PREDICT"]) > 0
    assert tokens["CROSS_FAMILY_LIVE_LABEL"] == "live Ollama"
    assert "digest" in tokens["CROSS_FAMILY_PROVENANCE_SUMMARY"]


def test_compute_cross_family_tokens_missing_artifact_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="run_cross_family"):
        compute_cross_family_tokens(tmp_path / "absent.json")


def test_compute_threshold_tokens_emits_separation_verdict() -> None:
    tokens = compute_threshold_tokens(_DATA / "sweep_results.json")
    assert tokens["THRESHOLD_SORTITION_VERDICT"] in {"separated", "overlapping"}
    assert "THRESHOLD_SORTITION_DELTA" in tokens
    assert "THRESHOLD_MEAN" in tokens
    assert "SORTITION_MEAN" in tokens


def test_compute_threshold_tokens_missing_strategy_raises(tmp_path: Path) -> None:
    bad = tmp_path / "rows.json"
    bad.write_text(
        '{"rows": [{"strategy": "random_selection", "panel_size": 3, '
        '"seed": 0, "eie_error": 0.2}]}'
    )
    with pytest.raises(ValueError, match="need both"):
        compute_threshold_tokens(bad)


def test_compute_extension_tokens_unions_all_three() -> None:
    tokens = compute_extension_tokens(_REPO)
    assert "POWER_TOTAL_CONTRASTS" in tokens
    assert "CROSS_FAMILY_DELTA" in tokens
    assert "THRESHOLD_SORTITION_VERDICT" in tokens


def test_compute_postdoc_panel_tokens_from_live_artifact() -> None:
    tokens = compute_postdoc_panel_tokens(
        _DATA / "postdoc_panel_results.json",
        _DATA / "postdoc_panel_alignment.json",
    )

    assert tokens["POSTDOC_MODEL"] == "gemma3:4b"
    assert len(tokens["POSTDOC_MODEL_DIGEST"]) == 12
    assert int(tokens["POSTDOC_N_APPLICATIONS"]) > 0
    assert int(tokens["POSTDOC_N_REVIEWERS"]) > 0
    assert tokens["POSTDOC_LIVE_LABEL"] == "live Ollama"
    assert "POSTDOC_ALIGNMENT_RATE" in tokens


def test_compute_postdoc_panel_tokens_missing_artifact_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="run_postdoc_panel"):
        compute_postdoc_panel_tokens(tmp_path / "absent.json")


def test_compute_alarm_tokens_from_benchmark_csv() -> None:
    tokens = compute_alarm_tokens(_DATA / "alarm_timings.csv")
    assert set(tokens) == {"ALARM_Q20_S", "ALARM_Q50_S", "ALARM_Q100_S", "ALARM_MAX_Q"}
    # Values are the measured seconds from the shipped CSV (1-decimal), not literals.
    assert float(tokens["ALARM_Q20_S"]) < float(tokens["ALARM_Q50_S"]) < float(
        tokens["ALARM_Q100_S"]
    )
    assert tokens["ALARM_MAX_Q"] == "30"


def test_compute_alarm_tokens_missing_csv_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="bench_alarm"):
        compute_alarm_tokens(tmp_path / "absent.csv")


def test_compute_alarm_tokens_incomplete_csv_raises(tmp_path: Path) -> None:
    bad = tmp_path / "alarm_timings.csv"
    bad.write_text("Q,seconds,misaligned\n20,0.5,False\n")  # missing 50, 100
    with pytest.raises(ValueError, match="missing Q"):
        compute_alarm_tokens(bad)


def test_generate_variables_unions_every_producer() -> None:
    tokens = generate_variables(_REPO)
    for key in ("RANK_BEST_STRATEGY", "ALARM_Q100_S", "TOLERANCE_VERDICT",
                "POSTDOC_MODEL", "POWER_TOTAL_CONTRASTS", "THRESHOLD_SORTITION_VERDICT"):
        assert key in tokens


def test_compute_alarm_tokens_uses_csv_value_not_hardcode(tmp_path: Path) -> None:
    """NEXT-1 regression: ALARM_Q100_S must come from the CSV, never the old 199.0 literal."""
    csv = tmp_path / "alarm_timings.csv"
    csv.write_text("Q,seconds,misaligned\n20,0.5,False\n50,7.0,False\n100,77.7,False\n")
    tokens = compute_alarm_tokens(csv)
    assert tokens["ALARM_Q100_S"] == "77.7"  # the fixture value, not 199.0
    assert tokens["ALARM_Q20_S"] == "0.5"
    assert tokens["ALARM_Q100_S"] != "199.0"


def test_compute_power_tokens_emits_significant_count() -> None:
    tokens = compute_power_tokens(_DATA / "sweep_results.json")
    assert "POWER_SIGNIFICANT_COUNT" in tokens
    # significant <= total, and consistent with integer parsing.
    assert 0 <= int(tokens["POWER_SIGNIFICANT_COUNT"]) <= int(tokens["POWER_TOTAL_CONTRASTS"])


def test_compute_cross_family_multiseed_tokens_from_fixture(tmp_path: Path) -> None:
    from ntqr_allotment.manuscript_variables import compute_cross_family_multiseed_tokens

    art = tmp_path / "cross_family_multiseed.json"
    art.write_text(
        '{"n_runs": 4, "mean_delta": -0.031, "std_delta": 0.012, '
        '"sign_stability": 0.75, "min_delta": -0.05, "max_delta": 0.004, '
        '"n_items": 150, "num_predict": 1, '
        '"judge_provenance": [{"model": "qwen2.5:3b", "family": "qwen2", '
        '"digest": "abcdef123456"}], '
        '"deltas": [-0.05, -0.04, -0.03, 0.004]}'
    )
    tokens = compute_cross_family_multiseed_tokens(art)
    assert tokens["CROSS_FAMILY_MS_RUNS"] == "4"
    assert tokens["CROSS_FAMILY_MS_SIGN_STABILITY"] == "0.750"
    assert float(tokens["CROSS_FAMILY_MS_MIN_DELTA"]) <= float(tokens["CROSS_FAMILY_MS_MAX_DELTA"])
    assert tokens["CROSS_FAMILY_MS_DELTA_CI95"].startswith("[")  # bootstrap CI present
    assert tokens["CROSS_FAMILY_MS_NEGATIVE_RUNS"] == "3"
    assert tokens["CROSS_FAMILY_MS_SIGN_TEST_P"] == "0.625"
    assert tokens["CROSS_FAMILY_MS_N_ITEMS"] == "150"
    assert tokens["CROSS_FAMILY_MS_NUM_PREDICT"] == "1"
    assert "abcdef12" in tokens["CROSS_FAMILY_MS_PROVENANCE_SUMMARY"]


def test_compute_cross_family_multiseed_tokens_missing_raises(tmp_path: Path) -> None:
    from ntqr_allotment.manuscript_variables import compute_cross_family_multiseed_tokens

    with pytest.raises(FileNotFoundError, match="run_cross_family_multiseed"):
        compute_cross_family_multiseed_tokens(tmp_path / "absent.json")
