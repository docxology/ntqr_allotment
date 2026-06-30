"""Tests for the study-specific power-analysis bridge (:mod:`power_study`).

Discipline mirrors the sibling suites: every expected value is hand-derived or
read from the project's REAL sweep output (``output/data/sweep_results.json``),
never produced by the function under test, and every behavioural claim is paired
with a negative control whose assertion flips on a degenerate or broken input.

The real fixture has all four strategies at both panel sizes 3 and 6 (3 seeds per
cell), so the branches it cannot reach -- the ``rep_vs_ideo_power`` skip path when
a strategy is absent at a panel size, the ``seeds_for_80 is None`` CSV cell, and
the three ``_verdict`` arms -- are exercised with small hand-built row lists.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from ntqr_allotment.power_study import (
    POWER_CSV_COLUMNS,
    PowerRow,
    _fmt,
    _verdict,
    analyze,
    contrast_power,
    group_eie_by_strategy,
    load_trial_rows,
    rep_vs_ideo_power,
    strategy_power_matrix,
    write_power_table,
)

# The authoritative real sweep output shipped with the project.
SWEEP_JSON = Path(__file__).resolve().parents[1] / "output" / "data" / "sweep_results.json"


def _real_rows() -> list[dict]:
    """Load the real sweep rows, skipping the test if the fixture is absent."""
    if not SWEEP_JSON.exists():
        pytest.skip(f"real sweep fixture not present at {SWEEP_JSON}")
    return load_trial_rows(SWEEP_JSON)


def _row(strategy: str, panel_size: int, seed: int, eie_error: float) -> dict:
    """Minimal per-seed row carrying exactly the keys the module reads."""
    return {
        "strategy": strategy,
        "panel_size": panel_size,
        "seed": seed,
        "eie_error": eie_error,
    }


# --------------------------------------------------------------------------- #
# load_trial_rows
# --------------------------------------------------------------------------- #
def test_load_trial_rows_reads_real_sweep() -> None:
    rows = _real_rows()
    assert isinstance(rows, list)
    assert len(rows) > 0
    # Each row must carry the keys the downstream functions index.
    sample = rows[0]
    for key in ("strategy", "panel_size", "seed", "eie_error"):
        assert key in sample


def test_load_trial_rows_rejects_missing_rows_key(tmp_path: Path) -> None:
    path = tmp_path / "no_rows.json"
    path.write_text(json.dumps({"meta": "no rows here"}))
    with pytest.raises(ValueError, match="non-empty 'rows'"):
        load_trial_rows(path)


def test_load_trial_rows_rejects_empty_rows_list(tmp_path: Path) -> None:
    # NEGATIVE CONTROL: an empty list is as invalid as a missing key.
    path = tmp_path / "empty_rows.json"
    path.write_text(json.dumps({"rows": []}))
    with pytest.raises(ValueError, match="non-empty 'rows'"):
        load_trial_rows(path)


# --------------------------------------------------------------------------- #
# group_eie_by_strategy
# --------------------------------------------------------------------------- #
def test_group_eie_by_strategy_filters_panel_and_sorts_by_seed() -> None:
    rows = [
        _row("alpha", 3, seed=2, eie_error=0.20),
        _row("alpha", 3, seed=0, eie_error=0.10),
        _row("alpha", 6, seed=0, eie_error=0.99),  # wrong panel: excluded
        _row("beta", 3, seed=1, eie_error=0.50),
    ]
    groups = group_eie_by_strategy(rows, panel_size=3)
    assert set(groups) == {"alpha", "beta"}
    # Seed-sorted for determinism: seed 0 value precedes seed 2 value.
    assert groups["alpha"] == [0.10, 0.20]
    assert groups["beta"] == [0.50]
    # NEGATIVE CONTROL: the panel-6 row never leaks into the panel-3 grouping.
    assert 0.99 not in groups["alpha"]


def test_group_eie_by_strategy_empty_when_panel_absent() -> None:
    rows = [_row("alpha", 3, seed=0, eie_error=0.1)]
    assert group_eie_by_strategy(rows, panel_size=99) == {}


def test_group_eie_by_strategy_excludes_degenerate_sentinel() -> None:
    """NEGATIVE CONTROL: the -1.0 all-degenerate sentinel must never be pooled.

    ``eie_error`` is ``error_vs`` (non-negative by construction), so a negative
    value is the sweep's degeneracy sentinel, not a measurement. Pooling it would
    drag Cohen's d and the published means toward an impossible negative; this
    fails on the pre-fix code that called ``float(row["eie_error"])`` blindly.
    """
    rows = [
        _row("alpha", 3, seed=0, eie_error=0.40),
        _row("alpha", 3, seed=1, eie_error=-1.0),  # DEGENERATE sentinel: dropped
        _row("alpha", 3, seed=2, eie_error=0.60),
    ]
    groups = group_eie_by_strategy(rows, panel_size=3)
    assert groups["alpha"] == [0.40, 0.60]
    assert all(v >= 0.0 for v in groups["alpha"])  # no impossible negative pooled


# --------------------------------------------------------------------------- #
# _verdict  (all three branches)
# --------------------------------------------------------------------------- #
def test_verdict_significant_branch() -> None:
    # perm_p below alpha -> significant, regardless of the power flag.
    assert _verdict(perm_p=0.01, underpowered=True, alpha=0.05) == "significant"


def test_verdict_underpowered_null_branch() -> None:
    # Not significant AND underpowered -> the honest "we could not have seen it" verdict.
    assert _verdict(perm_p=0.50, underpowered=True, alpha=0.05) == "underpowered-null"


def test_verdict_well_powered_null_branch() -> None:
    # Not significant AND well-powered -> a genuine null the design could have detected.
    assert _verdict(perm_p=0.50, underpowered=False, alpha=0.05) == "well-powered-null"


def test_verdict_alpha_boundary_is_not_significant() -> None:
    # NEGATIVE CONTROL: the test is strict ``<`` -- p exactly at alpha is NOT significant.
    assert _verdict(perm_p=0.05, underpowered=False, alpha=0.05) == "well-powered-null"


# --------------------------------------------------------------------------- #
# contrast_power
# --------------------------------------------------------------------------- #
def test_contrast_power_separated_pair_is_significant_nonzero_d() -> None:
    # A clearly separated pair with 5 seeds per group. C(10,5)=252 distinct splits, so
    # the smallest attainable two-sided permutation p is ~2/252 ~ 0.008 -- below alpha
    # (a 3-vs-3 design cannot reach significance: its floor is ~2/20 = 0.1). This both
    # reaches the "significant" verdict and takes the d != 0.0 power branch.
    a = [0.00, 0.02, -0.02, 0.01, -0.01]
    b = [1.00, 1.02, 0.98, 1.01, 0.99]
    rowp = contrast_power(
        a,
        b,
        contrast="sep",
        group_a="A",
        group_b="B",
        panel_size=5,
        n_perm=2000,
        seed=0,
    )
    assert isinstance(rowp, PowerRow)
    assert rowp.n_per_group == 5
    assert rowp.observed_d != 0.0
    assert rowp.mean_a == pytest.approx(0.0, abs=1e-9)
    assert rowp.mean_b == pytest.approx(1.0, abs=1e-9)
    # The fully separated split is the most extreme labelling, so perm_p sits at the
    # design's floor, comfortably below alpha=0.05.
    assert rowp.perm_p < 0.05
    assert rowp.verdict == "significant"
    # NEGATIVE CONTROL: a non-significant verdict would never read "significant".
    assert rowp.verdict != "well-powered-null"


def test_contrast_power_constant_pair_has_zero_d_and_none_seed_budget() -> None:
    # Both groups constant (but at different levels): pooled variance is 0 -> d == 0.0.
    # That drives seeds_for_80 to None (no retrospective observed power is computed).
    a = [0.5, 0.5, 0.5]
    b = [0.9, 0.9, 0.9]
    rowp = contrast_power(
        a,
        b,
        contrast="flat",
        group_a="A",
        group_b="B",
        panel_size=3,
        n_perm=200,
        seed=1,
        alpha=0.05,
    )
    assert rowp.observed_d == 0.0
    assert not hasattr(rowp, "power_at_n")  # retrospective observed power is not reported
    assert rowp.seeds_for_80 is None  # diagnose_null returns None for a zero effect
    assert rowp.underpowered is True  # |0| < mde always


def test_contrast_power_is_deterministic() -> None:
    a = [0.1, 0.3, 0.2, 0.4]
    b = [0.9, 0.7, 1.1, 0.8]
    first = contrast_power(
        a, b, contrast="c", group_a="A", group_b="B", panel_size=4, n_perm=300, seed=7
    )
    second = contrast_power(
        a, b, contrast="c", group_a="A", group_b="B", panel_size=4, n_perm=300, seed=7
    )
    assert first == second
    # NEGATIVE CONTROL: a different permutation seed changes the (stochastic) p-value.
    other = contrast_power(
        a, b, contrast="c", group_a="A", group_b="B", panel_size=4, n_perm=300, seed=8
    )
    assert other.perm_p != first.perm_p


def test_contrast_power_rejects_empty_group() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        contrast_power(
            [], [1.0, 2.0], contrast="x", group_a="A", group_b="B", panel_size=3, seed=0
        )
    with pytest.raises(ValueError, match="non-empty"):
        contrast_power(
            [1.0, 2.0], [], contrast="x", group_a="A", group_b="B", panel_size=3, seed=0
        )


# --------------------------------------------------------------------------- #
# strategy_power_matrix
# --------------------------------------------------------------------------- #
def test_strategy_power_matrix_one_row_per_unordered_pair() -> None:
    rows = _real_rows()
    matrix = strategy_power_matrix(rows, panel_size=3, seed=0, n_perm=200)
    strategies = sorted(group_eie_by_strategy(rows, 3))
    expected_pairs = len(strategies) * (len(strategies) - 1) // 2
    assert len(matrix) == expected_pairs
    # Name-sorted determinism: every row's group_a precedes group_b alphabetically,
    # and the rows themselves follow combinations() over the sorted strategy names.
    for rowp in matrix:
        assert rowp.group_a < rowp.group_b
        assert rowp.panel_size == 3


def test_strategy_power_matrix_is_deterministic() -> None:
    rows = _real_rows()
    first = strategy_power_matrix(rows, panel_size=6, seed=3, n_perm=200)
    second = strategy_power_matrix(rows, panel_size=6, seed=3, n_perm=200)
    assert first == second


# --------------------------------------------------------------------------- #
# rep_vs_ideo_power  (including the absent-strategy skip branch)
# --------------------------------------------------------------------------- #
def test_rep_vs_ideo_power_real_fixture_has_both_panels() -> None:
    rows = _real_rows()
    result = rep_vs_ideo_power(rows, seed=100, n_perm=200)
    panels_present = sorted({int(r["panel_size"]) for r in rows})
    # The real fixture carries both strategies at every panel size -> one row each.
    assert [r.panel_size for r in result] == panels_present
    for rowp in result:
        assert rowp.group_a == "representative_sortition"
        assert rowp.group_b == "ideological_selection"
        assert rowp.contrast == f"rep_vs_ideo_p{rowp.panel_size}"


def test_rep_vs_ideo_power_skips_panel_missing_a_strategy() -> None:
    # Panel 3 has BOTH strategies; panel 6 is MISSING ideological_selection.
    rows = [
        _row("representative_sortition", 3, 0, 0.10),
        _row("representative_sortition", 3, 1, 0.12),
        _row("representative_sortition", 3, 2, 0.11),
        _row("ideological_selection", 3, 0, 0.30),
        _row("ideological_selection", 3, 1, 0.31),
        _row("ideological_selection", 3, 2, 0.29),
        _row("representative_sortition", 6, 0, 0.05),  # panel 6 lacks ideological
        _row("representative_sortition", 6, 1, 0.06),
    ]
    result = rep_vs_ideo_power(rows, seed=100, n_perm=200)
    # The skip/continue branch fired: only panel 3 produced a row.
    assert [r.panel_size for r in result] == [3]
    # NEGATIVE CONTROL: the incomplete panel 6 produced NO row.
    assert all(r.panel_size != 6 for r in result)


def test_rep_vs_ideo_power_skips_when_representative_absent() -> None:
    # Mirror case: a panel with ideological but no representative is also skipped.
    rows = [
        _row("ideological_selection", 3, 0, 0.30),
        _row("ideological_selection", 3, 1, 0.31),
        _row("random_selection", 3, 0, 0.40),
    ]
    assert rep_vs_ideo_power(rows, seed=100, n_perm=100) == []


# --------------------------------------------------------------------------- #
# analyze
# --------------------------------------------------------------------------- #
def test_analyze_orders_rep_vs_ideo_first_then_matrices() -> None:
    rows = _real_rows()
    results = analyze(SWEEP_JSON, seed=0, n_perm=200)
    panels = sorted({int(r["panel_size"]) for r in rows})
    rep_rows = rep_vs_ideo_power(rows, seed=100, n_perm=200)
    # The head of the result list is exactly the rep-vs-ideo block (one per panel).
    assert [r.contrast for r in results[: len(rep_rows)]] == [
        f"rep_vs_ideo_p{ps}" for ps in panels
    ]
    # The tail is the per-panel pairwise matrices, so the total is rep + sum(matrices).
    expected_total = len(rep_rows) + sum(
        len(strategy_power_matrix(rows, ps, seed=ps * 10, n_perm=200)) for ps in panels
    )
    assert len(results) == expected_total


def test_analyze_is_deterministic() -> None:
    if not SWEEP_JSON.exists():
        pytest.skip("real sweep fixture not present")
    first = analyze(SWEEP_JSON, seed=0, n_perm=200)
    second = analyze(SWEEP_JSON, seed=0, n_perm=200)
    assert first == second


# --------------------------------------------------------------------------- #
# write_power_table  (including the seeds_for_80-is-None empty-cell branch)
# --------------------------------------------------------------------------- #
def _power_row(*, contrast: str, seeds_for_80: int | None) -> PowerRow:
    """A hand-built PowerRow with a chosen seeds_for_80 to exercise the CSV branch."""
    return PowerRow(
        contrast=contrast,
        group_a="A",
        group_b="B",
        panel_size=3,
        n_per_group=3,
        mean_a=0.1,
        mean_b=0.2,
        observed_d=0.5,
        perm_p=0.33,
        mde_80=1.85,
        seeds_for_80=seeds_for_80,
        underpowered=True,
        verdict="underpowered-null",
    )


def test_write_power_table_writes_header_and_both_seed_cells(tmp_path: Path) -> None:
    results = [
        _power_row(contrast="has_budget", seeds_for_80=128),
        _power_row(contrast="no_budget", seeds_for_80=None),
    ]
    out = write_power_table(results, tmp_path / "nested" / "power.csv")
    assert out.exists()  # parent dirs were created

    with out.open(newline="") as handle:
        read = list(csv.reader(handle))
    assert tuple(read[0]) == POWER_CSV_COLUMNS
    body = {row[0]: row for row in read[1:]}
    seeds_idx = POWER_CSV_COLUMNS.index("seeds_for_80")
    # Non-None budget renders as the integer string...
    assert body["has_budget"][seeds_idx] == "128"
    # ...while a None budget renders as the empty-string branch (line 349).
    assert body["no_budget"][seeds_idx] == ""


def test_write_power_table_formats_floats_to_four_places(tmp_path: Path) -> None:
    results = [_power_row(contrast="fmt", seeds_for_80=10)]
    out = write_power_table(results, tmp_path / "fmt.csv")
    with out.open(newline="") as handle:
        rows = list(csv.reader(handle))
    body = rows[1]
    observed_idx = POWER_CSV_COLUMNS.index("observed_d")
    # observed_d == 0.5 formatted by _fmt -> "0.5000".
    assert body[observed_idx] == "0.5000"


# --------------------------------------------------------------------------- #
# _fmt
# --------------------------------------------------------------------------- #
def test_fmt_renders_four_decimal_places() -> None:
    assert _fmt(0.5) == "0.5000"
    assert _fmt(1.0 / 3.0) == "0.3333"
    # NEGATIVE CONTROL: it is NOT the bare repr -- trailing zeros are kept to 4 places.
    assert _fmt(2.0) == "2.0000"


# --- paired size contrast (regime+seed-controlled size effect) ----------------
def _full_row(strategy, panel_size, seed, me, bias, eie):
    return {
        "strategy": strategy, "panel_size": panel_size, "seed": seed,
        "mean_expertise": me, "bias_std": bias, "expertise_heterogeneity": 0.08,
        "n_items": 300, "prevalence_a": 0.5, "n_experts": 96,
        "n_trios": 4, "eie_error": eie, "mv_error": eie,
    }


def test_paired_size_contrast_pairs_within_cell_and_resolves_real_effect() -> None:
    """A constant +0.2 size effect (every matched cell) resolves; a null does not.

    Also confirms pairing is WITHIN (regime, seed): only cells present at both
    sizes contribute, and the degenerate sentinel is excluded.
    """
    from ntqr_allotment.power_study import paired_size_contrast

    rows = []
    for seed in range(10):
        for me in (0.6, 0.7):
            for bias in (0.1, 0.2):
                rows.append(_full_row("rise", 3, seed, me, bias, 0.30))
                rows.append(_full_row("rise", 6, seed, me, bias, 0.50))  # +0.20 every cell
                rows.append(_full_row("flat", 3, seed, me, bias, 0.30))
                rows.append(_full_row("flat", 6, seed, me, bias, 0.30))  # no change

    r = paired_size_contrast(rows, "rise", 3, 6)
    assert r["n_pairs"] == 40
    assert r["mean_diff"] == pytest.approx(0.20)
    assert r["ci_low"] == pytest.approx(0.20) and r["ci_high"] == pytest.approx(0.20)
    assert r["verdict"] == "resolved"

    f = paired_size_contrast(rows, "flat", 3, 6)
    assert f["mean_diff"] == pytest.approx(0.0)
    assert f["verdict"] == "within-noise"


def test_paired_size_contrast_excludes_degenerate_and_unmatched_cells() -> None:
    from ntqr_allotment.power_study import paired_size_contrast

    rows = [
        _full_row("s", 3, 0, 0.6, 0.1, 0.40),
        _full_row("s", 6, 0, 0.6, 0.1, 0.50),   # matched pair -> +0.10
        _full_row("s", 3, 1, 0.6, 0.1, -1.0),   # degenerate sentinel -> excluded
        _full_row("s", 6, 1, 0.6, 0.1, 0.50),   # its partner is now unmatched -> dropped
        _full_row("s", 3, 2, 0.6, 0.1, 0.40),   # size-6 missing -> unmatched -> dropped
    ]
    r = paired_size_contrast(rows, "s", 3, 6)
    assert r["n_pairs"] == 1
    assert r["mean_diff"] == pytest.approx(0.10)
