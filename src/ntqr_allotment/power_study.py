"""Apply the power-analysis toolkit to the study's own real sweep output.

This is the bridge between :mod:`ntqr_allotment.power_analysis` (general, study-agnostic
statistics) and the concrete contrasts this paper reports. It reads the per-seed sweep
rows that ``scripts/run_sweep.py`` already wrote (``output/data/sweep_results.json``,
key ``"rows"``) and, for every strategy contrast, computes:

* the **observed standardized effect** (Cohen's d) between the two strategies' per-seed
  ``eie_error`` values,
* a **distribution-free permutation p-value** on the same data (no normality assumed),
* the **analytic power** the design had to detect that observed effect,
* the **minimum detectable effect** (MDE) at the design's observation count, and
* the **per-group observations needed** to reach 80% power for an effect of the
  observed size.

Each per-strategy group pools every non-degenerate sweep row at a panel size,
**marginalized across the regime grid** (mean-expertise x bias-spread cells over
all seeds). The per-group observation count is therefore the number of
regime-grid cells, not the seed count; it is a marginal-over-regimes contrast
(conservative: ignoring the blocking inflates the pooled SD). The degenerate
sentinel (``eie_error == -1.0`` for an all-degenerate panel) is excluded here
exactly as :func:`ntqr_allotment.sweeps.aggregate` does, so an impossible
negative value never enters Cohen's d, the permutation test, or the published
means.

The point of the exercise: the project's recurring "inconclusive" / CI-crosses-zero
verdicts must be read against design adequacy. The MDE makes the design's
blindness explicit, and ``seeds_for_80`` turns each soft null into an actionable
observation budget. Nothing here hardcodes a result -- every number is recomputed
from the sweep JSON.
"""

from __future__ import annotations

import csv
import json
from itertools import combinations
from pathlib import Path
from typing import NamedTuple, Sequence

from .power_analysis import (
    cohens_d_safe,
    diagnose_null,
    permutation_test,
)
from .sweeps import _is_degenerate_error


class PowerRow(NamedTuple):
    """One contrast's power characterization, all fields recomputed from real data."""

    contrast: str
    group_a: str
    group_b: str
    panel_size: int
    n_per_group: int
    mean_a: float
    mean_b: float
    observed_d: float
    perm_p: float
    mde_80: float
    seeds_for_80: int | None
    underpowered: bool
    verdict: str


#: CSV column order for the written power table (kept stable for downstream tokens).
POWER_CSV_COLUMNS = (
    "contrast",
    "group_a",
    "group_b",
    "panel_size",
    "n_per_group",
    "mean_a",
    "mean_b",
    "observed_d",
    "perm_p",
    "mde_80",
    "seeds_for_80",
    "underpowered",
    "verdict",
)


def load_trial_rows(json_path: str | Path) -> list[dict]:
    """Load the per-seed trial rows from a sweep results JSON file.

    Args:
        json_path: Path to ``sweep_results.json`` (must contain a ``"rows"`` list).

    Returns:
        The list of per-seed row dicts (each has ``strategy``, ``panel_size``,
        ``seed``, ``eie_error``).

    Raises:
        ValueError: If the file has no ``"rows"`` list.
    """
    data = json.loads(Path(json_path).read_text())
    rows = data.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("sweep results JSON must contain a non-empty 'rows' list")
    return rows


def group_eie_by_strategy(
    rows: Sequence[dict], panel_size: int
) -> dict[str, list[float]]:
    """Collect per-seed ``eie_error`` values per strategy at one panel size.

    Args:
        rows: Per-seed trial rows from :func:`load_trial_rows`.
        panel_size: The panel size to filter on.

    Returns:
        A mapping ``strategy -> [eie_error per seed]``, seed-sorted for determinism.
    """
    groups: dict[str, list[tuple[int, float]]] = {}
    for row in rows:
        if int(row["panel_size"]) != panel_size:
            continue
        eie = float(row["eie_error"])
        # Drop the all-degenerate-panel sentinel (-1.0 / non-finite) exactly as
        # sweeps.aggregate does; recovery error is non-negative by construction,
        # so a negative value is the sentinel, never a measurement.
        if _is_degenerate_error(eie):
            continue
        strat = str(row["strategy"])
        groups.setdefault(strat, []).append((int(row["seed"]), eie))
    return {
        strat: [val for _, val in sorted(pairs)] for strat, pairs in groups.items()
    }


def paired_size_contrast(
    rows: Sequence[dict],
    strategy: str,
    size_from: int,
    size_to: int,
    *,
    seed: int = 0,
    n_boot: int = 5000,
    alpha: float = 0.05,
) -> dict[str, object]:
    """Paired trio-vs-larger panel-size contrast for one strategy.

    Each (regime, seed) cell that has a non-degenerate error at BOTH sizes
    contributes one matched difference ``error(size_to) - error(size_from)``.
    Pairing within the regime+seed removes the large between-regime variance that
    inflates the *unpaired* size curve, so this is the powered test of whether
    panel size changes recovery error for a given strategy. Returns the mean
    paired difference, its bootstrap CI, the matched-cell count, and a verdict
    that reads ``resolved`` only when the CI excludes zero (otherwise the size
    change is within noise -- not evidence of no effect, just unresolved here).
    """
    from .statistics_analysis import bootstrap_ci

    by_cell: dict[tuple, dict[int, float]] = {}
    for row in rows:
        if str(row["strategy"]) != strategy:
            continue
        eie = float(row["eie_error"])
        if _is_degenerate_error(eie):
            continue
        cell = (
            row["mean_expertise"],
            row["bias_std"],
            row["expertise_heterogeneity"],
            row["n_items"],
            row["prevalence_a"],
            row["n_experts"],
            int(row["seed"]),
        )
        by_cell.setdefault(cell, {})[int(row["panel_size"])] = eie
    diffs = [
        sizes[size_to] - sizes[size_from]
        for sizes in by_cell.values()
        if size_from in sizes and size_to in sizes
    ]
    if not diffs:
        raise ValueError(f"no matched ({size_from},{size_to}) pairs for strategy {strategy!r}")
    mean_diff = sum(diffs) / len(diffs)
    ci_lo, ci_hi = bootstrap_ci(diffs, n_boot=n_boot, alpha=alpha, seed=seed)
    return {
        "strategy": strategy,
        "size_from": size_from,
        "size_to": size_to,
        "n_pairs": len(diffs),
        "mean_diff": mean_diff,
        "ci_low": ci_lo,
        "ci_high": ci_hi,
        "verdict": "resolved" if not (ci_lo <= 0.0 <= ci_hi) else "within-noise",
    }


def _verdict(perm_p: float, underpowered: bool, alpha: float) -> str:
    """Honest one-line verdict combining significance and design adequacy."""
    if perm_p < alpha:
        return "significant"
    if underpowered:
        return "underpowered-null"
    return "well-powered-null"


def contrast_power(
    values_a: Sequence[float],
    values_b: Sequence[float],
    *,
    contrast: str,
    group_a: str,
    group_b: str,
    panel_size: int,
    n_perm: int = 5000,
    seed: int,
    target_power: float = 0.8,
    alpha: float = 0.05,
) -> PowerRow:
    """Characterize one two-group contrast: effect, permutation p, power, MDE, budget.

    Args:
        values_a: Per-seed metric values for group A.
        values_b: Per-seed metric values for group B.
        contrast: Stable contrast key (e.g. ``"rep_vs_ideo_p3"``).
        group_a: Name of group A.
        group_b: Name of group B.
        panel_size: Panel size the contrast was measured at.
        n_perm: Permutations for the distribution-free p-value.
        seed: Seed for the permutation RNG.
        target_power: Power level for MDE and the seed-budget figure.
        alpha: Significance level.

    Returns:
        A fully-populated :class:`PowerRow`.

    Raises:
        ValueError: If either group is empty.
    """
    a = list(values_a)
    b = list(values_b)
    if not a or not b:
        raise ValueError("both groups must be non-empty")

    n = min(len(a), len(b))
    mean_a = sum(a) / len(a)
    mean_b = sum(b) / len(b)
    d = cohens_d_safe(a, b)
    perm_p = permutation_test(a, b, n_perm=n_perm, seed=seed, alternative="two-sided")
    diag = diagnose_null(d, n, target_power=target_power, alpha=alpha)
    # Deliberately NO retrospective/observed power (analytic_power at the observed d):
    # the design adequacy is reported prospectively via the MDE and seeds_for_80 only,
    # honoring the manuscript's "no retrospective observed power" contract.
    return PowerRow(
        contrast=contrast,
        group_a=group_a,
        group_b=group_b,
        panel_size=panel_size,
        n_per_group=n,
        mean_a=mean_a,
        mean_b=mean_b,
        observed_d=d,
        perm_p=perm_p,
        mde_80=diag.mde,
        seeds_for_80=diag.seeds_for_target,
        underpowered=diag.underpowered,
        verdict=_verdict(perm_p, diag.underpowered, alpha),
    )


def strategy_power_matrix(
    rows: Sequence[dict],
    panel_size: int,
    *,
    seed: int = 0,
    n_perm: int = 5000,
    target_power: float = 0.8,
    alpha: float = 0.05,
) -> list[PowerRow]:
    """Power characterization for every pairwise strategy contrast at one panel size.

    Args:
        rows: Per-seed trial rows.
        panel_size: Panel size to analyze.
        seed: Base permutation seed (each pair derives a stable offset).
        n_perm: Permutations per contrast.
        target_power: Power level for MDE / observation budget.
        alpha: Significance level.

    Returns:
        One :class:`PowerRow` per unordered strategy pair, name-sorted for determinism.
    """
    groups = group_eie_by_strategy(rows, panel_size)
    strategies = sorted(groups)
    out: list[PowerRow] = []
    for offset, (sa, sb) in enumerate(combinations(strategies, 2)):
        out.append(
            contrast_power(
                groups[sa],
                groups[sb],
                contrast=f"pair_{sa}__vs__{sb}_p{panel_size}",
                group_a=sa,
                group_b=sb,
                panel_size=panel_size,
                n_perm=n_perm,
                seed=seed + offset,
                target_power=target_power,
                alpha=alpha,
            )
        )
    return out


def rep_vs_ideo_power(
    rows: Sequence[dict],
    *,
    seed: int = 100,
    n_perm: int = 5000,
    target_power: float = 0.8,
    alpha: float = 0.05,
) -> list[PowerRow]:
    """The headline representative-vs-ideological contrast at each available panel size.

    Args:
        rows: Per-seed trial rows.
        seed: Base permutation seed.
        n_perm: Permutations per contrast.
        target_power: Power level for MDE / observation budget.
        alpha: Significance level.

    Returns:
        One :class:`PowerRow` per panel size where both strategies are present.
    """
    panel_sizes = sorted({int(r["panel_size"]) for r in rows})
    out: list[PowerRow] = []
    for offset, ps in enumerate(panel_sizes):
        groups = group_eie_by_strategy(rows, ps)
        if "representative_sortition" not in groups or "ideological_selection" not in groups:
            continue
        out.append(
            contrast_power(
                groups["representative_sortition"],
                groups["ideological_selection"],
                contrast=f"rep_vs_ideo_p{ps}",
                group_a="representative_sortition",
                group_b="ideological_selection",
                panel_size=ps,
                n_perm=n_perm,
                seed=seed + offset,
                target_power=target_power,
                alpha=alpha,
            )
        )
    return out


def analyze(
    json_path: str | Path,
    *,
    seed: int = 0,
    n_perm: int = 5000,
    target_power: float = 0.8,
    alpha: float = 0.05,
) -> list[PowerRow]:
    """Full power study: rep-vs-ideo plus every pairwise strategy matrix per panel size.

    Args:
        json_path: Path to the sweep results JSON.
        seed: Base permutation seed.
        n_perm: Permutations per contrast.
        target_power: Power level for MDE / budget.
        alpha: Significance level.

    Returns:
        The ordered list of all :class:`PowerRow` results (rep-vs-ideo first).
    """
    rows = load_trial_rows(json_path)
    results = rep_vs_ideo_power(
        rows, seed=seed + 100, n_perm=n_perm, target_power=target_power, alpha=alpha
    )
    for ps in sorted({int(r["panel_size"]) for r in rows}):
        results.extend(
            strategy_power_matrix(
                rows,
                ps,
                seed=seed + ps * 10,
                n_perm=n_perm,
                target_power=target_power,
                alpha=alpha,
            )
        )
    return results


def _fmt(value: float) -> str:
    return f"{value:.4f}"


def write_power_table(results: Sequence[PowerRow], csv_path: str | Path) -> Path:
    """Write the power rows to a tidy CSV at ``csv_path``.

    Args:
        results: Power rows from :func:`analyze` (or a subset).
        csv_path: Destination CSV path (parent dirs are created).

    Returns:
        The written path.
    """
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(POWER_CSV_COLUMNS)
        for r in results:
            writer.writerow(
                [
                    r.contrast,
                    r.group_a,
                    r.group_b,
                    r.panel_size,
                    r.n_per_group,
                    _fmt(r.mean_a),
                    _fmt(r.mean_b),
                    _fmt(r.observed_d),
                    _fmt(r.perm_p),
                    _fmt(r.mde_80),
                    "" if r.seeds_for_80 is None else r.seeds_for_80,
                    r.underpowered,
                    r.verdict,
                ]
            )
    return path


__all__ = [
    "PowerRow",
    "POWER_CSV_COLUMNS",
    "load_trial_rows",
    "group_eie_by_strategy",
    "paired_size_contrast",
    "contrast_power",
    "strategy_power_matrix",
    "rep_vs_ideo_power",
    "analyze",
    "write_power_table",
]
