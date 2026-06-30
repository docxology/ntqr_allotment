from __future__ import annotations

import csv
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import yaml

from .statistics_analysis import bootstrap_slope_ci

DEFAULT_CSV_PATH = (
    Path(__file__).resolve().parents[2] / "output" / "data" / "sweep_aggregated.csv"
)
REQUIRED_COLUMNS = ("strategy", "panel_size", "eie_mean", "eie_ci95", "n")
GRID_TOKENS = {
    "N_SEEDS": "3",
    "N_EXPERTS": "48",
    "N_ITEMS": "150",
    "N_TRIOS": "4",
    "SWEEP_PROFILE": "smoke",
    "SWEEP_CONFIG_HASH": "unknown",
}
#: Alarm token NAMES (the honesty contract); VALUES are computed from the shipped
#: benchmark artifact ``output/data/alarm_timings.csv`` by :func:`compute_alarm_tokens`,
#: never hardcoded. ``ALARM_MAX_Q`` is the opt-in policy cap, not a measurement.
ALARM_TOKEN_NAMES = ("ALARM_Q20_S", "ALARM_Q50_S", "ALARM_Q100_S", "ALARM_MAX_Q")
ALARM_MAX_Q = "30"
REPRESENTATIVE_STRATEGY = "representative_sortition"
IDEOLOGICAL_STRATEGY = "ideological_selection"
REQUIRED_PANEL_SIZES = (3, 6)

#: Columns required of the error-correlation tolerance sweep CSV. ``corr_mean``
#: is the NTQR-measured realized correlation; ``eie_mean`` the recovery error.
INDEPENDENCE_COLUMNS = ("rho", "eie_mean", "corr_mean", "n")
#: Sentinel value the independence sweep writes for an all-degenerate cell.
_INDEPENDENCE_DEGENERATE = -1.0
#: Token set produced by :func:`compute_independence_tokens` (the honesty gate
#: requires every prose ``{{TOKEN}}`` to be in the union of producible sets).
INDEPENDENCE_TOKEN_NAMES = frozenset(
    {
        "TOLERANCE_SLOPE",
        "TOLERANCE_SLOPE_CI95",
        "TOLERANCE_VERDICT",
        "CORR_AT_RHO0",
        "CORR_AT_RHO_HIGH",
        "EIE_AT_RHO0",
        "EIE_AT_RHO_HIGH",
        "INDEP_N_EXPERTS",
        "INDEP_N_ITEMS",
    }
)


@dataclass(frozen=True)
class SweepRow:
    strategy: str
    panel_size: int
    eie_mean: float
    eie_ci95: float
    n: int


def _parse_float(row: dict[str, str], column: str) -> float:
    value = row.get(column)
    if value is None or value == "":
        raise ValueError(f"Missing or empty required column: {column}")
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"Invalid value for column {column}: {value}") from exc


def _parse_int(row: dict[str, str], column: str) -> int:
    value = row.get(column)
    if value is None or value == "":
        raise ValueError(f"Missing or empty required column: {column}")
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"Invalid value for column {column}: {value}") from exc


def _humanize_strategy(strategy: str) -> str:
    return strategy.replace("_", " ")


def _token_strategy_name(strategy: str) -> str:
    return strategy.upper()


def _weighted_mean(rows: Iterable[SweepRow]) -> float:
    row_list = list(rows)
    total_n = sum(row.n for row in row_list)
    if total_n <= 0:
        raise ValueError("Encountered a group with no positive sample weight.")
    return sum(row.eie_mean * row.n for row in row_list) / total_n


def _pooled_ci(rows: Iterable[SweepRow]) -> float:
    row_list = list(rows)
    total_n = sum(row.n for row in row_list)
    if total_n <= 0:
        raise ValueError("Encountered a group with no positive sample weight.")
    return math.sqrt(sum((row.eie_ci95**2) * row.n for row in row_list) / total_n)


def _format_float(value: float) -> str:
    return f"{value:.3f}"


def _format_optional_float(value: object, *, digits: int = 1) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.{digits}f}"


def _format_p_value(value: float) -> str:
    return f"{value:.3f}"


def _provenance_summary(records: object) -> str:
    if not isinstance(records, list) or not records:
        return "not pinned"
    seen: dict[tuple[str, str], str] = {}
    for record in records:
        if not isinstance(record, dict):
            continue
        model = str(record.get("model") or "unknown-model")
        family = str(record.get("family") or "unknown-family")
        digest = record.get("digest")
        digest_text = "no-digest" if digest in (None, "") else str(digest)[:8]
        seen.setdefault((model, family), digest_text)
    if not seen:
        return "not pinned"
    return "; ".join(
        f"{model} ({family}, digest {digest})"
        for (model, family), digest in sorted(seen.items())
    )


def expected_token_names(strategies: Iterable[str]) -> set[str]:
    strategy_list = sorted(set(strategies))
    tokens = {
        "RANK_BEST_STRATEGY",
        "RANK_WORST_STRATEGY",
        "REP_VS_IDEO_P3_EFFECT",
        "REP_VS_IDEO_P3_CI",
        "REP_VS_IDEO_P3_VERDICT",
        "REP_VS_IDEO_P6_EFFECT",
        "REP_VS_IDEO_P6_CI",
        "REP_VS_IDEO_P6_VERDICT",
        *GRID_TOKENS.keys(),
    }
    for strategy in strategy_list:
        strategy_token = _token_strategy_name(strategy)
        tokens.add(f"RANK_{strategy_token}_EIE")
        tokens.add(f"RANK_{strategy_token}_CI")
        tokens.add(f"POWER_{strategy_token}_SIZE3")
        tokens.add(f"POWER_{strategy_token}_SIZE6")
        tokens.add(f"POWER_{strategy_token}_DIRECTION")
    return tokens


def _read_rows(csv_path: Path) -> list[SweepRow]:
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    rows: list[SweepRow] = []
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        for column in REQUIRED_COLUMNS:
            if column not in fieldnames:
                raise ValueError(f"Missing required column: {column}")

        for row in reader:
            strategy = row.get("strategy")
            if strategy is None or strategy == "":
                raise ValueError("Missing or empty required column: strategy")
            eie_mean = _parse_float(row, "eie_mean")
            if eie_mean == -1.0 or not math.isfinite(eie_mean):
                continue
            rows.append(
                SweepRow(
                    strategy=strategy,
                    panel_size=_parse_int(row, "panel_size"),
                    eie_mean=eie_mean,
                    eie_ci95=_parse_float(row, "eie_ci95"),
                    n=_parse_int(row, "n"),
                )
            )
    return rows


def _require_rows(
    rows: list[SweepRow], *, strategy: str | None = None, panel_size: int | None = None
) -> list[SweepRow]:
    filtered = [
        row
        for row in rows
        if (strategy is None or row.strategy == strategy)
        and (panel_size is None or row.panel_size == panel_size)
    ]
    if not filtered:
        details: list[str] = []
        if strategy is not None:
            details.append(f"strategy={strategy}")
        if panel_size is not None:
            details.append(f"panel_size={panel_size}")
        raise ValueError(f"No non-degenerate rows available for {', '.join(details)}.")
    return filtered


def _modal_str(values: list[str], fallback: str) -> str:
    """Most common non-empty value as a string, else ``fallback``."""
    cleaned = [v for v in values if v not in (None, "")]
    if not cleaned:
        return fallback
    return str(Counter(cleaned).most_common(1)[0][0])


def _derive_grid_tokens(csv_path: Path) -> dict[str, str]:
    """Derive grid tokens from the CSV (and config for N_TRIOS), not hardcoded.

    Falls back to :data:`GRID_TOKENS` for any value the CSV/config does not
    provide, so fixtures without the optional columns still resolve. N_SEEDS is
    the max per-cell seed count ``n``; N_EXPERTS/N_ITEMS are read from their
    columns; N_TRIOS comes from ``manuscript/config.yaml`` when present.
    """
    n_items: list[str] = []
    n_experts: list[str] = []
    profile_names: list[str] = []
    config_hashes: list[str] = []
    seed_counts: list[int] = []
    with Path(csv_path).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("profile_name"):
                profile_names.append(row["profile_name"])
            if row.get("config_hash"):
                config_hashes.append(row["config_hash"])
            if row.get("n_items"):
                n_items.append(row["n_items"])
            if row.get("n_experts"):
                n_experts.append(row["n_experts"])
            if row.get("n"):
                try:
                    seed_counts.append(int(row["n"]))
                except ValueError:
                    pass
    n_trios = GRID_TOKENS["N_TRIOS"]
    cfg = Path(csv_path).resolve().parents[2] / "manuscript" / "config.yaml"
    if cfg.exists():
        data = yaml.safe_load(cfg.read_text()) or {}
        profiles = data.get("experiment_profiles") or {}
        profile_name = data.get("default_profile", "smoke")
        profile = profiles.get(profile_name, {}) if isinstance(profiles, dict) else {}
        experiment = profile.get("experiment") if isinstance(profile, dict) else {}
        if not isinstance(experiment, dict):
            experiment = data.get("experiment") or {}
        if "n_trios" in experiment:
            n_trios = str(experiment["n_trios"])
    return {
        "N_SEEDS": str(max(seed_counts)) if seed_counts else GRID_TOKENS["N_SEEDS"],
        "N_EXPERTS": _modal_str(n_experts, GRID_TOKENS["N_EXPERTS"]),
        "N_ITEMS": _modal_str(n_items, GRID_TOKENS["N_ITEMS"]),
        "N_TRIOS": n_trios,
        "SWEEP_PROFILE": _modal_str(profile_names, GRID_TOKENS["SWEEP_PROFILE"]),
        "SWEEP_CONFIG_HASH": _modal_str(config_hashes, GRID_TOKENS["SWEEP_CONFIG_HASH"]),
    }


def compute_alarm_tokens(csv_path: Path) -> dict[str, str]:
    """Alarm-timing tokens computed from the shipped benchmark artifact.

    Reads ``output/data/alarm_timings.csv`` (written by ``scripts/bench_alarm.py``)
    and emits the measured wall-clock seconds at Q in {20, 50, 100}. Nothing is
    hardcoded: a missing artifact or missing Q row fails loudly rather than printing
    a frozen literal. ``ALARM_MAX_Q`` is the opt-in policy cap (not a measurement).
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Regenerate with "
            "`uv run python scripts/bench_alarm.py --qs 20 50 100`."
        )
    seconds: dict[int, float] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            seconds[int(row["Q"])] = float(row["seconds"])
    missing = [q for q in (20, 50, 100) if q not in seconds]
    if missing:
        raise ValueError(
            f"alarm_timings.csv is missing Q={missing}; run "
            "`scripts/bench_alarm.py --qs 20 50 100`."
        )
    return {
        "ALARM_Q20_S": f"{seconds[20]:.1f}",
        "ALARM_Q50_S": f"{seconds[50]:.1f}",
        "ALARM_Q100_S": f"{seconds[100]:.1f}",
        "ALARM_MAX_Q": ALARM_MAX_Q,
    }


def compute_tokens(csv_path: Path) -> dict[str, str]:
    rows = _read_rows(csv_path)
    if not rows:
        raise ValueError("No non-degenerate rows available after filtering sentinel values.")

    strategies = sorted({row.strategy for row in rows})
    tokens: dict[str, str] = {}

    ranking_data = []
    for strategy in strategies:
        strategy_rows = _require_rows(rows, strategy=strategy)
        strategy_token = _token_strategy_name(strategy)
        mean = _weighted_mean(strategy_rows)
        ci = _pooled_ci(strategy_rows)
        ranking_data.append((strategy, mean))
        tokens[f"RANK_{strategy_token}_EIE"] = _format_float(mean)
        tokens[f"RANK_{strategy_token}_CI"] = _format_float(ci)

        size_means: dict[int, float] = {}
        for panel_size in REQUIRED_PANEL_SIZES:
            panel_rows = _require_rows(rows, strategy=strategy, panel_size=panel_size)
            panel_mean = _weighted_mean(panel_rows)
            size_means[panel_size] = panel_mean
            tokens[f"POWER_{strategy_token}_SIZE{panel_size}"] = _format_float(panel_mean)
        # Size-direction label DERIVED from the trio->six-seat (two smallest) panel
        # means (never hardcoded): the prose Direction column reads from this token
        # so it cannot drift from the table cells. A change below the ~0.01 CI scale
        # is "roughly flat" rather than a noise direction. The 0.01 band MUST match
        # figure_parts/power.py _FLAT_DIRECTION_TOL so the table and the power-curve
        # figure label agree.
        ordered_sizes = sorted(size_means)
        step_delta = size_means[ordered_sizes[1]] - size_means[ordered_sizes[0]]
        if abs(step_delta) < 0.01:
            direction = "roughly flat"
        else:
            direction = "error rises" if step_delta > 0 else "error falls"
        tokens[f"POWER_{strategy_token}_DIRECTION"] = direction

    ranking_data.sort(key=lambda item: (item[1], item[0]))
    tokens["RANK_BEST_STRATEGY"] = _humanize_strategy(ranking_data[0][0])
    tokens["RANK_WORST_STRATEGY"] = _humanize_strategy(ranking_data[-1][0])

    for panel_size in REQUIRED_PANEL_SIZES:
        representative_rows = _require_rows(
            rows, strategy=REPRESENTATIVE_STRATEGY, panel_size=panel_size
        )
        ideological_rows = _require_rows(
            rows, strategy=IDEOLOGICAL_STRATEGY, panel_size=panel_size
        )
        representative_mean = _weighted_mean(representative_rows)
        representative_ci = _pooled_ci(representative_rows)
        ideological_mean = _weighted_mean(ideological_rows)
        ideological_ci = _pooled_ci(ideological_rows)
        effect = ideological_mean - representative_mean
        effect_ci = math.sqrt(representative_ci**2 + ideological_ci**2)
        verdict = (
            "inconclusive (95% CI crosses zero)"
            if abs(effect) <= effect_ci
            else "supported"
        )
        prefix = f"REP_VS_IDEO_P{panel_size}"
        tokens[f"{prefix}_EFFECT"] = _format_float(effect)
        tokens[f"{prefix}_CI"] = _format_float(effect_ci)
        tokens[f"{prefix}_VERDICT"] = verdict

    tokens.update(_derive_grid_tokens(csv_path))

    if set(tokens) != expected_token_names(strategies):
        raise ValueError("Computed token set does not match the documented token contract.")
    return tokens


@dataclass(frozen=True)
class IndependenceCsvRow:
    """One non-degenerate cell of the error-correlation tolerance sweep CSV."""

    rho: float
    eie_mean: float
    corr_mean: float
    n: int


def _ols_slope(xs: list[float], ys: list[float]) -> float:
    """Ordinary-least-squares slope of ``ys`` regressed on ``xs``.

    Mirrors :func:`ntqr_allotment.theory.fit_error_correlation_slope` so the
    tolerance token reports the same quantity the sweep module computes. Raises
    ``ValueError`` if fewer than two points or the ``xs`` have no variance.
    """
    if len(xs) < 2:
        raise ValueError("at least two (corr, eie) points are required")
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    denominator = sum((x - mean_x) ** 2 for x in xs)
    if math.isclose(denominator, 0.0, abs_tol=0.0):
        raise ValueError("realized correlations must have positive variance")
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=True))
    return numerator / denominator


def _read_independence_rows(csv_path: Path) -> list[IndependenceCsvRow]:
    """Read the tolerance-sweep CSV, dropping all-degenerate cells.

    A cell whose ``eie_mean`` is the degenerate sentinel (``-1.0``) or non-finite
    is excluded, exactly as the sweep aggregator records it; degenerate cells have
    no recovery error to regress against correlation.
    """
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    rows: list[IndependenceCsvRow] = []
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        for column in INDEPENDENCE_COLUMNS:
            if column not in fieldnames:
                raise ValueError(f"Missing required column: {column}")
        for row in reader:
            eie_mean = _parse_float(row, "eie_mean")
            if eie_mean == _INDEPENDENCE_DEGENERATE or not math.isfinite(eie_mean):
                continue
            rows.append(
                IndependenceCsvRow(
                    rho=_parse_float(row, "rho"),
                    eie_mean=eie_mean,
                    corr_mean=_parse_float(row, "corr_mean"),
                    n=_parse_int(row, "n"),
                )
            )
    return rows


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _independence_grid_dims(csv_path: Path) -> tuple[int, int]:
    """Read the modal ``n_experts`` and ``n_items`` of the tolerance-sweep CSV.

    These are constant across the grid by construction; the modal value tolerates
    a stray row. Raises ``ValueError`` if the columns are absent or unparsable so
    the tokens never silently fall back to a stale literal.
    """
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        for column in ("n_experts", "n_items"):
            if column not in fieldnames:
                raise ValueError(f"Missing required column: {column}")
        experts: Counter[int] = Counter()
        items: Counter[int] = Counter()
        for row in reader:
            experts[_parse_int(row, "n_experts")] += 1
            items[_parse_int(row, "n_items")] += 1
    if not experts or not items:
        raise ValueError("independence sweep CSV has no rows for grid dims")
    return experts.most_common(1)[0][0], items.most_common(1)[0][0]


def compute_independence_tokens(csv_path: Path) -> dict[str, str]:
    """Emit the error-correlation tolerance tokens from the sweep CSV.

    Produces exactly :data:`INDEPENDENCE_TOKEN_NAMES`:

    * ``TOLERANCE_SLOPE`` — OLS slope of recovery error (``eie_mean``) on the
      NTQR-measured realized correlation (``corr_mean``) across non-degenerate
      cells. A positive slope is the centerpiece "tolerance" signal.
    * ``CORR_AT_RHO0`` / ``CORR_AT_RHO_HIGH`` — mean realized correlation at the
      lowest / highest injected ``rho`` (across all non-degenerate cells there).
    * ``EIE_AT_RHO0`` / ``EIE_AT_RHO_HIGH`` — mean recovery error at the lowest /
      highest injected ``rho``.

    Raises ``ValueError`` if the CSV has no usable rows or the realized
    correlations have no variance (the slope is then undefined).
    """
    rows = _read_independence_rows(csv_path)
    if not rows:
        raise ValueError("No non-degenerate independence rows available.")

    indep_n_experts, indep_n_items = _independence_grid_dims(csv_path)

    rhos = sorted({row.rho for row in rows})
    rho_low, rho_high = rhos[0], rhos[-1]
    low_rows = [row for row in rows if row.rho == rho_low]
    high_rows = [row for row in rows if row.rho == rho_high]

    # Defensive de-duplication: regress over UNIQUE (corr, eie) cells so that any
    # size-invariant duplicate trio rows (a confound the sweep grid is designed to
    # avoid) cannot double-count points and fake the slope's effective N.
    unique_cells = sorted({(row.corr_mean, row.eie_mean) for row in rows})
    xs = [corr for corr, _ in unique_cells]
    ys = [eie for _, eie in unique_cells]
    slope = _ols_slope(xs, ys)
    ci_lo, ci_hi = bootstrap_slope_ci(xs, ys, n_boot=5000, alpha=0.05, seed=0)
    if ci_lo <= 0.0 <= ci_hi:
        # A predicate phrase that reads grammatically as "the slope is <verdict>";
        # prose supplies the "at this grid" / CI context so the sentence does not
        # double-print it (the token is used in 3 sections). "Indistinguishable
        # from zero" means unresolved (too few cells to resolve a sign), NOT that
        # the effect is known to be absent.
        verdict = "statistically indistinguishable from zero"
    elif ci_lo > 0.0:
        verdict = "positive (recovery error rises with realized correlation)"
    else:
        verdict = "negative (recovery error falls as realized correlation rises)"
    return {
        "TOLERANCE_SLOPE": _format_float(slope),
        "TOLERANCE_SLOPE_CI95": f"[{_format_float(ci_lo)}, {_format_float(ci_hi)}]",
        "TOLERANCE_VERDICT": verdict,
        "CORR_AT_RHO0": _format_float(_mean([row.corr_mean for row in low_rows])),
        "CORR_AT_RHO_HIGH": _format_float(_mean([row.corr_mean for row in high_rows])),
        "EIE_AT_RHO0": _format_float(_mean([row.eie_mean for row in low_rows])),
        "EIE_AT_RHO_HIGH": _format_float(_mean([row.eie_mean for row in high_rows])),
        "INDEP_N_EXPERTS": str(indep_n_experts),
        "INDEP_N_ITEMS": str(indep_n_items),
    }


_REPO_ROOT = Path(__file__).resolve().parents[2]
SWEEP_JSON_PATH = _REPO_ROOT / "output" / "data" / "sweep_results.json"
CROSS_FAMILY_JSON_PATH = _REPO_ROOT / "output" / "data" / "cross_family_results.json"
POSTDOC_PANEL_JSON_PATH = _REPO_ROOT / "output" / "data" / "postdoc_panel_results.json"
POSTDOC_ALIGNMENT_JSON_PATH = (
    _REPO_ROOT / "output" / "data" / "postdoc_panel_alignment.json"
)
INDEPENDENCE_CSV_PATH = _REPO_ROOT / "output" / "data" / "independence_sweep.csv"


def compute_power_tokens(json_path: Path | None = None) -> dict[str, str]:
    """Statistical-power design-budget tokens, recomputed from the real sweep JSON.

    Runs the power study over every pairwise strategy contrast and summarizes the
    design: how many contrasts are well-powered vs underpowered at the current
    observation count, the minimum detectable effect, and the per-group
    observation range required to reach 80% power. Nothing is hardcoded -- every
    value comes from
    :func:`ntqr_allotment.power_study.analyze` on the sweep rows.
    """
    from .power_study import analyze as _analyze

    path = Path(json_path) if json_path is not None else SWEEP_JSON_PATH
    rows = _analyze(path, seed=0, n_perm=5000)
    if not rows:
        raise ValueError(f"power tokens: no contrasts produced from {path}")
    from .statistics_analysis import holm_bonferroni

    total = len(rows)
    well = sum(1 for row in rows if not row.underpowered)
    significant = sum(1 for row in rows if row.perm_p < 0.05)
    holm = holm_bonferroni([row.perm_p for row in rows])
    seeds = [row.seeds_for_80 for row in rows if row.seeds_for_80 is not None]
    # The competence-first vs representative-sortition contrast at the trio grain,
    # so prose can state its power verdict without hand-transcribing a number.
    trio_size = min(row.panel_size for row in rows)
    threshold_sortition = next(
        (
            row
            for row in rows
            if {row.group_a, row.group_b}
            == {"expertise_threshold", "representative_sortition"}
            and row.panel_size == trio_size
        ),
        None,
    )
    return {
        "POWER_TOTAL_CONTRASTS": str(total),
        "POWER_WELL_POWERED_COUNT": str(well),
        "POWER_UNDERPOWERED_COUNT": str(total - well),
        "POWER_SIGNIFICANT_COUNT": str(significant),
        "POWER_SIGNIFICANT_HOLM_COUNT": str(holm.n_rejected),
        "POWER_MDE80": _format_float(min(row.mde_80 for row in rows)),
        "POWER_MIN_SEEDS_FOR_80": str(min(seeds)) if seeds else "n/a",
        "POWER_MAX_SEEDS_FOR_80": str(max(seeds)) if seeds else "n/a",
        "THRESHOLD_SORTITION_POWER_VERDICT": (
            threshold_sortition.verdict if threshold_sortition is not None else "n/a"
        ),
    }


def compute_cross_family_tokens(json_path: Path | None = None) -> dict[str, str]:
    """Cross-family decorrelation tokens, read from the live Qwen x Gemma artifact.

    Mirrors the existing independence-token contract: the artifact must be present
    (regenerate with ``scripts/run_cross_family.py``). Values are the live,
    n-limited run; the label is carried verbatim so prose stays honest.
    """
    path = Path(json_path) if json_path is not None else CROSS_FAMILY_JSON_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Regenerate with "
            "`uv run python scripts/run_cross_family.py`."
        )
    data = json.loads(path.read_text())
    pair_corr = data.get("pair_abs_corr", {})
    total_pairs = len(pair_corr)
    nonzero_pairs = sum(1 for value in pair_corr.values() if abs(float(value)) > 0.0)
    return {
        "CROSS_FAMILY_SAME_CORR": _format_float(float(data["mean_abs_same_family"])),
        "CROSS_FAMILY_CROSS_CORR": _format_float(float(data["mean_abs_cross_family"])),
        "CROSS_FAMILY_DELTA": _format_float(float(data["delta_cross_minus_same"])),
        "CROSS_FAMILY_CONFIG_HASH": str(data.get("config_hash", "unknown")),
        "CROSS_FAMILY_REPLICATES_PER_MODEL": str(
            int(data.get("replicates_per_model", 2))
        ),
        "CROSS_FAMILY_RUN_COUNT": str(int(data.get("run_count", 1))),
        "CROSS_FAMILY_N_SAME": str(int(data["n_same_pairs"])),
        "CROSS_FAMILY_N_CROSS": str(int(data["n_cross_pairs"])),
        "CROSS_FAMILY_N_ITEMS": str(int(data["n_items"])),
        "CROSS_FAMILY_NUM_PREDICT": str(int(data.get("num_predict", 0))),
        "CROSS_FAMILY_TIMEOUT": _format_optional_float(data.get("timeout"), digits=1),
        "CROSS_FAMILY_LIVE_LABEL": "live Ollama" if data.get("live_ollama") else "offline",
        "CROSS_FAMILY_PROVENANCE_SUMMARY": _provenance_summary(
            data.get("judge_provenance")
        ),
        "CROSS_FAMILY_NONZERO_PAIRS": str(nonzero_pairs),
        "CROSS_FAMILY_TOTAL_PAIRS": str(total_pairs),
        "CROSS_FAMILY_LABEL": str(data.get("provenance_label", "live empirical, n-limited")),
    }


CROSS_FAMILY_MULTISEED_JSON_PATH = (
    _REPO_ROOT / "output" / "data" / "cross_family_multiseed.json"
)


def compute_cross_family_multiseed_tokens(json_path: Path | None = None) -> dict[str, str]:
    """Multi-seed cross-family tokens (sign-stability across independent runs).

    Reads the aggregate written by ``scripts/run_cross_family_multiseed.py``. Like
    the other artifact producers it requires the file (regenerate the script);
    every value is the recomputed aggregate, never hardcoded.
    """
    path = Path(json_path) if json_path is not None else CROSS_FAMILY_MULTISEED_JSON_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Regenerate with "
            "`uv run python scripts/run_cross_family_multiseed.py`."
        )
    from .statistics_analysis import bootstrap_ci, exact_sign_test

    data = json.loads(path.read_text())
    deltas = [float(d) for d in data["deltas"]]
    sign = exact_sign_test(deltas)
    tokens = {
        "CROSS_FAMILY_MS_RUNS": str(int(data["n_runs"])),
        "CROSS_FAMILY_MS_CONFIG_HASH": str(data.get("config_hash", "unknown")),
        "CROSS_FAMILY_MS_REPLICATES_PER_MODEL": str(
            int(data.get("replicates_per_model", 2))
        ),
        "CROSS_FAMILY_MS_TOTAL_PAIRS_PER_RUN": str(
            int(data.get("total_pairs_per_run", 0))
        ),
        "CROSS_FAMILY_MS_SAME_PAIRS_PER_RUN": str(
            int(data.get("same_pairs_per_run", 0))
        ),
        "CROSS_FAMILY_MS_CROSS_PAIRS_PER_RUN": str(
            int(data.get("cross_pairs_per_run", 0))
        ),
        "CROSS_FAMILY_MS_MEAN_DELTA": _format_float(float(data["mean_delta"])),
        "CROSS_FAMILY_MS_STD_DELTA": _format_float(float(data["std_delta"])),
        "CROSS_FAMILY_MS_SIGN_STABILITY": _format_float(float(data["sign_stability"])),
        "CROSS_FAMILY_MS_MIN_DELTA": _format_float(float(data["min_delta"])),
        "CROSS_FAMILY_MS_MAX_DELTA": _format_float(float(data["max_delta"])),
        "CROSS_FAMILY_MS_NEGATIVE_RUNS": str(sign.negative),
        "CROSS_FAMILY_MS_SIGN_TEST_P": _format_p_value(sign.p_value),
        "CROSS_FAMILY_MS_N_ITEMS": str(int(data["n_items"])),
        "CROSS_FAMILY_MS_NUM_PREDICT": str(int(data.get("num_predict", 0))),
        "CROSS_FAMILY_MS_PROVENANCE_SUMMARY": _provenance_summary(
            data.get("judge_provenance")
        ),
    }
    # A real bootstrap CI on the per-run deltas (not just the min/max range).
    ci_lo, ci_hi = bootstrap_ci(deltas, n_boot=10000, alpha=0.05, seed=0)
    tokens["CROSS_FAMILY_MS_DELTA_CI95"] = (
        f"[{_format_float(ci_lo)}, {_format_float(ci_hi)}]"
    )
    return tokens


def compute_postdoc_panel_tokens(
    json_path: Path | None = None,
    alignment_path: Path | None = None,
) -> dict[str, str]:
    """Manuscript tokens for the Gemma postdoctoral reviewer-panel track."""
    path = Path(json_path) if json_path is not None else POSTDOC_PANEL_JSON_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Regenerate with `uv run python scripts/run_postdoc_panel.py`."
        )
    align_path = (
        Path(alignment_path) if alignment_path is not None else POSTDOC_ALIGNMENT_JSON_PATH
    )
    data = json.loads(path.read_text())
    alignment = json.loads(align_path.read_text()) if align_path.exists() else {}
    live_aggregates = [
        row for row in data.get("aggregates", []) if row.get("track") == "live"
    ]
    if not live_aggregates:
        raise ValueError("postdoc panel tokens require live aggregate rows")
    best = min(live_aggregates, key=lambda row: float(row["eie_mean"]))
    worst = max(live_aggregates, key=lambda row: float(row["eie_mean"]))
    by_strategy = {
        str(row["strategy"]): row
        for row in live_aggregates
        if int(row["panel_size"]) == min(int(r["panel_size"]) for r in live_aggregates)
    }
    representative = by_strategy.get("representative_sortition", live_aggregates[0])
    same_bias = by_strategy.get("ideological_selection", live_aggregates[0])
    expertise = by_strategy.get("expertise_threshold", live_aggregates[0])
    random = by_strategy.get("random_selection", live_aggregates[0])
    provenance = data.get("model_provenance") or {}
    config = data.get("config") or {}
    live_track = (data.get("tracks") or {}).get("live") or {}
    return {
        "POSTDOC_MODEL": str(data.get("model", "gemma3:4b")),
        "POSTDOC_MODEL_DIGEST": str(provenance.get("digest") or "unknown")[:12],
        "POSTDOC_CONFIG_HASH": str(data.get("config_hash", "unknown")),
        "POSTDOC_RUN_COUNT": str(len(config.get("seed_list", []))),
        "POSTDOC_N_APPLICATIONS": str(int(config.get("n_applications", 0))),
        "POSTDOC_N_REVIEWERS": str(int(config.get("n_reviewers", 0))),
        "POSTDOC_PANEL_SIZES": ", ".join(
            str(size) for size in config.get("panel_sizes", [])
        ),
        "POSTDOC_STRATEGY_COUNT": str(len(config.get("strategies", []))),
        "POSTDOC_NUM_PREDICT": str(int((config or {}).get("num_predict", 0))),
        "POSTDOC_TIMEOUT": _format_optional_float((config or {}).get("timeout"), digits=1),
        "POSTDOC_LIVE_LABEL": "live Ollama" if data.get("live_ollama") else "offline",
        "POSTDOC_BEST_STRATEGY": str(best.get("strategy_label")),
        "POSTDOC_BEST_EIE": _format_float(float(best["eie_mean"])),
        "POSTDOC_WORST_STRATEGY": str(worst.get("strategy_label")),
        "POSTDOC_WORST_EIE": _format_float(float(worst["eie_mean"])),
        "POSTDOC_REPRESENTATIVE_EIE": _format_float(float(representative["eie_mean"])),
        "POSTDOC_SAME_BIAS_EIE": _format_float(float(same_bias["eie_mean"])),
        "POSTDOC_EXPERTISE_EIE": _format_float(float(expertise["eie_mean"])),
        "POSTDOC_RANDOM_EIE": _format_float(float(random["eie_mean"])),
        "POSTDOC_REPRESENTATIVE_AGE_DISPARITY": _format_float(
            float(representative["age_disparity_mean"])
        ),
        "POSTDOC_SAME_BIAS_AGE_DISPARITY": _format_float(
            float(same_bias["age_disparity_mean"])
        ),
        "POSTDOC_ALIGNMENT_RATE": _format_float(
            float(alignment.get("agreement_rate_resolved", 0.0))
        ),
        "POSTDOC_ALIGNMENT_CELLS": str(int(alignment.get("n_cells", 0))),
        "POSTDOC_ALIGNMENT_RESOLVED_CELLS": str(
            int(alignment.get("n_resolved_cells", 0))
        ),
        "POSTDOC_VOTE_CACHE_ENTRIES": str(
            int((live_track.get("vote_cache_provenance") or {}).get("entries", 0))
        ),
    }


def compute_threshold_tokens(json_path: Path | None = None) -> dict[str, str]:
    """Threshold-vs-sortition CI-separation tokens (the Advisor's separation guard).

    Compares competence-first selection and representative sortition by separately
    bootstrapped mean intervals at the trio, emitting an explicit CI-overlap verdict
    that must read ``separated`` before any "beats" wording is justified.
    """
    from .power_study import group_eie_by_strategy, load_trial_rows
    from .statistics_analysis import strategy_separation

    path = Path(json_path) if json_path is not None else SWEEP_JSON_PATH
    rows = load_trial_rows(path)
    groups = group_eie_by_strategy(rows, panel_size=3)
    threshold = groups.get("expertise_threshold", [])
    sortition = groups.get("representative_sortition", [])
    if not threshold or not sortition:
        raise ValueError(
            "threshold tokens: need both expertise_threshold and "
            "representative_sortition at panel_size=3"
        )
    sep = strategy_separation(threshold, sortition, seed=0)
    return {
        "THRESHOLD_SORTITION_VERDICT": sep.verdict,
        "THRESHOLD_SORTITION_DELTA": _format_float(sep.signed_difference),
        "THRESHOLD_MEAN": _format_float(sep.mean_a),
        "SORTITION_MEAN": _format_float(sep.mean_b),
    }


def compute_size_effect_tokens(json_path: Path | None = None) -> dict[str, str]:
    """Paired (regime+seed-controlled) trio-to-six-seat size-effect tokens.

    For each strategy, reports the matched-pair mean change in EIE error from the
    trio (size 3) to the six-seat ensemble (size 6), its bootstrap CI, and a
    resolved/within-noise verdict. Pairing within regime+seed removes the large
    between-regime variance that inflates the unpaired pooled curve, so these are
    the powered test behind H3 -- replacing a direction read off a noisy pooled
    curve with a contrast that is actually resolved or not.
    """
    from .power_study import load_trial_rows, paired_size_contrast

    path = Path(json_path) if json_path is not None else SWEEP_JSON_PATH
    rows = load_trial_rows(path)
    strategies = sorted({str(r["strategy"]) for r in rows})
    tokens: dict[str, str] = {}
    resolved = 0
    for strat in strategies:
        st = _token_strategy_name(strat)
        result = paired_size_contrast(rows, strat, 3, 6, seed=0)
        tokens[f"SIZE_{st}_3TO6_DELTA"] = f"{float(result['mean_diff']):+.3f}"
        tokens[f"SIZE_{st}_3TO6_CI"] = (
            f"[{float(result['ci_low']):+.3f}, {float(result['ci_high']):+.3f}]"
        )
        tokens[f"SIZE_{st}_3TO6_VERDICT"] = str(result["verdict"])
        if result["verdict"] == "resolved":
            resolved += 1
    tokens["SIZE_PAIRED_RESOLVED_COUNT"] = str(resolved)
    return tokens


def compute_mechanism_tokens(json_path: Path | None = None) -> dict[str, str]:
    """Per-trio mechanism tokens: realized error-correlation vs panel size.

    Backs the H3 "why does size hurt" claim with the measured per-trio diagnostic
    (``output/data/trio_conditioning.json`` from ``run_trio_conditioning.py``). The
    headline: the realized ``|error-correlation|`` -- the quantity the EIE solver
    assumes is zero -- does NOT grow with panel size, so the size penalty is not an
    error-independence breakdown but an aggregation effect of the trio-only solver.
    """
    path = (
        Path(json_path)
        if json_path is not None
        else SWEEP_JSON_PATH.parent / "trio_conditioning.json"
    )
    data = json.loads(Path(path).read_text())
    by_size = {int(k): float(v) for k, v in data["mean_abs_corr_by_size"].items()}
    lo, hi = min(by_size.values()), max(by_size.values())
    tokens = {
        "MECH_N_TRIO_RECORDS": f"{int(data['n_records']):,}",
        "MECH_N_SEEDS": str(int(data["n_seeds"])),
        "MECH_CORR_MIN": f"{lo:.4f}",
        "MECH_CORR_MAX": f"{hi:.4f}",
        # The solver assumes zero error-correlation; "flat" means enlarging the
        # panel does not move the realized correlation off that small baseline.
        "MECH_CORR_VERDICT": "essentially flat" if (hi - lo) < 0.002 else "size-dependent",
    }
    for size in (3, 6, 9, 12):
        if size in by_size:
            tokens[f"MECH_CORR_SIZE{size}"] = f"{by_size[size]:.4f}"
    return tokens


def compute_extension_tokens(repo_root: Path | None = None) -> dict[str, str]:
    """Union of the Session-6 extension producers (power / cross-family / threshold)."""
    root = Path(repo_root) if repo_root is not None else _REPO_ROOT
    data_dir = root / "output" / "data"
    tokens: dict[str, str] = {}
    tokens.update(compute_power_tokens(data_dir / "sweep_results.json"))
    tokens.update(compute_size_effect_tokens(data_dir / "sweep_results.json"))
    tokens.update(compute_mechanism_tokens(data_dir / "trio_conditioning.json"))
    tokens.update(compute_cross_family_tokens(data_dir / "cross_family_results.json"))
    tokens.update(compute_threshold_tokens(data_dir / "sweep_results.json"))
    return tokens


def compute_bloc_phase_tokens(summary_path: Path) -> dict[str, str]:
    """Tokens for the bloc-confound phase study (the confirm-then-extend headline).

    Reads ``output/data/bloc_phase_summary.json`` (written by
    ``scripts/run_bloc_phase.py``). Surfaces the strategy x within-bloc-coupling
    recovery means at the lowest and highest coupling, the
    ideological-minus-representative separation at each end (collapse at rho=0,
    fan-out at rho_hi), and the realized trio error-correlation at rho_hi that
    drives the effect. Nothing is hand-transcribed.
    """
    path = Path(summary_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing required data file: {path}. Regenerate it with "
            "`uv run python scripts/run_bloc_phase.py`."
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    cells = payload.get("cells")
    headline = payload.get("headline")
    if not isinstance(cells, list) or not isinstance(headline, dict):
        raise ValueError(f"bloc phase summary malformed: {path}")

    rho_lo = float(headline["rho_lo"])
    rho_hi = float(headline["rho_hi"])

    def _cell(strategy: str, rho: float) -> dict[str, object]:
        for entry in cells:
            if entry.get("strategy") == strategy and abs(
                float(entry.get("bloc_correlation", -1)) - rho
            ) < 1e-9:
                return entry
        raise ValueError(f"bloc phase summary missing cell {strategy} @ rho={rho}")

    tokens: dict[str, str] = {
        "BLOC_RHO_HI": f"{rho_hi:.2f}",
        "BLOC_N_RHO_LEVELS": str(len({float(c["bloc_correlation"]) for c in cells})),
        "BLOC_SEP_LO": _format_float(float(headline["sep_lo"])),
        "BLOC_SEP_HI": _format_float(float(headline["sep_hi"])),
        # Mean non-degenerate trial count across all (strategy, rho) points, so the
        # prose "~N trials per point" is a true average rather than one cell's value.
        "BLOC_N_PER_POINT": str(
            round(
                sum(int(c["n"]) for c in cells if int(c.get("n", 0)) > 0)
                / max(1, sum(1 for c in cells if int(c.get("n", 0)) > 0))
            )
        ),
    }
    for strategy in (
        "expertise_threshold",
        "representative_sortition",
        "random_selection",
        "ideological_selection",
    ):
        token_name = _token_strategy_name(strategy)
        lo = _cell(strategy, rho_lo)
        hi = _cell(strategy, rho_hi)
        tokens[f"BLOC_{token_name}_LO"] = _format_float(float(lo["eie_mean"]))
        tokens[f"BLOC_{token_name}_HI"] = _format_float(float(hi["eie_mean"]))
        tokens[f"BLOC_CORR_{token_name}_HI"] = _format_float(float(hi["corr_mean"]))

    # Per-regime robustness (paired Simpson guard): the headline is not a pooling
    # artifact if ideological > representative holds across regimes at high coupling.
    robustness = payload.get("robustness")
    if isinstance(robustness, dict):
        tokens["BLOC_ROBUST_N_PAIRED"] = str(int(robustness["n_paired"]))
        tokens["BLOC_ROBUST_N_IDEO_GT_REP"] = str(int(robustness["n_ideo_gt_rep"]))
        tokens["BLOC_ROBUST_FRAC"] = (
            f"{int(robustness['n_ideo_gt_rep'])}/{int(robustness['n_paired'])}"
        )
        tokens["BLOC_PAIRED_DIFF"] = _format_float(float(robustness["paired_mean_diff"]))
        tokens["BLOC_PAIRED_CI"] = _format_float(float(robustness["paired_ci95"]))

    # Negative control: confound keyed on an axis sortition does NOT balance.
    control = payload.get("negative_control")
    if isinstance(control, dict) and isinstance(control.get("headline"), dict):
        ch = control["headline"]
        ccells = control.get("cells", [])

        def _ccell(strategy: str, rho: float) -> dict[str, object]:
            for entry in ccells:
                if entry.get("strategy") == strategy and abs(
                    float(entry.get("bloc_correlation", -1)) - rho
                ) < 1e-9:
                    return entry
            raise ValueError(f"negative-control summary missing {strategy} @ rho={rho}")

        tokens["BLOC_CTRL_AXIS"] = str(control.get("axis", "expertise_tier"))
        tokens["BLOC_CTRL_REP_LO"] = _format_float(float(ch["representative_lo"]))
        tokens["BLOC_CTRL_REP_HI"] = _format_float(float(ch["representative_hi"]))
        tokens["BLOC_CTRL_SEP_HI"] = _format_float(float(ch["sep_hi"]))
        tokens["BLOC_CTRL_EXPERTISE_HI"] = _format_float(
            float(_ccell("expertise_threshold", 0.9)["eie_mean"])
        )
        tokens["BLOC_CTRL_CORR_EXPERTISE_HI"] = _format_float(
            float(_ccell("expertise_threshold", 0.9)["corr_mean"])
        )
        tokens["BLOC_CTRL_CORR_IDEO_HI"] = _format_float(
            float(_ccell("ideological_selection", 0.9)["corr_mean"])
        )

    # Continuous representativeness dial: recovery error at the balanced vs
    # single-bloc extremes and the fraction of dial steps that increase error.
    concentration = payload.get("concentration")
    if isinstance(concentration, dict):
        tokens["BLOC_DIAL_RHO"] = f"{float(concentration['bloc_correlation']):.2f}"
        tokens["BLOC_DIAL_BALANCED"] = _format_float(float(concentration["eie_balanced"]))
        tokens["BLOC_DIAL_CONCENTRATED"] = _format_float(float(concentration["eie_concentrated"]))
        tokens["BLOC_DIAL_MONOTONE_FRAC"] = _format_float(
            float(concentration["monotone_increasing_fraction"])
        )
        cells = concentration.get("cells", [])
        tokens["BLOC_DIAL_N_LEVELS"] = str(len(cells))
    return tokens


def generate_variables(repo_root: Path | None = None) -> dict[str, str]:
    """Every manuscript {{TOKEN}} value, unioned across all producers.

    Single entry point the render pipeline hydrates from (via
    ``scripts/z_generate_manuscript_variables.py``). Every value is recomputed
    from real artifacts under ``output/data/`` -- nothing is hand-transcribed.
    Reads ``sweep_aggregated.csv`` (ranking / power-by-size / rep-vs-ideo / grid
    / alarm), ``independence_sweep.csv`` (error-correlation tolerance), the sweep
    per-seed JSON (statistical power + threshold separation), and the live
    cross-family artifact.
    """
    root = Path(repo_root) if repo_root is not None else _REPO_ROOT
    data_dir = root / "output" / "data"
    tokens: dict[str, str] = {}
    tokens.update(compute_tokens(data_dir / "sweep_aggregated.csv"))
    tokens.update(compute_alarm_tokens(data_dir / "alarm_timings.csv"))
    tokens.update(compute_independence_tokens(data_dir / "independence_sweep.csv"))
    tokens.update(compute_extension_tokens(root))
    tokens.update(
        compute_postdoc_panel_tokens(
            data_dir / "postdoc_panel_results.json",
            data_dir / "postdoc_panel_alignment.json",
        )
    )
    tokens.update(
        compute_cross_family_multiseed_tokens(data_dir / "cross_family_multiseed.json")
    )
    tokens.update(compute_bloc_phase_tokens(data_dir / "bloc_phase_summary.json"))
    return tokens


def main() -> None:
    print(json.dumps(generate_variables(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
