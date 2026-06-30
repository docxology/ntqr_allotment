#!/usr/bin/env python3
"""Measure the per-trio mechanism behind the panel-size effect (thin orchestrator).

For a 12-seed subset of the reported regime grid (a structural association, not a
headline CI), records every usable trio's (eie_error, |error-correlation|,
judge-accuracy, trio_rank) via :func:`ntqr_allotment.trio_conditioning.panel_trio_diagnostics`,
then writes the raw records plus a by-size / by-rank / association summary to
``output/data/trio_conditioning.json``. Lets the manuscript report a measured
diagnostic that rules out the size-growing error-correlation explanation without
claiming a positive mechanism for the tiny paired size deltas.
"""
from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from itertools import product
from pathlib import Path

from ntqr_allotment.config import load_experiment_profile
from ntqr_allotment.pipeline import TrialConfig
from ntqr_allotment.trio_conditioning import (
    panel_trio_diagnostics,
    pearson,
)

ROOT = Path(__file__).resolve().parents[1]
N_SEEDS = 12  # mechanism is structural; a subset keeps this analysis affordable
WORKERS = 12


def _configs() -> tuple[list[TrialConfig], int]:
    grid = load_experiment_profile(ROOT / "manuscript" / "config.yaml", "manuscript_contrast").grid
    configs: list[TrialConfig] = []
    for strat, size, me, het, bias, n_items, prev, seed in product(
        grid.strategies,
        grid.panel_sizes,
        grid.mean_expertises,
        grid.expertise_heterogeneities,
        grid.bias_stds,
        grid.n_items_values,
        grid.prevalence_as,
        range(N_SEEDS),
    ):
        configs.append(
            TrialConfig(
                strategy=strat,
                panel_size=size,
                n_experts=grid.n_experts,
                n_items=n_items,
                prevalence_a=prev,
                mean_expertise=me,
                expertise_heterogeneity=het,
                bias_std=bias,
                seed=seed,
            )
        )
    return configs, grid.n_trios


def _run(args: tuple[TrialConfig, int]) -> list[dict]:
    config, n_trios = args
    return [asdict(rec) for rec in panel_trio_diagnostics(config, n_trios=n_trios)]


def main() -> None:
    from ntqr_allotment.determinism import ensure_deterministic_hashing

    ensure_deterministic_hashing()  # representative_sortition is hash-order sensitive
    configs, n_trios = _configs()
    records: list[dict] = []
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        for batch in pool.map(_run, [(c, n_trios) for c in configs], chunksize=8):
            records.extend(batch)

    sizes = sorted({r["panel_size"] for r in records})
    by_size_err = mean_by_size_from_dicts(records, "eie_error")
    by_size_corr = mean_by_size_from_dicts(records, "mean_abs_corr")
    by_size_acc = mean_by_size_from_dicts(records, "mean_judge_accuracy")
    # First-trio (rank 0) vs later-trio (rank > 0) mean error, by size.
    rank0 = [r["eie_error"] for r in records if r["trio_rank"] == 0]
    rankN = [r["eie_error"] for r in records if r["trio_rank"] > 0]
    # Per-trio associations (do error and correlation/accuracy/rank move together?).
    err = [r["eie_error"] for r in records]
    assoc = {
        "error_vs_corr": pearson(err, [r["mean_abs_corr"] for r in records]),
        "error_vs_accuracy": pearson(err, [r["mean_judge_accuracy"] for r in records]),
        "error_vs_trio_rank": pearson(err, [float(r["trio_rank"]) for r in records]),
    }
    payload = {
        "schema_version": 1,
        "n_seeds": N_SEEDS,
        "n_records": len(records),
        "sizes": sizes,
        "mean_error_by_size": by_size_err,
        "mean_abs_corr_by_size": by_size_corr,
        "mean_judge_accuracy_by_size": by_size_acc,
        "mean_error_first_trio": sum(rank0) / len(rank0) if rank0 else None,
        "mean_error_later_trios": sum(rankN) / len(rankN) if rankN else None,
        "associations_pearson": assoc,
        "records": records,
    }
    out = ROOT / "output" / "data" / "trio_conditioning.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {out} ({len(records)} trio records, {N_SEEDS} seeds)")
    print("mean error by size:", {k: round(v, 4) for k, v in by_size_err.items()})
    print("mean |corr| by size:", {k: round(v, 4) for k, v in by_size_corr.items()})
    print("first-trio err:", round(payload["mean_error_first_trio"], 4),
          "| later-trios err:", round(payload["mean_error_later_trios"], 4))
    print("associations:", {k: round(v, 3) for k, v in assoc.items()})


def mean_by_size_from_dicts(records: list[dict], field: str) -> dict[int, float]:
    buckets: dict[int, list[float]] = {}
    for rec in records:
        buckets.setdefault(int(rec["panel_size"]), []).append(float(rec[field]))
    return {size: sum(v) / len(v) for size, v in sorted(buckets.items())}


if __name__ == "__main__":
    main()
