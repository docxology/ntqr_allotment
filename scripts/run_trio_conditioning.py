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
from itertools import product
from pathlib import Path

from ntqr_allotment.config import load_experiment_profile
from ntqr_allotment.pipeline import TrialConfig
from ntqr_allotment.trio_conditioning import (
    TrioDiagnostic,
    panel_trio_diagnostics,
    summarize_trio_diagnostics,
)

ROOT = Path(__file__).resolve().parents[1]
N_SEEDS = 12  # mechanism is structural; a subset keeps this analysis affordable
WORKERS = 12


def _configs() -> tuple[list[TrialConfig], int]:
    grid = load_experiment_profile(ROOT / "docs" / "manuscript" / "config.yaml", "manuscript_contrast").grid
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


def _run(args: tuple[TrialConfig, int]) -> list[TrioDiagnostic]:
    config, n_trios = args
    return panel_trio_diagnostics(config, n_trios=n_trios)


def main() -> None:
    from ntqr_allotment.determinism import ensure_deterministic_hashing

    ensure_deterministic_hashing()  # representative_sortition is hash-order sensitive
    configs, n_trios = _configs()
    records: list[TrioDiagnostic] = []
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        for batch in pool.map(_run, [(c, n_trios) for c in configs], chunksize=8):
            records.extend(batch)

    _write_summary(
        ROOT / "output" / "data" / "trio_conditioning.json",
        records,
        n_seeds=N_SEEDS,
    )


def _write_summary(out: Path, records: list[TrioDiagnostic], *, n_seeds: int) -> None:
    payload = summarize_trio_diagnostics(records, n_seeds=n_seeds)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {out} ({len(records)} trio records, {n_seeds} seeds)")
    print("mean error by size:", {k: round(v, 4) for k, v in payload["mean_error_by_size"].items()})
    print("mean |corr| by size:", {k: round(v, 4) for k, v in payload["mean_abs_corr_by_size"].items()})
    first = payload["mean_error_first_trio"]
    later = payload["mean_error_later_trios"]
    print("first-trio err:", round(first, 4) if first is not None else "unavailable",
          "| later-trios err:", round(later, 4) if later is not None else "unavailable")
    print("associations:", {
        k: round(v, 3) if v is not None else None
        for k, v in payload["associations_pearson"].items()
    })


if __name__ == "__main__":
    main()
