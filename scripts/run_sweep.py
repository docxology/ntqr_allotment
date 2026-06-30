from __future__ import annotations

import csv
import json
import argparse
from dataclasses import asdict
from pathlib import Path
from typing import Any

from ntqr_allotment.config import ExperimentProfile, load_experiment_profile
from ntqr_allotment.sweeps import (
    aggregate,
    representative_vs_ideological,
    run_sweep_parallel,
    strategy_ranking,
)


def _write_aggregates_csv(
    output_path: Path,
    aggregates: list[Any],
    profile: ExperimentProfile,
) -> None:
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
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
        )
        for item in aggregates:
            strategy, panel_size, mean_expertise, expertise_heterogeneity, bias_std, n_items, prevalence_a, n_experts = item.config_key
            writer.writerow(
                [
                    profile.name,
                    profile.config_hash,
                    strategy,
                    panel_size,
                    mean_expertise,
                    expertise_heterogeneity,
                    bias_std,
                    n_items,
                    prevalence_a,
                    n_experts,
                    item.n,
                    item.eie_mean,
                    item.eie_std,
                    item.eie_ci95,
                    item.mv_mean,
                    item.mv_std,
                    item.mv_ci95,
                ]
            )


def main() -> None:
    from ntqr_allotment.determinism import ensure_deterministic_hashing

    ensure_deterministic_hashing()  # representative_sortition is hash-order sensitive
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", help="experiment profile from manuscript/config.yaml")
    parser.add_argument("--workers", type=int, default=1, help="parallel worker processes")
    parser.add_argument(
        "--trial-timeout-s",
        type=float,
        help="mark a row degenerate if one trial exceeds this many seconds",
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config_path = root / "manuscript" / "config.yaml"
    profile = load_experiment_profile(config_path, args.profile)
    grid = profile.grid

    rows = run_sweep_parallel(
        grid,
        workers=args.workers,
        trial_timeout_s=args.trial_timeout_s,
    )
    aggregates = aggregate(rows)
    ranking = strategy_ranking(aggregates)
    effects = representative_vs_ideological(aggregates)

    output_dir = root / "output" / "data"
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "sweep_results.json"
    csv_path = output_dir / "sweep_aggregated.csv"

    payload = {
        "schema_version": 2,
        "metadata": {
            **profile.metadata(),
            "config_path": str(config_path.relative_to(root)),
            "seed_list": list(grid.seeds),
            "degenerate_rows": sum(1 for row in rows if row.n_trios == 0),
            "row_count": len(rows),
            "aggregate_count": len(aggregates),
            "workers": args.workers,
            "trial_timeout_s": args.trial_timeout_s,
            "command": (
                "uv run python scripts/run_sweep.py"
                + (f" --profile {args.profile}" if args.profile else "")
                + (f" --workers {args.workers}" if args.workers != 1 else "")
                + (f" --trial-timeout-s {args.trial_timeout_s}" if args.trial_timeout_s else "")
            ),
        },
        "rows": [asdict(row) for row in rows],
        "ranking": ranking,
        "rep_vs_ideo": [asdict(effect) for effect in effects],
    }
    json_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_aggregates_csv(csv_path, aggregates, profile)

    print(json_path.resolve())
    print(csv_path.resolve())


if __name__ == "__main__":
    main()
