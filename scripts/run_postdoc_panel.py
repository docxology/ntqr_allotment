#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from ntqr_allotment.config import load_live_postdoc_panel_config
from ntqr_allotment.postdoc_panel import build_postdoc_alignment, run_postdoc_panel_study


def main() -> None:
    args = _parse_args()
    root = Path(__file__).resolve().parents[1]
    config = load_live_postdoc_panel_config(root / args.config)
    data_dir = root / "output" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    seeds = tuple(args.seeds) if args.seeds else config.seeds
    panel_sizes = tuple(args.panel_sizes) if args.panel_sizes else config.panel_sizes
    strategies = tuple(args.strategies) if args.strategies else config.strategies
    n_reviewers = args.n_reviewers or config.n_reviewers
    n_applications = args.n_applications or config.n_applications
    runtime_config = {
        **config.metadata(),
        "seed_list": list(seeds),
        "n_reviewers": n_reviewers,
        "n_applications": n_applications,
        "panel_sizes": list(panel_sizes),
        "strategies": list(strategies),
    }
    runtime_hash = _hash_payload(runtime_config)

    common = {
        "seeds": seeds,
        "strategies": strategies,
        "panel_sizes": panel_sizes,
        "n_reviewers": n_reviewers,
        "n_applications": n_applications,
        "n_trios": config.n_trios,
        "prevalence_strong": config.prevalence_strong,
        "age_min": config.age_min,
        "age_max": config.age_max,
        "mean_expertise": config.mean_expertise,
        "expertise_heterogeneity": config.expertise_heterogeneity,
        "age_bias_std": config.age_bias_std,
        "model": config.model,
        "base_url": config.base_url,
        "temperature": config.temperature,
        "num_predict": config.num_predict,
        "timeout": config.timeout,
        "progress_every": config.progress_every,
    }
    analytical = run_postdoc_panel_study(
        **common,
        track="analytical",
        config_hash=runtime_hash,
        require_live=False,
        cache_path=None,
    )
    tracks: dict[str, object] = {"analytical": analytical}
    live = None
    if not args.offline:
        live = run_postdoc_panel_study(
            **common,
            track="live",
            config_hash=runtime_hash,
            require_live=config.require_live if args.require_live is None else args.require_live,
            cache_path=root / config.cache_path,
        )
        tracks["live"] = live

    alignment = build_postdoc_alignment(analytical, live if live is not None else analytical)
    combined = {
        "schema_version": 1,
        "config_hash": runtime_hash,
        "source_config_hash": config.config_hash,
        "config": runtime_config,
        "tracks": tracks,
        "rows": analytical["rows"] + ((live or {"rows": []})["rows"]),
        "aggregates": analytical["aggregates"] + ((live or {"aggregates": []})["aggregates"]),
        "alignment": alignment,
        "live_ollama": bool(live and live.get("live_ollama")),
        "model": config.model,
        "model_provenance": (live or analytical).get("model_provenance"),
        "caveat": (
            "Synthetic applicants and age metadata only; the live track uses one "
            "local Gemma model prompted as synthetic reviewers and is not a "
            "human-review substitute."
        ),
    }
    results_path = data_dir / "postdoc_panel_results.json"
    alignment_path = data_dir / "postdoc_panel_alignment.json"
    results_path.write_text(json.dumps(combined, indent=2, sort_keys=True), encoding="utf-8")
    alignment_path.write_text(json.dumps(alignment, indent=2, sort_keys=True), encoding="utf-8")
    print(results_path.resolve())
    print(alignment_path.resolve())


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="manuscript/config.yaml")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--require-live", dest="require_live", action="store_true")
    parser.add_argument("--allow-unavailable-live", dest="require_live", action="store_false")
    parser.set_defaults(require_live=None)
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--panel-sizes", type=int, nargs="+")
    parser.add_argument("--strategies", nargs="+")
    parser.add_argument("--n-reviewers", type=int)
    parser.add_argument("--n-applications", type=int)
    return parser.parse_args()


def _hash_payload(payload: object) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]


if __name__ == "__main__":
    main()
