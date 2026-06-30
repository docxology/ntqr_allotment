#!/usr/bin/env python3
"""Multi-seed cross-FAMILY decorrelation experiment (thin orchestrator).

Runs the cross-family panel over several independent corpora (one per seed),
measures the same-vs-cross family error-correlation delta on each, and aggregates
the deltas into a sign-stability summary (:func:`aggregate_contrasts`). This
upgrades the single-run existence demonstration toward a measured estimate: it
reports how often the cross<same direction holds across runs, with mean and spread.

Uses live Ollama when reachable, otherwise the deterministic offline core. The
result is labeled "live empirical, n-limited (multi-seed)" — more runs, not a
different claim class. Writes ``output/data/cross_family_multiseed.json``.

Usage:
    uv run python scripts/run_cross_family_multiseed.py [--seeds 0 1 2 3] \
        [--n-items 150] [--models qwen2.5:3b gemma3:4b]
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from ntqr_allotment.config import load_live_cross_family_config
from ntqr_allotment.corpus import make_arithmetic_corpus
from ntqr_allotment.cross_family import (
    aggregate_contrasts,
    build_pairwise_error_correlation_matrix,
    collect_live_family_votes,
    contrast_same_vs_cross_family,
    make_family_correlated_votes,
    model_family,
    ollama_panel_available,
    pair_correlation_records,
    summarize_pair_groups,
)
from ntqr_allotment.personas import OllamaJudge, PersonaSpec, provenance_for_judge

_ROOT = Path(__file__).resolve().parent.parent


def _hash_payload(payload: dict[str, object]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]


def _panel_for_seed(args, seed, live):
    corpus = make_arithmetic_corpus(args.n_items, seed=seed, max_operand=9999, max_error=5)
    persona = PersonaSpec("cf", "competent", "neutral", "center", args.target_accuracy)
    if live:
        live_models = [
            m for m in args.models for _ in range(args.replicates_per_model)
        ]
        panel = collect_live_family_votes(
            models=live_models,
            persona=persona,
            corpus=corpus,
            base_url=args.base_url,
            temperature=args.temperature,
            timeout=args.timeout,
            num_predict=args.num_predict,
            progress_every=args.progress_every,
        )
    else:
        families: list[str] = []
        for model in args.models:
            fam = model_family(model)
            families.extend([fam for _ in range(args.replicates_per_model)])
        panel = make_family_correlated_votes(
            families=families,
            corpus=corpus,
            target_accuracy=args.target_accuracy,
            shared_strength=args.shared_strength,
            seed=seed,
        )
    return panel, corpus


def _live_judge_provenance(
    models, replicates_per_model, base_url, temperature, timeout, num_predict
):
    records = []
    live_models = [m for m in models for _ in range(replicates_per_model)]
    for idx, model in enumerate(live_models):
        judge = OllamaJudge(
            model=model,
            base_url=base_url,
            timeout=timeout,
            temperature=temperature,
            seed=idx,
            num_predict=num_predict,
        )
        record = provenance_for_judge(judge).to_dict()
        record["judge_id"] = f"{model_family(model)}-{model}-{idx}"
        records.append(record)
    return records


def _offline_judge_provenance(models, replicates_per_model=2):
    records = []
    for idx, model in enumerate([m for m in models for _ in range(replicates_per_model)]):
        family = model_family(model)
        records.append(
            {
                "judge_id": f"{family}-{idx}",
                "model": model,
                "family": family,
                "digest": None,
                "temperature": 0.0,
                "seed": idx,
                "num_predict": 0,
            }
        )
    return records


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=_ROOT / "manuscript" / "config.yaml")
    ap.add_argument("--seeds", type=int, nargs="+")
    ap.add_argument("--n-items", type=int)
    ap.add_argument("--shared-strength", type=float)
    ap.add_argument("--target-accuracy", type=float)
    ap.add_argument("--models", nargs="+")
    ap.add_argument("--replicates-per-model", type=int)
    ap.add_argument("--base-url")
    ap.add_argument("--temperature", type=float)
    ap.add_argument("--num-predict", type=int)
    ap.add_argument("--timeout", type=float)
    ap.add_argument("--progress-every", type=int)
    ap.add_argument("--require-live", action="store_true")
    args = ap.parse_args()
    live_config = load_live_cross_family_config(args.config)
    args.seeds = list(args.seeds or live_config.seeds)
    args.n_items = live_config.n_items if args.n_items is None else args.n_items
    args.shared_strength = (
        live_config.shared_strength if args.shared_strength is None else args.shared_strength
    )
    args.target_accuracy = (
        live_config.target_accuracy if args.target_accuracy is None else args.target_accuracy
    )
    args.models = list(args.models or live_config.models)
    args.replicates_per_model = (
        live_config.replicates_per_model
        if args.replicates_per_model is None
        else args.replicates_per_model
    )
    if args.replicates_per_model < 1:
        raise SystemExit("--replicates-per-model must be >= 1")
    args.base_url = live_config.base_url if args.base_url is None else args.base_url
    args.temperature = live_config.temperature if args.temperature is None else args.temperature
    args.num_predict = live_config.num_predict if args.num_predict is None else args.num_predict
    args.timeout = live_config.timeout if args.timeout is None else args.timeout
    args.progress_every = (
        live_config.progress_every if args.progress_every is None else args.progress_every
    )
    args.require_live = args.require_live or live_config.require_live

    live = ollama_panel_available(args.models, base_url=args.base_url)
    if args.require_live and not live:
        raise SystemExit("required live Ollama panel is unavailable")
    judge_provenance = (
        _live_judge_provenance(
            args.models,
            args.replicates_per_model,
            args.base_url,
            args.temperature,
            args.timeout,
            args.num_predict,
        )
        if live
        else _offline_judge_provenance(args.models, args.replicates_per_model)
    )
    run_settings = {
        "models": args.models,
        "replicates_per_model": args.replicates_per_model,
        "seeds": args.seeds,
        "n_items": args.n_items,
        "target_accuracy": args.target_accuracy,
        "shared_strength": args.shared_strength,
        "base_url": args.base_url,
        "temperature": args.temperature,
        "num_predict": args.num_predict,
        "timeout": args.timeout,
        "require_live": args.require_live,
    }
    per_run = []
    contrasts = []
    for seed in args.seeds:
        panel, corpus = _panel_for_seed(args, seed, live)
        matrix = build_pairwise_error_correlation_matrix(panel, corpus)
        contrast = contrast_same_vs_cross_family(matrix)
        pair_records = pair_correlation_records(matrix)
        group_summaries = summarize_pair_groups(matrix, seed=seed)
        nonzero_pairs = sum(
            1 for value in matrix.pair_abs_corr.values() if abs(value) > 0.0
        )
        contrasts.append(contrast)
        per_run.append(
            {
                "seed": seed,
                "mean_abs_same_family": round(contrast.mean_abs_same_family, 6),
                "mean_abs_cross_family": round(contrast.mean_abs_cross_family, 6),
                "delta_cross_minus_same": round(contrast.delta_cross_minus_same, 6),
                "n_same_pairs": contrast.n_same_pairs,
                "n_cross_pairs": contrast.n_cross_pairs,
                "nonzero_pairs": nonzero_pairs,
                "total_pairs": len(matrix.pair_abs_corr),
                "pair_records": [
                    {**record._asdict(), "abs_corr": round(record.abs_corr, 6)}
                    for record in pair_records
                ],
                "pair_group_summaries": [
                    {
                        **summary._asdict(),
                        "mean_abs_corr": round(summary.mean_abs_corr, 6),
                        "std_abs_corr": round(summary.std_abs_corr, 6),
                        "ci_low": round(summary.ci_low, 6),
                        "ci_high": round(summary.ci_high, 6),
                    }
                    for summary in group_summaries
                ],
                "judge_provenance": judge_provenance,
            }
        )

    agg = aggregate_contrasts(contrasts)
    payload = {
        "schema_version": 2,
        "live_ollama": live,
        "source_config_hash": live_config.config_hash,
        "config_hash": _hash_payload(run_settings),
        "config_path": str(args.config.resolve().relative_to(_ROOT)),
        "models": list(args.models),
        "replicates_per_model": args.replicates_per_model,
        "judge_provenance": judge_provenance,
        "n_items": args.n_items,
        "temperature": args.temperature,
        "num_predict": args.num_predict,
        "timeout": args.timeout,
        "require_live": args.require_live,
        "run_count": agg.n_runs,
        "n_runs": agg.n_runs,
        "same_pairs_per_run": per_run[0]["n_same_pairs"],
        "cross_pairs_per_run": per_run[0]["n_cross_pairs"],
        "total_pairs_per_run": per_run[0]["total_pairs"],
        "nonzero_pairs_per_run": [row["nonzero_pairs"] for row in per_run],
        "mean_delta": round(agg.mean_delta, 6),
        "std_delta": round(agg.std_delta, 6),
        "min_delta": round(agg.min_delta, 6),
        "max_delta": round(agg.max_delta, 6),
        "sign_stability": round(agg.sign_stability, 6),
        "deltas": [round(d, 6) for d in agg.deltas],
        "per_run": per_run,
        "provenance_label": agg.label,
        "interpretation": (
            "sign_stability is the fraction of independent runs whose cross-minus-"
            "same delta is negative (the decorrelation direction). 1.0 means every "
            "run agreed; ~0.5 means the sign is a coin-flip. n-limited."
        ),
    }
    out_dir = _ROOT / "output" / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "cross_family_multiseed.json"
    out_path.write_text(json.dumps(payload, indent=2))
    print(str(out_path))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
