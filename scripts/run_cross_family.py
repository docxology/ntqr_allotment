#!/usr/bin/env python3
"""Cross-FAMILY decorrelation experiment (thin orchestrator).

Builds a mixed-family panel, measures realized pairwise error-correlations with
NTQR's supervised estimator, and reports the SIGNED same-vs-cross family
contrast. Uses live Ollama models when a server is reachable, otherwise the
deterministic offline core (families derived from the requested model tags).

The contrast is reported with its honest provenance label
("live empirical, n-limited") and as a SIGNED magnitude — never as
"sortition validated" in absolute terms.

Usage:
    uv run python scripts/run_cross_family.py [--n-items 400] [--shared-strength 0.85] \
        [--models qwen2.5:3b gemma3:4b]
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from ntqr_allotment.config import load_live_cross_family_config
from ntqr_allotment.corpus import make_arithmetic_corpus
from ntqr_allotment.cross_family import (
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=_ROOT / "manuscript" / "config.yaml")
    ap.add_argument("--n-items", type=int)
    ap.add_argument("--seed", type=int)
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
    models = list(args.models or live_config.models)
    replicates_per_model = (
        live_config.replicates_per_model
        if args.replicates_per_model is None
        else args.replicates_per_model
    )
    if replicates_per_model < 1:
        raise SystemExit("--replicates-per-model must be >= 1")
    seed = live_config.seeds[0] if args.seed is None else args.seed
    n_items = live_config.n_items if args.n_items is None else args.n_items
    shared_strength = (
        live_config.shared_strength if args.shared_strength is None else args.shared_strength
    )
    target_accuracy = (
        live_config.target_accuracy if args.target_accuracy is None else args.target_accuracy
    )
    base_url = live_config.base_url if args.base_url is None else args.base_url
    temperature = live_config.temperature if args.temperature is None else args.temperature
    num_predict = live_config.num_predict if args.num_predict is None else args.num_predict
    timeout = live_config.timeout if args.timeout is None else args.timeout
    progress_every = (
        live_config.progress_every if args.progress_every is None else args.progress_every
    )
    require_live = args.require_live or live_config.require_live

    corpus = make_arithmetic_corpus(
        n_items, seed=seed, max_operand=9999, max_error=5
    )
    persona = PersonaSpec("cf", "competent", "neutral", "center", target_accuracy)

    live = ollama_panel_available(models, base_url=base_url)
    if require_live and not live:
        raise SystemExit("required live Ollama panel is unavailable")
    if live:
        live_models = [m for m in models for _ in range(replicates_per_model)]
        panel = collect_live_family_votes(
            models=live_models,
            persona=persona,
            corpus=corpus,
            base_url=base_url,
            temperature=temperature,
            timeout=timeout,
            num_predict=num_predict,
            progress_every=progress_every,
        )
        judge_provenance = _live_judge_provenance(
            live_models, base_url, temperature, timeout, num_predict
        )
    else:
        families: list[str] = []
        for model in models:
            fam = model_family(model)
            families.extend([fam for _ in range(replicates_per_model)])
        panel = make_family_correlated_votes(
            families=families,
            corpus=corpus,
            target_accuracy=target_accuracy,
            shared_strength=shared_strength,
            seed=seed,
        )
        judge_provenance = _offline_judge_provenance(models, replicates_per_model)

    matrix = build_pairwise_error_correlation_matrix(panel, corpus)
    contrast = contrast_same_vs_cross_family(matrix)
    pair_records = pair_correlation_records(matrix)
    group_summaries = summarize_pair_groups(matrix, seed=seed)
    nonzero_pairs = sum(1 for value in matrix.pair_abs_corr.values() if abs(value) > 0.0)
    run_settings = {
        "models": models,
        "replicates_per_model": replicates_per_model,
        "seed": seed,
        "n_items": n_items,
        "target_accuracy": target_accuracy,
        "shared_strength": shared_strength,
        "base_url": base_url,
        "temperature": temperature,
        "num_predict": num_predict,
        "timeout": timeout,
        "require_live": require_live,
    }

    payload = {
        "schema_version": 2,
        "live_ollama": live,
        "source_config_hash": live_config.config_hash,
        "config_hash": _hash_payload(run_settings),
        "config_path": str(args.config.resolve().relative_to(_ROOT)),
        "models": models,
        "replicates_per_model": replicates_per_model,
        "run_count": 1,
        "seed": seed,
        "n_items": n_items,
        "target_accuracy": target_accuracy,
        "shared_strength": shared_strength,
        "temperature": temperature,
        "num_predict": num_predict,
        "timeout": timeout,
        "require_live": require_live,
        "n_judges": len(panel),
        "n_families": len(set(matrix.families)),
        "judge_ids": list(matrix.judge_ids),
        "families": list(matrix.families),
        "judge_provenance": judge_provenance,
        "pair_abs_corr": {k: round(v, 6) for k, v in matrix.pair_abs_corr.items()},
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
        "nonzero_pairs": nonzero_pairs,
        "total_pairs": len(matrix.pair_abs_corr),
        "mean_abs_same_family": round(contrast.mean_abs_same_family, 6),
        "mean_abs_cross_family": round(contrast.mean_abs_cross_family, 6),
        "delta_cross_minus_same": round(contrast.delta_cross_minus_same, 6),
        "n_same_pairs": contrast.n_same_pairs,
        "n_cross_pairs": contrast.n_cross_pairs,
        "provenance_label": contrast.label,
        "interpretation": (
            "Signed magnitude (cross minus same). A negative delta means cross-"
            "family pairs were less error-correlated on this sample. NOT an "
            "absolute 'sortition validated' claim; n-limited."
        ),
    }
    out_dir = _ROOT / "output" / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "cross_family_results.json"
    out_path.write_text(json.dumps(payload, indent=2))
    print(str(out_path))
    print(json.dumps(payload, indent=2))


def _live_judge_provenance(
    models: list[str],
    base_url: str,
    temperature: float,
    timeout: float,
    num_predict: int,
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for idx, model in enumerate(models):
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


def _offline_judge_provenance(
    models: list[str], replicates_per_model: int = 2
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for idx, model in enumerate([m for m in models for _ in range(replicates_per_model)]):
        records.append(
            {
                "judge_id": f"{model_family(model)}-{idx}",
                "model": model,
                "family": model_family(model),
                "digest": None,
                "temperature": 0.0,
                "seed": idx,
                "num_predict": 0,
            }
        )
    return records


if __name__ == "__main__":
    main()
