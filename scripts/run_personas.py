#!/usr/bin/env python3
"""Empirical centerpiece (thin orchestrator): LLM-persona judges -> NTQR -> oracle.

Builds personas from a synthetic population, draws a representative trio, has
live Ollama models judge a labeled arithmetic corpus, then runs the same NTQR
evaluation used by the synthetic track. ``--offline`` explicitly selects seeded
synthetic judges and writes diagnostics separately from recorded live results.

Usage:
    uv run python scripts/run_personas.py [--model qwen2.5:3b] [--n-items 60]
    uv run python scripts/run_personas.py --offline --output-dir /tmp/personas
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ntqr_allotment.corpus import make_arithmetic_corpus
from ntqr_allotment.experts import generate_population
from ntqr_allotment.ntqr_eval import (
    closest_solution,
    error_independent_solutions,
    supervised_oracle,
)
from ntqr_allotment.personas import (
    DeterministicJudge,
    OllamaJudge,
    judge_corpus,
    personas_from_population,
    provenance_for_judge,
)
from ntqr_allotment.sortition import representative_sortition

_ROOT = Path(__file__).resolve().parent.parent


def _live_judges(model: str, seed: int) -> list[OllamaJudge]:
    """Configure the mixed-model protocol without contacting Ollama."""
    return [
        OllamaJudge(model=model, temperature=0.6, seed=seed + 1),
        OllamaJudge(model="gemma3:4b", temperature=0.6, seed=seed + 2),
        OllamaJudge(model=model, temperature=1.0, seed=seed + 3),
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen2.5:3b")
    ap.add_argument("--offline", action="store_true",
                    help="use synthetic seeded judges, without contacting Ollama")
    ap.add_argument("--output-dir", type=Path,
                    help="results/cache directory; offline defaults to output/diagnostics/personas")
    ap.add_argument("--n-items", type=int, default=60)
    ap.add_argument("--n-experts", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-operand", type=int, default=9999,
                    help="larger operands produce harder arithmetic items")
    args = ap.parse_args()
    live_dir = _ROOT / "output" / "data"
    out_dir = args.output_dir
    if out_dir is None:
        out_dir = _ROOT / "output" / "diagnostics" / "personas" if args.offline else live_dir
    if args.offline and out_dir.resolve() == live_dir.resolve():
        ap.error("--offline requires an output directory separate from recorded live results")

    corpus = make_arithmetic_corpus(args.n_items, seed=args.seed, max_operand=args.max_operand, max_error=5)
    population = generate_population(args.n_experts, seed=args.seed, mean_expertise=0.78)
    personas = personas_from_population(population)

    # Draw a representative trio by auditable lottery, then map ids -> personas.
    panel = representative_sortition(population, 3, seed=args.seed + 13)
    by_id = {p.id: p for p in personas}
    trio = [by_id[i] for i in panel.expert_ids]

    live = not args.offline
    # Model/decode variation and distinct synthetic seeds do not establish
    # conditional error independence; the solver can still be degenerate.
    cache_namespaces: list[str | None] = [None] * 3
    if live:
        judges = _live_judges(args.model, args.seed)
        for judge in {judge.model: judge for judge in judges}.values():
            if not judge.available():
                ap.error(
                    f"live Ollama model {judge.model!r} is unavailable; "
                    "use --offline explicitly for a synthetic diagnostic"
                )
        judge_labels = [f"{judge.model}@{judge.temperature}" for judge in judges]
        live_provenance = [provenance_for_judge(judge) for judge in judges]
        for judge, provenance in zip(judges, live_provenance):
            if not provenance.digest or not provenance.digest.strip():
                raise ValueError(
                    f"cannot cache live votes for {judge.model!r}: "
                    "Ollama did not provide an immutable model digest"
                )
        cache_namespaces = [provenance.digest for provenance in live_provenance]
    else:
        judges = [DeterministicJudge(corpus, seed=args.seed + i + 1) for i in range(3)]
        judge_labels = [f"deterministic-offline@seed={judge.seed}" for judge in judges]

    # Collect votes + the supervised oracle (always defined). The unsupervised
    # error-independent solver can be DEGENERATE when judges are too correlated
    # -- an honest finding, reported rather than crashed.
    votes = [
        judge_corpus(judges[i], trio[i], corpus,
                     cache_path=out_dir / f"persona_votes_j{i}_{judge_labels[i].replace(':','-').replace('@','-')}.json",
                     cache_namespace=cache_namespaces[i])
        for i in range(3)
    ]
    oracle = supervised_oracle(votes, corpus)
    empirical_acc = [
        round(sum(1 for it, v in zip(corpus, vj) if v == it.true_label) / len(corpus), 4)
        for vj in votes
    ]
    eie_sols = error_independent_solutions(votes)
    degenerate = not eie_sols
    eie = None if degenerate else closest_solution(eie_sols, oracle)

    payload = {
        "judge_models": judge_labels,
        "live_ollama": live,
        "seed": args.seed,
        "judge_seeds": [judge.seed for judge in judges],
        "independence_note": (
            "Distinct models, decode seeds or synthetic hash seeds do not establish "
            "conditional error independence."
        ),
        "n_items": args.n_items,
        "max_operand": args.max_operand,
        "panel_audit_hash": panel.audit_hash,
        "persona_ids": [p.id for p in trio],
        "persona_specs": [
            {"id": p.id, "expertise_level": p.expertise_level, "leaning": p.leaning,
             "ideology": p.ideology, "target_accuracy": round(p.target_accuracy, 4)}
            for p in trio
        ],
        "empirical_per_judge_accuracy": empirical_acc,
        "oracle_prevalence_a": round(oracle.prevalence_a, 4),
        "oracle_accuracies": [round(a, 4) for a in oracle.accuracies],
        "ntqr_degenerate": degenerate,
        "degenerate_note": (
            "No physical error-independent solution was found for these votes; "
            "this does not establish why the fitted model is degenerate."
        ) if degenerate else None,
        "eie_prevalence_a": None if degenerate else round(eie.prevalence_a, 4),
        "eie_accuracies": None if degenerate else [round(a, 4) for a in eie.accuracies],
        "eie_error": None if degenerate else round(eie.error_vs(oracle), 4),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "persona_results.json"
    out_path.write_text(json.dumps(payload, indent=2))
    print(str(out_path))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
