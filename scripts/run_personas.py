#!/usr/bin/env python3
"""Empirical centerpiece (thin orchestrator): LLM-persona judges -> NTQR -> oracle.

Builds personas from a synthetic population, draws a representative trio, has a
live Ollama model (or the deterministic offline judge if no server) judge a
labeled arithmetic corpus, then runs the same NTQR evaluation used by the
synthetic track and writes the result to ``output/data/persona_results.json``.

Usage:
    uv run python scripts/run_personas.py [--model qwen2.5:3b] [--n-items 60]
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
)
from ntqr_allotment.sortition import representative_sortition

_ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen2.5:3b")
    ap.add_argument("--n-items", type=int, default=60)
    ap.add_argument("--n-experts", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-operand", type=int, default=9999,
                    help="larger operands -> harder items -> more independent model errors")
    args = ap.parse_args()

    corpus = make_arithmetic_corpus(args.n_items, seed=args.seed, max_operand=args.max_operand, max_error=5)
    population = generate_population(args.n_experts, seed=args.seed, mean_expertise=0.78)
    personas = personas_from_population(population)

    # Draw a representative trio by auditable lottery, then map ids -> personas.
    panel = representative_sortition(population, 3, seed=args.seed + 13)
    by_id = {p.id: p for p in personas}
    trio = [by_id[i] for i in panel.expert_ids]

    live = OllamaJudge(model=args.model).available()
    out_dir = _ROOT / "output" / "data"

    # A trio of error-INDEPENDENT judges. Live: different models + sampling so
    # their errors decorrelate (what NTQR needs). Offline: one deterministic judge.
    if live:
        judges = [
            OllamaJudge(model="qwen2.5:3b", temperature=0.6, seed=args.seed + 1),
            OllamaJudge(model="gemma3:4b", temperature=0.6, seed=args.seed + 2),
            OllamaJudge(model="qwen2.5:3b", temperature=1.0, seed=args.seed + 3),
        ]
        judge_labels = ["qwen2.5:3b@0.6", "gemma3:4b@0.6", "qwen2.5:3b@1.0"]
    else:
        det = DeterministicJudge(corpus, seed=args.seed)
        judges = [det, det, det]
        judge_labels = ["deterministic-offline"] * 3

    # Collect votes + the supervised oracle (always defined). The unsupervised
    # error-independent solver can be DEGENERATE when judges are too correlated
    # -- an honest finding, reported rather than crashed.
    votes = [
        judge_corpus(judges[i], trio[i], corpus,
                     cache_path=out_dir / f"persona_votes_j{i}_{judge_labels[i].replace(':','-').replace('@','-')}.json")
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
            "judges too correlated/accurate -> no real error-independent solution; "
            "this is the diversity NTQR requires and the case for sortition"
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
