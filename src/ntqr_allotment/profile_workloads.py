"""Small deterministic workloads using real NTQR and CBC, without model calls."""

from __future__ import annotations
import hashlib
import json
from .runtime_profile import Workload


def workloads():
    from .ensemble import panel_agreement_moments
    from .experts import generate_population, sample_items, sample_votes
    from .pipeline import TrialConfig, run_trial_ensemble
    from .ntqr_eval import alarm_misaligned
    from .sortition import representative_sortition

    population = generate_population(12, seed=103)
    items = sample_items(96, prevalence_a=0.6, seed=104)
    votes = [
        sample_votes(expert, items, seed=105 + i) for i, expert in enumerate(population)
    ]
    specs = [
        (
            "agreement",
            lambda: panel_agreement_moments(votes),
            "12 judges, 96 seeded synthetic items; exact NTQR agreement moments",
            votes,
        ),
        (
            "recovery",
            lambda: run_trial_ensemble(
                TrialConfig(
                    strategy="random_selection",
                    n_experts=12,
                    panel_size=3,
                    n_items=96,
                    seed=108,
                ),
                n_trios=1,
            ),
            "One real trio solve with synthetic oracle scoring",
            {"seed": 108, "n_items": 96, "n_experts": 12},
        ),
        (
            "alarm",
            lambda: alarm_misaligned([row[:6] for row in votes[:3]], max_q=6),
            "Real finite-key alarm: three judges, six items, safety factors (2,2)",
            [row[:6] for row in votes[:3]],
        ),
    ]
    specs.append(
        (
            "sortition",
            lambda: representative_sortition(population, 3, seed=109, panel_count=4),
            "Native CBC maximin lottery over a seeded population; parent CPU excludes solver children",
            {"seed": 109, "panel_count": 4, "population_seed": 103, "panel_size": 3},
        )
    )
    return [
        Workload(
            name,
            run,
            description,
            hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest(),
        )
        for name, run, description, inputs in specs
    ]
