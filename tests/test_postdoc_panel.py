from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ntqr_allotment.postdoc_panel import (
    POSTDOC_STRATEGY_LABELS,
    ReviewerProfile,
    aggregate_postdoc_rows,
    analytical_review_vote_probability,
    build_postdoc_alignment,
    make_postdoc_application_corpus,
    make_reviewer_population,
    reviewers_to_experts,
    run_postdoc_panel_study,
    sample_analytical_reviewer_votes,
)


def test_postdoc_corpus_is_deterministic_and_age_is_independent_of_quality() -> None:
    a = make_postdoc_application_corpus(400, seed=3)
    b = make_postdoc_application_corpus(400, seed=3)

    assert a == b
    ages = np.array([item.age for item in a])
    quality = np.array([item.quality_score for item in a])
    labels = [item.true_label for item in a]
    assert int(ages.min()) >= 28
    assert int(ages.max()) <= 52
    assert abs(float(np.corrcoef(ages, quality)[0, 1])) < 0.12
    assert 0.4 < labels.count("a") / len(labels) < 0.6
    assert "Age metadata" in a[0].text


def test_reviewer_bias_moves_recommendation_probabilities_by_age_direction() -> None:
    old_item = make_postdoc_application_corpus(1, seed=9, age_min=52, age_max=52)[0]
    young_item = make_postdoc_application_corpus(1, seed=10, age_min=28, age_max=28)[0]
    older_favoring = ReviewerProfile("r001", 0.74, 1.0, "older-favoring", "medium")
    younger_favoring = ReviewerProfile("r002", 0.74, -1.0, "younger-favoring", "medium")

    assert analytical_review_vote_probability(
        older_favoring, old_item
    ) > analytical_review_vote_probability(older_favoring, young_item)
    assert analytical_review_vote_probability(
        younger_favoring, old_item
    ) < analytical_review_vote_probability(younger_favoring, young_item)


def test_reviewer_expertise_improves_analytical_oracle_accuracy() -> None:
    items = make_postdoc_application_corpus(200, seed=21)
    high = ReviewerProfile("r010", 0.9, 0.0, "near-neutral", "high")
    low = ReviewerProfile("r011", 0.58, 0.0, "near-neutral", "low")

    high_votes = sample_analytical_reviewer_votes(high, items, seed=1)
    low_votes = sample_analytical_reviewer_votes(low, items, seed=1)
    high_accuracy = sum(v == item.true_label for v, item in zip(high_votes, items)) / len(items)
    low_accuracy = sum(v == item.true_label for v, item in zip(low_votes, items)) / len(items)

    assert high_accuracy > low_accuracy


def test_reviewer_population_maps_same_bias_groups_to_sortition_features() -> None:
    reviewers = make_reviewer_population(24, seed=4)
    experts = reviewers_to_experts(reviewers)

    assert {expert.ideology for expert in experts} <= set(POSTDOC_STRATEGY_LABELS) | {
        "younger-favoring",
        "near-neutral",
        "older-favoring",
    }
    assert {expert.expertise_tier for expert in experts}
    assert all(expert.id.startswith("r") for expert in experts)


def test_run_postdoc_panel_offline_artifact_schema(tmp_path: Path) -> None:
    payload = run_postdoc_panel_study(
        seeds=(0, 1),
        strategies=(
            "representative_sortition",
            "random_selection",
            "expertise_threshold",
        ),
        panel_sizes=(3,),
        n_reviewers=18,
        n_applications=36,
        n_trios=1,
        prevalence_strong=0.5,
        age_min=28,
        age_max=52,
        mean_expertise=0.74,
        expertise_heterogeneity=0.08,
        age_bias_std=0.65,
        track="analytical",
        config_hash="abc123def456",
        require_live=False,
        cache_path=tmp_path / "unused.json",
    )

    assert payload["schema_version"] == 1
    assert payload["model"] == "gemma3:4b"
    assert payload["live_ollama"] is False
    assert payload["config_hash"] == "abc123def456"
    assert payload["rows"]
    assert payload["aggregates"]
    assert any(float(row["eie_error"]) >= 0.0 for row in payload["rows"])
    assert payload["vote_cache_provenance"]["key_fields"] == [
        "config_hash",
        "seed",
        "reviewer_id",
        "application_id",
        "model_digest",
        "decode_params",
    ]
    assert "qwen" not in json.dumps(payload).lower()


def test_run_postdoc_panel_all_degenerate_rows_are_not_aggregated(tmp_path: Path) -> None:
    payload = run_postdoc_panel_study(
        seeds=(0,),
        strategies=("representative_sortition",),
        panel_sizes=(3,),
        n_reviewers=9,
        n_applications=1,
        n_trios=1,
        prevalence_strong=0.5,
        age_min=28,
        age_max=52,
        mean_expertise=0.74,
        expertise_heterogeneity=0.08,
        age_bias_std=0.65,
        track="analytical",
        config_hash="abc123def456",
        require_live=False,
        cache_path=tmp_path / "unused.json",
    )

    assert payload["rows"]
    assert payload["aggregates"] == []
    assert all(float(row["eie_error"]) < 0.0 for row in payload["rows"])
    assert all(int(row["usable_trios"]) == 0 for row in payload["rows"])


def test_postdoc_alignment_compares_analytical_and_gemma_signs() -> None:
    rows = [
        {
            "track": "analytical",
            "strategy": "representative_sortition",
            "strategy_label": "representative sortition",
            "panel_size": 3,
            "eie_error": 0.1,
            "mv_error": 0.2,
            "age_disparity_old_minus_young": 0.05,
            "eie_ci_low": 0.08,
            "eie_ci_high": 0.12,
            "usable_trios": 1,
            "degenerate_trios": 0,
            "panel_mean_expertise": 0.74,
            "panel_mean_age_bias": 0.1,
        },
        {
            "track": "live",
            "strategy": "representative_sortition",
            "strategy_label": "representative sortition",
            "panel_size": 3,
            "eie_error": 0.12,
            "mv_error": 0.22,
            "age_disparity_old_minus_young": 0.02,
            "eie_ci_low": 0.1,
            "eie_ci_high": 0.14,
            "usable_trios": 1,
            "degenerate_trios": 0,
            "panel_mean_expertise": 0.72,
            "panel_mean_age_bias": 0.08,
        },
    ]
    analytical = {"config_hash": "a", "aggregates": aggregate_postdoc_rows(rows[:1])}
    live = {
        "config_hash": "b",
        "model": "gemma3:4b",
        "model_provenance": {"digest": "digest"},
        "aggregates": aggregate_postdoc_rows(rows[1:]),
    }

    alignment = build_postdoc_alignment(analytical, live)

    assert alignment["schema_version"] == 1
    assert alignment["model"] == "gemma3:4b"
    assert alignment["cells"][0]["sign_agrees"] is True
    assert alignment["agreement_rate_resolved"] == 1.0
