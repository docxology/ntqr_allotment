"""Gemma-based postdoctoral-review panel stress test.

This module is the manuscript-facing live Ollama track. It deliberately does
not compare LLM families. A single local ``gemma3:4b`` model is prompted as
different synthetic reviewers whose expertise and irrelevant age-bias factors
vary; the measurement target is how reviewer sampling changes the NTQR and
majority-vote summaries in a fictitious postdoctoral-application setting.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import combinations
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .experts import Expert
from .ntqr_eval import (
    closest_solution,
    error_independent_solutions,
    majority_voting_solutions,
    supervised_oracle,
)
from .personas import ModelProvenance, OllamaJudge, fetch_model_provenance
from .sortition import STRATEGIES
from .statistics_analysis import bootstrap_ci

POSTDOC_STRATEGY_LABELS: Mapping[str, str] = {
    "representative_sortition": "representative sortition",
    "random_selection": "random selection",
    "ideological_selection": "same-bias selection",
    "expertise_threshold": "expertise threshold",
}

BIAS_GROUPS = ("younger-favoring", "near-neutral", "older-favoring")


@dataclass(frozen=True)
class ApplicationItem:
    """Fictitious postdoctoral application with independent age metadata."""

    index: int
    text: str
    true_label: str
    age: int
    quality_score: float
    field: str

    @property
    def age_centered(self) -> float:
        return (self.age - 40.0) / 12.0


@dataclass(frozen=True)
class ReviewerProfile:
    """Synthetic reviewer prompt profile.

    ``age_bias > 0`` means the reviewer erroneously favors older applicants;
    ``age_bias < 0`` means the reviewer erroneously favors younger applicants.
    Expertise controls sensitivity to merit evidence.
    """

    id: str
    expertise: float
    age_bias: float
    bias_group: str
    expertise_tier: str

    @property
    def target_accuracy(self) -> float:
        return self.expertise

    def build_prompt(self, item_text: str) -> tuple[str, str]:
        direction = (
            "older applicants"
            if self.age_bias > 0.12
            else "younger applicants"
            if self.age_bias < -0.12
            else "neither age group"
        )
        strength = abs(self.age_bias)
        system = (
            "You are a synthetic reviewer in a controlled bias stress test for "
            "postdoctoral fellowship panels. Age is not a legitimate selection "
            "feature in real review. For this simulation, your profile has "
            f"expertise={self.expertise:.2f} and an irrelevant age-bias tendency "
            f"toward {direction} with strength {strength:.2f}. Apply the profile "
            "when evidence is ambiguous, then answer with exactly one lowercase "
            "letter: a for shortlist, b for do not shortlist."
        )
        user = (
            f"{item_text}\n\n"
            "Decision: should this fictitious applicant be shortlisted for a "
            "postdoctoral fellowship? Answer exactly `a` or `b`."
        )
        return system, user


@dataclass(frozen=True)
class PanelEvaluationRow:
    seed: int
    strategy: str
    strategy_label: str
    panel_size: int
    track: str
    eie_error: float
    mv_error: float
    usable_trios: int
    degenerate_trios: int
    age_disparity_old_minus_young: float
    panel_mean_expertise: float
    panel_mean_age_bias: float
    panel_abs_age_bias: float
    panel_ids: tuple[str, ...]
    panel_composition: dict[str, dict[str, int]]
    audit_hash: str | None


def make_postdoc_application_corpus(
    n_items: int,
    *,
    seed: int,
    prevalence_strong: float = 0.5,
    age_min: int = 28,
    age_max: int = 52,
) -> tuple[ApplicationItem, ...]:
    """Generate fictitious applications; quality is independent of age."""
    if n_items <= 0:
        raise ValueError("n_items must be positive")
    if not 0.0 < prevalence_strong < 1.0:
        raise ValueError("prevalence_strong must be in (0, 1)")
    if age_min > age_max:
        raise ValueError("age_min must be at or below age_max")

    rng = np.random.default_rng(seed)
    fields = (
        "computational neuroscience",
        "social choice theory",
        "statistical learning",
        "cognitive science",
        "complex systems",
    )
    thresholds = np.quantile(rng.normal(size=4000), 1.0 - prevalence_strong)
    items: list[ApplicationItem] = []
    for index in range(n_items):
        quality = float(rng.normal())
        age = int(rng.integers(age_min, age_max + 1))
        field = fields[index % len(fields)]
        label = "a" if quality >= thresholds else "b"
        pubs = max(0, int(round(5 + 2.5 * quality + rng.normal(0, 1.2))))
        methods = "clear causal design" if quality >= 0 else "underdeveloped design"
        letters = "specific and convergent" if quality >= 0 else "mixed or generic"
        award = "strong independent trajectory" if label == "a" else "promising but not yet independent"
        text = (
            f"Applicant {index:03d} is a fictitious postdoctoral candidate in {field}. "
            f"Age metadata: {age}. Merit evidence: {pubs} relevant publications, "
            f"{methods}, letters are {letters}, and the dossier suggests {award}. "
            "The age metadata is synthetic and unrelated to true latent quality."
        )
        items.append(
            ApplicationItem(
                index=index,
                text=text,
                true_label=label,
                age=age,
                quality_score=quality,
                field=field,
            )
        )
    return tuple(items)


def make_reviewer_population(
    n_reviewers: int,
    *,
    seed: int,
    mean_expertise: float = 0.74,
    expertise_heterogeneity: float = 0.08,
    age_bias_std: float = 0.65,
) -> tuple[ReviewerProfile, ...]:
    """Generate synthetic reviewers with expertise and irrelevant age bias."""
    if n_reviewers < 3:
        raise ValueError("n_reviewers must be at least 3")
    if not 0.0 < mean_expertise < 1.0:
        raise ValueError("mean_expertise must be in (0, 1)")
    if expertise_heterogeneity < 0.0:
        raise ValueError("expertise_heterogeneity must be non-negative")
    if age_bias_std < 0.0:
        raise ValueError("age_bias_std must be non-negative")

    rng = np.random.default_rng(seed + 17_003)
    profiles: list[ReviewerProfile] = []
    for index in range(n_reviewers):
        expertise = float(
            np.clip(rng.normal(mean_expertise, expertise_heterogeneity), 0.52, 0.96)
        )
        age_bias = float(np.clip(rng.normal(0.0, age_bias_std), -1.4, 1.4))
        if age_bias > 0.18:
            group = "older-favoring"
        elif age_bias < -0.18:
            group = "younger-favoring"
        else:
            group = "near-neutral"
        tier = "high" if expertise >= 0.78 else "medium" if expertise >= 0.68 else "low"
        profiles.append(
            ReviewerProfile(
                id=f"r{index:03d}",
                expertise=expertise,
                age_bias=age_bias,
                bias_group=group,
                expertise_tier=tier,
            )
        )
    return tuple(profiles)


def reviewers_to_experts(reviewers: Sequence[ReviewerProfile]) -> tuple[Expert, ...]:
    """Map reviewer profiles into the existing sortition strategy interface."""
    return tuple(
        Expert(
            id=reviewer.id,
            accuracy_a=reviewer.expertise,
            accuracy_b=reviewer.expertise,
            expertise=reviewer.expertise,
            bias=reviewer.age_bias,
            ideology=reviewer.bias_group,
            expertise_tier=reviewer.expertise_tier,
        )
        for reviewer in reviewers
    )


def analytical_review_vote_probability(
    reviewer: ReviewerProfile,
    item: ApplicationItem,
    *,
    bias_weight: float = 0.16,
) -> float:
    """Probability of a shortlist vote under the analytical postdoc model."""
    merit = reviewer.expertise if item.true_label == "a" else 1.0 - reviewer.expertise
    biased = merit + bias_weight * reviewer.age_bias * item.age_centered
    return float(np.clip(biased, 0.02, 0.98))


def sample_analytical_reviewer_votes(
    reviewer: ReviewerProfile,
    items: Sequence[ApplicationItem],
    *,
    seed: int,
) -> tuple[str, ...]:
    rng = np.random.default_rng(seed)
    return tuple(
        "a" if rng.random() < analytical_review_vote_probability(reviewer, item) else "b"
        for item in items
    )


def evaluate_panel_votes(
    votes_by_reviewer: Mapping[str, Sequence[str]],
    reviewers_by_id: Mapping[str, ReviewerProfile],
    items: Sequence[ApplicationItem],
    *,
    seed: int,
    strategy: str,
    panel_size: int,
    panel_ids: Sequence[str],
    panel_composition: dict[str, dict[str, int]],
    audit_hash: str | None,
    track: str,
    n_trios: int,
) -> PanelEvaluationRow:
    """Evaluate selected reviewers via trio EIE/MV and age-disparity summaries."""
    if panel_size < 3:
        raise ValueError("panel_size must be at least 3")
    trio_indices = list(combinations(tuple(panel_ids), 3))
    rng = np.random.default_rng(seed + 41_911 + panel_size)
    rng.shuffle(trio_indices)
    trio_indices = trio_indices[: max(1, min(n_trios, len(trio_indices)))]

    eie_errors: list[float] = []
    mv_errors: list[float] = []
    degenerate = 0
    for trio_ids in trio_indices:
        trio_votes = [tuple(votes_by_reviewer[reviewer_id]) for reviewer_id in trio_ids]
        try:
            oracle = supervised_oracle(trio_votes, items)
            eie = error_independent_solutions(trio_votes)
            mv = majority_voting_solutions(trio_votes)
            if not eie or not mv:
                degenerate += 1
                continue
            eie_errors.append(closest_solution(eie, oracle).error_vs(oracle))
            mv_errors.append(closest_solution(mv, oracle).error_vs(oracle))
        except (ValueError, KeyError, IndexError, ZeroDivisionError):
            degenerate += 1
            continue

    if not eie_errors:
        eie_error = -1.0
        mv_error = -1.0
    else:
        eie_error = float(np.mean(eie_errors))
        mv_error = float(np.mean(mv_errors))

    rates_by_item: list[tuple[int, float]] = []
    for idx, item in enumerate(items):
        votes = [votes_by_reviewer[reviewer_id][idx] for reviewer_id in panel_ids]
        rates_by_item.append((item.age, sum(v == "a" for v in votes) / len(votes)))
    median_age = float(np.median([age for age, _ in rates_by_item]))
    young = [rate for age, rate in rates_by_item if age <= median_age]
    old = [rate for age, rate in rates_by_item if age > median_age]
    age_disparity = float(np.mean(old) - np.mean(young)) if old and young else 0.0

    reviewers = [reviewers_by_id[reviewer_id] for reviewer_id in panel_ids]
    return PanelEvaluationRow(
        seed=seed,
        strategy=strategy,
        strategy_label=POSTDOC_STRATEGY_LABELS.get(strategy, strategy.replace("_", " ")),
        panel_size=panel_size,
        track=track,
        eie_error=eie_error,
        mv_error=mv_error,
        usable_trios=len(eie_errors),
        degenerate_trios=degenerate,
        age_disparity_old_minus_young=age_disparity,
        panel_mean_expertise=float(np.mean([reviewer.expertise for reviewer in reviewers])),
        panel_mean_age_bias=float(np.mean([reviewer.age_bias for reviewer in reviewers])),
        panel_abs_age_bias=float(np.mean([abs(reviewer.age_bias) for reviewer in reviewers])),
        panel_ids=tuple(panel_ids),
        panel_composition=panel_composition,
        audit_hash=audit_hash,
    )


def run_postdoc_panel_study(
    *,
    seeds: Sequence[int],
    strategies: Sequence[str],
    panel_sizes: Sequence[int],
    n_reviewers: int,
    n_applications: int,
    n_trios: int,
    prevalence_strong: float,
    age_min: int,
    age_max: int,
    mean_expertise: float,
    expertise_heterogeneity: float,
    age_bias_std: float,
    track: str,
    config_hash: str,
    model: str = "gemma3:4b",
    base_url: str = "http://localhost:11434",
    temperature: float = 0.2,
    num_predict: int = 1,
    timeout: float = 20.0,
    require_live: bool = True,
    cache_path: Path | None = None,
    progress_every: int = 25,
) -> dict[str, object]:
    """Run the analytical or live Gemma postdoc panel study."""
    if track not in {"analytical", "live"}:
        raise ValueError("track must be 'analytical' or 'live'")
    if model != "gemma3:4b":
        raise ValueError("postdoc live manuscript track is single-model gemma3:4b")
    if len(set(seeds)) != len(tuple(seeds)):
        raise ValueError("seeds must be unique")
    for strategy in strategies:
        if strategy not in STRATEGIES:
            raise ValueError(f"unknown strategy {strategy!r}")
    if any(size < 3 for size in panel_sizes):
        raise ValueError("panel_sizes must be >= 3")
    if n_reviewers < max(panel_sizes):
        raise ValueError("n_reviewers must cover the largest panel")

    provenance = _postdoc_model_provenance(
        model=model,
        base_url=base_url,
        temperature=temperature,
        num_predict=num_predict,
        timeout=timeout,
        require_live=require_live,
        track=track,
    )
    judge: OllamaJudge | None = None
    if track == "live":
        judge = OllamaJudge(
            model=model,
            base_url=base_url,
            temperature=temperature,
            seed=0,
            num_predict=num_predict,
            timeout=timeout,
        )
        if require_live and not judge.available():
            raise RuntimeError(f"required live Ollama model {model!r} is unavailable")

    cache = _load_vote_cache(cache_path)
    rows: list[PanelEvaluationRow] = []
    panel_draws: list[dict[str, object]] = []
    vote_cache_hits = 0
    vote_cache_misses = 0
    for seed in seeds:
        items = make_postdoc_application_corpus(
            n_applications,
            seed=seed,
            prevalence_strong=prevalence_strong,
            age_min=age_min,
            age_max=age_max,
        )
        reviewers = make_reviewer_population(
            n_reviewers,
            seed=seed,
            mean_expertise=mean_expertise,
            expertise_heterogeneity=expertise_heterogeneity,
            age_bias_std=age_bias_std,
        )
        experts = reviewers_to_experts(reviewers)
        reviewers_by_id = {reviewer.id: reviewer for reviewer in reviewers}

        for strategy in strategies:
            for panel_size in panel_sizes:
                draw = STRATEGIES[strategy](
                    experts,
                    int(panel_size),
                    seed=seed * 10_000 + panel_size * 101 + len(rows),
                )
                panel_draws.append(
                    {
                        "seed": seed,
                        "strategy": strategy,
                        "strategy_label": POSTDOC_STRATEGY_LABELS[strategy],
                        "panel_size": panel_size,
                        "expert_ids": list(draw.expert_ids),
                        "composition": draw.composition,
                        "audit_hash": draw.audit_hash,
                    }
                )
                selected = [reviewers_by_id[reviewer_id] for reviewer_id in draw.expert_ids]
                votes_by_reviewer: dict[str, tuple[str, ...]] = {}
                for reviewer in selected:
                    if track == "live":
                        assert judge is not None
                        votes, hits, misses = _live_votes_for_reviewer(
                            judge,
                            reviewer,
                            items,
                            seed=seed,
                            config_hash=config_hash,
                            provenance=provenance,
                            cache=cache,
                            progress_every=progress_every,
                        )
                        vote_cache_hits += hits
                        vote_cache_misses += misses
                        _save_vote_cache(cache_path, cache)
                    else:
                        votes = sample_analytical_reviewer_votes(
                            reviewer,
                            items,
                            seed=seed * 100_000 + int(reviewer.id[1:]),
                        )
                    votes_by_reviewer[reviewer.id] = votes
                rows.append(
                    evaluate_panel_votes(
                        votes_by_reviewer,
                        reviewers_by_id,
                        items,
                        seed=seed,
                        strategy=strategy,
                        panel_size=int(panel_size),
                        panel_ids=draw.expert_ids,
                        panel_composition=draw.composition,
                        audit_hash=draw.audit_hash,
                        track=track,
                        n_trios=n_trios,
                    )
                )
        _save_vote_cache(cache_path, cache)

    row_dicts = [asdict(row) for row in rows]
    # Honest liveness: true only when the live track actually contacted the model
    # (fresh calls this run) or carries a real model digest — not merely because
    # the track is named "live" (a fully cache-hit run against a down server with
    # require_live=False would otherwise mislabel itself).
    genuinely_live = track == "live" and (
        vote_cache_misses > 0 or bool(provenance.get("digest"))
    )
    return {
        "schema_version": 1,
        "track": track,
        "model": model,
        "live_ollama": genuinely_live,
        "require_live": require_live,
        "config_hash": config_hash,
        "seed_list": list(seeds),
        "n_reviewers": n_reviewers,
        "n_applications": n_applications,
        "panel_sizes": list(panel_sizes),
        "strategies": list(strategies),
        "strategy_labels": dict(POSTDOC_STRATEGY_LABELS),
        "n_trios": n_trios,
        "prevalence_strong": prevalence_strong,
        "age_range": [age_min, age_max],
        "mean_expertise": mean_expertise,
        "expertise_heterogeneity": expertise_heterogeneity,
        "age_bias_std": age_bias_std,
        "decode_params": {
            "temperature": temperature,
            "num_predict": num_predict,
            "timeout": timeout,
        },
        "model_provenance": provenance,
        "vote_cache_provenance": {
            "path": cache_path.name if cache_path else None,
            "key_fields": [
                "config_hash",
                "seed",
                "reviewer_id",
                "application_id",
                "model_digest",
                "decode_params",
            ],
            "hits": vote_cache_hits,
            "misses": vote_cache_misses,
            "entries": len(cache),
        },
        "reviewer_population_metadata": _reviewer_population_metadata(row_dicts),
        "application_age_distribution": {
            "generated_independently_of_quality": True,
            "age_min": age_min,
            "age_max": age_max,
        },
        "panel_draws": panel_draws,
        "rows": row_dicts,
        "aggregates": aggregate_postdoc_rows(row_dicts),
    }


def aggregate_postdoc_rows(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    """Aggregate per-seed rows by track, strategy, and panel size."""
    groups: dict[tuple[str, str, int], list[Mapping[str, object]]] = {}
    for row in rows:
        if float(row["eie_error"]) < 0.0:
            continue
        key = (str(row["track"]), str(row["strategy"]), int(row["panel_size"]))
        groups.setdefault(key, []).append(row)
    aggregates: list[dict[str, object]] = []
    for (track, strategy, panel_size), group in sorted(groups.items()):
        eie = [float(row["eie_error"]) for row in group]
        mv = [float(row["mv_error"]) for row in group]
        disparity = [float(row["age_disparity_old_minus_young"]) for row in group]
        cie = bootstrap_ci(eie, seed=panel_size) if len(eie) > 1 else (eie[0], eie[0])
        cdisp = (
            bootstrap_ci(disparity, seed=panel_size + 13)
            if len(disparity) > 1
            else (disparity[0], disparity[0])
        )
        aggregates.append(
            {
                "track": track,
                "strategy": strategy,
                "strategy_label": POSTDOC_STRATEGY_LABELS.get(strategy, strategy),
                "panel_size": panel_size,
                "n": len(group),
                "eie_mean": float(np.mean(eie)),
                "eie_ci_low": float(cie[0]),
                "eie_ci_high": float(cie[1]),
                "mv_mean": float(np.mean(mv)),
                "age_disparity_mean": float(np.mean(disparity)),
                "age_disparity_ci_low": float(cdisp[0]),
                "age_disparity_ci_high": float(cdisp[1]),
                "usable_trios_mean": float(np.mean([float(row["usable_trios"]) for row in group])),
                "degenerate_trios_mean": float(
                    np.mean([float(row["degenerate_trios"]) for row in group])
                ),
                "panel_mean_expertise": float(
                    np.mean([float(row["panel_mean_expertise"]) for row in group])
                ),
                "panel_mean_age_bias": float(
                    np.mean([float(row["panel_mean_age_bias"]) for row in group])
                ),
            }
        )
    return aggregates


def build_postdoc_alignment(
    analytical_payload: Mapping[str, object],
    live_payload: Mapping[str, object],
) -> dict[str, object]:
    """Compare directional analytical predictions with live Gemma observations."""
    analytical = {
        (row["strategy"], int(row["panel_size"])): row
        for row in analytical_payload.get("aggregates", [])
        if isinstance(row, dict)
    }
    live = {
        (row["strategy"], int(row["panel_size"])): row
        for row in live_payload.get("aggregates", [])
        if isinstance(row, dict)
    }
    cells: list[dict[str, object]] = []
    for key in sorted(set(analytical) & set(live), key=lambda item: (item[1], item[0])):
        a = analytical[key]
        g = live[key]
        synthetic_sign = _sign(float(a["age_disparity_mean"]))
        gemma_sign = _sign(float(g["age_disparity_mean"]))
        cells.append(
            {
                "strategy": key[0],
                "strategy_label": POSTDOC_STRATEGY_LABELS.get(key[0], key[0]),
                "panel_size": key[1],
                "analytical_eie_mean": a["eie_mean"],
                "gemma_eie_mean": g["eie_mean"],
                "analytical_age_disparity": a["age_disparity_mean"],
                "gemma_age_disparity": g["age_disparity_mean"],
                "predicted_disparity_sign": synthetic_sign,
                "observed_gemma_sign": gemma_sign,
                "sign_agrees": synthetic_sign == gemma_sign and synthetic_sign != "zero",
                "unresolved": synthetic_sign == "zero" or gemma_sign == "zero",
            }
        )
    resolved = [cell for cell in cells if not cell["unresolved"]]
    agreement_rate = (
        sum(bool(cell["sign_agrees"]) for cell in resolved) / len(resolved)
        if resolved
        else 0.0
    )
    return {
        "schema_version": 1,
        "analytical_config_hash": analytical_payload.get("config_hash"),
        "live_config_hash": live_payload.get("config_hash"),
        "model": live_payload.get("model"),
        "model_provenance": live_payload.get("model_provenance"),
        "cells": cells,
        "agreement_rate_resolved": agreement_rate,
        "n_cells": len(cells),
        "n_resolved_cells": len(resolved),
        "caveat": (
            "Directional alignment is descriptive; the live track uses one local "
            "Gemma model prompted as synthetic reviewers, not human reviewers."
        ),
    }


def _postdoc_model_provenance(
    *,
    model: str,
    base_url: str,
    temperature: float,
    num_predict: int,
    timeout: float,
    require_live: bool,
    track: str,
) -> dict[str, object]:
    if track == "live":
        provenance = fetch_model_provenance(model, host=base_url, timeout=timeout)
        if require_live and (provenance is None or not provenance.digest):
            raise RuntimeError(f"required live model provenance missing for {model!r}")
        if provenance is None:
            provenance = ModelProvenance(
                model=model,
                family="gemma3",
                digest=None,
                temperature=temperature,
                seed=0,
                num_predict=num_predict,
            )
    else:
        provenance = ModelProvenance(
            model=model,
            family="gemma3",
            digest=None,
            temperature=temperature,
            seed=0,
            num_predict=num_predict,
        )
    data = asdict(provenance)
    data.update(
        {
            "base_url": base_url,
            "temperature": temperature,
            "num_predict": num_predict,
            "timeout": timeout,
            "decode_boundary": "single-token a/b response",
        }
    )
    return data


def _live_votes_for_reviewer(
    judge: OllamaJudge,
    reviewer: ReviewerProfile,
    items: Sequence[ApplicationItem],
    *,
    seed: int,
    config_hash: str,
    provenance: Mapping[str, object],
    cache: dict[str, str],
    progress_every: int,
) -> tuple[tuple[str, ...], int, int]:
    votes: list[str] = []
    hits = 0
    misses = 0
    digest = str(provenance.get("digest") or "no-digest")
    decode = {
        "temperature": provenance.get("temperature"),
        "num_predict": provenance.get("num_predict"),
        "timeout": provenance.get("timeout"),
    }
    for idx, item in enumerate(items):
        key = json.dumps(
            {
                "config_hash": config_hash,
                "seed": seed,
                "reviewer_id": reviewer.id,
                "application_id": item.index,
                "model_digest": digest,
                "decode_params": decode,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        if key in cache:
            vote = cache[key]
            hits += 1
        else:
            vote = judge.judge(reviewer, item.text)
            cache[key] = vote
            misses += 1
        votes.append(vote)
        if progress_every > 0 and misses and misses % progress_every == 0:
            print(
                f"live postdoc votes: reviewer={reviewer.id} misses={misses} item={idx}",
                flush=True,
            )
    return tuple(votes), hits, misses


def _load_vote_cache(cache_path: Path | None) -> dict[str, str]:
    if cache_path is None or not cache_path.is_file():
        return {}
    data = json.loads(cache_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{cache_path} must contain a JSON object")
    return {str(key): str(value) for key, value in data.items()}


def _save_vote_cache(cache_path: Path | None, cache: Mapping[str, str]) -> None:
    if cache_path is None:
        return
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(cache, indent=2, sort_keys=True), encoding="utf-8")


def _reviewer_population_metadata(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    return {
        "panel_rows": len(rows),
        "panel_mean_expertise_min": min(
            (float(row["panel_mean_expertise"]) for row in rows), default=0.0
        ),
        "panel_mean_expertise_max": max(
            (float(row["panel_mean_expertise"]) for row in rows), default=0.0
        ),
        "panel_mean_age_bias_min": min(
            (float(row["panel_mean_age_bias"]) for row in rows), default=0.0
        ),
        "panel_mean_age_bias_max": max(
            (float(row["panel_mean_age_bias"]) for row in rows), default=0.0
        ),
    }


def _sign(value: float, *, tolerance: float = 1e-9) -> str:
    if value > tolerance:
        return "positive"
    if value < -tolerance:
        return "negative"
    return "zero"


__all__ = [
    "ApplicationItem",
    "ReviewerProfile",
    "PanelEvaluationRow",
    "POSTDOC_STRATEGY_LABELS",
    "make_postdoc_application_corpus",
    "make_reviewer_population",
    "reviewers_to_experts",
    "analytical_review_vote_probability",
    "sample_analytical_reviewer_votes",
    "evaluate_panel_votes",
    "run_postdoc_panel_study",
    "aggregate_postdoc_rows",
    "build_postdoc_alignment",
]
