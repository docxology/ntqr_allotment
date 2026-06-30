"""Tests for the cross-FAMILY decorrelation experiment (no mocks).

The deterministic offline core uses :func:`make_family_correlated_votes`
(genuine seeded logic, not a mock) and measures realized error-correlations with
NTQR's real supervised estimator. The single live test is marked
``requires_ollama`` and skips when no server is reachable.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from ntqr_allotment.corpus import make_arithmetic_corpus
from ntqr_allotment.cross_family import (
    FamilyContrast,
    PairwiseCorrelationMatrix,
    TaggedJudgeVotes,
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
from ntqr_allotment.personas import PersonaSpec

_SRC = (
    Path(__file__).resolve().parent.parent
    / "src"
    / "ntqr_allotment"
    / "cross_family.py"
)


# ---- family tagging ---------------------------------------------------------

@pytest.mark.parametrize(
    "model,family",
    [
        ("qwen2.5:3b", "qwen2"),
        ("gemma3:4b", "gemma3"),
        ("llama3.1:8b", "llama3"),
        ("gemma3", "gemma3"),  # no dotted minor, no tag
        ("QWEN2.5:3B", "qwen2"),  # case-insensitive
    ],
)
def test_model_family_mapping(model, family):
    assert model_family(model) == family


def test_model_family_rejects_empty():
    with pytest.raises(ValueError):
        model_family("")
    with pytest.raises(ValueError):
        model_family("   ")


def test_distinct_families_are_distinct():
    assert model_family("qwen2.5:3b") != model_family("gemma3:4b")


# ---- deterministic offline core ---------------------------------------------

def test_make_family_correlated_votes_deterministic():
    corpus = make_arithmetic_corpus(200, seed=1)
    a = make_family_correlated_votes(
        families=["qwen2", "qwen2", "gemma3"], corpus=corpus, seed=5
    )
    b = make_family_correlated_votes(
        families=["qwen2", "qwen2", "gemma3"], corpus=corpus, seed=5
    )
    assert a == b
    assert all(set(jv.votes) <= {"a", "b"} for jv in a)
    assert all(len(jv.votes) == len(corpus) for jv in a)
    assert [jv.family for jv in a] == ["qwen2", "qwen2", "gemma3"]


def test_make_family_correlated_votes_validation():
    corpus = make_arithmetic_corpus(10, seed=1)
    with pytest.raises(ValueError):
        make_family_correlated_votes(families=[], corpus=corpus)
    with pytest.raises(ValueError):
        make_family_correlated_votes(families=["qwen2"], corpus=[])
    with pytest.raises(ValueError):
        make_family_correlated_votes(
            families=["qwen2"], corpus=corpus, target_accuracy=1.5
        )
    with pytest.raises(ValueError):
        make_family_correlated_votes(
            families=["qwen2"], corpus=corpus, shared_strength=2.0
        )


def test_make_family_correlated_votes_hits_target_accuracy():
    corpus = make_arithmetic_corpus(800, seed=2)
    # shared_strength=0 -> fully independent judges; accuracy should track target.
    panel = make_family_correlated_votes(
        families=["qwen2"], corpus=corpus, target_accuracy=0.75,
        shared_strength=0.0, seed=3,
    )
    votes = panel[0].votes
    acc = sum(1 for it, v in zip(corpus, votes) if v == it.true_label) / len(corpus)
    assert acc == pytest.approx(0.75, abs=0.05)


# ---- pairwise matrix --------------------------------------------------------

def test_pairwise_matrix_value_symmetry_and_self():
    corpus = make_arithmetic_corpus(400, seed=4)
    panel = make_family_correlated_votes(
        families=["qwen2", "qwen2", "gemma3"], corpus=corpus, seed=4
    )
    matrix = build_pairwise_error_correlation_matrix(panel, corpus)
    assert isinstance(matrix, PairwiseCorrelationMatrix)
    assert matrix.value(0, 1) == matrix.value(1, 0)  # symmetric
    with pytest.raises(ValueError):
        matrix.value(0, 0)  # no self-correlation
    with pytest.raises(ValueError):
        matrix.value(0, 99)  # out of range


def test_build_matrix_validation():
    corpus = make_arithmetic_corpus(50, seed=5)
    one = make_family_correlated_votes(families=["qwen2"], corpus=corpus, seed=5)
    with pytest.raises(ValueError):
        build_pairwise_error_correlation_matrix(one, corpus)  # < 2 judges
    panel = make_family_correlated_votes(
        families=["qwen2", "gemma3"], corpus=corpus, seed=5
    )
    bad = [panel[0], TaggedJudgeVotes("x", "gemma3", panel[1].votes[:-1])]
    with pytest.raises(ValueError):
        build_pairwise_error_correlation_matrix(bad, corpus)  # length mismatch


def test_pipeline_is_deterministic():
    corpus = make_arithmetic_corpus(400, seed=6)
    fams = ["qwen2", "qwen2", "gemma3", "gemma3"]
    m1 = build_pairwise_error_correlation_matrix(
        make_family_correlated_votes(families=fams, corpus=corpus, seed=6), corpus
    )
    m2 = build_pairwise_error_correlation_matrix(
        make_family_correlated_votes(families=fams, corpus=corpus, seed=6), corpus
    )
    assert m1.pair_abs_corr == m2.pair_abs_corr


def test_pair_records_and_group_summaries_count_same_cross_pairs() -> None:
    corpus = make_arithmetic_corpus(500, seed=16)
    panel = make_family_correlated_votes(
        families=["qwen2", "qwen2", "gemma3", "gemma3", "llama3"],
        corpus=corpus,
        target_accuracy=0.75,
        shared_strength=0.85,
        seed=16,
    )
    matrix = build_pairwise_error_correlation_matrix(panel, corpus)

    records = pair_correlation_records(matrix)
    summaries = {summary.relation: summary for summary in summarize_pair_groups(matrix)}

    assert len(records) == 10
    assert sum(1 for record in records if record.relation == "same") == 2
    assert sum(1 for record in records if record.relation == "cross") == 8
    assert summaries["same"].n_pairs == 2
    assert summaries["cross"].n_pairs == 8
    assert summaries["same"].nonzero_pairs <= summaries["same"].n_pairs
    assert summaries["cross"].ci_low <= summaries["cross"].mean_abs_corr
    assert summaries["cross"].mean_abs_corr <= summaries["cross"].ci_high


# ---- the validating contrast (end-to-end offline) ---------------------------

def test_cross_family_less_correlated_than_same_family():
    """The measurement recovers the injected structure: cross |corr| < same |corr|.

    NEGATIVE CONTROL: this assertion fails if make_family_correlated_votes were
    reverted to independent errors (shared_strength -> 0), because then both
    groups would have near-zero |corr| and the ordering would vanish into noise.
    """
    corpus = make_arithmetic_corpus(600, seed=7)
    panel = make_family_correlated_votes(
        families=["qwen2", "qwen2", "gemma3", "gemma3"],
        corpus=corpus,
        target_accuracy=0.75,
        shared_strength=0.85,
        seed=7,
    )
    matrix = build_pairwise_error_correlation_matrix(panel, corpus)
    contrast = contrast_same_vs_cross_family(matrix)
    assert isinstance(contrast, FamilyContrast)
    assert contrast.n_same_pairs == 2  # (q,q) and (g,g)
    assert contrast.n_cross_pairs == 4  # 2 qwen x 2 gemma
    assert contrast.mean_abs_cross_family < contrast.mean_abs_same_family
    assert contrast.delta_cross_minus_same < 0


def test_negative_control_independent_has_no_ordering():
    """With shared_strength=0 the cross<same ordering must NOT hold robustly.

    This pins the validity of the positive test: the structure comes from the
    injected family-shared latent, not from the measurement machinery.
    """
    corpus = make_arithmetic_corpus(600, seed=8)
    panel = make_family_correlated_votes(
        families=["qwen2", "qwen2", "gemma3", "gemma3"],
        corpus=corpus,
        target_accuracy=0.75,
        shared_strength=0.0,  # independent -> no family signal
        seed=8,
    )
    contrast = contrast_same_vs_cross_family(
        build_pairwise_error_correlation_matrix(panel, corpus)
    )
    # Both groups hover near zero AND the separation collapses. The positive test
    # reported a strong negative delta (|delta| ~ 0.13); with shared_strength=0 the
    # mechanism is gone, so the delta magnitude must itself collapse toward zero.
    # Asserting the magnitude collapse (not just a small same-mean) is what makes
    # this a real negative control: a generator that still produced cross < same by
    # some measurement artifact would FAIL here.
    assert contrast.mean_abs_same_family < 0.1
    assert abs(contrast.delta_cross_minus_same) < 0.05


def test_contrast_handles_empty_groups():
    corpus = make_arithmetic_corpus(300, seed=9)
    # single family -> no cross-family pairs
    panel = make_family_correlated_votes(
        families=["qwen2", "qwen2"], corpus=corpus, seed=9
    )
    contrast = contrast_same_vs_cross_family(
        build_pairwise_error_correlation_matrix(panel, corpus)
    )
    assert contrast.n_cross_pairs == 0
    assert math.isnan(contrast.mean_abs_cross_family)
    assert math.isnan(contrast.delta_cross_minus_same)


# ---- anti-overclaim (ISC-88) ------------------------------------------------

def test_contrast_label_is_honest_provenance():
    corpus = make_arithmetic_corpus(300, seed=10)
    panel = make_family_correlated_votes(
        families=["qwen2", "gemma3"], corpus=corpus, seed=10
    )
    contrast = contrast_same_vs_cross_family(
        build_pairwise_error_correlation_matrix(panel, corpus)
    )
    assert contrast.label == "live empirical, n-limited"


def test_source_documents_signed_magnitude_not_absolute_claim():
    src = _SRC.read_text()
    # The signed/n-limited framing is documented...
    assert "n-limited" in src
    assert "signed magnitude" in src.lower()
    # ...and every mention of the overclaim phrase sits inside a negation, so
    # the phrase is only ever used to FORBID the absolute reading.
    negation_cues = ("never", "not ", "no ")
    occurrences = [
        line for line in src.splitlines() if "sortition validated" in line.lower()
    ]
    assert occurrences, "expected the overclaim phrase to be explicitly forbidden"
    # The phrase can wrap across lines; join each occurrence with its neighbours
    # and require a negation cue in that window.
    lines = src.splitlines()
    for idx, line in enumerate(lines):
        if "sortition validated" in line.lower():
            window = " ".join(lines[max(0, idx - 1) : idx + 2]).lower()
            assert any(cue in window for cue in negation_cues), window


def test_contrast_docstring_carries_n_limited():
    assert "n-limited" in contrast_same_vs_cross_family.__doc__


# ---- guard ------------------------------------------------------------------

def test_ollama_panel_available_false_on_empty_and_dead_port():
    assert ollama_panel_available([]) is False
    assert ollama_panel_available(["qwen2.5:3b"], base_url="http://localhost:6553") is False


def test_collect_live_family_votes_rejects_empty_models():
    corpus = make_arithmetic_corpus(5, seed=11)
    persona = PersonaSpec("p", "competent", "neutral", "center", 0.75)
    with pytest.raises(ValueError):
        collect_live_family_votes(models=[], persona=persona, corpus=corpus)


# ---- live Ollama (skips without a server) -----------------------------------

@pytest.mark.requires_ollama
def test_collect_live_family_votes_runs_live():
    models = ["qwen2.5:3b", "gemma3:4b"]
    if not ollama_panel_available(models):
        pytest.skip("no live Ollama server")
    corpus = make_arithmetic_corpus(8, seed=12, max_operand=99, max_error=2)
    persona = PersonaSpec("p", "competent", "neutral", "center", 0.75)
    tagged = collect_live_family_votes(models=models, persona=persona, corpus=corpus)
    assert [t.family for t in tagged] == ["qwen2", "gemma3"]
    assert all(set(t.votes) <= {"a", "b"} for t in tagged)
    assert all(len(t.votes) == len(corpus) for t in tagged)


def test_filler_trio_invariance_pair01_is_independent_of_third_judge():
    """H4 regression: the pair-(0,1) error correlation must be invariant to the
    third (filler) judge.

    Every cross-family number depends on routing a *pair* through NTQR's trio-only
    ``pair_label_error_correlation`` by supplying a filler at slot 2 and reading the
    (0,1) entry. If a future ntqr changed how it marginalizes over the third slot,
    every cross-family result would silently corrupt while the higher-level tests
    stayed green. Pin the invariance directly: the (0,1) correlation is identical
    across three genuinely different third-slot fillers on a fixed corpus.
    """
    from ntqr_allotment.dependence import measure_error_correlations

    corpus = make_arithmetic_corpus(600, seed=11)
    panel = make_family_correlated_votes(
        families=["qwen2", "gemma3", "qwen2", "gemma3"],
        corpus=corpus,
        target_accuracy=0.75,
        shared_strength=0.6,
        seed=11,
    )
    v0, v1, v2, v3 = (judge.votes for judge in panel)

    def pair01(filler):
        report = measure_error_correlations([v0, v1, filler], corpus)
        return tuple(report.pair_correlations[f"(0,1)|{label}"] for label in ("a", "b"))

    # Three distinct fillers: copy-of-0 (the production choice), judge 2, judge 3.
    base = pair01(v0)
    assert pair01(v2) == pytest.approx(base, abs=1e-9)
    assert pair01(v3) == pytest.approx(base, abs=1e-9)
    # Sanity: the fillers really are different sequences (so the test isn't vacuous).
    assert not (v2 == v0 and v3 == v0)


# ---- multi-seed aggregation (sign stability across runs) ---------------------

def test_aggregate_contrasts_recovers_sign_stability():
    """Across independent offline seeds the correlated regime gives a stable
    negative delta (cross < same): sign_stability should be high."""
    contrasts = []
    for s in range(6):
        corpus = make_arithmetic_corpus(400, seed=100 + s)
        panel = make_family_correlated_votes(
            families=["qwen2", "qwen2", "gemma3", "gemma3"],
            corpus=corpus, target_accuracy=0.75, shared_strength=0.85, seed=100 + s,
        )
        contrasts.append(
            contrast_same_vs_cross_family(
                build_pairwise_error_correlation_matrix(panel, corpus)
            )
        )
    agg = aggregate_contrasts(contrasts)
    assert agg.n_runs == 6
    assert agg.mean_delta < 0  # decorrelation direction
    assert agg.sign_stability >= 0.8  # stable across seeds
    assert agg.min_delta <= agg.mean_delta <= agg.max_delta
    assert len(agg.deltas) == 6


def test_aggregate_contrasts_independent_regime_unstable_sign():
    """NEGATIVE CONTROL: with shared_strength=0 the delta hovers at zero, so the
    sign is NOT robustly negative — sign_stability must fall away from 1.0."""
    contrasts = []
    for s in range(6):
        corpus = make_arithmetic_corpus(400, seed=200 + s)
        panel = make_family_correlated_votes(
            families=["qwen2", "qwen2", "gemma3", "gemma3"],
            corpus=corpus, target_accuracy=0.75, shared_strength=0.0, seed=200 + s,
        )
        contrasts.append(
            contrast_same_vs_cross_family(
                build_pairwise_error_correlation_matrix(panel, corpus)
            )
        )
    agg = aggregate_contrasts(contrasts)
    assert abs(agg.mean_delta) < 0.05  # no real separation
    assert agg.sign_stability < 0.8    # sign is not robustly negative


def test_aggregate_contrasts_rejects_empty():
    with pytest.raises(ValueError, match="non-empty"):
        aggregate_contrasts([])
