"""NTQR_allotment: sortition upstream of NTQR.

Study how forming an expert panel (representative lottery vs ideological vs
competence-first) and its size shape the oracle-referenced error of no-answer-key
`ntqr` evaluations for noisy experts of known precision, bias, and heterogeneity.

The package spans two tracks. The synthetic-generator track is the deterministic
spine (experts -> sortition -> NTQR -> oracle), swept across strategy/size/
correlation; the Gemma postdoc-reviewer panel is the live empirical companion. Scientific
extensions: ``dependence`` (controlled error-correlation), ``independence_sweep``
(the measured correlation-tolerance curve), ``ternary`` (R=3 axiom-CONSISTENCY,
not recovery), ``ensemble`` (N-judge consistency + alarm power), ``fairness``
(maximin selection-probability), ``theory`` (symbolic axioms / O(Q^3) growth),
and ``statistics_analysis`` (bootstrap / Cohen's d / OLS).
"""

from .dependence import (
    CorrelationReport,
    measure_error_correlations,
    sample_votes_correlated,
)
from .ensemble import (
    alarm_power_curve,
    observed_vote_counts,
    panel_agreement_moments,
)
from .experts import (
    Expert,
    Item,
    generate_population,
    sample_items,
    sample_votes,
)
from .fairness import (
    FairnessReport,
    maximin_fairness,
    representation_error,
)
from .independence_sweep import (
    IndependenceAggregate,
    IndependenceGrid,
    IndependenceRow,
    aggregate_independence,
    run_independence_sweep,
    tolerance_slope,
)
from .ntqr_eval import (
    Evaluation,
    alarm_misaligned,
    error_independent_solutions,
    majority_voting_solutions,
    supervised_oracle,
)
from .pipeline import (
    TrialConfig,
    TrialResult,
    run_trial,
    run_trial_ensemble,
)
from .cross_family import (
    FamilyContrast,
    MultiSeedContrast,
    PairCorrelationRecord,
    PairGroupSummary,
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
from .config import (
    ExperimentProfile,
    LivePostdocPanelConfig,
    load_experiment_profile,
    load_live_postdoc_panel_config,
)
from .personas import (
    ModelProvenance,
    fetch_model_provenance,
    provenance_for_judge,
)
from .power_analysis import (
    NullDiagnosis,
    analytic_power,
    cohens_d_safe,
    diagnose_null,
    min_detectable_effect,
    norm_cdf,
    norm_ppf,
    permutation_test,
    power_curve_over_effect,
    power_curve_over_n,
    sample_size_for_power,
    simulate_power,
)
from .power_study import (
    POWER_CSV_COLUMNS,
    PowerRow,
    analyze,
    contrast_power,
    group_eie_by_strategy,
    load_trial_rows,
    rep_vs_ideo_power,
    strategy_power_matrix,
    write_power_table,
)
from .sortition import PanelDraw, STRATEGIES
from .statistics_analysis import (
    HolmResult,
    MeanCIResult,
    OLSResult,
    SeparationResult,
    SignTestResult,
    bootstrap_ci,
    bootstrap_slope_ci,
    ci_overlap_verdict,
    exact_sign_test,
    holm_bonferroni,
    cohens_d,
    mean_ci_summary,
    ols_slope,
    strategy_separation,
)
from .sweeps import (
    aggregate,
    representative_vs_ideological,
    run_sweep,
    run_sweep_parallel,
    strategy_ranking,
)
from .ternary import (
    TernaryConfusion,
    is_axiom_consistent,
    make_ternary_confusion,
)
from .theory import (
    fit_error_correlation_slope,
    number_apriori_evaluations,
    symbolic_single_classifier_axiom,
)

__version__ = "0.1.0"

__all__ = [
    # generator
    "Expert",
    "Item",
    "generate_population",
    "sample_items",
    "sample_votes",
    # ntqr evaluation
    "Evaluation",
    "alarm_misaligned",
    "error_independent_solutions",
    "majority_voting_solutions",
    "supervised_oracle",
    # sortition
    "PanelDraw",
    "STRATEGIES",
    # pipeline
    "TrialConfig",
    "TrialResult",
    "run_trial",
    "run_trial_ensemble",
    # sweep
    "run_sweep",
    "run_sweep_parallel",
    "aggregate",
    "strategy_ranking",
    "representative_vs_ideological",
    # dependence + independence
    "CorrelationReport",
    "sample_votes_correlated",
    "measure_error_correlations",
    "IndependenceGrid",
    "IndependenceRow",
    "IndependenceAggregate",
    "run_independence_sweep",
    "aggregate_independence",
    "tolerance_slope",
    # ternary consistency (R=3 axioms; NOT recovery)
    "TernaryConfusion",
    "make_ternary_confusion",
    "is_axiom_consistent",
    # N-judge ensemble consistency
    "observed_vote_counts",
    "panel_agreement_moments",
    "alarm_power_curve",
    # fairness / maximin
    "FairnessReport",
    "maximin_fairness",
    "representation_error",
    # theory
    "symbolic_single_classifier_axiom",
    "number_apriori_evaluations",
    "fit_error_correlation_slope",
    # statistics
    "bootstrap_ci",
    "cohens_d",
    "ols_slope",
    "OLSResult",
    "SeparationResult",
    "SignTestResult",
    "bootstrap_slope_ci",
    "ci_overlap_verdict",
    "exact_sign_test",
    "strategy_separation",
    "HolmResult",
    "holm_bonferroni",
    "MeanCIResult",
    "mean_ci_summary",
    # power analysis (study-agnostic)
    "norm_cdf",
    "norm_ppf",
    "cohens_d_safe",
    "analytic_power",
    "sample_size_for_power",
    "min_detectable_effect",
    "permutation_test",
    "simulate_power",
    "power_curve_over_n",
    "power_curve_over_effect",
    "NullDiagnosis",
    "diagnose_null",
    # power study (applied to this study's sweeps)
    "PowerRow",
    "POWER_CSV_COLUMNS",
    "load_trial_rows",
    "group_eie_by_strategy",
    "contrast_power",
    "strategy_power_matrix",
    "rep_vs_ideo_power",
    "analyze",
    "write_power_table",
    # cross-family decorrelation (the validation track)
    "model_family",
    "TaggedJudgeVotes",
    "PairwiseCorrelationMatrix",
    "PairCorrelationRecord",
    "PairGroupSummary",
    "build_pairwise_error_correlation_matrix",
    "pair_correlation_records",
    "summarize_pair_groups",
    "FamilyContrast",
    "contrast_same_vs_cross_family",
    "MultiSeedContrast",
    "aggregate_contrasts",
    "make_family_correlated_votes",
    "ollama_panel_available",
    "collect_live_family_votes",
    "ExperimentProfile",
    "LivePostdocPanelConfig",
    "load_experiment_profile",
    "load_live_postdoc_panel_config",
    # persona provenance (reproducibility pinning)
    "ModelProvenance",
    "fetch_model_provenance",
    "provenance_for_judge",
    "__version__",
]
