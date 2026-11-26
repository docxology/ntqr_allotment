# Study overview

## The question

The `ntqr` package's exact error-independent evaluator returns two possible
logical evaluations for **three** binary judges from how often they agree and
disagree — *if* their errors are independent in the package's test-level sense.
Sortition is a way to *engineer* a panel. So:

> As a function of statistical-power features (expert precision, bias,
> heterogeneity, corpus size), how does the way we **form the panel upstream**
> — representative lottery vs. ideological selection vs. expertise threshold vs.
> random — change the no-answer-key evaluation error downstream, and **how many
> experts do we need**?

## What we found

The *rule* that forms the panel dominates; the *number* of judges is essentially
neutral. Competence-first selection gave the lowest no-answer-key recovery error on
synthetic judges — but that advantage reversed under a live prompted model, where
"expertise" is only an instruction. Representative selection protects recovery only
when the lottery balances the exact attribute a shared error rides on — a property
captured by a closed-form concentration (Herfindahl) index over that attribute,
which a continuous representativeness dial confirms is monotone. The protection is
conditional, not magical: balanced on the wrong axis, it disappears.

The synthetic layer knows the ground truth, so every no-answer-key `ntqr`
evaluation is scored against the **supervised oracle** and uses the oracle to choose the better of the two candidate branches. The retained empirical companion used one local
`gemma3:4b` model prompted as synthetic postdoctoral reviewers with expertise
and irrelevant age-bias factors, judging fictitious applications with synthetic
age metadata.

The manuscript's scholarship boundary is explicit: peer-review reliability and
bias, research-funding lottery proposals, deliberative democracy, diversity,
protected-attribute auditing, and LLM-as-judge sources motivate the setting and
the variables, while all reported effects remain bounded to regenerated synthetic
artifacts and the single-model Gemma companion.

## Architecture (two-layer, thin-orchestrator)

All business logic lives in `src/ntqr_allotment/`; `scripts/` are thin
orchestrators that only do I/O and call into `src/`.

```
src/ntqr_allotment/
  # synthetic deterministic spine
  experts.py            # synthetic population: precision/bias/heterogeneity -> votes
  corpus.py             # item/corpus generation
  sortition.py          # allotment adapter + 4 panel-formation strategies
  ntqr_eval.py          # ntqr adapter: trio EIE/MV solver, supervised oracle, alarm
  pipeline.py           # population -> panel -> NTQR -> metrics (one trial)
  sweeps.py             # deterministic grid sweep + degenerate-aware aggregation
  dependence.py         # controllable error-correlation generator (rho knob)
  independence_sweep.py # rho x strategy tolerance sweep -> recovery-vs-correlation
  ensemble.py           # ensemble-of-trios + N-judge alarm power
  ternary.py            # R=3 axiom-consistency track (consistency only)
  fairness.py           # maximin selection-probability diagnostic
  # statistics / theory
  statistics_analysis.py# bootstrap CIs, Holm-Bonferroni, CI-overlap separation
  power_analysis.py     # analytic two-sample power, MDE, permutation test (no scipy)
  power_study.py        # power study over the real sweep contrasts
  theory.py             # symbolic NTQR axioms (sympy), monotone predictions
  contrast_analysis.py  # analytical directional predictions vs observed cells
  # live empirical companion
  personas.py           # OllamaJudge (real HTTP) + DeterministicJudge (seeded)
  postdoc_panel.py      # required-live gemma3:4b reviewer-panel; analytical+live
  cross_family.py       # cross-family decorrelation (live, n-limited)
  # manuscript + outputs
  manuscript_variables.py # token producer: every number from output/data/, 0 hardcoded
  figures.py, figure_parts/ # matplotlib figures (shared readability theme)
  cover.py, stego.py, web_explorer.py, config.py
```

## Figures

All figures read regenerated source artifacts and share a font/claim-title
readability theme; their titles and captions summarize artifact-bounded claims,
not evidence beyond the source tables.

| Figure | What it shows |
| --- | --- |
| `method_pipeline_schematic` | Fig 1: the deterministic instrument (population -> formation -> blind NTQR EIE over trios -> oracle scoring). Front matter, not a result. |
| `strategy_ranking` | Mean oracle-referenced EIE error by formation strategy, showing expertise-threshold separated from the bunched bottom cluster. |
| `rep_vs_ideo_heatmap`, `pre_post_ntqr_heatmap`, `theory_vs_observed_alignment` | Regime-grid contrasts and analytical-prediction alignment. |
| `power_curve`, `power_design_diagnosis`, `power_vs_n` | Size-vs-error per strategy; observed effect vs MDE; prospective power-vs-n budgets. |
| `error_vs_correlation`, `strategy_correlation` | Error-correlation tolerance and per-strategy realized correlation. |
| `bloc_phase_diagram` | Monotone bloc-confound fan-out: recovery error and measured trio correlation vs within-bloc coupling, one line per strategy; representative sortition stays flat while single-bloc selection fans out. |
| `bloc_concentration_dial` | Continuous representativeness dial: recovery error and measured correlation vs single-bloc concentration (Herfindahl index 1/B to 1) at fixed coupling; error rises monotonically, tracing the closed-form Herfindahl law. |
| `alarm_cost_curve`, `alarm_power`, `fairness_maximin` | O(Q^3) alarm cost, N-judge alarm power, maximin selection probabilities. |
| `track_ranking_inversion` | Cross-track ranking inversion at the matched three-seat grain (ranks compared, magnitudes not pooled). |
| `postdoc_strategy_ranking`, `postdoc_age_bias_heatmap`, `postdoc_empirical_alignment` | Recorded Gemma reviewer-panel ranking, age-disparity, and analytical-vs-Gemma alignment. |
