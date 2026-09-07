# Results: what resolved and what stayed bounded

Numbers below are token-injected by
`src/ntqr_allotment/manuscript_variables.py` from the artifact named in each
section: `sweep_aggregated.csv` for strategy and size summaries,
`independence_sweep.csv` for tolerance, `postdoc_panel_results.json` and
`postdoc_panel_alignment.json` for the live Gemma reviewer-panel companion,
`power_analysis.csv` / `sweep_results.json` for design budgets, and
`alarm_timings.csv` for alarm timing. None are hand-transcribed. Errors are mean
EIE recovery error against the supervised oracle, aggregated over {{N_SEEDS}}
seeds in the active sweep profile.

This section adjudicates the five hypotheses from the Introduction. H1, H3, and
H5 each resolve in one subsection. H2 and H4 are addressed in two stages: first
against the baseline grid (where, by construction, both are design-limited because
the baseline judges err independently), then resolved together by the
composition-coupled confound. Remaining subsections report supporting diagnostics.
The Discussion returns the per-hypothesis verdicts.

## Synthetic deterministic results: controlled spine for H1-H4

The synthetic results are generated from seeded populations and corpora with a
known oracle. They support strategy, regime, tolerance, power-budget, alarm, and
diagnostic claims for the active deterministic profile only. Throughout, a *regime*
is one (expert-stringency × bias-spread × panel-size) cell of the sweep grid
(sixteen cells in the active profile); contrasts are evaluated cell by cell and
averaged only where the text says so.

### Formation strategy sets the recovery floor

**H1 (formation strategy is the dominant lever).** The strategies do not fan out
into a graded four-way ranking; instead one rule stands far apart and the other
three collapse together ([@fig:ranking]). Competence-first selection recovers at
mean EIE error {{RANK_EXPERTISE_THRESHOLD_EIE}} — roughly a quarter of the error of
any other rule — while representative sortition, random selection, and single-bloc
ideological selection are **statistically indistinguishable from one another**,
clustered at {{RANK_REPRESENTATIVE_SORTITION_EIE}}–{{RANK_RANDOM_SELECTION_EIE}}:

| Strategy | Mean EIE error | 95% CI |
| -------- | -------------- | ------ |
| expertise threshold (far best) | {{RANK_EXPERTISE_THRESHOLD_EIE}} | ±{{RANK_EXPERTISE_THRESHOLD_CI}} |
| representative sortition | {{RANK_REPRESENTATIVE_SORTITION_EIE}} | ±{{RANK_REPRESENTATIVE_SORTITION_CI}} |
| random selection | {{RANK_RANDOM_SELECTION_EIE}} | ±{{RANK_RANDOM_SELECTION_CI}} |
| single-bloc ideological selection | {{RANK_IDEOLOGICAL_SELECTION_EIE}} | ±{{RANK_IDEOLOGICAL_SELECTION_CI}} |

![Horizontal bars give the mean oracle-referenced EIE recovery error for each of the four panel-formation strategies, ordered best (lowest, top) to worst (highest, bottom); the whiskers are 95% confidence intervals over {{N_SEEDS}} seeds and the value beside each bar is the mean +/- half-interval, all from source `output/data/sweep_aggregated.csv` (profile {{SWEEP_PROFILE}}, hash {{SWEEP_CONFIG_HASH}}). Read it as the ceiling each upstream choice imposes on the downstream blind estimator: competence-first selection sits far left (near-zero error) while the other three strategies cluster together to its right. Claim: the upstream formation rule, not the estimator, sets the no-answer-key error ceiling, and the competence-first-vs-rest gap dwarfs anything the estimator does on a fixed panel; caveat: only the competence-first separation is resolved — representative, random, and single-bloc selection are statistically indistinguishable from one another, as the power/separation layer certifies.](../output/figures/strategy_ranking.png){#fig:ranking width=75%}

The dominant lever is therefore *whether the panel is curated for competence at
all*, not a graded property of the formation rule. The competence-first-vs-sortition
contrast is a **resolved** separation on this instrument: the trio-level bootstrap
separation gate reads **{{THRESHOLD_SORTITION_VERDICT}}** (means {{THRESHOLD_MEAN}}
vs {{SORTITION_MEAN}}), the two strategies' pooled confidence intervals in the table
above are disjoint ({{RANK_EXPERTISE_THRESHOLD_EIE}} ± {{RANK_EXPERTISE_THRESHOLD_CI}}
versus {{RANK_REPRESENTATIVE_SORTITION_EIE}} ± {{RANK_REPRESENTATIVE_SORTITION_CI}}),
and the power analysis resolves it as **{{THRESHOLD_SORTITION_POWER_VERDICT}}**
(well-powered, not design-limited). Every contrast *among* the other three is, by
contrast, design-indistinguishable — including the representative-vs-single-bloc
pair, whose effect is inconclusive (its 95% CI crosses zero,
{{REP_VS_IDEO_P3_VERDICT}}). So competence-first selection beats the representative
draw by a resolved interval, but representativeness, randomness, and single-bloc
concentration are interchangeable for recovery on this instrument — a sharper and
more honest result than a four-way ranking would suggest.

### Sortition only separates when the confound rides on the balanced axis

**H2 (concentrating correlated error degrades recovery).** [@fig:repideo] reports
the single-bloc-minus-representative EIE error contrast across the full active regime
grid. Positive cells mean the representative draw has lower post-NTQR recovery error.
The axes expose the design levers directly: expert stringency (`mean_expertise`),
ideological bias spread (`bias_std`), and sortition size (`panel_size`).

![Faceted heatmaps of the single-bloc-minus-representative EIE error contrast across expert stringency (mean expertise, rows), ideological bias spread (columns), and panel size (one facet per size), from source `output/data/sweep_aggregated.csv` (profile {{SWEEP_PROFILE}}, hash {{SWEEP_CONFIG_HASH}}), with analytical sign predictions overlaid from `output/data/analytical_predictions.json`. Colour encodes the signed contrast on a diverging scale centred at zero: positive (red) cells mean the representative draw recovers with lower error than the single-bloc draw in that regime, negative (blue) the reverse. Statistic: cell-level mean contrast over {{N_SEEDS}} seeds; stars mark descriptive 95% intervals excluding zero. Read across a row to see how stringency modulates the gap and down a column to see the effect of bias concentration. Claim: the representative-vs-single-bloc distinction is not a single number but is regime-dependent, becoming visible only as bias, stringency, and size are jointly varied; caveat: the intervals are descriptive and synthetic-profile bounded, and they are deliberately not pooled with the real Ollama evidence, which lives in the separate Gemma postdoc companion in [@fig:postdocrank], [@fig:postdocbias], and [@fig:postdocalign].](../output/figures/rep_vs_ideo_heatmap.png){#fig:repideo width=92%}

On this baseline grid the contrast is design-limited and must be read cell by cell:
the analytical prediction is directional (single-bloc ideological selection should
not beat representative sortition when bias is the manipulated dependence source),
and the heatmap shows where the regenerated synthetic data aligns, where it remains
uncertain, and where the active design is too small to resolve a sign. This grid
cannot fan the strategies apart because it never realizes H2's premise — its judges
err independently; the composition-coupled confound that supplies the missing
channel, and resolves H2, is reported in §Composition-coupled correlation fans the
strategies apart.

### NTQR beats majority voting only in selected regimes

For the pre/post comparison, "pre-NTQR" means the supervised majority-vote
baseline already stored as `mv_error`; "post-NTQR" means the ground-truth-free EIE
recovery error stored as `eie_error`. [@fig:prepost] plots `EIE - MV`, so negative
cells mean the NTQR recovery estimate is closer to the supervised oracle than the
majority-vote baseline for that regime.

![Faceted heatmaps contrasting the post-NTQR error-independent estimate against the pre-NTQR supervised majority-vote baseline, broken out by strategy, panel size, expert stringency, and bias spread, from source `output/data/analytical_predictions.json`, derived in turn from `output/data/sweep_aggregated.csv` (profile {{SWEEP_PROFILE}}, hash {{SWEEP_CONFIG_HASH}}). Colour encodes the signed difference on a diverging scale: negative (blue) cells are where the blind NTQR recovery lands closer to the oracle than the majority-vote baseline did, positive (red) where it does not. Statistic: cell-level mean difference over {{N_SEEDS}} seeds; metric is `eie_mean - mv_mean`, so the figure isolates what the estimator adds (or costs) on top of naive voting. Claim: panel size and bias act on each formation strategy differently before and after NTQR recovery, so there is no uniform pre/post improvement; caveat: this is oracle-referenced simulation on synthetic labels, not a live-judge validation claim.](../output/figures/pre_post_ntqr_heatmap.png){#fig:prepost width=95%}

The companion alignment map ([@fig:theoryalign]) makes the analytical layer
auditable rather than rhetorical. Each cell reports how many expertise levels in
that size-bias slice match the directional prediction that
ideological-minus-representative EIE should be positive. The same JSON artifact
also records the monotone checks for bias and expertise.

![Audit heatmap of how often the analytical directional predictions match the regenerated synthetic cells, from source `output/data/analytical_predictions.json`. Each cell of a panel-size x bias-spread slice reports the count of expertise levels whose observed contrast sign agrees with the predicted sign; darker cells mean more of the expertise levels in that slice align with the prediction. Statistic: aligned expertise-level cells per panel-size x bias slice, over source sweep profile {{SWEEP_PROFILE}}, {{N_SEEDS}} seeds. The figure exists to make the analytical layer falsifiable rather than rhetorical: a prediction that systematically disagreed with the data would show as pale cells. Claim: analytical expectations are checked against regenerated artifacts rather than asserted in prose; caveat: the predictions are directional and order constraints only, not closed-form numerical EIE laws, so partial agreement is expected and is reported honestly.](../output/figures/theory_vs_observed_alignment.png){#fig:theoryalign width=72%}

### Larger panels are a neutral sampling knob here

**H3 (size is a sampling knob, not a uniform improvement).** [@fig:powercurve]
plots EIE error against panel/ensemble size for each strategy. If size were a clean
power knob, every curve would fall from size 3 to size 6. It does not:

| Strategy | Size 3 | Size 6 | Pooled direction |
| -------- | ------ | ------ | ---------------- |
| expertise threshold | {{POWER_EXPERTISE_THRESHOLD_SIZE3}} | {{POWER_EXPERTISE_THRESHOLD_SIZE6}} | {{POWER_EXPERTISE_THRESHOLD_DIRECTION}} |
| representative sortition | {{POWER_REPRESENTATIVE_SORTITION_SIZE3}} | {{POWER_REPRESENTATIVE_SORTITION_SIZE6}} | {{POWER_REPRESENTATIVE_SORTITION_DIRECTION}} |
| random selection | {{POWER_RANDOM_SELECTION_SIZE3}} | {{POWER_RANDOM_SELECTION_SIZE6}} | {{POWER_RANDOM_SELECTION_DIRECTION}} |
| ideological selection | {{POWER_IDEOLOGICAL_SELECTION_SIZE3}} | {{POWER_IDEOLOGICAL_SELECTION_SIZE6}} | {{POWER_IDEOLOGICAL_SELECTION_DIRECTION}} |

![One line per panel-formation strategy tracing mean EIE recovery error against panel/ensemble size (3, 6, 9, 12 members), from source `output/data/sweep_results.json`, aggregated over {{N_SEEDS}} seeds with per-point 95% confidence intervals and a colour-matched end-label summarising each curve's trio-to-six-seat direction. If size were a clean power knob every curve would fall monotonically left to right; instead the curves cross. Read the vertical spread at any size as the strategy gap and each curve's slope as the strategy-specific effect of adding experts. Claim: a paired regime-controlled test (`paired_size_contrast`) resolves a trio-to-six-seat size effect for three of the four strategies, but every resolved increase is tiny and single-bloc is within noise, so size is essentially neutral at this grid and the dominant lever is which strategy forms the panel; caveat: this pooled curve marginalizes over sixteen regimes and is bounded to the active sweep profile.](../output/figures/power_curve.png){#fig:powercurve width=75%}

Those Size-3/Size-6 cells and the figure's end-labels are **pooled point
estimates** over sixteen regimes. The powered test is a **paired** contrast that
matches each regime-and-seed cell across the two sizes (`paired_size_contrast` in
`src/ntqr_allotment/power_study.py`), removing the between-regime variance. Under
that test {{SIZE_PAIRED_RESOLVED_COUNT}} of the four strategies show a *resolved*
trio-to-six-seat change, and every resolved change is a small **increase** in
error: random selection ({{SIZE_RANDOM_SELECTION_3TO6_DELTA}}, 95% CI
{{SIZE_RANDOM_SELECTION_3TO6_CI}}), representative sortition
({{SIZE_REPRESENTATIVE_SORTITION_3TO6_DELTA}}, CI
{{SIZE_REPRESENTATIVE_SORTITION_3TO6_CI}}), and competence-first selection
({{SIZE_EXPERTISE_THRESHOLD_3TO6_DELTA}}, CI {{SIZE_EXPERTISE_THRESHOLD_3TO6_CI}});
only single-bloc selection ({{SIZE_IDEOLOGICAL_SELECTION_3TO6_DELTA}}, CI
{{SIZE_IDEOLOGICAL_SELECTION_3TO6_CI}}) is within noise. These effects are
**resolved but negligible** — the largest, {{SIZE_RANDOM_SELECTION_3TO6_DELTA}},
is about a tenth of the bottom-tier baseline near {{RANK_RANDOM_SELECTION_EIE}}. So
more experts do not help, and at most very slightly hurt: we reject the simple
hypothesis that more experts always help, but the honest reading is that **size is
essentially neutral** at this grid, and the dominant lever is **strategy**, not
size.

**What the size diagnostic rules out.** The error-independent solver assumes the
three judges' errors are uncorrelated, so the natural guess is that larger panels
feed the ensemble more error-correlated trios. A per-trio diagnostic refutes that
guess ([@fig:triocond], `src/ntqr_allotment/trio_conditioning.py`, over
{{MECH_N_TRIO_RECORDS}} usable trios). The realized mean absolute pairwise
error-correlation — the quantity the solver assumes is zero — is
**{{MECH_CORR_VERDICT}}** across the size ladder for every strategy
({{MECH_CORR_SIZE3}} at the trio to {{MECH_CORR_SIZE12}} at twelve seats), so
enlarging the panel does not feed the ensemble more correlated trios. Correlation
*is* a relevant axis — within a strategy, higher-correlation trios recover worse
(Pearson up to +0.70 for competence-first) — but because correlation does not
grow with size, it does not identify a positive mechanism for the tiny paired size
deltas. No single per-trio summary statistic (correlation, judge accuracy, or scan
position) predicts the per-trio error strongly enough to pin the remaining cost on
worse judges. We therefore report the small paired size deltas after ruling out the
main correlation explanation, not as an affirmative aggregation effect.

![Two-panel per-trio mechanism diagnostic for the panel-size contrast, from source `output/data/trio_conditioning.json` ({{MECH_N_TRIO_RECORDS}} usable trios over {{MECH_N_SEEDS}} seeds of the reported regime grid). Panel A plots the mean absolute pairwise error-correlation of the usable trios against panel size, one line per formation strategy, with a dashed reference at zero (the value the error-independent solver assumes); every line holds the small {{MECH_CORR_SIZE3}}-to-{{MECH_CORR_MAX}} baseline rather than rising, so enlarging the panel does not pull in more error-correlated trios. Panel B plots the within-strategy Pearson correlation between a trio's recovery error and its absolute error-correlation; the bars are positive (up to +0.70 for competence-first), so correlation genuinely predicts per-trio error — it simply does not grow with size. Statistic: mean absolute error-correlation by size (A) and within-strategy Pearson of per-trio error against absolute error-correlation (B), measured over the same usable trios the ensemble-of-trios averages. Claim: the diagnostic rules out a size-growing error-correlation mechanism; caveat: it does not identify a positive mechanism and remains a {{MECH_N_SEEDS}}-seed structural diagnostic on synthetic labels, not a headline confidence interval.](../output/figures/trio_conditioning.png){#fig:triocond width=90%}

### Global injected correlation is measurable but recovery-limited

**H4 (error-correlation is measurable; recovery should degrade with it).** The
centerpiece measurement injects a controlled error-correlation $\rho$ into a
trio's votes, measures the *realized* correlation NTQR itself reports
(`mean_abs_pair`), computes the unlabeled EIE evaluation, and scores it against
the supervised oracle (`output/data/independence_sweep.csv`, produced by
`src/ntqr_allotment/independence_sweep.py`). The robust, monotone finding is that
**the realized correlation rises with the injected $\rho$**: averaged over the
non-degenerate cells it climbs from {{CORR_AT_RHO0}} at the lowest injected $\rho$
to {{CORR_AT_RHO_HIGH}} at the highest. The injection knob does what it claims —
it manufactures measurable error-correlation, and NTQR's own correlation
diagnostic registers it.

![Oracle-referenced EIE error (y-axis) versus the realized pairwise error correlation $\rho_{\mathrm{NTQR}}$ that the NTQR estimator itself measures from the votes (x-axis), shown as a scatter with one point per non-degenerate ($\rho$, strategy) cell of the tolerance sweep, from source `output/data/independence_sweep.csv`. The x-axis is the quantity the exact solver assumes is zero; the controlled injection knob moves points rightward, which verifies the diagnostic, while the y-axis tests whether more correlation actually costs recovery accuracy. Claim: injected $\rho$ creates measurable NTQR error correlation (points do move right), but the fitted recovery-error trend is unresolved at this grid; statistic: OLS error-vs-correlation slope {{TOLERANCE_SLOPE}} with 95% bootstrap CI {{TOLERANCE_SLOPE_CI95}}, which crosses zero; caveat: this figure supports the correlation diagnostic only, not a resolved recovery-effect law.](../output/figures/error_vs_correlation.png){#fig:tolerance width=75%}

Recovery error against the oracle, by contrast, shows **no detectable monotone
relationship** with that correlation on this grid. The sweep fixes the panel at
the trio (the exact solver's unit) and varies only $\rho$ and strategy over up to
six seeds per cell, so each ($\rho$, strategy) contributes one independent cell to the
regression — no size-invariant duplicate trio is double-counted. The
ordinary-least-squares slope of recovery error on realized correlation across
those unique cells is {{TOLERANCE_SLOPE}}, but its 95% bootstrap confidence
interval is {{TOLERANCE_SLOPE_CI95}} — it **crosses zero**. The honest verdict is
therefore that the fitted slope is **{{TOLERANCE_VERDICT}}** at this
{{INDEP_N_EXPERTS}}-expert / {{INDEP_N_ITEMS}}-item grid. Crucially this is an
*unresolved* result, not evidence of *no* effect: a CI that spans zero is
consistent with a small positive slope, a small negative one, or none, and this
grid is simply too small to choose between them (mean EIE error {{EIE_AT_RHO0}} at
the lowest injected `rho` and {{EIE_AT_RHO_HIGH}} at the highest, with a
non-monotone dip in between). We report the interval rather than a
bare point estimate precisely because a single noisy slope on eight cells would
overclaim a precision the data do not support.

What the instrument *does* establish here is narrower and robust: the realized
correlation tracks the injected correlation (the monotone rise above). The
*analytical* expectation encoded in `theory.py`
(`predicted_error_vs_correlation`) is that recovery error should **not decrease**
as positive error-correlation grows, because correlation violates the
error-independence the exact solver assumes. That expectation is **not confirmed
on this global-injection grid**: `independence_sweep.csv` shows a slight,
non-monotone *decrease* ({{EIE_AT_RHO0}} at the lowest injected `rho` to
{{EIE_AT_RHO_HIGH}} at the highest, slope 95% CI crossing zero). We now attribute
that non-monotonicity to a **disclosed limitation of this particular injection
model**, not to small-grid noise alone: `dependence.sample_votes_correlated` mixes
a shared and an independent *uniform* latent, and a convex combination of uniforms
is not uniform, so the model's realized per-judge accuracy is **not preserved as
`rho` varies** — it inflates and then deflates, peaking near `rho=0.5`. That
accuracy confound moves recovery error in its own right and contaminates the
recovery-vs-correlation slope. Rather than re-engineer this diagnostic (which would
perturb a shipped result), we draw the H4 recovery conclusion from the
**marginal-accuracy-preserving composition-coupled instrument** of the following
subsection (§Composition-coupled correlation fans the strategies apart), which holds
each judge's accuracy fixed by construction and resolves the recovery-vs-correlation
relationship in the affirmative ([@fig:blocphase]). The global-injection sweep
remains in the paper as the correlation *diagnostic* (the realized correlation does
rise with `rho`) with its accuracy artifact disclosed. This is a
**measured behaviour under controlled correlation**,
validated in simulation only; it does *not* show that sortition restores low
oracle-referenced error on real prompted judges. The live Gemma postdoctoral-review
panel that probes the same sampling mechanism on a local LLM is reported in the
Real-Ollama results subsection.

### Composition-coupled correlation exposes the sortition mechanism

The H4 slope above is unresolved for a specific, fixable reason. The tolerance
sweep injects a *global* correlation onto a *fixed* trio, identically for every
strategy, so it measures sensitivity to correlation **decoupled from how the panel
was formed**. That is the wrong instrument for H2, whose premise is that
single-bloc selection *seats judges whose errors are correlated* — a premise the
baseline generator never realizes, because there ideology shifts only each judge's
*marginal* accuracy and every judge errs from an independent stream. Under that
generator representative, random, and single-bloc panels are indistinguishable not
by coincidence but **by construction**: the channel that would separate them does
not exist, so no parameter sweep over the baseline can fan them out.

We close that gap with a composition-coupled confound
(`src/ntqr_allotment/bloc_confound.py`). Judges who share an ideological bloc draw
a shared latent *error shock* through a Gaussian copula of within-group strength
$\rho$; the construction preserves each judge's per-label accuracy exactly, so any
change in recovery is attributable to error *correlation*, not to an accuracy
shift, and $\rho=0$ reproduces the independent baseline. (The shared channel is a
symmetric competence shock, not directional bias. $\rho$ is the latent within-group
correlation; what we plot as "realized correlation" is NTQR's own
label-conditional error-correlation statistic, which is much smaller in magnitude
than $\rho$ — we report the quantity the solver assumes is zero, never the latent
$\rho$.) We sweep $\rho$ across {{BLOC_N_RHO_LEVELS}} levels,
aggregating on average {{BLOC_N_PER_POINT}} non-degenerate trials per (strategy, $\rho$) point over
bias-spread, stringency, panel-size, and seed regimes ([@fig:blocphase]).

The result is a clean, graded fan-out. At $\rho=0$ the three composition
strategies collapse exactly as the baseline reported — the
ideological-minus-representative gap is {{BLOC_SEP_LO}}. As coupling rises they fan
out: representative sortition stays essentially flat
({{BLOC_REPRESENTATIVE_SORTITION_LO}} to {{BLOC_REPRESENTATIVE_SORTITION_HI}} EIE
error), random selection degrades ({{BLOC_RANDOM_SELECTION_LO}} to
{{BLOC_RANDOM_SELECTION_HI}}), and single-bloc ideological selection degrades most
({{BLOC_IDEOLOGICAL_SELECTION_LO}} to {{BLOC_IDEOLOGICAL_SELECTION_HI}}), widening
the gap to {{BLOC_SEP_HI}} at $\rho={{BLOC_RHO_HI}}$. NTQR's own correlation
diagnostic makes the mechanism legible: at high coupling a representative trio
carries measured error-correlation {{BLOC_CORR_REPRESENTATIVE_SORTITION_HI}} while
a single-bloc trio carries {{BLOC_CORR_IDEOLOGICAL_SELECTION_HI}} — bloc-balancing
*decorrelates* the shared shock that bloc-concentration *concentrates*. This
resolves **H2** (concentrating correlated error does degrade recovery, once the
correlation is composition-coupled) and answers the open **H4** recovery slope in
the affirmative under a correctly specified, marginal-preserving instrument. It is
not a pooling artifact: in a paired per-regime test at $\rho={{BLOC_RHO_HI}}$,
single-bloc error exceeds representative error in {{BLOC_ROBUST_FRAC}} matched
regimes (paired mean {{BLOC_PAIRED_DIFF}}, 95% CI ±{{BLOC_PAIRED_CI}}). Nor is the
averaged subsample cherry-picked by the degenerate-trio skip: at high coupling the
single-bloc panels have the *lowest* degenerate-trio rate of the four strategies
(tracked per strategy in `bloc_phase_summary.json`), so dropping ill-posed trios
cannot be what manufactures their degradation — if anything it makes the reported
gap conservative.

This robustness is **conditional, not magical**, and a negative control says so.
The protection appears because representative sortition balances the very axis —
ideology — that the confound rides on. When the shared shock is re-keyed to an
orthogonal axis the lottery does *not* balance (expertise tier), representative
sortition loses its immunity: its error climbs from {{BLOC_CTRL_REP_LO}} to
{{BLOC_CTRL_REP_HI}}, and the large ideological-minus-representative gap of the
matched axis ({{BLOC_SEP_HI}} at $\rho={{BLOC_RHO_HI}}$) nearly closes under the
orthogonal one ({{BLOC_CTRL_SEP_HI}}). The point is not that some *other* strategy
inherits the protection — competence-first selection draws the top experts, who
span expertise tiers, so it does not maximally concentrate the tier axis either —
but that representativeness on ideology stops mattering once the confound no longer
rides on ideology. The defensible claim is therefore precise: **balancing a panel
on the axis a shared error rides on preserves no-answer-key recovery; balancing the
wrong axis does not.** That is a statement about sortition design in simulation, not
a blanket endorsement of representative panels.

![Two-panel bloc-confound phase diagram from source `output/data/bloc_phase_summary.json`, aggregated over bias-spread, stringency, panel-size, and seed regimes. Left: mean oracle-referenced EIE recovery error (y) versus within-bloc error coupling $\rho$ (x), one line per panel-formation strategy with 95% CI bands; the lines coincide at $\rho=0$ (the reproduced baseline collapse) and fan out as $\rho$ rises, representative sortition staying flat while single-bloc ideological selection climbs. Right: the NTQR-measured realized trio error correlation (y) versus $\rho$ — the mechanism — showing representative sortition suppressing the shared confound (flat, low) while single-bloc selection concentrates it (steeply rising). Claim: composition-coupled error correlation makes panel-formation rule the dominant lever on recovery, with representativeness protective specifically when the panel balances the axis the confound rides on; caveat: synthetic, marginal-preserving simulation against a known oracle, and the protection is axis-conditional as the expertise-tier negative control demonstrates.](../output/figures/bloc_phase_diagram.png){#fig:blocphase width=95%}

Representativeness is not only a four-way contrast but a **continuous dial**. Fixing
the coupling at $\rho={{BLOC_DIAL_RHO}}$ and forming panels with a tunable single-bloc
concentration $c$ — from balanced ($c=0$, Herfindahl index $1/B$) to single-bloc
($c=1$, Herfindahl index $1$) — recovery error climbs monotonically from
{{BLOC_DIAL_BALANCED}} to {{BLOC_DIAL_CONCENTRATED}} across the {{BLOC_DIAL_N_LEVELS}}
dial levels (fraction of steps that increase error: {{BLOC_DIAL_MONOTONE_FRAC}};
[@fig:dial]). This is the closed-form Herfindahl account of §Methods made visible:
the panel's concentration index over the confound axis, a closed-form combinatorial statistic,
sets its shared-confound exposure and hence its no-answer-key recovery error.

![Single-panel concentration-dial figure from source `output/data/bloc_phase_summary.json` (concentration block), aggregated over bias-spread, stringency, and seed regimes at fixed within-bloc coupling. The x-axis is the single-bloc concentration dial $c$ (the panel's Herfindahl index runs $1/B$ at $c=0$ to $1$ at $c=1$). Left y-axis (circles, with 95% CI band): mean oracle-referenced EIE recovery error; right y-axis (squares): the NTQR-measured realized trio error correlation. Recovery error rises monotonically with concentration, while the realized correlation rises and then saturates once the scored trio becomes single-bloc. Claim: recovery error is a graded function of panel concentration over the confound axis, tracing the closed-form Herfindahl prediction rather than a binary representative-vs-single-bloc split; caveat: synthetic, marginal-accuracy-preserving simulation at one coupling level against a known oracle, and the protection implied by low concentration is axis-conditional — it holds only for the axis the confound rides on, as the expertise-tier negative control in [@fig:blocphase] shows.](../output/figures/bloc_concentration_dial.png){#fig:dial width=70%}

### Power budgets distinguish ranking from resolved contrasts

Before any "beats" wording, competence-first selection and representative sortition
are compared by separately bootstrapped mean intervals at the trio
(`strategy_separation`, `src/ntqr_allotment/statistics_analysis.py`): mean recovery
error {{THRESHOLD_MEAN}} (competence-first) versus {{SORTITION_MEAN}}
(representative), a signed difference of {{THRESHOLD_SORTITION_DELTA}} with a
CI-overlap verdict of **{{THRESHOLD_SORTITION_VERDICT}}** — that is, the two
strategies' separately bootstrapped mean intervals do not overlap, so the
difference is not an artifact of within-strategy spread. The power study then
separates resolved contrasts from design-limited nulls: of
{{POWER_TOTAL_CONTRASTS}} pairwise strategy contrasts,
{{POWER_WELL_POWERED_COUNT}} are well-powered and
{{POWER_UNDERPOWERED_COUNT}} are underpowered at the current observation count, while
{{POWER_SIGNIFICANT_COUNT}} of {{POWER_TOTAL_CONTRASTS}} reach raw permutation-test
significance ({{POWER_SIGNIFICANT_HOLM_COUNT}} after Holm-Bonferroni across the
{{POWER_TOTAL_CONTRASTS}}-test family). The strategy *ranking* is therefore a point
estimate ordering plus a set of explicitly tested contrasts, not a blanket claim
that every neighboring rank is a significant win. [@fig:powervn] shows the analytic
power-vs-sample-size curves that set these budgets. For a two-sample contrast, the
design quantities are standardized effect $d$, per-group observation count $n$
(seeded trials across the active profile cells), Type-I error $\alpha$,
target power $1-\beta$, and the minimum detectable effect (MDE) at the chosen $n$. The
budget is reported with a minimum detectable effect of {{POWER_MDE80}} and between
{{POWER_MIN_SEEDS_FOR_80}} and {{POWER_MAX_SEEDS_FOR_80}} per-group observations
required to reach 80% power **for effects of the magnitudes observed in these
contrasts** — a prospective design target keyed to the observed effect sizes,
never retrospective observed power ([@fig:diagnosis],
`output/data/power_analysis.csv`).

![Design-adequacy diagnosis for every pairwise strategy contrast, from source `output/data/power_analysis.csv`: each row plots the observed standardized effect size (Cohen's $d$) against the minimum detectable effect (MDE) at 80% power, annotated with the permutation p-value and the per-group observation budgets needed to resolve an effect of that size. A contrast whose observed $|d|$ sits below its MDE marker is design-limited: the study could not have detected it even if it were real, so its non-significance is a statement about sample size, not about the absence of an effect. Claim: {{POWER_UNDERPOWERED_COUNT}} of {{POWER_TOTAL_CONTRASTS}} contrasts are design-limited at the current observation count, which is why several neighboring-rank comparisons remain inconclusive; caveat: the budgets are keyed to the observed effect magnitudes and are prospective design targets, not retrospective observed power evidence.](../output/figures/power_design_diagnosis.png){#fig:diagnosis width=80%}

![Analytic two-sample power $1-\beta$ (y-axis) versus samples-per-group $n$ (x-axis, log scale), shown as a family of curves with one curve per representative Cohen's $d$ value, computed from source `output/data/power_analysis.csv` / `src/ntqr_allotment/power_analysis.py`; the horizontal dashed line marks the 80% power target at the chosen $\alpha$, and where each curve crosses it gives the $n$ that effect size requires. Read it as a budgeting tool: the smaller the standardized effect, the further right its curve crosses the dashed line, i.e. the more per-group observations are needed. Claim: small standardized effects require many more seeds per group than the current design provides, which is the mechanism behind the design-limited nulls; caveat: this is a prospective design-budget curve and an MDE visual, not retrospective observed power.](../output/figures/power_vs_n.png){#fig:powervn width=72%}

The remaining "inconclusive" verdicts are therefore statements about *design size*,
made explicit through the minimum detectable effect — not evidence of no effect, and
never reported as retrospective observed power.

### Companion diagnostics bound cost, correlation, fairness, and consistency

The companion alarm's answer-key enumeration is roughly cubic in corpus size
([@fig:alarm]): about {{ALARM_Q20_S}} s at $Q=20$,
{{ALARM_Q50_S}} s at $Q=50$, and {{ALARM_Q100_S}} s at $Q=100$ (measured by
`scripts/bench_alarm.py`, written to `output/data/alarm_timings.csv`). This is a real
ceiling on the alarm track, so it is opt-in and capped at $Q\leq{{ALARM_MAX_Q}}$. We report
it as a finding: the alarm is usable as a small-corpus consistency check, not as a
sweep-scale primitive.

![Measured alarm wall-clock time (seconds) versus corpus size $Q$ on log-log axes, from source `output/data/alarm_timings.csv`, with a cubic $O(Q^3)$ reference line overlaid. On log-log axes a power law is a straight line whose slope is its exponent, so the measured points tracking the reference slope is the visual evidence that the answer-key-enumeration alarm scales cubically in $Q$. The practical consequence is a hard ceiling: the wall-clock cost rises steeply enough that the alarm is usable only as a small-corpus consistency check. Claim: at the current implementation the alarm is small-corpus only and is therefore opt-in and capped; caveat: the absolute wall-clock constants are machine-local and load-dependent, so it is the cubic scaling, not the individual timings, that is the robust finding.](../output/figures/alarm_cost_curve.png){#fig:alarm width=70%}

Three further companion tracks measure structural properties of the pipeline rather than
recovery error, and we report them as diagnostics. The error-correlation track
records the mean realized correlation each formation strategy induces
([@fig:strategycorr]): single-bloc selection sits highest, consistent with its
status as the deliberately correlated comparator. The maximin fairness track
characterizes the representative lottery's selection-probability distribution over
the population ([@fig:fairness]) — the maximin objective is the floor on who can be
seated, independent of any downstream recovery number. The N-judge alarm-power track
records a saturated small-$Q$ alarm-firing rate across the plotted panel sizes
([@fig:alarmpower]). The ternary ($R=3$) track is consistency/feasibility only — it
confirms three-way vote profiles satisfy the NTQR axioms and is never an $R=3$ recovery
claim (out of scope) — so it yields a pass/fail check rather than a plotted number.

![Bar chart of the mean realized pairwise error correlation that each panel-formation strategy induces among its judges, measured by NTQR's own supervised estimator over the tolerance sweep, from source `output/data/independence_sweep.csv`. Higher bars mean the strategy seats judges whose mistakes are more correlated, which is precisely the error-independence assumption the exact trio solver leans on. Single-bloc selection is the tallest bar, consistent with its design as the deliberately correlated comparator, while representative and random draws sit lower. Read this as a structural property of the draw itself, upstream of any recovery number. Caveat: this is a structural diagnostic of the formed panel, not a recovery-effect claim about downstream EIE error.](../output/figures/strategy_correlation.png){#fig:strategycorr width=70%}

![Per-candidate selection probabilities under the representative maximin sortition lottery, computed from `src/ntqr_allotment/fairness.py` over the feasible panel draws. Each bar is one expert's probability of being seated across the lottery; the maximin objective explicitly maximises the smallest of these probabilities, so the figure should be read by its floor (the shortest bar) rather than its average — a fairer lottery lifts the worst-off candidate's chance of selection. This characterises the representation properties of the draw and is fully independent of any downstream evaluation number. Caveat: this describes panel-formation fairness only, not NTQR recovery error.](../output/figures/fairness_maximin.png){#fig:fairness width=70%}

![N-judge alarm firing rate as a function of panel size at small corpus size $Q$, computed live from `src/ntqr_allotment/ensemble.py`. The alarm fires when no single answer key can make all seated judges simultaneously axiom-consistent at the stated safety specification; the curve shows how often that happens as the panel grows. At the tight safety setting plotted here the rate is already saturated across the panel sizes shown, so the figure demonstrates that the N-judge alarm is executable and panel-size-indexed rather than establishing a monotone growth law (a looser setting would be needed to see a rising curve). Caveat: the alarm track is a consistency signal only, never a recovery method, and is bounded by the same $O(Q^3)$ answer-key enumeration, which confines it to small $Q$.](../output/figures/alarm_power.png){#fig:alarmpower width=70%}

\clearpage

## Real-Ollama postdoctoral panel results: live H5 companion

The live-Ollama results are separate empirical companion artifacts. They use one
local `{{POSTDOC_MODEL}}` model prompted as synthetic postdoctoral reviewers over
fictitious applications with synthetic age metadata. The result is not a
model-family comparison, not a human-review validation, and not evidence that age
belongs in real admissions or hiring review.

### Gemma ranking asks the same sampling question under prompt labels

The live artifact uses {{POSTDOC_RUN_COUNT}} seeds, {{POSTDOC_N_REVIEWERS}}
reviewers, {{POSTDOC_N_APPLICATIONS}} applications per seed, panel sizes
{{POSTDOC_PANEL_SIZES}}, and {{POSTDOC_LIVE_LABEL}} provenance
(`{{POSTDOC_MODEL}}` digest {{POSTDOC_MODEL_DIGEST}}, config hash
{{POSTDOC_CONFIG_HASH}}, vote-cache entries {{POSTDOC_VOTE_CACHE_ENTRIES}}).
The best live postdoc EIE point estimate is {{POSTDOC_BEST_STRATEGY}} at
{{POSTDOC_BEST_EIE}}; the worst is {{POSTDOC_WORST_STRATEGY}} at
{{POSTDOC_WORST_EIE}}. For the three-seat panels, representative sortition has
EIE {{POSTDOC_REPRESENTATIVE_EIE}}, same-bias selection has
{{POSTDOC_SAME_BIAS_EIE}}, expertise-threshold selection has
{{POSTDOC_EXPERTISE_EIE}}, and random selection has {{POSTDOC_RANDOM_EIE}}.

![Side-by-side strategy ranking for the analytical vote model (square markers) and the live Gemma reviewer panel (circle markers), one row per sampling strategy, from source `output/data/postdoc_panel_results.json`. Metric: oracle-referenced EIE error (lower is better), aggregated by track, sampling strategy, and panel size, with descriptive intervals over {{POSTDOC_RUN_COUNT}} seeds. The two marker shapes are juxtaposed, never pooled, so the reader can see where the live model echoes the analytical ordering and where it departs; the horizontal gap between a strategy's square and circle is exactly that analytical-vs-live divergence. Claim: the live single-model panel is analysed as a within-model sampling-strategy stress test, not as an LLM-family comparison; caveat: it uses synthetic applications and age metadata only, one local Gemma model, and carries no human-review validation.](../output/figures/postdoc_strategy_ranking.png){#fig:postdocrank width=80%}

### Same-bias panels expose age-conditioned recommendations

The age-bias outcome is older-minus-younger recommendation rate. Positive values
mean older synthetic applicants are recommended more often; negative values mean
younger synthetic applicants are recommended more often. At the three-seat panel
grain, representative sortition's live disparity is
{{POSTDOC_REPRESENTATIVE_AGE_DISPARITY}}, while same-bias selection's live disparity
is {{POSTDOC_SAME_BIAS_AGE_DISPARITY}}. Age bias here *is* simply illegitimate:
because true quality is generated independently of age, any age-conditioned shift
is a reviewer acting on an irrelevant attribute. That is exactly why age is a
useful *probe* — it gives a clean, known-illegitimate signal whose magnitude we
can read directly — so the question is not whether age bias is acceptable (it is
not) but whether the upstream sampling rule **amplifies or contains** an
illegitimate bias the reviewers already carry. The disparities here are negative
across the board, meaning this model favors younger synthetic applicants; the
experiment measures that illegitimate behavior to compare bias containment across
sampling rules (age is a probe; see Ethics).

![Heatmap of the older-minus-younger recommendation-rate disparity expressed by the live Gemma reviewer panel, by sampling strategy (rows) and panel size (columns), from source `output/data/postdoc_panel_results.json`. Because true latent quality is generated independently of age, any non-zero cell is reviewer age-bias expression rather than signal: positive values mean older synthetic applicants are recommended more often, negative values mean younger applicants are. Metric: age-conditioned recommendation-rate difference, aggregated over {{POSTDOC_RUN_COUNT}} seeds; read down a column to compare how each sampling rule amplifies or dampens the irrelevant age signal. Claim: same-bias (single-bloc) sampling is the explicit bias-amplification stress test among the strategies; caveat: all applicants and ages are synthetic, and this figure does not validate Gemma or endorse age-aware real review.](../output/figures/postdoc_age_bias_heatmap.png){#fig:postdocbias width=72%}

### Analytical and Gemma cells stay juxtaposed, not pooled

The alignment artifact compares analytical prediction signs with live Gemma
observations by strategy and panel size. {{POSTDOC_ALIGNMENT_RESOLVED_CELLS}} of
{{POSTDOC_ALIGNMENT_CELLS}} cells are resolved after zero-sign cells are marked
unresolved; the resolved-cell sign-agreement rate is {{POSTDOC_ALIGNMENT_RATE}}.
This agreement is a **weak directional check, not an independent match**: every
live Gemma age-disparity sign in this run is negative (the model uniformly
favors younger synthetic applicants), so the rate mostly measures how often the
analytical sign is also negative rather than a cell-by-cell coincidence of two
freely varying signals. This is the intended bridge between the synthetic and
live tracks: same causal question, shared sampling vocabulary, separate evidence
levels.

![Cell-by-cell alignment grid comparing the analytical age-disparity direction with the live Gemma direction, one cell per strategy x panel-size combination, from source `output/data/postdoc_panel_alignment.json`. Each cell is marked agree, disagree, or unresolved (a zero-sign cell on either track), so the figure functions as the explicit bridge between the controlled and the live track while keeping their uncertainties separate. Statistic: sign agreement between the analytical and live age-disparity directions; resolved-cell agreement {{POSTDOC_ALIGNMENT_RATE}} over {{POSTDOC_ALIGNMENT_RESOLVED_CELLS}} resolved cells. Because every live disparity sign is negative in this run, the agreement rate is a weak directional check rather than an independent match, and should be read as such. Claim: the analytical and empirical tracks can be compared cell by cell without pooling their uncertainty; caveat: single-model live evidence is descriptive and n-limited.](../output/figures/postdoc_empirical_alignment.png){#fig:postdocalign width=72%}

### Synthetic strategy ranking does not transfer to the live track

**H5 (does the synthetic ranking transfer to a live single-model panel?).** The
synthetic and live tracks are never pooled, and a matched-grain comparison shows
why pooling would mislead. At the shared three-seat panel grain
([@fig:trackinversion]) the strategy *rank order* inverts between tracks. The
rule with by far the lowest synthetic recovery error — expertise threshold at
{{POWER_EXPERTISE_THRESHOLD_SIZE3}} — is the **worst** under the live Gemma panel
at {{POSTDOC_EXPERTISE_EIE}}. The other three strategies are bunched on *both*
tracks — synthetically at
{{POWER_RANDOM_SELECTION_SIZE3}}–{{POWER_IDEOLOGICAL_SELECTION_SIZE3}} and live at
{{POSTDOC_REPRESENTATIVE_EIE}}–{{POSTDOC_RANDOM_EIE}} — so they neither clearly
invert nor clearly transfer. The robust component of the non-transfer is therefore
the expertise-threshold flip alone — it is the lone clear outlier on both tracks
(best synthetic, worst live) — so the non-transfer claim rests on the
competence-first rule reversing, not on a precisely resolved ordering of the other
three (which are statistically indistinguishable on each track). We compare ranks, not magnitudes, because
the two tracks own different oracles and different uncertainty, so the figure is
a qualitative non-transfer result rather than a pooled effect size.

![Cross-track strategy-ranking inversion at the matched three-seat grain, shown as a slope chart, from source `output/data/sweep_aggregated.csv` (synthetic `POWER_*_SIZE3`, left column) and source `output/data/postdoc_panel_results.json` (live `POSTDOC_*_EIE`, right column). Each strategy is a coloured line connecting its rank under the synthetic track to its rank under the live track; lines that cross are strategies whose standing reverses, and the steepest crossers are expertise threshold (synthetic best to live worst) and single-bloc selection (synthetic worst to live rank-two), while representative sortition is near-best on both tracks. The y-axis is ordinal rank, with the raw error annotated at each node so the reader can see that the live top three are tightly bunched while expertise threshold is the lone outlier. Statistic: ordinal EIE-error rank per track over {{N_SEEDS}} synthetic seeds and {{POSTDOC_RUN_COUNT}} live seeds; ranks compared, magnitudes not pooled (the two tracks own different oracles and uncertainty). Claim: the formation rule that is best blind on synthetic data is worst under one local Gemma model, so the synthetic ranking does not transfer to the live single-model panel; caveat: single-model live evidence is descriptive and n-limited, not a human-review validation.](../output/figures/track_ranking_inversion.png){#fig:trackinversion width=80%}

**Why the ranking inverts.** The two tracks do genuinely disagree, and for a
principled reason. In the synthetic track the generator *sets* each judge's
accuracy directly, so competence-first selection seats genuinely higher-accuracy
judges and the exact solver recovers them cleanly — the ordering is, in part,
built into the data-generating process. Live, "expertise" is only a *prompt
instruction*: the local `{{POSTDOC_MODEL}}` model need not behave more accurately,
or with more independent errors, when it is told it is an expert reviewer.
Selecting personas by their stated expertise therefore seats no better live
judges, and the rule that wins by construction on synthetic data carries no
guaranteed live advantage — here it is the worst. Put differently, the synthetic
oracle rewards a property (controlled judge accuracy) that the prompted model does
not inherit from the persona label. This is a hypothesis about the mechanism, not
a measured causal claim, and it is the empirical reason the manuscript keeps the
synthetic and live tracks at distinct inference levels rather than reporting a
single cross-track strategy winner.
