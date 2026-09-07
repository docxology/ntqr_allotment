# Discussion: claim boundaries and implications

## Hypothesis verdicts before interpretation

We return an explicit verdict on each pre-stated hypothesis (Introduction, H1–H5)
before interpreting it, so the contribution is the adjudication itself rather than
a single narrative.

- **H1 (formation strategy is the dominant lever) — supported.** The four panels
  do not separate into a graded four-way ladder. Competence-first selection
  (**{{RANK_BEST_STRATEGY}}**) is clearly best at
  {{RANK_EXPERTISE_THRESHOLD_EIE}}, while representative sortition, single-bloc
  selection, and random selection cluster around
  {{RANK_REPRESENTATIVE_SORTITION_EIE}}-{{RANK_RANDOM_SELECTION_EIE}} with
  overlapping intervals. The resolved result is competence-first versus the
  bottom cluster, not a precise ordering inside that cluster.
- **H2 (concentrating correlated error degrades recovery) — unresolved on the
  baseline grid, then RESOLVED by the composition-coupled confound.** On the
  baseline generator the representative-vs-single-bloc contrast is design-limited,
  for a structural reason: that generator never realizes H2's premise, because its
  judges err independently and ideology shifts only marginal accuracy, so the
  strategies coincide *by construction* and no regime sweep can fan them out. Once a
  composition-coupled, marginal-accuracy-preserving error confound supplies the
  missing channel ([@fig:blocphase]), the contrast resolves cleanly: single-bloc
  error exceeds representative error in {{BLOC_ROBUST_FRAC}} matched regimes (paired
  mean {{BLOC_PAIRED_DIFF}}, 95% CI ±{{BLOC_PAIRED_CI}}), the gap widening from
  {{BLOC_SEP_LO}} to {{BLOC_SEP_HI}} as coupling rises. The advantage is real but
  **conditional**: an orthogonal-axis negative control removes it, so we claim
  "balancing the axis the confound rides on preserves recovery," not a universal
  sortition law.
- **H3 (size is a sampling knob, not guaranteed improvement) — supported only in
  the bounded sense.** A paired, regime-controlled test resolves a trio-to-six-seat
  size effect for {{SIZE_PAIRED_RESOLVED_COUNT}} of the four strategies, and every
  resolved effect is a tiny increase in error: random selection
  ({{SIZE_RANDOM_SELECTION_3TO6_DELTA}}), representative sortition
  ({{SIZE_REPRESENTATIVE_SORTITION_3TO6_DELTA}}), and competence-first selection
  ({{SIZE_EXPERTISE_THRESHOLD_3TO6_DELTA}}). Single-bloc is within noise. So more
  experts do not help here, but they also do not create a material penalty; size is
  essentially neutral at this grid, and the dominant lever is strategy, not size.
- **H4 (error-correlation is measurable, and recovery degrades with it) —
  diagnostic confirmed; recovery slope resolved by the composition-coupled
  instrument.** The diagnostic works — the realized correlation tracks the injected
  coupling ({{CORR_AT_RHO0}} to {{CORR_AT_RHO_HIGH}}). The *global-injection*
  tolerance sweep leaves the recovery-vs-correlation slope unresolved
  ({{TOLERANCE_SLOPE}}, 95% CI {{TOLERANCE_SLOPE_CI95}} crosses zero), and we now
  read that as a limitation of that instrument — it couples correlation uniformly
  to a fixed trio and uses a small grid — rather than as evidence of no effect. The
  composition-coupled, marginal-accuracy-preserving sweep ([@fig:blocphase]), with
  far more observations per point, resolves the recovery half in the affirmative:
  recovery error rises with within-bloc coupling for the panels that
  concentrate the confound, and the closed-form Herfindahl account
  (§Methods) explains the ordering. So the recovery half is a result, not an open
  question, under a correctly specified confound.
- **H5 (the synthetic ranking transfers to a live single-model panel) —
  rejected.** At the matched three-seat grain the ranking inverts: the rule that is
  best blind on synthetic data (expertise threshold,
  {{POWER_EXPERTISE_THRESHOLD_SIZE3}}) is the worst under the live
  `{{POSTDOC_MODEL}}` panel ({{POSTDOC_EXPERTISE_EIE}}). The robust component is that
  expertise-threshold flip; the live top three are bunched and the evidence is
  single-model and n-limited, so we report non-transfer of the best-synthetic rule
  rather than a precise live winner. The suggested caution, scoped to this one
  companion model: "choose the most expert judges" most cleanly minimized
  blind-recovery error on judges of *known* accuracy, yet it was the worst rule once
  "expert" was merely a prompt label the model need not honor — a hypothesis that
  self-asserted expertise may be an unsafe blind-evaluation selection criterion,
  worth testing beyond one model rather than an established result.

The remaining subsections elaborate the mechanism behind each verdict.

## Practical lesson: selection rule before panel size

Three usable lessons follow for anyone selecting judges to be evaluated without an
answer key. (i) *Which* rule forms the panel matters far more than how many judges
it seats: competence-first selection set the lowest blind-recovery error here, while
panel size was essentially neutral (H1, H3). (ii) Representative selection protects
unsupervised recovery *only* when the lottery balances the very attribute a shared
error rides on; balanced on the wrong axis it gives no protection (H2/H4, negative
control), and the exposure it controls is a closed-form concentration (Herfindahl)
index you can compute on a *proposed* panel before any votes are cast. (iii) The
advantage of picking "expert" judges did not survive when judges were a prompted
live model rather than parameterized synthetic ones (H5), so a selection rule
validated on controllable judges cannot be assumed to carry over to real ones.

## Formation strategy is the measured lever

Studying NTQR *upstream* — at the panel-formation step rather than at the estimator
— reframes ground-truth-free evaluation as a selection problem. The strongest
finding is that competence-first selection sets a much lower downstream
no-answer-key error floor than the other panel-formation rules. The other three
strategies cluster tightly enough that their point-estimate order should not be
read as a substantive ranking. The scientific claim is therefore about the
competence-first-vs-rest separation, not about naming a bottom-cluster winner or
loser.

That framing matters for application review because peer-review scholarship
already treats expert judgment as socially situated and panel-dependent rather
than mechanically objective [Lamont (2009)](https://doi.org/10.4159/9780674054158);
[Lee et al. (2013)](https://doi.org/10.1002/asi.22784), and because empirical
studies find substantial reviewer disagreement on the same submitted work
[Cole et al. (1981)](https://doi.org/10.1126/science.7302566);
[Pier et al. (2018)](https://doi.org/10.1073/pnas.1714379115), score-model
uncertainty large enough to alter the implied funded set
[Johnson (2008)](https://doi.org/10.1073/pnas.0804538105), and limited
grant-productivity predictiveness of NIH percentiles
[Fang et al. (2016)](https://doi.org/10.7554/eLife.13323). The manuscript's
increment is narrower: it instruments one selection mechanism and asks whether
different panel draws change an unlabeled evaluator's oracle-referenced error.
The claim is about generated artifacts and one local Gemma stress test, not about
the global reliability of academic review.

This is, in part, a result *against* the intuitive case for sortition. A
representative lottery is the fair, auditable way to form a panel, but on this
instrument it does not minimize oracle-referenced EIE error — competence-first does. We
report that plainly rather than engineering a narrative in which sortition wins.
That is not a general refutation of sortition, deliberative participation, or the
"diversity can beat ability" result. Those arguments rely on different objectives
and premises: democratic legitimacy and public consultation
[Fishkin (2009)](https://global.oup.com/academic/product/when-the-people-speak-9780199604432),
search diversity [Hong and Page (2004)](https://doi.org/10.1073/pnas.0403723101),
and jury-theorem aggregation under competence and conditional-independence
assumptions [Grofman et al. (1983)](https://doi.org/10.1007/BF00125672).
The present result is narrower: in this binary noisy-judge instrument,
competence-first sampling gives the lowest oracle-referenced EIE error.
Relative to the single-bloc comparator the representative draw's point estimate is
now reported over the full active regime grid rather than collapsed to two
panel-size means. Some cells resolve in the predicted direction, others remain
descriptive or design-limited, so we claim artifact-bounded regime structure, not a
general sortition advantage over single-bloc selection.

## Design-limited nulls remain results

Two results are bounded rather than universal, and we do not dress them up.

1. **On the baseline grid, representative vs ideological is design-limited.** Varying
   expert stringency, bias spread, and panel size jointly, the heatmap reports which
   regenerated synthetic cells align with the directional prediction, which resolve by
   descriptive intervals, and which remain uncertain — but the pooled contrast is not
   resolved there, *by construction*, because the baseline judges err independently.
   It resolves only once the composition-coupled confound supplies the correlation
   channel (see the H2 verdict and [@fig:blocphase]); the null is a property of the
   baseline design, not of sortition.

2. **Size is not a uniform power knob.** A paired, regime-controlled contrast
   resolves a trio-to-six-seat size change for {{SIZE_PAIRED_RESOLVED_COUNT}} of the
   four strategies, and each resolved change is a small increase in error. The
   largest delta is {{SIZE_RANDOM_SELECTION_3TO6_DELTA}}, so more experts do not
   help and at most very slightly hurt. The clean "more experts always helps" story
   is rejected, but the result is essentially neutral rather than a material size
   penalty. That is consistent with peer-review
   jury-theorem work: adding reviewers helps only under assumptions about
   competence, dependence, and aggregation that must be checked rather than
   presumed [Arvan et al. (2025)](https://doi.org/10.1086/719117).

Reporting these nulls is the point of building a measurement instrument rather than
a demonstration.

## Independence explains why strategy ordering changes

NTQR's EIE solver rests on the judges' errors being approximately independent. On
the baseline generator, single-bloc selection is indistinguishable from
representative and random selection **by construction**: ideology there shifts only
each judge's marginal accuracy, every judge errs from an independent stream, and an
agreement-only estimator cannot be moved by composition. Single-bloc becomes the
*separable* adversarial comparator only once the composition-coupled confound
supplies a genuine cross-judge error-correlation channel ([@fig:blocphase]).
Competence-first panels pair high accuracy with whatever independence the population
affords, giving the solver the easiest system to invert. The fair-lottery argument
should therefore be made from auditability, representation, and bounded empirical
performance rather than from an unqualified error-minimization win.

## Error independence must be measured before interpretation

The assumption the whole ordering hangs on — that judges' errors are approximately
independent — is, in this instrument, no longer an assumption but a measured
quantity. The controlled-correlation sweep confirms the *knob works*: the realized
pairwise correlation NTQR reports rises with the injected coupling
({{CORR_AT_RHO0}} to {{CORR_AT_RHO_HIGH}}). What it does **not** yet resolve, at
this grid, is whether that correlation degrades recovery — the fitted slope is
**{{TOLERANCE_VERDICT}}** ({{TOLERANCE_SLOPE}}, 95% CI
{{TOLERANCE_SLOPE_CI95}} spans zero), unresolved rather than absent — and the power
layer explains why that is unsurprising rather than disappointing:
{{POWER_WELL_POWERED_COUNT}} of {{POWER_TOTAL_CONTRASTS}} contrasts are well-powered
at the current seed count (MDE {{POWER_MDE80}}), {{POWER_SIGNIFICANT_COUNT}} reach
nominal significance, and {{POWER_SIGNIFICANT_HOLM_COUNT}} survive Holm correction.
The honest reading is that the current design resolves the largest separations but
leaves smaller neighboring contrasts design-limited. The strategy *ranking* remains
an ordering of point estimates; the analysis says exactly how many seeds would let
the unresolved contrasts resolve. That global-injection slope is, in any case, the
wrong instrument for the recovery question — it couples correlation to a fixed trio
regardless of how the panel was formed; the marginal-preserving composition-coupled
sweep ([@fig:blocphase]) resolves the recovery half in the affirmative, as
adjudicated in the H4 verdict above.

The Gemma postdoctoral panel is the direct live look at the same sampling mechanism
under a real local LLM. It does not ask whether one model family beats another; it
asks whether representative, random, same-bias, and expertise-first sampling leave
different traces when one `{{POSTDOC_MODEL}}` model is prompted as reviewers with
different expertise and irrelevant age-bias profiles. The live artifact reports
{{POSTDOC_N_APPLICATIONS}} fictitious applications per seed and model provenance
(digest {{POSTDOC_MODEL_DIGEST}}), while the alignment artifact juxtaposes
analytical signs with Gemma signs over {{POSTDOC_ALIGNMENT_CELLS}} strategy-size
cells. We still deliberately under-claim it: the applications and ages are
synthetic, the reviewer personas are prompts rather than humans, and the resolved
agreement rate ({{POSTDOC_ALIGNMENT_RATE}} over
{{POSTDOC_ALIGNMENT_RESOLVED_CELLS}} resolved cells) is descriptive companion
evidence, not validation that Gemma or any age-aware real review process is
appropriate. The competence-first versus representative comparison is likewise
gated by an explicit CI-overlap verdict (**{{THRESHOLD_SORTITION_VERDICT}}**, means
{{THRESHOLD_MEAN}} vs {{SORTITION_MEAN}}), so "beats" is never asserted across
overlapping intervals.

## Scholarship frames the stress test, not the evidence level

The postdoctoral-review setting is intentionally close to a literature where
selection, status, and bias are known concerns. Cumulative-advantage accounts of
scientific recognition [Merton (1968)](https://doi.org/10.1126/science.159.3810.56),
resubmission experiments showing fragility in journal review
[Peters and Ceci (1982)](https://doi.org/10.1017/S0140525X00011183), blind-review
experiments and observational bias studies
[Tomkins et al. (2017)](https://doi.org/10.1073/pnas.1707323114);
[Helmer et al. (2017)](https://doi.org/10.7554/eLife.21718), and empirical
studies of fellowship or grant outcomes
[Wennerås and Wold (1997)](https://doi.org/10.1038/387341a0);
[Ginther et al. (2011)](https://doi.org/10.1126/science.1196783) make it
reasonable to study reviewer sampling, not only evaluator algebra. The age axis
has the same status: ageism and age-discrimination findings motivate it as a
protected-attribute stress test
[North and Fiske (2013)](https://doi.org/10.1177/0146167213480043);
[Neumark et al. (2019)](https://doi.org/10.1086/701029), but the manuscript does
not infer anything about real postdoctoral age discrimination from prompted Gemma
votes.

The synthetic and Gemma tracks therefore answer different questions. The
synthetic track can make controlled claims because it owns the oracle label and
the expert parameters. The live Gemma track can only show whether the same
sampling vocabulary produces measurable traces under one local model with
serialized provenance. Prompted LLM evaluation is itself an active measurement
problem, not a neutral readout [Zheng et al. (2023)](https://arxiv.org/abs/2306.05685),
and language-model risk scholarship cautions against treating model text as a
transparent substitute for human judgment
[Bender et al. (2021)](https://doi.org/10.1145/3442188.3445922), so the correct
inference level is empirical feasibility plus directional stress testing.
Lottery and collective-allocation proposals in science funding
[Bollen et al. (2014)](https://doi.org/10.1002/embr.201338068);
[Fang and Casadevall (2016)](https://doi.org/10.1128/mBio.00422-16), and
maverick-science arguments for lotteries
[Avin (2019)](https://doi.org/10.1016/j.shpsa.2018.11.006) make randomized
institutional design a legitimate comparator, but they do not license a claim
that lottery-formed reviewer panels optimize NTQR recovery. Scholarship supplies
the problem context and the variables worth stress-testing; regenerated artifacts
supply the evidence.

The pre-1800 sources sharpen that boundary rather than broadening the claim.
Aristotle, Aquinas, Contarini, Montesquieu, Rousseau, Borda, and Condorcet show
that lot, choice, mixed selection, and probabilistic group judgment have long been
treated as procedural responses to faction, legitimacy, and uncertainty. They do
not license a claim that historical sortition "validates" this synthetic NTQR
instrument. The contribution here is narrower: a regenerated experiment that
keeps historical and modern motivations upstream of the evidentiary claim, then
tests how panel formation changes oracle-referenced blind recovery.

The historical sources also draw a useful negative boundary. We exclude gambling
lotteries, divinatory lots, and broad political-theory claims that are not about
selecting evaluators or aggregating judgments. The manuscript's analogy is
procedural: randomness can distribute evaluative authority when deterministic
selection is capture-prone or status-weighted. Whether that helps an unlabeled
evaluator is not answered by the history; it is answered by the regenerated
artifacts above.

## Limitations: synthetic scope, single-model live evidence, historical analogy

The reported `{{SWEEP_PROFILE}}` grid fixes prevalence and corpus size while varying
mean expertise, bias, panel size, and strategy; it uses
{{N_EXPERTS}} experts, {{N_ITEMS}} as the modal item count in the rendered tokens,
and up to {{N_TRIOS}} trios per panel over {{N_SEEDS}} seeds. The repository now
defines broader sensitivity and finer panel-ladder profiles, but those profiles are
configuration surfaces until regenerated and audited as manuscript evidence; the
reported results remain bounded to the active profile. The
oracle-closest tie-break is deliberately charitable to the unsupervised estimate, so
reported errors are a lower bound on what a blind tie-break would incur. The alarm's
$O(Q^3)$ cost confines the consistency-alarm track to small corpora
($Q\leq{{ALARM_MAX_Q}}$), so the alarm cannot yet serve as a sweep-scale signal. The
statistical-power analysis shows most strategy contrasts are underpowered at the
current seed count, so several headline comparisons are design-limited rather than
settled. The Gemma postdoctoral panel is also bounded: it uses fictitious
applications, synthetic age metadata, prompted reviewer personas, and one local
model. It tests whether the sampling mechanism is visible under that empirical
stress test; it does not establish human-review performance or a policy claim about
age. Finally, the synthetic generator is a model of noisy judges, not a guarantee
about real ones.

## Synthetic and live tracks operate at different inference levels

This manuscript follows a standard division between controlled experiment and
empirical companion evidence. The deterministic synthetic track is the controlled
Results spine: it generates the strategy-ranking, panel-size, controlled-correlation,
power-budget, alarm-cost, and analytical-alignment numbers from regenerated local
artifacts. Those claims are validated against known oracle labels because the
generator owns the truth labels.

The real-Ollama track is reported as separate live artifacts and empirical companion evidence, not a pooled extension of the synthetic sweep.
It was performed locally with required-live `{{POSTDOC_MODEL}}` (full provenance in Methods), showing the same sampling vocabulary run on a real local model prompted as different reviewers.
It does not validate the full synthetic regime grid, establish a population effect size, or prove that Gemma substitutes for human reviewers.

The combined interpretation is therefore deliberately tiered: synthetic experiments
support the controlled mechanism and regime maps; analytical checks test directional
expectations against those regenerated artifacts; and live Ollama runs demonstrate
empirical feasibility plus n-limited directional support. Wider parameter sweeps and
larger empirical panels are the next steps before any stronger general claim.

## Data, code, and generated-artifact availability

All source code, methods, and documentation are openly available at the public
repository [`docxology/ntqr_allotment`](https://github.com/docxology/ntqr_allotment):
the deterministic synthetic instrument, the bloc-confound and Herfindahl modules,
the live Gemma vote cache with serialized model provenance, every figure, and the
manuscript regeneration pipeline. Every reported number is
token-injected from `output/data/` by
`src/ntqr_allotment/manuscript_variables.py`, so no result is hand-transcribed and
the manuscript regenerates from source under a zero-orphan token contract. The
synthetic track is fully deterministic under fixed seeds (profile config hash
`{{SWEEP_CONFIG_HASH}}`); the live track reproduces against a local Ollama
`{{POSTDOC_MODEL}}` instance (digest `{{POSTDOC_MODEL_DIGEST}}`, config hash
`{{POSTDOC_CONFIG_HASH}}`) using the serialized, resumable per-vote cache keyed on
the config hash, seed, reviewer, application, model digest, and decode parameters.
A steganographic provenance variant of the PDF additionally carries an extractable
hash of the current source PDF, verified by `scripts/verify_stego.py`. The archival
DOI is [10.5281/zenodo.21083779](https://doi.org/10.5281/zenodo.21083779).

## Ethics, protected attributes, and competing interests

The postdoctoral-review setting is entirely synthetic: the applications, latent
quality labels, reviewer personas, and age metadata are all generated, and no human
subjects, real applicants, or real review records are involved. True latent quality
is generated independently of age; age enters only as a protected-attribute stress
test for bias expression under sampling, and its use here is diagnostic, not an
endorsement of using age in real admissions, hiring, or fellowship review. The live
language-model outputs are treated as an instrumented measurement of a prompted
system, not as a substitute for human judgment. The authors declare no competing
interests.
