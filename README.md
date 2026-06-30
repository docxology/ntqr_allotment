# NTQR_allotment

**Sortition upstream of NTQR** — a research testbed for how *forming an expert
panel* and *how many experts you draw* shape what no-answer-key evaluation can
infer about noisy experts of known precision, bias, and heterogeneity.

Here "ground-truth-free" is project shorthand for evaluation from unlabeled
decision data: labels are hidden from the estimator, then used only afterward to
score the synthetic oracle-referenced error.

Two upstream libraries are imported and used for real:

- [`allotment`](https://github.com/Citizen-Infra/allotment) (AGPL-3.0) — an
  auditable fair-sortition engine (maximin stratified lottery, Flanigan et al.
  *Nature* 2021). Draws a representative mini-public by reproducible, SHA-256
  audited lottery.
- [`ntqr`](https://ntqr.readthedocs.io/) (MIT) — algebraic logic tools for
  unsupervised evaluation of noisy binary classifiers from unlabeled decision
  data. In this project, its error-independent three-classifier evaluator returns
  the possible logically consistent prevalence and per-classifier accuracy
  evaluations implied by agreement/disagreement patterns, **without an answer
  key**.

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
evaluation is scored against the **supervised oracle** — an honest comparator,
not a rigged win. The empirical layer uses one required-live local
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

## Setup

```bash
uv sync            # locks clean deps + allotment(git) + vendored ntqr wheel
uv run pytest      # tests
```

### Dependency note

`ntqr` ships a working wheel but a broken sdist (its flit build references a
README absent from the tarball), so `uv` cannot lock it from the index. The
published wheel is vendored under `vendor/` and referenced via
`[tool.uv.sources]`. `allotment` is not on PyPI and is installed from its git
source. See `pyproject.toml`.

## Reproduce

Every reported number regenerates from `output/data/` artifacts; nothing in the
manuscript is hand-transcribed (`manuscript_variables.py` emits the tokens and a
zero-orphan gate enforces it). The analysis scripts run in a dependency-correct
order declared by `analysis.scripts` in `manuscript/config.yaml` — producers
first, then the token injector, then the figure consumers:

```bash
# data producers -> token injector -> figures (see analysis.scripts order)
uv run python scripts/run_sweep.py
uv run python scripts/run_analytical_predictions.py
uv run python scripts/run_power_analysis.py
uv run python scripts/run_independence_sweep.py
uv run python scripts/run_postdoc_panel.py --require-live   # live gemma3:4b (slow)
uv run python scripts/z_generate_manuscript_variables.py    # tokens + injection
uv run python scripts/make_figures.py
uv run python scripts/make_extended_figures.py
```

The synthetic track is byte-deterministic under fixed seeds. The live postdoc
track is reproducible against a local Ollama `gemma3:4b` via a serialized,
resumable per-vote cache (keyed on config hash, seed, reviewer, application,
model digest, and decode params); it is `--require-live` by design and resumes
incrementally if interrupted. Render the PDF/HTML/slides from the template repo
(see the steganographic-provenance order below).

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
| `bloc_phase_diagram` | Bloc-confound phase transition: recovery error and measured trio correlation vs within-bloc coupling, one line per strategy; representative sortition stays flat while single-bloc selection fans out. |
| `bloc_concentration_dial` | Continuous representativeness dial: recovery error and measured correlation vs single-bloc concentration (Herfindahl index 1/B to 1) at fixed coupling; error rises monotonically, tracing the closed-form Herfindahl law. |
| `alarm_cost_curve`, `alarm_power`, `fairness_maximin` | O(Q^3) alarm cost, N-judge alarm power, maximin selection probabilities. |
| `track_ranking_inversion` | Cross-track ranking inversion at the matched three-seat grain (ranks compared, magnitudes not pooled). |
| `postdoc_strategy_ranking`, `postdoc_age_bias_heatmap`, `postdoc_empirical_alignment` | Live Gemma reviewer-panel ranking, age-disparity, and analytical-vs-Gemma alignment. |

## Status

The deterministic spine, live Gemma postdoc-panel companion, statistical-power
layer, shared figure-readability theme, caption contract, front matter, cover,
and manuscript render gates are implemented and covered by the project test
suite. The current manuscript contrast profile regenerates a 96-seed, 300-item
grid over panel sizes 3/6/9/12, expertise 0.62/0.68/0.74/0.80, and bias spreads
0.10/0.20/0.35/0.50; the representative-vs-ideological and pre/post NTQR results
are reported as heatmaps, with directional-prediction checks in
`output/data/analytical_predictions.json`. The clean rerun result is intentionally
bounded: expertise-threshold is the only resolved best strategy; representative,
random, and single-bloc selection are statistically bunched; size effects are
tiny and essentially neutral at this grid; and the trio-conditioning diagnostic
rules out a size-growing error-correlation mechanism without asserting a positive
replacement mechanism. That three-way bunching is structural to the baseline
generator (independent errors; ideology shifts only marginal accuracy), and a
composition-coupled error confound (`bloc_confound.py`, `fig:blocphase`) resolves
it: when judges sharing a group also share a marginal-accuracy-preserving latent
error shock, the strategies fan out into a phase transition — representative
sortition stays flat while single-bloc selection degrades — with an
`expertise_tier` negative control showing the protection is axis-conditional, not
innate. The effect has a closed-form backbone (`theory.py`): realized
error-correlation, hence recovery degradation, is governed by the panel's
Herfindahl concentration index over the confound axis, and a continuous
representativeness dial confirms recovery error rises monotonically with it. The live companion is required-live Ollama evidence over
`gemma3:4b` only: one model prompted as reviewers in a synthetic postdoctoral
application setting, with model digest, decode params, vote-cache provenance,
panel composition, age-disparity summaries, and analytical-vs-Gemma alignment
serialized.

Current open work is proofreading and publication packaging, followed by evidence
expansion only when regenerated artifacts support stronger claims. Broader
`research_broad` and `panel_ladder` profiles plus finer live postdoc
bias/application strata remain future work. A deterministic schematic cover is
generated at `output/figures/ntqr_cover.png` with provenance in
`output/data/cover_manifest.json`; it is front matter, not an empirical result.
A generated local artifact explorer is available at
`output/web/ntqr_explorer.html` after regeneration; it is a QA/readback surface,
not a publishing surface. The static figures are manuscript reader surfaces:
their titles and captions summarize artifact-bounded claims, not additional
evidence beyond the generated source tables.

### Steganographic provenance variant

The stego PDF is expected to look the same as the normal PDF. Its payload is
plain JSON provenance appended as PDF comments near `%%EOF`, and the same payload
is mirrored in the least-significant bits of `output/figures/ntqr_cover_stego.png`.
It is not encryption, not a visible watermark, and not a scientific result.

The required local order is:

```bash
cd <path-to-template-repo>
uv run python scripts/03_render_pdf.py --project working/NTQR_allotment
cd <path-to-this-repo>
uv run python scripts/make_stego_pdf.py
uv run python scripts/verify_stego.py
```

`verify_stego.py` checks that the PNG and PDF payloads extract to the same bytes,
that their hash matches `output/data/stego_manifest.json`, and that the embedded
`source_pdf_sha256` still matches the current normal PDF.

## License

Project code: see repository. Note `allotment` is AGPL-3.0; this is a
local-only research project and is not distributed as a combined work.
