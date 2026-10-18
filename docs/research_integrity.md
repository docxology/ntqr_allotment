# Research integrity, evidence and publication artifacts

`research_map.json` is the machine-readable claim registry. Each claim states
its scope, bibliography keys, evidence paths and limitations. Selected primary
records carry their verified source URL and review date. This is a curated
core bibliography review, not a claim to exhaustive literature coverage.

## Checks

```bash
uv run python scripts/check_research.py
```

The command is offline and read-only. It rejects duplicate bibliography keys,
missing creators/titles/years, malformed identifiers, unresolved Pandoc citations,
dangling evidence paths, duplicate claim IDs and claims lacking limitations.
Its parser supports the brace-valued BibTeX subset maintained here. It does not
resolve arbitrary BibTeX macros or verify every cited work against the internet.
CI runs this gate independently of the numerical regression suite.

## Inference boundaries

| Object | What it establishes | What it cannot establish |
|---|---|---|
| Observed vote variation | A necessary input condition for the implemented binary-trio solve | Identifiability or physical roots by itself |
| Physical solution | Finite probability parameters compatible with the fitted model | Conditional independence or correct label orientation |
| Observed vote covariance | Unconditional agreement structure | Conditional error covariance without labels or assumptions |
| Oracle-referenced error | Agreement with the labeled benchmark convention | A deployable label-free branch-selection rule |
| Binomial SE reference | A descriptive Bernoulli proportion scale | A universal lower bound on recovery MAE |
| Scenario bootstrap | Conditional variation under the declared resampling scheme | Prompt/model-selection or external-population uncertainty |
| Logical inconsistency | Incompatibility with the stated exact empirical constraints | A calibrated population-level zero-false-positive guarantee |
| Passing software/citation checks | Declared implementation and reference integrity | Scientific correctness, causal fairness or external validity |

Allman, Matias and Rhodes' [latent-model identifiability paper](https://arxiv.org/abs/0809.5032)
provides the relevant distinction between generic identifiability and exceptional
nonidentifiable parameters. Model and rank assumptions matter; an algebraic
root alone does not remove them.

## Figures and sources

Static scientific figures export a raster PNG and a selectable-text SVG with a
title/description. Shared exporters write complete files atomically, preserve
existing file modes, contain style changes, and close newly created figures on
failure. Direct labels, marker shapes and numeric axes carry meaning alongside
color. Figure data sidecars and the recorded source artifacts carry the numbers;
visual polish does not add evidence.

The manuscript's empirical numbers continue to come from token generators.
Authored prose states the scope of those numbers. Benchmarks live under
`benchmarks/runtime/` and never masquerade as additional study observations.
An archived release DOI identifies its archive; cite the Git commit as well
when using later main-branch code or regenerated figures.

## Offline figure and manuscript regeneration

```bash
uv run python scripts/make_figures.py
uv run python scripts/z_generate_manuscript_variables.py
uv run python scripts/z_generate_manuscript_variables.py --check
```

Figure builders remain in `ntqr_allotment.figure_parts`; scripts load artifacts
and call them. `ntqr_allotment.figure_export.export_figure` is reusable for a
caller-supplied Matplotlib figure. The source manuscript remains under
`docs/manuscript/`, and hydrated deliverables remain under `output/manuscript/`.
