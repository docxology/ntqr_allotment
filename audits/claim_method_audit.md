# NTQR Allotment Claim And Method Audit

This ledger is source-owned and bounded to the artifacts in this repository. It
does not certify claims outside the shipped synthetic generator, local Ollama
artifacts, and rendered manuscript.

## Upstream NTQR Source Truth

The manuscript treats `ntqr` as a package name and local shorthand, not as an
expanded phrase. The current upstream PyPI page identifies `ntqr` 0.8 as “Tools
for the logic of evaluation using unlabeled data,” authored by Andres
Corrada-Emmanuel under the MIT license. The ReadTheDocs overview frames NTQR as
logic and algebra for unsupervised evaluation from agreement/disagreement
patterns, not as network theory. The exact binary evaluator used here is
`ntqr.r2.evaluators.ErrorIndependentEvaluation`: it evaluates exactly three
binary classifiers under the package's error-independence assumption and returns
two possible logically consistent evaluations. `MajorityVotingEvaluation` returns
two majority-vote evaluations, assuming the crowd is right or wrong, and
`SupervisedEvaluation` is reserved for labeled/oracle paths. Primary sources:
<https://pypi.org/project/ntqr/>,
<https://ntqr.readthedocs.io/en/latest/>,
<https://ntqr.readthedocs.io/en/latest/notebooks/ExactAlgebraicSolutionErrorIndependent.html>,
<https://ntqr.readthedocs.io/en/latest/autoapi/ntqr/r2/evaluators/index.html>.

| Claim or method surface | Source artifact or code | Statistical support | Caveat | Status |
| --- | --- | --- | --- | --- |
| `ntqr` is algebraic logic for unsupervised evaluation using unlabeled data; the earlier network-theory expansion is not supported by upstream sources. | PyPI and ReadTheDocs URLs above; manuscript and README terminology. | Source-document verification, not a statistical claim. | `NTQR` remains local shorthand after the package is defined; no expansion is asserted. | Verified by primary source; regression-tested. |
| Formation strategy is the dominant synthetic lever for EIE recovery. | `output/data/sweep_aggregated.csv`; `output/data/sweep_results.json`; `src/ntqr_allotment/sweeps.py` | Weighted mean EIE, pooled 95% CI, multi-seed profile metadata. | Competence-first separates from the bottom cluster; representative, random, and single-bloc are not promoted as a resolved internal ranking. | Verified by local artifact; wording bounded. |
| Strategy ranking in the active profile is competence-first far best, with representative sortition, single-bloc selection, and random selection statistically bunched. | `output/data/sweep_results.json` ranking field; rendered Results table. | Mean EIE order over the active `manuscript_contrast` profile after the physical-solution filter and deterministic 96-seed rerun. | The bottom-cluster order is a point estimate only; it is not a scientific claim. | Verified by local artifact; regression-tested. |
| Representative-vs-ideological is regime-dependent across expert stringency, bias spread, and panel size. | `output/data/sweep_aggregated.csv`; `output/data/analytical_predictions.json`; `output/figures/rep_vs_ideo_heatmap.png`. | Cell-level ideological-minus-representative EIE over 96 seeds; descriptive intervals and directional prediction checks. | Synthetic-profile bounded; cells are read individually rather than collapsed into a universal sortition advantage. | Verified by local artifact. |
| Size is not a uniform power knob. | `output/data/sweep_results.json`; `output/data/trio_conditioning.json`; power-curve and trio-conditioning figures. | Paired per-strategy size-3 vs size-6 contrast; per-trio correlation diagnostic. | Resolved size increases are tiny and essentially neutral; the diagnostic rules out a size-growing error-correlation mechanism but does not establish a positive aggregation mechanism. | Verified by local artifact; prior prose corrected. |
| Controlled error-correlation injection is measurable by NTQR diagnostics. | `output/data/independence_sweep.csv`; `src/ntqr_allotment/independence_sweep.py`. | Realized correlation rises from the lowest to highest injected `rho`. | Recovery-error slope CI crosses zero; larger-grid positive-slope language remains conjectural. | Verified by local artifact; bounded. |
| Single-model Gemma postdoc panel is the manuscript-facing live companion. | `output/data/postdoc_panel_results.json`; `src/ntqr_allotment/postdoc_panel.py`. | Live and analytical rows by strategy and panel size; EIE error, majority-vote error, age-disparity summaries, usable-trio counts, and panel composition. | Synthetic applications and ages only; one prompted local model is not a human-review substitute. | Verified by local artifact; n-limited. |
| Analytical and Gemma postdoc signs are juxtaposed cell by cell. | `output/data/postdoc_panel_alignment.json`. | Strategy x panel-size sign agreement, unresolved-cell flags, and agreement rate over resolved cells. | Descriptive companion evidence; no pooled uncertainty with the synthetic sweep. | Verified by local artifact; n-limited. |
| Live Gemma model provenance is pinned. | `model_provenance` and `vote_cache_provenance` blocks in the postdoc panel JSON artifact; `src/ntqr_allotment/postdoc_panel.py`. | Model, digest, temperature, `num_predict`, timeout, config hash, seed/reviewer/application vote-cache key fields serialized. | Reproducibility depends on local Ollama availability and retained model tag. | Verified by local artifact. |
| The manuscript contrast profile is regenerated and broader research profiles remain configured. | `manuscript/config.yaml`; `output/data/sweep_results.json`; `src/ntqr_allotment/config.py`. | `manuscript_contrast` covers panel sizes 3/6/9/12, expertise 0.62/0.68/0.74/0.80, bias 0.10/0.20/0.35/0.50, 96 seeds, 300 items; profile validation covers axes and metadata hashes. | `research_broad` and `panel_ladder` remain configured future surfaces until separately regenerated and audited as reported results. | Verified by code/config tests and regenerated artifact metadata. |
| Local web explorer exposes source tables and figure contracts. | `src/ntqr_allotment/web_explorer.py`; `scripts/make_web_explorer.py`. | Reads generated artifact tables, pair rows, multiseed runs, and figures into a local HTML filter surface. | QA/readback surface only; it does not publish or expand PDF claims. | Verified by local tests. |
| Manuscript figures use a shared readability and claim-title contract. | `src/ntqr_allotment/figure_parts/_common.py`; plot modules under `figure_parts/`; Results captions. | Minimum readable text-size regression tests, direct labels, claim-oriented plot titles, and caption source/statistic/caveat checks. | Visual polish does not add evidence or strengthen scientific claims; captions remain the claim boundary. | Verified by local tests after this pass. |
| Front-matter cover is schematic. | `output/figures/ntqr_cover.png`; `output/data/cover_manifest.json`; `src/ntqr_allotment/cover.py`. | Deterministic visual contract: audited lottery, usable trios, agreement matrix, ranked intervals. | It is explanatory front matter, not empirical evidence and not counted as a manuscript data figure. | Verified by local artifact; regression-tested. |
| Steganographic PDF version carries provenance only. | `src/ntqr_allotment/stego.py`; `scripts/make_stego_pdf.py`; `output/figures/ntqr_cover_stego.png`; `output/pdf/NTQR_allotment_combined_stego.pdf`; `output/data/stego_manifest.json`. | Extractable plain JSON payload pins source PDF/cover/config/manuscript-variable hashes and manuscript data-figure count. | This is not encryption, not confidential storage, and not empirical evidence; it is a local provenance variant. | Verified by round-trip tests and generated manifest. |
| Power/MDE layer separates resolved contrasts from design-limited nulls. | `output/data/power_analysis.csv`; `src/ntqr_allotment/power_study.py`; `src/ntqr_allotment/statistics_analysis.py`. | Cohen's d, permutation p-values, Holm correction, MDE at 80% power, seeds-for-80 budgets. | Budgets are keyed to observed effect magnitudes; not retrospective observed power evidence. | Verified by local artifact. |
| Bootstrap intervals are descriptive uncertainty summaries, not small-n proof. | `src/ntqr_allotment/statistics_analysis.py`; Methods `tbl:falsification`; Efron and Tibshirani bibliography entry. | Percentile/bootstrap summaries over seeds, cells, or live runs depending on artifact. | Does not turn six live corpora or overlapping sweep intervals into population validation. | Verified by local code/prose; primary statistical source cited. |
| Holm-adjusted significance counts are reported at the pairwise-contrast family level. | `output/data/power_analysis.csv`; `statistics_analysis.holm_bonferroni`; Holm bibliography entry. | Raw permutation p-values adjusted by step-down Holm procedure before reporting rejected count. | Applies only to the defined contrast family in the generated artifact. | Verified by local artifact; primary statistical source cited. |
| Main claim families have explicit falsification checks. | Methods `tbl:falsification`; source artifacts named in that table. | Negative controls include degenerate roots, sign mismatches, rising error with panel size, unresolved tolerance slope, and nonnegative live deltas. | The table is a manuscript contract, not a new result source. | Verified by source prose; regression-tested. |
| Alarm enumeration has an O(Q^3) scaling limit. | `output/data/alarm_timings.csv`; `scripts/bench_alarm.py`; alarm-cost figure. | Measured timings plotted on log-log axes with cubic reference. | Constants are machine-local; scaling and cap are the claim. | Verified by local artifact. |
| Ternary R=3 track is only axiom consistency. | `src/ntqr_allotment/ternary.py`; Methods and Results caveat. | Pass/fail consistency checks. | No R=3 recovery claim is made. | Verified by local code and prose. |
| ISA/TODO current state is complete with evidence-hardening upcoming work. | `ISA.md`; final gate records. | Latest recorded ruff, pytest, render, and live-Ollama artifact facts. | Upcoming items are scoped future evidence expansion, not current bring-up blockers. | Updated in this pass. |

## Method Ledger

| Method | Unit of analysis | Aggregation or uncertainty | Artifact boundary |
| --- | --- | --- | --- |
| Synthetic sweep | Strategy x panel size x profile cell x seed | Mean, SD, CI, degenerate-row exclusion, config hash, worker count, trial timeout | `sweep_results.json`, `sweep_aggregated.csv` |
| Trio EIE recovery | One usable NTQR trio | Oracle-referenced recovery error | `src/ntqr_allotment/ntqr_eval.py` |
| Ensemble-of-trios | Up to configured usable trios per panel | Average oracle-referenced trio errors | `src/ntqr_allotment/pipeline.py` |
| Tolerance regression | Unique non-degenerate (`rho`, strategy) cells | OLS slope with bootstrap CI | `independence_sweep.csv` |
| Pairwise strategy power | Per-strategy EIE observations at a fixed panel size | Cohen's d, permutation p, Holm family, MDE, per-group observation budget for 80% power | `power_analysis.csv` |
| Gemma postdoc panel | Strategy x panel-size x seed | EIE error, MV error, usable/degenerate trios, older-minus-younger recommendation-rate disparity, model/vote-cache provenance | `postdoc_panel_results.json` |
| Postdoc analytical-vs-Gemma alignment | Strategy x panel-size cells | Directional sign agreement, unresolved-cell count, caveat | `postdoc_panel_alignment.json` |
| Alarm timing | Corpus size Q | Measured seconds and cubic reference | `alarm_timings.csv` |
| Front-matter cover | Schematic pipeline visual | Config hash, concept references, visual contract, non-evidence caveat | `cover_manifest.json`, `ntqr_cover.png` |
| Figure readability contract | Manuscript-facing static PNG figures | Shared typography floor, claim title, direct labels, source/statistic/caveat caption fields | `figure_parts/_common.py`, `tests/test_figures.py`, Results captions |
| Steganographic provenance variant | Source PDF plus generated cover image | Plain JSON payload embedded in PNG LSBs and mirrored as PDF comments; extraction tests verify round-trip | `stego_manifest.json`, `ntqr_cover_stego.png`, `NTQR_allotment_combined_stego.pdf` |
| Local web explorer | Generated source tables and figures | Filters, row previews, figure contract notes | `output/web/ntqr_explorer.html` |
