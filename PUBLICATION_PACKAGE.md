# NTQR_allotment Publication Package

Prepared after the 2026-06-28 RedTeam proofreading pass.

## Canonical Metadata

Title: Sortition Upstream of NTQR

Subtitle: How Panel Formation and Size Shape Ground-Truth-Free Evaluation

Author: Daniel Ari Friedman

ORCID: 0000-0001-6232-9096

DOI: [10.5281/zenodo.21083779](https://doi.org/10.5281/zenodo.21083779) (concept; version DOI [10.5281/zenodo.21083780](https://doi.org/10.5281/zenodo.21083780), published)

GitHub: [docxology/ntqr_allotment](https://github.com/docxology/ntqr_allotment), release [v0.1.0](https://github.com/docxology/ntqr_allotment/releases/tag/v0.1.0)

Short description: A deterministic research testbed for how upstream expert-panel formation affects no-answer-key `ntqr` evaluation error, with a separate single-model Gemma postdoctoral-review companion.

Keywords: sortition; NTQR; unlabeled evaluation; expert panels; peer review; error independence; statistical power; panel formation; synthetic evaluation; LLM reviewers.

## Claim Boundary

- Current controlled evidence is bounded to the regenerated `manuscript_contrast` profile: 96 seeds, 300 items, 96 experts, panel sizes 3/6/9/12, expertise 0.62/0.68/0.74/0.80, and bias spreads 0.10/0.20/0.35/0.50.
- Main synthetic result (baseline generator): expertise-threshold selection is the only resolved best strategy; representative sortition, random selection, and single-bloc ideological selection are statistically bunched. This bunching is a *structural* property of the baseline generator (errors are independent across judges; ideology shifts only marginal accuracy), not a statistical accident.
- Composition-coupled result (`bloc_confound.py`): when judges who share a group attribute also share a latent, marginal-accuracy-preserving error confound of strength rho, the strategies fan out monotonically as coupling rises — representative sortition stays flat while single-bloc ideological selection degrades, with the measured trio error-correlation tracking the effect. This resolves H2/H4 that the global-injection tolerance sweep left design-limited. The robustness is **axis-conditional**: a full-rho `expertise_tier` negative control (an axis the lottery does not balance) removes representative sortition's immunity, so the claim is "balancing the axis the confound rides on preserves recovery," not a blanket sortition endorsement. Synthetic, oracle-scored, simulation only.
- Closed-form account (`theory.py`): the composition-to-exposure relationship is closed-form by construction — keying the shared shock on the grouping axis makes a trio's confound exposure equal to its same-group pair count, i.e. the panel's **Herfindahl–Hirschman concentration index**. Representative draws attain H = 1/B exactly, single-bloc H = 1, and this ordering predicts the measured correlation ordering. A continuous representativeness dial (`concentration_panel`) confirms recovery error rises monotonically with H. The genuinely falsifiable link is whether NTQR recovery actually degrades as that exposure rises — it does — making the result more than just an engineered instrument.
- Honest limitation (disclosed, not patched): the *legacy* global-injection model `dependence.sample_votes_correlated` is not marginal-accuracy-preserving (a convex combination of uniforms is triangular), so it inflates realized accuracy non-monotonically (peak near rho=0.5); this is the source of the tolerance sweep's "non-monotone dip." It is retained as the correlation *diagnostic* only; all recovery conclusions use the marginal-preserving copula.
- Size result: trio-to-six-seat effects are tiny and essentially neutral at this grid, not evidence that larger panels materially help or materially hurt.
- Mechanism result: the trio-conditioning diagnostic rules out a size-growing error-correlation mechanism; it does not establish a positive replacement mechanism.
- Live companion: the Gemma `gemma3:4b` postdoctoral-review run is synthetic-applicant, single-model, and n-limited; it is not human-review validation and not a policy argument for using age in real review.

## Publish-Readiness Assessment (2026-06-29 multi-agent review)

A multi-dimensional review (formalism, honesty, writing, publication-readiness;
each finding adversarially verified) was run and acted on. Verdict: **ready for an
arXiv preprint and a methods/simulation or sortition/evaluation venue, conditional
on framing the contribution as a formal + simulation result, not an empirical
discovery.**

- **Strengths**: fully deterministic and seeded; every manuscript number
  token-injected from regenerated artifacts under a zero-orphan gate; explicit
  H1–H5 falsification ledger with verdicts; a power/MDE layer separating resolved
  from design-limited nulls; an axis-conditional negative control; and now a
  closed-form Herfindahl law that makes the composition effect falsifiable and
  general rather than a property of one strategy.
- **Honest scope**: the sharp positive result (the monotone fan-out) is produced by
  a confound the authors inject; the Herfindahl composition-to-exposure relationship
  follows by construction from how the confound is keyed; the genuinely falsifiable
  link is whether NTQR recovery degrades along it — it does. The manuscript states
  this explicitly and scopes the lesson as a "simulation-bounded prediction."
- **Remaining gaps before a top-tier empirical venue** (out of scope for this
  pass, documented for honesty): (1) no real-data anchor — the only non-synthetic
  evidence is one small local `gemma3:4b` run; (2) no benchmarking against
  alternative unsupervised estimators (e.g. Platanios et al., Parisi et al.); (3)
  the oracle is used both to disambiguate the EIE two-solution branch and to score
  error. These are appropriate future-work items, not blockers for a preprint /
  methods venue that judges the formal + simulation contribution on its own terms.

## Artifact Set

| Purpose | Path | Publication note |
| --- | --- | --- |
| Main manuscript PDF | `output/pdf/NTQR_allotment_combined.pdf` | Primary reader artifact. |
| Stego provenance PDF | `output/pdf/NTQR_allotment_combined_stego.pdf` | Visually equivalent provenance variant; include only with the caveat that payloads are plain extractable JSON, not encryption. |
| Combined HTML | `output/web/index.html` | Web reader artifact; verify relative figure paths before hosting outside the repo. |
| Section HTML | `output/web/manuscript__*.html` | Optional per-section web exports. |
| Slides | `output/slides/*_slides.pdf` | Optional presentation companion, not the archival manuscript. |
| Figures | `output/figures/*.png` | Generated from source artifacts; captions in Results remain the claim boundary. |
| Data artifacts | `output/data/*.json`, `output/data/*.csv` | Include with source archive when the platform supports supplemental files. |
| Provenance manifest | `output/data/stego_manifest.json` | Pairs with stego PDF/cover. |
| Source manuscript | `manuscript/*.md`, `manuscript/references.bib`, `manuscript/config.yaml` | Source-owned text and config. |
| Rendered manuscript inputs | `output/manuscript/*.md`, `output/pdf/_combined_manuscript.tex` | Useful for arXiv-style source packages. |

## Platform Preparation

### Zenodo / OSF / Figshare

- Upload the main PDF, source archive, generated figures, generated data artifacts, and `PUBLICATION_PACKAGE.md`.
- Add the stego PDF and `stego_manifest.json` as optional provenance companions with the plain-JSON caveat.
- DOI minted and Zenodo record published (2026-06-30, concept 10.5281/zenodo.21083779 / version 10.5281/zenodo.21083780, deposition `21083780`, state `done`); the deposit carries only the final combined PDF, per publication policy for this release. `manuscript/config.yaml` updated (doi/version_doi/version_record/keywords), manuscript rerendered, stego rebuilt, gates rerun green.

### arXiv-style Preprint

- Use `output/pdf/NTQR_allotment_combined.pdf` as the reviewed PDF.
- If source upload is required, package `output/pdf/_combined_manuscript.tex`, `output/manuscript/references.bib`, and the referenced `output/figures/*.png` files.
- Keep the title/subtitle acronym-free except for `NTQR`, which is the upstream package name and is defined in the abstract.
- Do not upload private live vote-cache internals unless the target platform and privacy boundary are explicitly approved.

### Project Website / Static HTML

- Host `output/web/index.html` with the `output/figures/` assets in the same relative layout, or rewrite figure paths during deployment.
- Keep `output/web/ntqr_explorer.html` local QA by default. If it is published, label it as a readback/exploration surface that does not expand the manuscript claims.
- Include links to the main PDF, source archive, and generated artifact manifest.

### Repository Release

- Attach the main PDF, stego PDF, combined HTML, source archive, and generated data/figure archive.
- State dependency caveats: `allotment` is AGPL-3.0 and sourced from GitHub; `ntqr` is MIT and vendored as its published wheel because the sdist is not lockable in this environment.
- State the live dependency caveat: live tests require local Ollama with `gemma3:4b` digest `a2af6cc3eb7f`.

## Latest Gate Evidence

Refreshed 2026-06-29 after the multi-agent RedTeam + writing-quality pass:
composition-coupled bloc-confound fan-out study (`bloc_confound.py`,
`figure_parts/phase.py`, `fig:blocphase`), full-rho orthogonal-axis negative
control, continuous representativeness dial (`concentration_panel`, `fig:dial`),
closed-form Herfindahl account (`theory.py`), wide cover redesign, ToC trimmed to
one page (26 H3 headings), abstract rewritten (problem-first, panel-selection
framing, ~340 words), repo `docxology/ntqr_allotment` stated in abstract + Data
section, and a comprehensive writing/honesty refactor: load-bearing contradiction
(baseline "genuinely more correlated" claim) corrected; "phase transition" →
"monotone fan-out" throughout (0 remaining); by-construction vs. genuinely-
falsifiable links cleanly separated; age disclaimer consolidated to Ethics section;
changelog-leakage language removed; cross-reference direction corrected; per-point-n
token emits true mean; falsification-ledger cells updated.

- `ruff check` (new/changed sources) -> All checks passed.
- `scripts/z_generate_manuscript_variables.py --check` -> 174 generated, 118 used, 0 orphans.
- `scripts/run_bloc_phase.py --workers 8 --seeds 24` -> 10,080 cells (main + full-rho control + concentration dial); MAIN ideo-rep separation 0.0 -> +0.000, 0.9 -> +0.112; orthogonal-axis negative control representative 0.147 -> 0.229 with the gap collapsing to +0.026 (protection is axis-conditional); robustness ideo>rep in 180/205 paired regimes; concentration dial balanced 0.175 -> concentrated 0.263 (monotone fraction 1.00).
- `scripts/verify_stego.py` -> PDF/PNG payloads match manifest; source PDF hash is current (rebuilt against the refreshed combined PDF).
- `output/pdf/NTQR_allotment_combined.pdf` (4.14 MB) -> rendered; 0 literal `{{TOKEN}}` remain; `fig:blocphase` and `fig:dial` embedded; 20 labeled figures; new citations resolved (Platanios, Parisi, Emrich–Piedmonte, Nelsen).
- `pytest -m "not requires_ollama" --cov=src/ntqr_allotment --cov-fail-under=90` -> 598 passed, 4 deselected; total coverage ~93% (gate 90%).

Refreshed 2026-06-30 after Zenodo DOI reservation (production, unpublished draft):
concept DOI `10.5281/zenodo.21083779`, version DOI `10.5281/zenodo.21083780`
written into `manuscript/config.yaml`, the title page, and the Discussion's
data-availability sentence; manuscript rerendered and stego rebuilt against the
refreshed PDF.

- `ruff check .` -> All checks passed.
- `scripts/z_generate_manuscript_variables.py --check` -> 174 generated, 118 used, 0 orphans.
- `scripts/make_stego_pdf.py` + `scripts/verify_stego.py` -> PDF/PNG payloads match manifest; `source_pdf_hash_is_current: true` against the DOI-bearing PDF.
- `qpdf --check` on `output/pdf/NTQR_allotment_combined.pdf` and `..._combined_stego.pdf` -> no syntax/stream errors on either.
- `pytest -m "not requires_ollama" --cov=src/ntqr_allotment --cov-fail-under=90` -> 599 passed, 4 deselected; total coverage 93.04% (gate 90%).
- Pre-publish path-leakage audit: found and fixed absolute local-filesystem paths leaking into generated provenance metadata (`cover.py` concept-art references, `config_path`/`source`/`vote_cache_provenance.path` fields in `run_cross_family.py`, `run_cross_family_multiseed.py`, `contrast_analysis.py`, `postdoc_panel.py`) and in `README.md`'s reproduction snippet; fixed at the source and confirmed via a repo-wide grep for `/Users/` across all text files in the publish export. No leakage in the rendered PDF/HTML at any point.
- Zenodo deposition `21083780` published 2026-06-30 (`state: done`); DOI `10.5281/zenodo.21083780` resolves publicly; deposit carries the single final combined PDF only (`Friedman_2026_Sortition_73289489.pdf`, sha256 `73289489d2d1...c009061e4`).
- GitHub repository `docxology/ntqr_allotment` created public 2026-06-30 and pushed as a single squashed commit (no incremental development history) containing full source, tests, manuscript, and generated outputs; release `v0.1.0` published with the main and stego PDFs attached.
