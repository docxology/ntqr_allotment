# AGENTS.md — src/ntqr_allotment/

Layout: core modules — `pipeline.py`, `config.py`, `ensemble.py`, `experts.py`, `ternary.py`, `dependence.py`, `determinism.py`, `fairness.py`, `personas.py`, `postdoc_panel.py`, `bloc_confound.py`, `contrast_analysis.py`, `cross_family.py`, `independence_sweep.py`, `ntqr_eval.py`, `corpus.py`, `cover.py`, `stego.py`, `sortition.py`, `solver.py`, `artifacts.py`, `statistics_analysis.py`, `sweeps.py`, `theory.py`, `trio_conditioning.py`, `web_explorer.py`, `figures.py`, `manuscript_variables.py` — plus subpackages `figure_parts/` and `power_parts/`. (Full module list: see directory; every module has coverage in `tests/`.)
Invariants: this is the only place for project logic. Depends on the vendored `vendor/ntqr` wheel (pinned 0.9) — upgrade only with explicit user authorization. `manuscript_variables.py` computes the token values injected into the manuscript.
Verify: `uv run pytest tests/test_<module>.py` for focused checks.
- Repo-wide policy: `../AGENTS.md` (ongoing) and `/Volumes/external_drive/Git/template/projects/ongoing/AGENTS.md`; do not duplicate it here.
