# AGENTS.md — output/

Layout: generated artifacts only — `data/`, `figures/`, `manuscript/` (hydrated), `pdf/`, `slides/`, `web/`, plus top-level `manuscript_variables.json`. All contents are reproducible via `scripts/`.
Invariant: everything here is derived; do not hand-edit outputs to fix problems — fix `src/`/`scripts/` inputs and re-run. `manuscript_variables.json` is the single token source for manuscript numbers.
Verify: regenerate via `scripts/run_*.py` and figure scripts; `scripts/z_generate_manuscript_variables.py` refreshes variables.
- Repo-wide policy: `../AGENTS.md` (ongoing) and `/Volumes/external_drive/Git/template/projects/ongoing/AGENTS.md`; do not duplicate it here.
