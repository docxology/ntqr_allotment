# AGENTS.md — ntqr_allotment (root)

Layout: `src/ntqr_allotment/` (all logic), `scripts/` (thin I/O orchestrators), `tests/`, `docs/manuscript/` (source markdown), `output/` (generated: data, figures, pdf, slides, web, manuscript), `docs/`, `audits/`, `vendor/` (vendored ntqr wheel). `docs/AGENTS.md` exists — see it for docs-tree rules.
Invariants: thin-orchestrator layout — business logic lives only in `src/ntqr_allotment/`; scripts do I/O and call into src. `vendor/ntqr-0.9-py2.py3-none-any.whl` is the pinned dependency; upgrade only with explicit user authorization. Manuscript numbers are token-injected from `output/data/manuscript_variables.json` at render time — never hand-edit hydrated numbers in `docs/manuscript/` prose.
Verify: `uv run pytest` (tests). Generate variables via `scripts/z_generate_manuscript_variables.py`; analysis entry points are `scripts/run_*.py`.
- Repo-wide policy: `../AGENTS.md` (ongoing) and `/Volumes/external_drive/Git/template/projects/ongoing/AGENTS.md`; do not duplicate it here.
