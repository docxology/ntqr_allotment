# AGENTS.md — tests/

Layout: `test_*.py` mirrors of `src/ntqr_allotment/` modules (config, ensemble, dependence, fairness, personas, cross_family, bloc_confound, cover, determinism, figures, manuscript_variables, ntqr_eval, …), artifact/claim-contract tests, `__init__.py`, `README.md`.
Invariants: tests exercise real computation — no mock frameworks. Claim-contract and artifact-metadata tests guard manuscript-number provenance; treat their failures as blocking.
Verify: `uv run pytest` from the project root.
- Repo-wide policy: `../AGENTS.md` (ongoing) and `/Volumes/external_drive/Git/template/projects/ongoing/AGENTS.md`; do not duplicate it here.
