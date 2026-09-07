# AGENTS.md — src/

Layout: single package `ntqr_allotment/` + `README.md`. No other children.
Invariant: all project business logic lives in `src/ntqr_allotment/`; scripts under `scripts/` may import it but logic must not migrate into scripts.
Verify: `uv run pytest` runs the `tests/` suite against this package.
- Repo-wide policy: `../AGENTS.md` (ongoing) and `/Volumes/external_drive/Git/template/projects/ongoing/AGENTS.md`; do not duplicate it here.
