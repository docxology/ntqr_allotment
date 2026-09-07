# AGENTS.md — docs/manuscript/

Layout: `00_abstract.md` … `04_discussion.md`, `99_references.md`, `preamble.md`, `config.yaml` (paper metadata), `references.bib`, `README.md`.
Invariant: source-of-truth manuscript markdown. Numbers are `{{TOKEN}}` placeholders hydrated from `output/data/manuscript_variables.json` at render time — never paste literal statistics here; regenerate variables instead. `config.yaml` feeds PDF metadata/publication fields.
Gotcha: `output/manuscript/` is the hydrated copy — edit here, not there.
Verify: `uv run python -m infrastructure.validation.cli markdown docs/manuscript/ --repo-root <template root>` (run from the template checkout).
- Repo-wide policy: `../AGENTS.md` (ongoing) and `/Volumes/external_drive/Git/template/projects/ongoing/AGENTS.md`; do not duplicate it here.
