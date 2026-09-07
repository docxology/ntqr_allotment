# docs/manuscript/ — ntqr_allotment

Markdown source of the publication manuscript, one file per section, rendered to PDF/HTML/slides by the template repo's render stages.

- `00_abstract.md` … `04_discussion.md`, `99_references.md` — section sources
- `preamble.md` — front matter
- `config.yaml` — paper metadata (title, authors, publication info) and `analysis.scripts`: the declared dependency-correct order for analysis producers → token injector → figure consumers
- `references.bib` — BibTeX citations

Manuscript numbers are injected tokens produced by `scripts/z_generate_manuscript_variables.py` (zero hand-transcribed values; a zero-orphan gate enforces this).
