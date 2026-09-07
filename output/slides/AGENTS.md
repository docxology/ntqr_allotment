# AGENTS.md — output/slides/

Layout: per-section beamer slide outputs (`00_abstract_slides.*` … `99_references_slides.*`: `.pdf`, `.tex`, `.aux`, `.nav`, `.snm`, `.toc`, `.vrb`) + `_slides_math_header.tex` + `README.md`.
Invariant: build intermediates and PDFs are all generated; regenerate via the render pipeline rather than editing `.tex` here. `_slides_math_header.tex` is a shared preamble injected by the pipeline.
Verify: re-run the slides render stage from the template checkout.
- Repo-wide policy: `../AGENTS.md` (ongoing) and `/Volumes/external_drive/Git/template/projects/ongoing/AGENTS.md`; do not duplicate it here.
