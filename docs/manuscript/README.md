# Sortition Upstream of NTQR

The numbered Markdown files in this directory are the canonical source.

- [Abstract](00_abstract.md) and [introduction](01_introduction.md)
- [Methods](02_methods.md), [results](03_results.md), and [discussion](04_discussion.md)
- [References](99_references.md) and [BibTeX bibliography](references.bib)
- [Paper configuration](config.yaml), [LaTeX preamble](preamble.md), and [figure registry](figure_registry.json)

Edit the numbered Markdown to change the paper. Keep computed claims as
`{{TOKEN}}` placeholders. The bridge resolves them from the retained data
and writes the numbered render inputs under `output/manuscript/`.
Figures link to the tracked scientific artwork. The registry pairs their
manuscript labels and filenames with short alternative descriptions.

Follow [reproduction and rendering](../reproduction.md) to verify the tokens
and build PDF, HTML, and EPUB with the pinned Template runner.
