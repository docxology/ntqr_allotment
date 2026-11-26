# Reproduce the study offline

Run the following commands from a study checkout. The project pins
[Python 3.15](../.python-version) by default and CI covers Python 3.12,
3.13, and 3.15. See [dependencies](dependencies.md) for the vendored NTQR
wheel, pinned allotment source and CBC solver setup.

```bash
uv sync --locked --extra dev
uv run --no-sync pytest -m 'not requires_ollama'
uv run --no-sync python scripts/check_research.py
uv run --no-sync python scripts/z_generate_manuscript_variables.py --check
```

The hydration check reads the tracked results and compares its derived token
map, numbered sections, reference and figure metadata, and manifest against
the generated inputs in `output/`. The published observations include
deterministic simulations and the separate, recorded `gemma3:4b` companion.
No model server is needed to render or check these records.

## Seeded replay

The [canonical configuration](manuscript/config.yaml) defines the study designs
under `experiment_profiles`. The default is the full `manuscript_contrast`
grid. To exercise the full native seeded path on the small `smoke` profile,
run the command below in a separate clone because the sweep writes fixed
filenames under `output/data/`.

```bash
study_root=$(pwd)
replay_dir=$(mktemp -d)
git clone "$study_root" "$replay_dir/ntqr_allotment"
(
  cd "$replay_dir/ntqr_allotment"
  uv sync --locked --extra dev
  uv run --no-sync python scripts/run_sweep.py --profile smoke
)
```

This produces a new `sweep_results.json` and `sweep_aggregated.csv` with the
profile, seeds, outcome counts, and runtime provenance. Compare runs under
the same solver and dependencies. Keep the published full-grid artifacts
with their original provenance. For an analytical-only postdoc diagnostic,
use `scripts/run_postdoc_panel.py --offline --output-dir <temporary directory>`.
That mode generates analytical records. The archived Gemma companion used
synthetic applications judged by a single prompted model and is a separate
measurement.

## Pinned manuscript render

Install Git, uv, XeLaTeX, BibTeX, and a matching Pandoc and pandoc-crossref
pair on `PATH`. The local verification used Pandoc 3.1.13 and pandoc-crossref
0.3.17.1. From the study root, create a separate Template checkout:

```bash
study_root=$(pwd)
git clone https://github.com/docxology/template.git ../template-render
cd ../template-render
git checkout --detach 87704feae0a4490ef7dad9c931d268e8c30273c2
uv sync --locked --no-dev --python 3.14

git clone "$study_root" projects/working/ntqr_allotment
(
  cd projects/working/ntqr_allotment
  uv sync --locked --extra dev --python 3.14
  uv run --no-sync python scripts/z_generate_manuscript_variables.py --check
)

uv run --no-sync python scripts/pipeline/stage_03_render.py --project working/ntqr_allotment
uv run --no-sync python scripts/pipeline/stage_04_validate.py --project working/ntqr_allotment
```

Stage 03 hydrates the [canonical manuscript](manuscript/README.md) from
the tracked evidence and figures. The full Stage 02 analysis roster includes
live data producers. The targeted render above uses the recorded artifacts
and leaves those producers idle.

In the working checkout, look for
`output/pdf/ntqr_allotment_combined.pdf`, `output/web/index.html`, and
`output/epub/ntqr_allotment_combined.epub`. Stage 04 writes its validation
report and rendered provenance receipt in `output/reports/`.
Check the abstract, introduction, methods, results, discussion, and references,
as well as the cover, twenty body figures, and resolved citations. The preamble
keeps figures and captions with their text to prevent overflowing float queues. The reader
build may replace historical render files in its working checkout.

## Archived release

The [publication package](PUBLICATION_PACKAGE.md) describes the retained
v0.1.0 PDF, EPUB, HTML, and historical slides. The **Visually equivalent provenance variant**
retains the provenance payload made when it was built. Inspect the pairing from a clean checkout:

```bash
uv run --no-sync python scripts/verify_stego.py
```

The report compares the extractable PDF and cover provenance, checks it against
the archive's manifest, and compares the embedded `source_pdf_sha256` to
the retained source PDF. In the audited checkout, the embedded PDF and PNG
payloads match the manifest, while the tracked normal PDF has a different
SHA-256 from the payload's `source_pdf_sha256`. The verifier reports a stale
source binding. The current renderer uses the canonical manuscript and retained
study data independently of that archived pairing.
