# output/pdf/ — ntqr_allotment

Rendered PDF deliverables and LaTeX intermediates.

- `NTQR_allotment_combined.pdf` — the manuscript PDF
- `NTQR_allotment_combined_stego.pdf` — steganographic-provenance variant (JSON payload in PDF comments near `%%EOF`; looks identical)
- `_combined_manuscript.{md,tex,toc,aux,bbl,blg}`, `references.bib` — build intermediates

Stego order: render from the template repo, then `uv run python scripts/make_stego_pdf.py` and `scripts/verify_stego.py`.
