# AGENTS.md — scripts/

Layout: thin orchestrators — `run_*.py` (analysis: sweep, independence_sweep, bloc_phase, cross_family(+multiseed), trio_conditioning, personas, postdoc_panel, power_analysis, analytical_predictions), `make_*.py` (figures, extended figures, cover, web explorer, stego PDF), `bench_alarm.py`, `verify_stego.py`, `z_generate_manuscript_variables.py` (token injector), `README.md` (inventory table: script | purpose | delegates to | command).

Invariant: I/O and orchestration only — all computation imports from `src/ntqr_allotment/`. Adding logic here violates the thin-orchestrator pattern. Outputs land under `output/`.

## Script Inventory

| Script | Delegates to | Output |
|--------|--------------|--------|
| `run_sweep.py` | `ntqr_allotment.sweeps` | `output/data/sweep_results.json`, `sweep_aggregated.csv` |
| `run_independence_sweep.py` | `ntqr_allotment.independence_sweep` | `output/data/independence_sweep.csv` |
| `run_bloc_phase.py` | `ntqr_allotment.bloc_confound` | `output/data/bloc_phase.csv`, `bloc_concentration.csv`, `bloc_phase_summary.json` |
| `run_cross_family.py` | `ntqr_allotment.cross_family` | `output/data/cross_family_results.json` |
| `run_cross_family_multiseed.py` | `ntqr_allotment.cross_family` | `output/data/cross_family_multiseed.json` |
| `run_trio_conditioning.py` | `ntqr_allotment.trio_conditioning` | `output/data/trio_conditioning.csv` |
| `run_personas.py` | `ntqr_allotment.personas` | `output/data/persona_*.json/csv` |
| `run_postdoc_panel.py` | `ntqr_allotment.postdoc_panel` | `output/data/postdoc_panel_results.json`, `postdoc_panel_alignment.json` |
| `run_power_analysis.py` | `ntqr_allotment.power_study` | `output/data/power_table.csv` |
| `run_analytical_predictions.py` | `ntqr_allotment.contrast_analysis` | `output/data/analytical_predictions.json` |
| `z_generate_manuscript_variables.py` | `ntqr_allotment.manuscript_variables` | `output/manuscript_variables.json`, `output/manuscript/*` |
| `make_figures.py` | `ntqr_allotment.figures` | `output/figures/*.png` |
| `make_extended_figures.py` | `ntqr_allotment.figures`, `power_study`, `ensemble`, `fairness` | `output/figures/*.png` |
| `make_cover.py` | `ntqr_allotment.cover` | `output/figures/ntqr_cover.png`, `output/data/cover_manifest.json` |
| `make_web_explorer.py` | `ntqr_allotment.web_explorer` | `output/web/*` |
| `make_stego_pdf.py` | `ntqr_allotment.stego` | `output/pdf/*` |
| `verify_stego.py` | `ntqr_allotment.stego` | stdout verification report |
| `bench_alarm.py` | `ntqr_allotment.ntqr_eval` | stdout benchmark timings |

Verify: any script standalone, e.g. `uv run python scripts/run_sweep.py`.
- Repo-wide policy: `../AGENTS.md` (ongoing) and `/Volumes/external_drive/Git/template/projects/ongoing/AGENTS.md`; do not duplicate it here.
