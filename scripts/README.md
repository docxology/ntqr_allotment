# scripts/ — ntqr_allotment

Thin orchestrators: argparse, path bootstrap, logging, and a single delegated
call into `src/ntqr_allotment/` entrypoints. All computation is importable and
tested there. Run with `uv run python scripts/<name>.py`.

## Inventory

| Script | Purpose | Delegates to | Command |
|--------|---------|--------------|---------|
| `run_sweep.py` | Baseline strategy x panel x regime sweep | `ntqr_allotment.sweeps` (`run_sweep_parallel`, `aggregate`) | `uv run python scripts/run_sweep.py` |
| `run_independence_sweep.py` | Independence-assumption sweep | `ntqr_allotment.independence_sweep` (`run_independence_sweep`, `aggregate_independence`) | `uv run python scripts/run_independence_sweep.py` |
| `run_bloc_phase.py` | Bloc-confound phase sweep (confirm-then-extend) | `ntqr_allotment.bloc_confound` (`run_bloc_phase`, `run_concentration_sweep`, `build_bloc_phase_summary`) | `uv run python scripts/run_bloc_phase.py` |
| `run_cross_family.py` | Single-run cross-family decorrelation | `ntqr_allotment.cross_family` (`build_pairwise_error_correlation_matrix`, `contrast_same_vs_cross_family`) | `uv run python scripts/run_cross_family.py` |
| `run_cross_family_multiseed.py` | Multi-seed cross-family sign-stability aggregate | `ntqr_allotment.cross_family` (`aggregate_contrasts`) | `uv run python scripts/run_cross_family_multiseed.py` |
| `run_trio_conditioning.py` | Trio-conditioning diagnostics | `ntqr_allotment.trio_conditioning` (`panel_trio_diagnostics`) | `uv run python scripts/run_trio_conditioning.py` |
| `run_personas.py` | Persona-layer trio study | `ntqr_allotment.personas` (`personas_from_population`, `run_persona_trio`) | `uv run python scripts/run_personas.py` |
| `run_postdoc_panel.py` | Live gemma3:4b postdoc panel (`--require-live`, slow) | `ntqr_allotment.postdoc_panel` (`run_postdoc_panel_study`, `build_postdoc_alignment`) | `uv run python scripts/run_postdoc_panel.py` |
| `run_power_analysis.py` | Power table from sweep results | `ntqr_allotment.power_study` (`analyze`, `write_power_table`) | `uv run python scripts/run_power_analysis.py` |
| `run_analytical_predictions.py` | Analytical predictions from the aggregated sweep | `ntqr_allotment.contrast_analysis` (`write_analytical_predictions`) | `uv run python scripts/run_analytical_predictions.py` |
| `z_generate_manuscript_variables.py` | Token compute + inject + orphan gate (after producers, before figures) | `ntqr_allotment.manuscript_variables` (`generate_variables`) | `uv run python scripts/z_generate_manuscript_variables.py` |
| `make_figures.py` | Manuscript figures | `ntqr_allotment.figures` (`plot_*`) | `uv run python scripts/make_figures.py` |
| `make_extended_figures.py` | Companion/extended figures | `ntqr_allotment.figures`, `ntqr_allotment.power_study`, `ntqr_allotment.ensemble`, `ntqr_allotment.fairness` | `uv run python scripts/make_extended_figures.py` |
| `make_cover.py` | Cover art + manifest | `ntqr_allotment.cover` (`render_cover`) | `uv run python scripts/make_cover.py` |
| `make_web_explorer.py` | Local web explorer (non-publishing QA) | `ntqr_allotment.web_explorer` (`write_explorer`) | `uv run python scripts/make_web_explorer.py` |
| `make_stego_pdf.py` | Steganographic provenance artifacts | `ntqr_allotment.stego` (`render_stego_artifacts`) | `uv run python scripts/make_stego_pdf.py` |
| `verify_stego.py` | Verify stego artifacts | `ntqr_allotment.stego` (`verify_stego_artifacts`) | `uv run python scripts/verify_stego.py` |
| `bench_alarm.py` | Alarm cost benchmark | `ntqr_allotment.ntqr_eval` (`alarm_misaligned`) | `uv run python scripts/bench_alarm.py` |

Dependency-correct order is declared in `docs/manuscript/config.yaml` (`analysis.scripts`).
