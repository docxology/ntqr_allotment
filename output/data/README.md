# output/data/ — ntqr_allotment

Regenerated analysis artifacts — the single source of truth for every manuscript number.

- Sweep results: `sweep_results.json`, `sweep_aggregated.csv`, `independence_sweep.csv`, `cross_family_results.json`, `cross_family_multiseed.json`, `bloc_phase.{csv,json}`, `bloc_concentration.csv`, `trio_conditioning.json`
- Statistics/power: `analytical_predictions.json`, `power_analysis.csv`, `alarm_timings.csv`
- Live companion: `postdoc_panel_results.json`, `postdoc_panel_alignment.json`, `persona_results.json`, `postdoc_votes/` (per-vote Ollama cache)
- Provenance: `cover_manifest.json`, `stego_manifest.json`

Producers and their dependency order are declared in `manuscript/config.yaml` (`analysis.scripts`).
