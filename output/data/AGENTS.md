# AGENTS.md — output/data/

Layout: CSV/JSON analysis artifacts (`alarm_timings.csv`, `bloc_*.csv|json`, `cross_family*.json`, `persona_*.json`, `power_analysis.csv`, `independence_sweep.csv`, `postdoc_panel_*.json`, `analytical_predictions.json`, manifests) + `postdoc_votes/` + `README.md`.
Invariant: all files are script-generated; `manuscript_variables.json` (parent dir) is derived from these. Do not hand-edit — regenerate via `scripts/run_*.py`. Vote files are keyed by model/jitter tag; treat as immutable evidence.
Verify: re-run the matching `scripts/run_*.py` and diff.
- Repo-wide policy: `../AGENTS.md` (ongoing); the shared policy for template projects is maintained in the template repository, not restated here.
