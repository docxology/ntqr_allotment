# AGENTS.md — power_parts/

Layout: statistical-power subpackage — `analytic.py`, `diagnosis.py`, `effect_size.py`, `normal.py`, `permutation.py` + `README.md`. Consumed by power analysis scripts and `output/data/power_analysis.csv`.
Gotcha: numeric results here feed manuscript tokens via `manuscript_variables.py` — changes alter published numbers; keep tests (`test_*.py` under `tests/`) green before regenerating.
Verify: run the project test suite (`uv run pytest`).
- Repo-wide policy: `../AGENTS.md` (ongoing) and `/Volumes/external_drive/Git/template/projects/ongoing/AGENTS.md`; do not duplicate it here.
