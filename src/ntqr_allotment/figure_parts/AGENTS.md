# AGENTS.md — figure_parts/

Layout: figure-builder subpackage consumed by `figures.py` — `_common.py` (shared styling/helpers), `empirical.py`, `fairness.py`, `heatmaps.py`, `independence.py`, `mechanism.py`, `phase.py`, `power.py`, `ranking.py`, `schematic.py` + `README.md`.
Gotcha: plotting logic belongs here (or `figures.py`), not in `scripts/make_*.py`; scripts only call these builders.
Verify: `uv run pytest tests/test_figures.py`.
- Repo-wide policy: `../AGENTS.md` (ongoing) and `/Volumes/external_drive/Git/template/projects/ongoing/AGENTS.md`; do not duplicate it here.
