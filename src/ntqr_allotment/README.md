# src/ntqr_allotment/ — package core

Importable package holding all business logic (thin-orchestrator pattern; `scripts/` are I/O-only orchestrators).

- Deterministic synthetic spine: `experts.py`, `corpus.py`, `sortition.py`, `ntqr_eval.py`, `pipeline.py`, `sweeps.py`, `dependence.py`, `independence_sweep.py`, `ensemble.py`, `ternary.py`, `fairness.py`
- Confound mechanism: `bloc_confound.py`
- Statistics/theory: `statistics_analysis.py`, `power_analysis.py`, `power_study.py`, `theory.py`, `contrast_analysis.py`
- Live empirical companion: `personas.py`, `postdoc_panel.py`, `cross_family.py`
- Manuscript/outputs: `manuscript_variables.py`, `figures.py` + `figure_parts/`, `cover.py`, `stego.py`, `config.py`

Tests: `tests/` (one module per topic). Entry points: `scripts/` (see `scripts/README.md`).
