# docs/ — ntqr_allotment

NTQR allotment — ground-truth-free evaluation of noisy classifiers (v0.1.0 public release).

## Repository layout

- src/ tests/ scripts/ — package core
- docs/manuscript/ — publication package
- audits/ output/ — evaluation evidence

## Verify (real commands from repo metadata)

- `uv run pytest tests/`


- [Dependencies and NTQR 0.9 migration](dependencies.md): pins, native CBC, API compatibility and verification.

- [Valid study designs and resumable artifacts](reliability.md): input validation, failure provenance, paired support, native CBC and cache identities.


## Profiling and research integrity

- [Runtime profiling](runtime_profiling.md): Python 3.15, workload contracts,
  source-bound measurements, comparison rules and sampling availability.
- [Research integrity](research_integrity.md): inference boundaries, publication
  artifacts, bibliography checks and figure regeneration.
- [Claim registry](research_map.json): structured evidence paths, reviewed primary
  references and explicit limits.
