# Runtime measurement receipt

Python 3.15.0; CPython; arm64.

Source binding: `1aabcf6b292fdd13f13f1c84af715f7bdd4ffb12f6d4dc51c4c3bd631c202a26`.

All timing values below come from unprofiled samples after the declared warmup. Allocation peaks come from a separate instrumented pass.

| Workload | Median CPU ms | Median wall ms | Wall range ms | Python peak KiB |
|---|---:|---:|---:|---:|
| agreement | 280.585 | 719.330 | 524.596–1002.504 | 672.2 |
| recovery | 53.292 | 147.408 | 49.310–180.309 | 185.5 |
| alarm | 49.374 | 129.307 | 60.553–219.650 | 101.1 |
| sortition | 30.123 | 184.578 | 169.644–193.707 | 215.5 |

Ranges are descriptive, not confidence intervals. Parent CPU excludes external solver CPU; Python allocations exclude native heaps and process RSS.

## Same-context descriptive comparison

Ratios are current/baseline; values below one indicate lower observed cost. No significance or causal speedup claim is implied.

| Workload | Comparable | CPU ratio | Wall ratio | Reasons |
|---|---|---:|---:|---|
| agreement | True | 0.288 | 0.244 | |
| recovery | True | 0.935 | 0.673 | |
| alarm | True | 0.825 | 0.540 | |
| sortition | True | 1.265 | 1.071 | |

## Workloads and traces

### agreement

12 judges, 96 seeded synthetic items; exact NTQR agreement moments

Input SHA-256: `6de06133057634b753b321720809209577145d8f036261004e5726c61d32eaad`. Tracing backend: `profiling.tracing`; 2554376 instrumented calls.

Top cumulative tracing frames (instrumented time; nested frames overlap):

- `src/ntqr_allotment/profile_workloads.py:24 <lambda>`: 1 calls, 2.164456 seconds cumulative.
- `src/ntqr_allotment/ensemble.py:118 panel_agreement_moments`: 1 calls, 2.164340 seconds cumulative.
- `src/ntqr_allotment/agreement.py:37 centered_pair`: 135 calls, 2.159894 seconds cumulative.
- `src/ntqr_allotment/agreement.py:28 pair`: 135 calls, 1.882781 seconds cumulative.
- `~:0 <built-in method builtins.sum>`: 159 calls, 1.077427 seconds cumulative.

### recovery

One real trio solve with synthetic oracle scoring

Input SHA-256: `2d2a6137447062c456bc4107d095fc3b121e1ea8cc8afccc80d331d6cb38e259`. Tracing backend: `profiling.tracing`; 342771 instrumented calls.

Top cumulative tracing frames (instrumented time; nested frames overlap):

- `src/ntqr_allotment/profile_workloads.py:30 <lambda>`: 1 calls, 0.377022 seconds cumulative.
- `src/ntqr_allotment/pipeline.py:158 run_trial_ensemble`: 1 calls, 0.376988 seconds cumulative.
- `src/ntqr_allotment/ntqr_eval.py:170 error_independent_solutions`: 1 calls, 0.373789 seconds cumulative.
- `.venv315/lib/python3.15/site-packages/ntqr/r2/evaluators.py:376 __init__`: 1 calls, 0.373615 seconds cumulative.
- `.venv315/lib/python3.15/site-packages/sympy/simplify/simplify.py:435 simplify`: 14 calls, 0.353053 seconds cumulative.

### alarm

Real finite-key alarm: three judges, six items, safety factors (2,2)

Input SHA-256: `33b4de4658472efff83ee1471e4a6b23c8889416cbfd3f8aa9d8f3f5dd8c2b61`. Tracing backend: `profiling.tracing`; 554895 instrumented calls.

Top cumulative tracing frames (instrumented time; nested frames overlap):

- `src/ntqr_allotment/profile_workloads.py:45 <lambda>`: 1 calls, 0.319549 seconds cumulative.
- `src/ntqr_allotment/ntqr_eval.py:239 alarm_misaligned`: 1 calls, 0.319532 seconds cumulative.
- `.venv315/lib/python3.15/site-packages/ntqr/alarms.py:184 are_misaligned`: 1 calls, 0.319167 seconds cumulative.
- `.venv315/lib/python3.15/site-packages/ntqr/alarms.py:162 misalignment_trace`: 1 calls, 0.319161 seconds cumulative.
- `.venv315/lib/python3.15/site-packages/ntqr/alarms.py:179 <genexpr>`: 8 calls, 0.319149 seconds cumulative.

### sortition

Native CBC maximin lottery over a seeded population; parent CPU excludes solver children

Input SHA-256: `a53a5faceccf24ce0438ec0fc1881261fb221b436f2132f50b43812bfd411ed6`. Tracing backend: `profiling.tracing`; 9366 instrumented calls.

Top cumulative tracing frames (instrumented time; nested frames overlap):

- `src/ntqr_allotment/profile_workloads.py:53 <lambda>`: 1 calls, 0.184412 seconds cumulative.
- `src/ntqr_allotment/sortition.py:113 representative_sortition`: 1 calls, 0.184383 seconds cumulative.
- `.venv315/lib/python3.15/site-packages/allotment/selection_core/audit.py:23 run_draw`: 1 calls, 0.183500 seconds cumulative.
- `.venv315/lib/python3.15/site-packages/allotment/selection_core/maximin.py:23 draw_selection`: 1 calls, 0.183472 seconds cumulative.
- `.venv315/lib/python3.15/site-packages/pulp/pulp.py:2383 solve`: 5 calls, 0.177862 seconds cumulative.

## Availability and limits

Native sampling: unavailable. Cold import profile: available.

The JSON receipt retains individual samples, bounded caller–callee edges, allocation sites, dependency/build context, input hashes and source bindings. These software measurements are separate from scientific model observations.
