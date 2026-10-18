# Runtime measurement receipt

Python 3.15.0; CPython; arm64.

Source binding: `197e114901277eeffd3704a4b8cc98766869fe090342f03ffab431a3a1a16705`.

All timing values below come from unprofiled samples after the declared warmup. Allocation peaks come from a separate instrumented pass.

| Workload | Median CPU ms | Median wall ms | Wall range ms | Python peak KiB |
|---|---:|---:|---:|---:|
| agreement | 181.722 | 195.970 | 186.022–214.361 | 672.2 |
| recovery | 42.494 | 45.155 | 41.356–60.808 | 186.0 |
| alarm | 37.129 | 44.999 | 33.235–100.289 | 101.1 |
| sortition | 23.885 | 107.868 | 104.377–125.814 | 215.4 |

Ranges are descriptive, not confidence intervals. Parent CPU excludes external solver CPU; Python allocations exclude native heaps and process RSS.

## Same-context descriptive comparison

Ratios are current/baseline; values below one indicate lower observed cost. No significance or causal speedup claim is implied.

| Workload | Comparable | CPU ratio | Wall ratio | Reasons |
|---|---|---:|---:|---|
| agreement | True | 0.251 | 0.189 | |
| recovery | True | 1.262 | 0.781 | |
| alarm | True | 1.147 | 1.000 | |
| sortition | True | 0.953 | 0.798 | |

## Workloads and traces

### agreement

12 judges, 96 seeded synthetic items; exact NTQR agreement moments

Input SHA-256: `6de06133057634b753b321720809209577145d8f036261004e5726c61d32eaad`. Tracing backend: `profiling.tracing`; 2554376 instrumented calls.

Top cumulative tracing frames (instrumented time; nested frames overlap):

- `src/ntqr_allotment/profile_workloads.py:24 <lambda>`: 1 calls, 0.636097 seconds cumulative.
- `src/ntqr_allotment/ensemble.py:118 panel_agreement_moments`: 1 calls, 0.636000 seconds cumulative.
- `src/ntqr_allotment/agreement.py:37 centered_pair`: 135 calls, 0.633140 seconds cumulative.
- `src/ntqr_allotment/agreement.py:28 pair`: 135 calls, 0.513769 seconds cumulative.
- `~:0 <built-in method builtins.sum>`: 159 calls, 0.353073 seconds cumulative.

### recovery

One real trio solve with synthetic oracle scoring

Input SHA-256: `2d2a6137447062c456bc4107d095fc3b121e1ea8cc8afccc80d331d6cb38e259`. Tracing backend: `profiling.tracing`; 342759 instrumented calls.

Top cumulative tracing frames (instrumented time; nested frames overlap):

- `src/ntqr_allotment/profile_workloads.py:30 <lambda>`: 1 calls, 0.141103 seconds cumulative.
- `src/ntqr_allotment/pipeline.py:158 run_trial_ensemble`: 1 calls, 0.141079 seconds cumulative.
- `src/ntqr_allotment/ntqr_eval.py:170 error_independent_solutions`: 1 calls, 0.138710 seconds cumulative.
- `.venv315/lib/python3.15/site-packages/ntqr/r2/evaluators.py:376 __init__`: 1 calls, 0.138586 seconds cumulative.
- `.venv315/lib/python3.15/site-packages/sympy/simplify/simplify.py:435 simplify`: 14 calls, 0.130550 seconds cumulative.

### alarm

Real finite-key alarm: three judges, six items, safety factors (2,2)

Input SHA-256: `33b4de4658472efff83ee1471e4a6b23c8889416cbfd3f8aa9d8f3f5dd8c2b61`. Tracing backend: `profiling.tracing`; 554895 instrumented calls.

Top cumulative tracing frames (instrumented time; nested frames overlap):

- `src/ntqr_allotment/profile_workloads.py:45 <lambda>`: 1 calls, 0.178042 seconds cumulative.
- `src/ntqr_allotment/ntqr_eval.py:239 alarm_misaligned`: 1 calls, 0.178029 seconds cumulative.
- `.venv315/lib/python3.15/site-packages/ntqr/alarms.py:184 are_misaligned`: 1 calls, 0.177729 seconds cumulative.
- `.venv315/lib/python3.15/site-packages/ntqr/alarms.py:162 misalignment_trace`: 1 calls, 0.177724 seconds cumulative.
- `.venv315/lib/python3.15/site-packages/ntqr/alarms.py:179 <genexpr>`: 8 calls, 0.177715 seconds cumulative.

### sortition

Native CBC maximin lottery over a seeded population; parent CPU excludes solver children

Input SHA-256: `a53a5faceccf24ce0438ec0fc1881261fb221b436f2132f50b43812bfd411ed6`. Tracing backend: `profiling.tracing`; 9366 instrumented calls.

Top cumulative tracing frames (instrumented time; nested frames overlap):

- `src/ntqr_allotment/profile_workloads.py:53 <lambda>`: 1 calls, 0.112850 seconds cumulative.
- `src/ntqr_allotment/sortition.py:113 representative_sortition`: 1 calls, 0.112821 seconds cumulative.
- `.venv315/lib/python3.15/site-packages/allotment/selection_core/audit.py:23 run_draw`: 1 calls, 0.112048 seconds cumulative.
- `.venv315/lib/python3.15/site-packages/allotment/selection_core/maximin.py:23 draw_selection`: 1 calls, 0.112022 seconds cumulative.
- `.venv315/lib/python3.15/site-packages/pulp/pulp.py:2383 solve`: 5 calls, 0.108265 seconds cumulative.

## Availability and limits

Native sampling: unavailable. Cold import profile: available.

The JSON receipt retains individual samples, bounded caller–callee edges, allocation sites, dependency/build context, input hashes and source bindings. These software measurements are separate from scientific model observations.
