# Runtime measurement receipt

Python 3.15.0; CPython; arm64.

Source binding: `42adb6181166fe0c19651e8dbfcd4d1d9aa1fbc463174530151f7caf97c6f4b4`.

All timing values below come from unprofiled samples after the declared warmup. Allocation peaks come from a separate instrumented pass.

| Workload | Median CPU ms | Median wall ms | Wall range ms | Python peak KiB |
|---|---:|---:|---:|---:|
| agreement | 974.723 | 2952.848 | 2814.343–3095.795 | 674.9 |
| recovery | 56.998 | 219.123 | 122.430–333.321 | 186.8 |
| alarm | 59.822 | 239.545 | 88.084–926.122 | 101.1 |
| sortition | 23.815 | 172.396 | 107.977–225.443 | 215.5 |

Ranges are descriptive, not confidence intervals. Parent CPU excludes external solver CPU; Python allocations exclude native heaps and process RSS.

## Workloads and traces

### agreement

12 judges, 96 seeded synthetic items; exact NTQR agreement moments

Input SHA-256: `6de06133057634b753b321720809209577145d8f036261004e5726c61d32eaad`. Tracing backend: `profiling.tracing`; 8332656 instrumented calls.

Top cumulative tracing frames (instrumented time; nested frames overlap):

- `src/ntqr_allotment/profile_workloads.py:24 <lambda>`: 1 calls, 7.104127 seconds cumulative.
- `src/ntqr_allotment/ensemble.py:110 panel_agreement_moments`: 1 calls, 7.104064 seconds cumulative.
- `src/ntqr_allotment/ensemble.py:98 _pair_frequency_moment`: 135 calls, 7.085619 seconds cumulative.
- `src/ntqr_allotment/ensemble.py:82 _classifier_label_frequency`: 273 calls, 4.826591 seconds cumulative.
- `~:0 <built-in method builtins.sum>`: 411 calls, 3.894473 seconds cumulative.

### recovery

One real trio solve with synthetic oracle scoring

Input SHA-256: `2d2a6137447062c456bc4107d095fc3b121e1ea8cc8afccc80d331d6cb38e259`. Tracing backend: `profiling.tracing`; 342771 instrumented calls.

Top cumulative tracing frames (instrumented time; nested frames overlap):

- `src/ntqr_allotment/profile_workloads.py:30 <lambda>`: 1 calls, 0.886843 seconds cumulative.
- `src/ntqr_allotment/pipeline.py:158 run_trial_ensemble`: 1 calls, 0.886809 seconds cumulative.
- `src/ntqr_allotment/ntqr_eval.py:170 error_independent_solutions`: 1 calls, 0.868356 seconds cumulative.
- `site-packages/ntqr/r2/evaluators.py:376 __init__`: 1 calls, 0.868202 seconds cumulative.
- `site-packages/sympy/simplify/simplify.py:435 simplify`: 14 calls, 0.834980 seconds cumulative.

### alarm

Real finite-key alarm: three judges, six items, safety factors (2,2)

Input SHA-256: `33b4de4658472efff83ee1471e4a6b23c8889416cbfd3f8aa9d8f3f5dd8c2b61`. Tracing backend: `profiling.tracing`; 554895 instrumented calls.

Top cumulative tracing frames (instrumented time; nested frames overlap):

- `src/ntqr_allotment/profile_workloads.py:45 <lambda>`: 1 calls, 0.958432 seconds cumulative.
- `src/ntqr_allotment/ntqr_eval.py:239 alarm_misaligned`: 1 calls, 0.958423 seconds cumulative.
- `site-packages/ntqr/alarms.py:184 are_misaligned`: 1 calls, 0.958182 seconds cumulative.
- `site-packages/ntqr/alarms.py:162 misalignment_trace`: 1 calls, 0.958178 seconds cumulative.
- `site-packages/ntqr/alarms.py:179 <genexpr>`: 8 calls, 0.958170 seconds cumulative.

### sortition

Native CBC maximin lottery over a seeded population; parent CPU excludes solver children

Input SHA-256: `a53a5faceccf24ce0438ec0fc1881261fb221b436f2132f50b43812bfd411ed6`. Tracing backend: `profiling.tracing`; 9366 instrumented calls.

Top cumulative tracing frames (instrumented time; nested frames overlap):

- `src/ntqr_allotment/profile_workloads.py:53 <lambda>`: 1 calls, 0.212433 seconds cumulative.
- `src/ntqr_allotment/sortition.py:113 representative_sortition`: 1 calls, 0.212407 seconds cumulative.
- `site-packages/allotment/selection_core/audit.py:23 run_draw`: 1 calls, 0.211774 seconds cumulative.
- `site-packages/allotment/selection_core/maximin.py:23 draw_selection`: 1 calls, 0.211752 seconds cumulative.
- `site-packages/pulp/pulp.py:2383 solve`: 5 calls, 0.208279 seconds cumulative.

## Availability and limits

Native sampling: not requested. Cold import profile: available.

The JSON receipt retains individual samples, bounded caller–callee edges, allocation sites, dependency/build context, input hashes and source bindings. These software measurements are separate from scientific model observations.
