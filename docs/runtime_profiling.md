# Runtime profiling and Python 3.15

The project defaults to CPython 3.15 and retains a Python 3.12 package floor.
CI runs the locked environment on Python 3.12, 3.13 and 3.15. The lockfiles
include wheels usable on the new runtime; the profiling receipts record the
actual installed distributions and interpreter build.

## Reproduce

```bash
uv sync --locked --extra dev
uv run --no-sync python scripts/profile_runtime.py --output benchmarks/runtime/current.json --sample
uv run --no-sync python scripts/check_research.py
uv run --no-sync pytest
```

Choose a subset with repeated `--workload NAME` options. Use `--repeats` for a
bounded timing sample count, and `--compare PATH` for a descriptive comparison
with a compatible saved report. The CLI fixes an unspecified process hash seed to zero before workloads run.
Importing the package does not start profiling.
The package API accepts a `Workload` containing a callable, description and
input hash; `measure` returns JSON-compatible data without publishing files.

## Measurement contract

| Evidence | How measured | Interpretation |
|---|---|---|
| Unprofiled timings | Warm up once, then independent `perf_counter_ns` and `process_time_ns` samples | Wall and parent-process CPU observations; keep all samples |
| Call profile | Separate `profiling.tracing.Profile` pass on Python 3.15; compatible `cProfile` fallback on older runtimes | Exact call counts, self/cumulative costs and bounded caller–callee edges; instrumented times are not speedup evidence |
| Python allocations | Separate `tracemalloc` pass with bounded allocation-site summaries | Python allocation peak, not process RSS, NumPy native heaps or external solver memory |
| Cold imports | Fresh interpreter with `-X importtime` | Startup/import context, separate from warmed workload timings |
| Native sampling | Bounded `profiling.sampling` child workload | Optional stack sampling; status and failure remain explicit |
| Source and inputs | SHA-256 bindings for source files, lockfile and workload input descriptions | Evidence applies to that snapshot, not an unrelated checkout |

The local macOS sampling probe was denied access to process memory. Its receipts
mark sampling unavailable; no privilege elevation or system-security changes
were attempted. Tracing, unprofiled timings and memory measurements still ran.
On a host permitting native sampling, the same command can produce a flame graph.
See the official [Python 3.15 profiling overview](https://docs.python.org/3.15/library/profiling.html),
[tracing reference](https://docs.python.org/3.15/library/profiling.tracing.html),
and [sampling reference](https://docs.python.org/3.15/library/profiling.sampling.html).

## Comparisons and limits

`before.json` preserves the initial observations. A controlled baseline and
current receipt use the same harness, seeded inputs and installed environment,
with the pre-refactor implementation restored only in the isolated baseline.
The JSON bindings record the implementation actually measured. Both absolute
samples and source identities remain available; nothing rewrites the original
observation into a new result.

Comparisons require matching runtime/dependency context, input hashes, warmup
count, repetition count and fixed process hash seed. An incompatible or absent workload gets reasons
and null ratios. New workloads cannot acquire an invented baseline. Ratios are
one-host descriptive measurements, not significance tests or cross-runtime
performance claims. Tracing, memory collection, imports and figure rendering
are excluded from the unprofiled samples.

The generated PNG/SVG runtime summary shows timing medians, ranges and individual
samples, plus allocation peaks from their separate instrumented pass. It is
software performance evidence, separate from the scientific study artifacts.
No new LLM observations are collected by the profiling command.

## Allotment workloads and the refactor

- `agreement`: seeded binary observations for a panel, normalized through the
  real NTQR primitive. `AgreementMoments` prepares exact frequencies once and
  memoizes marginal and pair frequencies within one panel call. Pair-pattern
  scans remain; no asymptotic complexity claim is made.
- `recovery`: one real trio evaluation with synthetic oracle scoring.
- `alarm`: the real finite-answer-key alarm on a deliberately small corpus.
- `sortition`: a bounded maximin lottery using native CBC. Wall time includes
  child-process waiting; parent CPU timing excludes CBC child CPU.

The agreement regression compares the prepared view against the original NTQR
primitive calculations across every judge pair and both labels. The estimator,
probability filter, branch convention and recorded empirical results are not
changed by this normalization optimization.
