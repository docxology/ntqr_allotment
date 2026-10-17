# Valid study designs and resumable artifacts

Panel strategies now reject empty populations, duplicate expert IDs, noninteger
or impossible panel sizes, and unknown ideology selections. A request for more
experts than exist cannot silently return a smaller panel. Trial configurations
validate finite probabilities, nonnegative dispersions, integer counts and
nonnegative seeds before sampling or solving.

Sweep grids reject duplicate axis values and duplicate seeds; repeated seeds
cannot inflate the apparent replication count. Invalid configuration raises an
error rather than producing a degenerate research row. A valid observation with
no admissible recovery is represented separately by
`NoPhysicalSolutionError` or a zero-usable-trio result. Future `SweepRow` records
include `failure_reason` (`timeout` or `no_physical_solution`) and sweep metadata
summarizes the failure counts. The added field defaults to `None`, so historical
rows remain readable.

Aggregation includes the same eligible trials for both EI and majority-voting
errors, and its sample count refers to that common support. Partial method
failures cannot silently compare means from different trial sets.

`trial_timeout_s` must be finite and positive when supplied. Interval-timer
support is required, and an existing process interval timer cannot be replaced.
The native CBC adapter honors explicit executable overrides, rejects empty or
invalid overrides, and keeps its original bundled fallback separate from a
previous override. Solver version detection must succeed before a new run is
reported with solver provenance.

## Vote caches

`judge_corpus` uses versioned, content-bound keys including persona fields,
actual prompts, item text/index, and judge identity. The deterministic judge
binds its seed and labeled corpus. Ollama binds its endpoint, model tag and
decode settings; provide `cache_namespace` with an immutable model digest when
a mutable tag must be distinguished across model updates.

Old persona-ID/item-index keys remain in the cache as historical records but
are not reused without provenance. A custom cacheable judge supplies
`cache_identity()` or an explicit `cache_namespace`. Cached and fresh votes
must be binary labels; malformed caches and invalid votes fail explicitly.

The generic persona path checkpoints after each response. Generic and postdoc
vote caches, and the main sweep JSON, use atomic strict UTF-8 JSON replacement.
A serialization failure leaves the prior complete artifact in place. File
modes are preserved. These are single-writer checkpoints, not a protocol for
merging simultaneous writers to the same cache.

```bash
uv run pytest
uv run python scripts/z_generate_manuscript_variables.py --check
```

Known PuLP 3 deprecation messages are filtered in pytest because the project
explicitly constrains PuLP below 4; other warnings remain visible.

An alarm safety factor of one demands strictly better than 100% accuracy.
The adapter now returns the logically certain alarm immediately for that
specification, after validating observations and the corpus budget. Other
safety specifications continue through the real NTQR enumeration.
