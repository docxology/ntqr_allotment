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

Statistical APIs reject nonfinite observations and confidence-interval endpoints.
Cohen's d returns zero for equal observed means, but different means without an
estimable pooled spread raise `ValueError` rather than being flattened into a
false no-effect result. Effect-size and permutation calculations normalize inputs
to a common unit scale. Permutation statistics first remove a safe common
midpoint, so a large shared offset cannot erase significance through the tie
tolerance. Tests reject unknown tail selections, and power/verdict logic uses
the standard `p <= alpha` rejection boundary.

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

## Live votes and model identity

Free-text labels require a bare answer or complete answer/option/label expression
before taking precedence over explanatory whole-word cues. An ordinary article
in "a false statement" is not a true vote. Conflicting, negated, or ambiguous
responses are rejected. Empty, missing, non-string, and unparseable Ollama
responses raise instead of becoming cached `b` votes. Model availability
accepts the bare-name/`:latest` equivalence, rejects malformed tag payloads, and
panel preflight requires every distinct requested model. Cached live votes must
be bound to an immutable model digest through `cache_namespace`.

Live configuration rejects fractional and boolean counts/seeds before coercion,
including seed sequences that would collapse into duplicate replicates. Error
correlation reports reject missing true-label strata instead of returning NaN.

Persona and postdoc CLIs require explicit `--offline` for synthetic diagnostics.
Missing live persona models fail rather than silently substituting synthetic
votes. `--output-dir` can select a fresh diagnostic destination; offline defaults
are `output/diagnostics/personas` and `output/diagnostics/postdoc_panel`, and the
canonical live data directory is refused for offline output. Persona `--model`
reaches both requested-model slots; offline judge seeds and labels are distinct.
Neither distinct seeds nor model choices prove conditional error independence.

Analytical-only postdoc output has unavailable alignment and null live-model
provenance, never analytical-versus-itself agreement. Explicit zero count
overrides reach validation. Trio summaries with fewer than two usable records
report null associations; missing first/later rank partitions have null means
and render as unavailable rather than fabricated values.

## Numerical and publication contracts

The trio agreement descriptor includes observed three-way co-vote frequency in
the centered NTQR moment. Nested alarm panels reuse each seed's vote prefixes,
and EIE-degenerate candidates skip unusable majority-solver work. Strategy
ranking intervals pool both within-cell and between-cell variance. The pinned
PuLP adapter synchronizes external CBC permission state with executable selection.

Manuscript substitution rejects token-shaped values left after one-pass
injection. Publication preflights every source and payload before writing, uses
atomic replacement for owned files, and records SHA-256 hashes in
`output/data/render_manifest.json`. Per-file atomicity is not a multi-file
transaction; `--check` uses the manifest and byte comparisons to detect partial
or stale publication.

The local explorer renders dataset titles, sources, and column headers as text,
rejects nonfinite embedded JSON, and reports when a table shows only the first
rows of a larger source. Partial trio-conditioning plots use each strategy's
observed panel sizes, and empirical heatmaps reject duplicate strategy/panel
cells instead of silently taking the last row.
