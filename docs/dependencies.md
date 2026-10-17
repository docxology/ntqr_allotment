# Dependencies and NTQR 0.9 migration

The project pins the [NTQR 0.9 wheel](https://pypi.org/project/ntqr/0.9/)
under `vendor/`. SHA-256:
`83f3cbeeb54eb6a86fb1df8a7c86187f8b14511d69de94d3527ebe49d0a931d8`.
`uv.lock` records its hash and the full dependency graph, including new upstream
SciPy and Numba requirements. `.python-version` defaults to Python 3.12.

The wheel avoids the historical upstream source-distribution packaging failure.
The allotment Git source is pinned by full revision in both pyproject and lock.
PuLP is constrained below 4 because that allotment revision constructs the
legacy PULP_CBC_CMD API. A native CBC executable on PATH is preferred;
`NTQR_CBC_PATH` can select one explicitly. On Apple Silicon install with
`brew install cbc`. Other platforms can use PuLP's bundled solver.
`ntqr_allotment.solver.configure_cbc` sets the legacy solver executable path
before calling the real allotment engine; no engine functions are replaced.
The tested macOS solver is Homebrew CBC 2.10.13. The native cbcbox 2.935 and
2.933 wheels could not run on this host (invalid signatures in bundled
Fortran libraries), so the project does not depend on them. Future sweep
metadata records NTQR, PuLP and CBC versions. Different CBC builds may choose
different optimal panels; historical artifacts retain their original provenance.

The trio evaluator, binary/ternary axioms, and legacy single-classifier alarm
are unchanged between the 0.8 and 0.9 wheels. The adapter now validates nonempty,
aligned binary votes, labeled-item alignment, and positive integer safety
factors before invoking them. Single and ensemble trials share identical
seeded panel preparation. Empty or invalid observations produce clear errors.

NTQR 0.9 changes possible/consistent-set generators to sparse HashablePoint
rows and changes ResponseVariables mappings to accept actual counts/ql. This
project does not call those changed interfaces. The existing opt-in alarm
continues using the legacy enumeration algorithm and retains its corpus cap;
the migration does not claim an LP alarm speedup.

Source/config paths now consistently use `docs/manuscript/`, including token
hydration, live scripts, cover provenance, explorer and stego metadata. The
removed historical session diary is no longer a required test input.

```bash
uv sync --locked --extra dev
uv run --no-sync pytest
uv run --no-sync python scripts/z_generate_manuscript_variables.py --check
```

The lock was generated and verified with uv 0.13.1.

CI checks offline tests on Python 3.12 and 3.13. Live Ollama tests remain opt-in.
Published PDFs and historical model/sweep observations are not relabeled as
fresh NTQR 0.9 measurements. Token hydration continues to consume the recorded
data; numerical compatibility is checked using real solver fixtures.
