"""Cross-process reproducibility of the hash-order-sensitive sortition draw.

``representative_sortition`` goes through the external ``allotment`` lottery, whose
tie-breaking iterates a set and is therefore ``PYTHONHASHSEED``-sensitive. The
project claims byte-identical artifacts, so :func:`ensure_deterministic_hashing`
must make the draw reproducible across independent processes. No mocks: this runs
the real draw in two fresh subprocesses and compares the selected panels.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]

# A real script file (not ``python -c``) so the guard's os.execv re-exec preserves
# argv. Calls the guard FIRST, then forms a representative panel and prints its ids.
_DRAW_SCRIPT = """
from ntqr_allotment.determinism import ensure_deterministic_hashing
ensure_deterministic_hashing()
from ntqr_allotment.experts import generate_population
from ntqr_allotment.sortition import STRATEGIES
pop = generate_population(
    96, seed=3, mean_expertise=0.62, expertise_heterogeneity=0.08, bias_std=0.2
)
panel = STRATEGIES["representative_sortition"](pop, 6, seed=16)
print("PANEL:" + ",".join(panel.expert_ids))
"""


def _draw(script_path: Path, env: dict[str, str]) -> str:
    result = subprocess.run(
        [sys.executable, str(script_path)],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(_REPO),
    )
    assert result.returncode == 0, result.stderr
    lines = [ln for ln in result.stdout.splitlines() if ln.startswith("PANEL:")]
    assert lines, f"no panel printed; stdout={result.stdout!r} stderr={result.stderr!r}"
    return lines[-1]


def test_representative_sortition_is_deterministic_across_processes(tmp_path: Path) -> None:
    script = tmp_path / "draw.py"
    script.write_text(_DRAW_SCRIPT, encoding="utf-8")
    # Start from an environment with PYTHONHASHSEED UNSET, so each subprocess relies
    # on the guard (which pins it and re-execs). Without the guard these two draws
    # would select different panels.
    env = {k: v for k, v in os.environ.items() if k != "PYTHONHASHSEED"}
    first = _draw(script, env)
    second = _draw(script, env)
    assert first == second, (
        "representative_sortition is non-deterministic across processes even with "
        f"the determinism guard: {first!r} != {second!r}"
    )
