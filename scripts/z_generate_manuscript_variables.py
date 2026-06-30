#!/usr/bin/env python3
"""Thin orchestrator: generate manuscript {{TOKEN}} values and inject them.

Business logic lives in ``src/ntqr_allotment/manuscript_variables.py``. This script
only handles I/O. The render pipeline (``infrastructure.rendering._manuscript_source``)
auto-runs this before rendering, then renders from ``output/manuscript/`` when present.

It (1) computes every token via ``generate_variables`` (recomputed from the real
artifacts under ``output/data/``), (2) checks that every ``{{TOKEN}}`` used in
``manuscript/*.md`` is produced (no orphans), (3) writes
``output/manuscript_variables.json``, and (4) unless ``--check`` is passed, writes
token-substituted copies of every manuscript file (and ``config.yaml``) under
``output/manuscript/`` for the renderer to consume.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# The render pipeline invokes this script with the *template* interpreter, which
# lacks the project-only deps (sympy / ntqr / allotment live in the project's
# .venv). If the import fails, re-exec once under the project's .venv python so
# token hydration runs against real, freshly-computed values. Comparing resolved
# executables is unreliable here: both venvs symlink to the same shared uv cpython,
# so we key off an actual import failure instead.
_VENV_PY = PROJECT_ROOT / ".venv" / "bin" / "python"
try:
    from ntqr_allotment.manuscript_variables import generate_variables  # noqa: E402
except ModuleNotFoundError:
    if os.environ.get("_NTQR_HYDRATE_REEXEC") != "1" and _VENV_PY.is_file():
        os.environ["_NTQR_HYDRATE_REEXEC"] = "1"
        os.execv(str(_VENV_PY), [str(_VENV_PY), str(Path(__file__).resolve()), *sys.argv[1:]])
    raise

TOKEN_RE = re.compile(r"\{\{([A-Z0-9_]+)\}\}")


def used_tokens(manuscript_dir: Path) -> set[str]:
    """Every {{TOKEN}} referenced across the manuscript markdown."""
    tokens: set[str] = set()
    for md in sorted(manuscript_dir.glob("*.md")):
        tokens.update(TOKEN_RE.findall(md.read_text(encoding="utf-8")))
    return tokens


def inject(text: str, variables: dict[str, str]) -> str:
    return TOKEN_RE.sub(lambda m: variables.get(m.group(1), m.group(0)), text)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Only validate token coverage; do not write the injected manuscript.",
    )
    args = parser.parse_args()

    manuscript_dir = PROJECT_ROOT / "manuscript"
    output_dir = PROJECT_ROOT / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    variables = generate_variables(PROJECT_ROOT)

    used = used_tokens(manuscript_dir)
    missing = sorted(used - set(variables))
    if missing:
        print(f"ORPHAN TOKENS (used in prose, not generated): {missing}", file=sys.stderr)
        return 1

    vars_path = output_dir / "manuscript_variables.json"
    vars_path.write_text(json.dumps(variables, indent=2, sort_keys=True), encoding="utf-8")
    print(str(vars_path))
    print(f"tokens generated: {len(variables)} | tokens used in prose: {len(used)} | orphans: 0")

    if args.check:
        return 0

    injected_dir = output_dir / "manuscript"
    injected_dir.mkdir(parents=True, exist_ok=True)
    for md in sorted(manuscript_dir.glob("*.md")):
        out = injected_dir / md.name
        out.write_text(inject(md.read_text(encoding="utf-8"), variables), encoding="utf-8")
        print(str(out))
    config_src = manuscript_dir / "config.yaml"
    if config_src.is_file():
        shutil.copy2(config_src, injected_dir / "config.yaml")
        print(str(injected_dir / "config.yaml"))
    for bib in sorted(manuscript_dir.glob("*.bib")):
        shutil.copy2(bib, injected_dir / bib.name)
        print(str(injected_dir / bib.name))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
