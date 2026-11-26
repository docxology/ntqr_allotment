#!/usr/bin/env python3
"""Thin orchestrator: generate manuscript {{TOKEN}} values and inject them.

Business logic lives in ``src/ntqr_allotment/manuscript_variables.py``. This script
only handles I/O. The render pipeline (``infrastructure.rendering._manuscript_source``)
auto-runs this before rendering, then renders from ``output/manuscript/`` when present.

It (1) computes every token via ``generate_variables`` (recomputed from the real
artifacts under ``output/data/``), (2) checks that every ``{{TOKEN}}`` used in
``docs/manuscript/*.md`` is produced (no orphans), and (3) preflights complete
byte payloads for ``output/manuscript_variables.json``, the injected manuscript,
and a deterministic SHA-256 render manifest before publishing any of them.
``--check`` performs the same preflight and byte-compares every owned payload,
including the manifest, without writing.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
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
    from ntqr_allotment.artifacts import atomic_write_bytes  # noqa: E402
    from ntqr_allotment.manuscript_variables import generate_variables, numeric_evidence  # noqa: E402
except ModuleNotFoundError:
    if os.environ.get("_NTQR_HYDRATE_REEXEC") != "1" and _VENV_PY.is_file():
        os.environ["_NTQR_HYDRATE_REEXEC"] = "1"
        os.execv(str(_VENV_PY), [str(_VENV_PY), str(Path(__file__).resolve()), *sys.argv[1:]])
    raise

TOKEN_RE = re.compile(r"\{\{([A-Z0-9_]+)\}\}")


def source_markdown(manuscript_dir: Path) -> list[Path]:
    """Select render sources; directory guides are not manuscript sections."""
    if not manuscript_dir.is_dir():
        raise FileNotFoundError(f"manuscript source directory missing: {manuscript_dir}")
    files = sorted(manuscript_dir.glob("[0-9]*.md"))
    preamble = manuscript_dir / "preamble.md"
    return files + ([preamble] if preamble.is_file() else [])


def used_tokens(manuscript_dir: Path) -> set[str]:
    """Every {{TOKEN}} referenced across the manuscript markdown."""
    tokens: set[str] = set()
    for md in source_markdown(manuscript_dir):
        tokens.update(TOKEN_RE.findall(md.read_text(encoding="utf-8")))
    return tokens


def inject(
    text: str, variables: dict[str, str], source: str | Path = "<text>"
) -> str:
    """Substitute once, then reject tokens left by missing or token-valued values."""
    rendered = TOKEN_RE.sub(lambda m: variables.get(m.group(1), m.group(0)), text)
    remaining = sorted({match.group(0) for match in TOKEN_RE.finditer(rendered)})
    if remaining:
        raise ValueError(
            f"unresolved tokens after one-pass substitution in {source}: "
            + ", ".join(remaining)
        )
    return rendered


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify all generated payloads byte-for-byte without writing.",
    )
    args = parser.parse_args(argv)

    manuscript_dir = PROJECT_ROOT / "docs" / "manuscript"
    output_dir = PROJECT_ROOT / "output"

    variables = generate_variables(PROJECT_ROOT)

    used = used_tokens(manuscript_dir)
    missing = sorted(used - set(variables))
    if missing:
        print(f"ORPHAN TOKENS (used in prose, not generated): {missing}", file=sys.stderr)
        return 1

    vars_path = output_dir / "manuscript_variables.json"
    injected_dir = output_dir / "manuscript"
    payloads: dict[Path, bytes] = {
        vars_path: json.dumps(variables, indent=2, sort_keys=True).encode("utf-8")
    }
    payloads[output_dir / "data" / "manuscript_numeric_evidence.json"] = (
        json.dumps(numeric_evidence(variables), indent=2, sort_keys=True, allow_nan=False)
        + "\n"
    ).encode("utf-8")
    for md in source_markdown(manuscript_dir):
        rendered = inject(md.read_text(encoding="utf-8"), variables, source=md)
        rendered = rendered.replace("](../../output/figures/", "](../figures/")
        payloads[injected_dir / md.name] = rendered.encode("utf-8")
    config_src = manuscript_dir / "config.yaml"
    if config_src.is_file():
        payloads[injected_dir / "config.yaml"] = (
            b"# Generated manuscript ordering\n" + config_src.read_bytes()
        )
    registry = manuscript_dir / "figure_registry.json"
    if registry.is_file():
        payloads[output_dir / "figures" / registry.name] = registry.read_bytes()
    for bib in sorted(manuscript_dir.glob("*.bib")):
        payloads[injected_dir / bib.name] = bib.read_bytes()

    # Per-file atomic writes are not a multi-file transaction. The manifest makes
    # interrupted or partial publication detectable by --check, not preventable.
    manifest = {
        "schema_version": 1,
        "files": {
            str(path.relative_to(PROJECT_ROOT)): hashlib.sha256(source).hexdigest()
            for path, source in sorted(payloads.items())
        },
    }
    manifest_path = output_dir / "data" / "render_manifest.json"
    payloads[manifest_path] = (
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")

    print(str(vars_path))
    print(f"tokens generated: {len(variables)} | tokens used in prose: {len(used)} | orphans: 0")

    if args.check:
        stale = [
            str(path.relative_to(PROJECT_ROOT))
            for path, source in payloads.items()
            if not path.is_file() or path.read_bytes() != source
        ]
        if stale:
            print("Stale generated artifacts: " + ", ".join(stale), file=sys.stderr)
            return 1
        print(f"OK — {len(payloads)} generated artifacts match their sources.")
        return 0

    # All source reads and token validation finished before the first write.
    for path, source in payloads.items():
        atomic_write_bytes(path, source)
        if path != vars_path:
            print(str(path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
