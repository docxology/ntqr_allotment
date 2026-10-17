"""Atomic, strict JSON for resumable votes and generated analysis results."""

from __future__ import annotations

import json
import os
from pathlib import Path
import stat
import tempfile


def write_json(path: str | Path, payload) -> Path:
    """Serialize before touching disk, then atomically replace a complete UTF-8 file."""
    path = Path(path)
    source = (
        json.dumps(
            payload, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False
        )
        + "\n"
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = stat.S_IMODE(path.stat().st_mode) if path.exists() else 0o644
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as handle:
            temporary = Path(handle.name)
            os.chmod(temporary, mode)
            handle.write(source)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return path
