"""Select a real CBC executable for allotment's legacy PuLP API.

The pinned allotment revision constructs PULP_CBC_CMD internally. Configure its
executable path, preserving all upstream optimization logic. A system CBC is
preferred to PuLP 3's bundled Intel-only macOS executable.
"""

import os
from pathlib import Path
import platform
import re
import shutil
import subprocess

import ntqr
import pulp


_BUNDLED_CBC_PATH = pulp.PULP_CBC_CMD.pulp_cbc_path


def configure_cbc() -> str:
    """Select NTQR_CBC_PATH, CBC on PATH, or the compatible bundled executable."""
    override = os.environ.get("NTQR_CBC_PATH")
    if override is not None and not override.strip():
        raise RuntimeError("NTQR_CBC_PATH must name an executable CBC solver")
    external = override if override is not None else shutil.which("cbc")
    if external:
        path = shutil.which(external)
        if path is None:
            raise RuntimeError("NTQR_CBC_PATH must name an executable CBC solver")
    else:
        if platform.system() == "Darwin" and platform.machine() == "arm64":
            raise RuntimeError(
                "native CBC required on Apple Silicon: brew install cbc, or set NTQR_CBC_PATH"
            )
        path = _BUNDLED_CBC_PATH
    if not Path(path).is_file() or not os.access(path, os.X_OK):
        raise RuntimeError("CBC executable missing; install CBC or set NTQR_CBC_PATH")
    pulp.PULP_CBC_CMD.pulp_cbc_path = str(path)
    return str(path)


def runtime_provenance() -> dict[str, str]:
    """Dependency/solver versions without machine-specific absolute paths."""
    result = subprocess.run(
        [configure_cbc(), "-stop"],
        capture_output=True,
        text=True,
        timeout=10,
        check=True,
    )
    match = re.search(r"Version:\s*([^\s]+)", result.stdout)
    if match is None:
        raise RuntimeError("selected CBC executable did not report its solver version")
    return {
        "ntqr_version": ntqr.__version__,
        "pulp_version": pulp.__version__,
        "cbc_version": match.group(1),
    }
