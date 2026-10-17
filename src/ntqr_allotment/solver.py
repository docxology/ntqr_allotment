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


def configure_cbc() -> str:
    """Select NTQR_CBC_PATH, CBC on PATH, or the compatible bundled executable."""
    override = os.environ.get("NTQR_CBC_PATH")
    external = override or shutil.which("cbc")
    if external:
        path = shutil.which(external)
        if path is None:
            raise RuntimeError("NTQR_CBC_PATH must name an executable CBC solver")
    else:
        if platform.system() == "Darwin" and platform.machine() == "arm64":
            raise RuntimeError(
                "native CBC required on Apple Silicon: brew install cbc, or set NTQR_CBC_PATH"
            )
        path = pulp.PULP_CBC_CMD.pulp_cbc_path
    if not Path(path).is_file():
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
    return {
        "ntqr_version": ntqr.__version__,
        "pulp_version": pulp.__version__,
        "cbc_version": match.group(1) if match else "unknown",
    }
