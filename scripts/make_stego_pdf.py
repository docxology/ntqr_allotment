#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

from ntqr_allotment.stego import render_stego_artifacts


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    paths = render_stego_artifacts(root)
    print(paths.cover.resolve())
    print(paths.pdf.resolve())
    print(paths.manifest.resolve())


if __name__ == "__main__":
    main()
