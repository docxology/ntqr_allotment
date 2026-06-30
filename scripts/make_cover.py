#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

from ntqr_allotment.cover import render_cover


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    paths = render_cover(root)
    print(paths.cover.resolve())
    print(paths.manifest.resolve())


if __name__ == "__main__":
    main()
