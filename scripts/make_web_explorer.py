#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

from ntqr_allotment.web_explorer import write_explorer


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    print(write_explorer(root).resolve())


if __name__ == "__main__":
    main()
