#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

from ntqr_allotment.stego import verify_stego_artifacts


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    verification = verify_stego_artifacts(root)
    print(json.dumps(verification.to_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
