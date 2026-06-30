from __future__ import annotations

import json
from pathlib import Path

from ntqr_allotment.contrast_analysis import write_analytical_predictions


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    data_dir = root / "output" / "data"
    sweep_path = data_dir / "sweep_results.json"
    metadata = {}
    if sweep_path.is_file():
        payload = json.loads(sweep_path.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and isinstance(payload.get("metadata"), dict):
            metadata = dict(payload["metadata"])
    output_path = write_analytical_predictions(
        data_dir / "sweep_aggregated.csv",
        data_dir / "analytical_predictions.json",
        metadata=metadata,
    )
    print(output_path.resolve())


if __name__ == "__main__":
    main()
