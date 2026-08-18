"""Small, dependency-free serialization helpers."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


def write_rows_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Write a homogeneous sequence of mappings as CSV."""

    if not rows:
        raise ValueError("cannot write an empty row sequence")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    expected = set(fieldnames)
    for index, row in enumerate(rows):
        if set(row.keys()) != expected:
            raise ValueError(f"row {index} has keys inconsistent with the first row")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write stable, human-readable JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
