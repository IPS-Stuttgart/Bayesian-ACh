"""Plot held-out replay model-recovery scores."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", type=Path)
    args = parser.parse_args()

    with args.csv_path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    generators = sorted({row["generator"] for row in rows})
    for generator in generators:
        selected = [row for row in rows if row["generator"] == generator]
        selected.sort(key=lambda row: int(row["rank"]))
        candidates = [row["candidate"] for row in selected]
        scores = [float(row["test_mean_log_likelihood"]) for row in selected]

        plt.figure(figsize=(8, 4))
        plt.bar(candidates, scores)
        plt.title(f"Replay generator: {generator}")
        plt.ylabel("Held-out mean log likelihood")
        plt.xticks(rotation=35, ha="right")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
