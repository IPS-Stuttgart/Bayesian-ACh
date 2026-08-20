"""Plot held-out ACh measurement-model scores by generating signal."""

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
        selected.sort(
            key=lambda row: float(row["marginal_test_log_likelihood"]),
            reverse=True,
        )
        candidates = [row["candidate"] for row in selected]
        scores = [float(row["marginal_test_log_likelihood"]) for row in selected]
        best = max(scores)
        relative = [score - best for score in scores]

        plt.figure(figsize=(9, 4))
        plt.bar(candidates, relative)
        plt.title(f"Generating signal: {generator}")
        plt.ylabel("Held-out log likelihood relative to winner")
        plt.xticks(rotation=35, ha="right")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
