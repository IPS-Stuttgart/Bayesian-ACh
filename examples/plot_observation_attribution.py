"""Plot evidence margins from ``bayesian-ach observation-benchmark``."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt

from bayesian_ach.attribution import MECHANISM_NAMES


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", type=Path)
    args = parser.parse_args()

    with args.csv_path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    margins = [
        [
            float(row["evidence_margin"])
            for row in rows
            if row["true_mechanism"] == mechanism
        ]
        for mechanism in MECHANISM_NAMES
    ]
    labels = [mechanism.replace("_", "\n") for mechanism in MECHANISM_NAMES]

    plt.figure(figsize=(9, 4.5))
    plt.boxplot(margins, tick_labels=labels)
    plt.ylabel("Winning minus runner-up post-change log evidence")
    plt.title("Partial-observation causal-attribution recovery")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
