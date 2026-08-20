"""Plot sequence-wise regime-evidence margins."""

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

    regimes = ["known_context_switch", "structural_reset"]
    values = [
        [
            float(row["context_minus_changepoint"])
            for row in rows
            if row["true_regime"] == regime
        ]
        for regime in regimes
    ]

    plt.figure(figsize=(7, 4))
    plt.boxplot(values, labels=regimes)
    plt.axhline(0.0, linewidth=1)
    plt.ylabel("Context minus change-point log evidence")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
