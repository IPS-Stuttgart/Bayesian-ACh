import csv
import json
from pathlib import Path

from bayesian_ach.cli import main


def test_observation_benchmark_cli_writes_exact_evidence(tmp_path: Path) -> None:
    output = tmp_path / "observation"
    assert main(
        [
            "observation-benchmark",
            "--output",
            str(output),
            "--sequences-per-class",
            "3",
            "--pre-length",
            "8",
            "--post-length",
            "12",
            "--seed",
            "11",
        ]
    ) == 0

    with (output / "summary.json").open(encoding="utf-8") as handle:
        summary = json.load(handle)
    assert summary["experiment"] == "partial_observation_causal_attribution"
    assert summary["n_sequences"] == 9
    assert summary["balanced_accuracy"] == 1.0
    assert summary["all_sequences_correct"] is True

    with (output / "observation_sequences.csv").open(encoding="utf-8") as handle:
        assert len(list(csv.DictReader(handle))) == 9
    with (output / "observation_trials.csv").open(encoding="utf-8") as handle:
        assert len(list(csv.DictReader(handle))) == 9 * (1 + 8 + 12)
