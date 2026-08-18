import csv
import json
from pathlib import Path

from bayesian_ach.cli import main


def test_dissociate_cli_writes_evidence(tmp_path: Path) -> None:
    output = tmp_path / "dissociation"
    assert main(["dissociate", "--output", str(output), "--pairs", "16", "--seed", "3"]) == 0

    with (output / "summary.json").open(encoding="utf-8") as handle:
        summary = json.load(handle)
    assert summary["experiment"] == "matched_confidence"
    assert summary["max_paired_mismatch_for_matched_quantities"]["surprise"] < 1e-12

    with (output / "matched_confidence.csv").open(encoding="utf-8") as handle:
        assert len(list(csv.DictReader(handle))) == 32


def test_benchmark_cli_recovers_candidates(tmp_path: Path) -> None:
    output = tmp_path / "benchmark"
    assert main(
        [
            "benchmark",
            "--output",
            str(output),
            "--trials",
            "1800",
            "--noise-std",
            "0.10",
            "--seed",
            "5",
        ]
    ) == 0

    with (output / "summary.json").open(encoding="utf-8") as handle:
        summary = json.load(handle)
    assert summary["all_generators_recovered"] is True
