import csv
import json
from pathlib import Path

from bayesian_ach.cli import main


def test_regime_cli_writes_recovery_evidence(tmp_path: Path) -> None:
    output = tmp_path / "regime"
    assert main(
        [
            "regime-benchmark",
            "--output",
            str(output),
            "--sequences-per-class",
            "4",
            "--pre-length",
            "16",
            "--post-length",
            "24",
            "--seed",
            "5",
        ]
    ) == 0

    with (output / "summary.json").open(encoding="utf-8") as handle:
        summary = json.load(handle)
    assert summary["experiment"] == "known_context_switch_vs_structural_reset"
    assert summary["balanced_accuracy"] >= 0.75

    with (output / "regime_sequences.csv").open(encoding="utf-8") as handle:
        assert len(list(csv.DictReader(handle))) == 8
    with (output / "regime_trials.csv").open(encoding="utf-8") as handle:
        assert len(list(csv.DictReader(handle))) == 8 * 40
