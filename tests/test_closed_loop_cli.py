import csv
import json
from pathlib import Path

from bayesian_ach.cli import main


def test_closed_loop_cli_writes_complete_evidence(tmp_path: Path) -> None:
    output = tmp_path / "closed-loop"
    assert (
        main(
            [
                "closed-loop-benchmark",
                "--output",
                str(output),
                "--subjects",
                "5",
                "--sessions-per-subject",
                "4",
                "--train-sessions-per-subject",
                "2",
                "--opportunities-per-session",
                "72",
                "--seed",
                "7",
            ]
        )
        == 0
    )

    expected = {
        "closed_loop_generators.csv",
        "closed_loop_fits.csv",
        "closed_loop_pairs.csv",
        "closed_loop_opportunities.csv",
        "summary.json",
    }
    assert {path.name for path in output.iterdir()} == expected

    with (output / "summary.json").open(encoding="utf-8") as handle:
        summary = json.load(handle)
    assert summary["all_generators_recovered"] is True
    assert summary["recovery_count"] == 5
    assert summary["maximum_active_sham_command_time_difference"] == 0.0
    assert summary["accepted_false_trigger_count"] > 0

    with (output / "closed_loop_generators.csv").open(encoding="utf-8") as handle:
        generators = list(csv.DictReader(handle))
    assert len(generators) == 5
    assert all(row["correct"] == "True" for row in generators)
