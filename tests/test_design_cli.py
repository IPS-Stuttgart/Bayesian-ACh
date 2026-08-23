import json
from pathlib import Path

from bayesian_ach.design_cli import main


def test_design_cli_writes_complete_evidence(tmp_path: Path) -> None:
    output = tmp_path / "design"
    assert main(
        [
            "--output",
            str(output),
            "--budget",
            "24",
            "--replicates",
            "12",
            "--seed",
            "5",
        ]
    ) == 0
    with (output / "summary.json").open(encoding="utf-8") as handle:
        summary = json.load(handle)
    assert summary["experiment"] == "prospective_maximin_trial_design"
    assert summary["designs"][0]["trial_count"] == 24
    assert (output / "design_allocation.csv").is_file()
    assert (output / "design_pairwise_geometry.csv").is_file()
    assert (output / "design_optimization_trace.csv").is_file()
