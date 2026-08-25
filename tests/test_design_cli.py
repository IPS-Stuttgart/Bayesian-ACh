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
            "--target-log-score-gap",
            "3.0",
            "--seed",
            "5",
        ]
    ) == 0
    with (output / "summary.json").open(encoding="utf-8") as handle:
        summary = json.load(handle)
    assert summary["experiment"] == "prospective_maximin_trial_design"
    assert summary["designs"][0]["trial_count"] == 24
    assert summary["config"]["target_log_score_gap"] == 3.0
    assert "target_log_bf" not in summary["config"]
    assert "expected_profiled_log_score_gap_per_trial" in summary["designs"][0]
    assert (output / "design_allocation.csv").is_file()
    assert (output / "design_pairwise_geometry.csv").is_file()
    assert (output / "design_optimization_trace.csv").is_file()


def test_design_cli_retains_deprecated_target_alias(tmp_path: Path) -> None:
    output = tmp_path / "legacy-design"
    assert main(
        [
            "--output",
            str(output),
            "--budget",
            "12",
            "--replicates",
            "1",
            "--target-log-bf",
            "2.5",
        ]
    ) == 0
    with (output / "summary.json").open(encoding="utf-8") as handle:
        summary = json.load(handle)
    assert summary["config"]["target_log_score_gap"] == 2.5
    assert summary["designs"][0]["trials_for_expected_log_score_gap_target"] > 0
