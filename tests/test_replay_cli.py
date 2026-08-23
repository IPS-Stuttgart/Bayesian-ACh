import json
from pathlib import Path

from bayesian_ach.cli_ext import main
from bayesian_ach.replay_cli import main as replay_main


def test_replay_cli_writes_complete_evidence(tmp_path: Path) -> None:
    output = tmp_path / "replay"
    assert replay_main(
        [
            "--output",
            str(output),
            "--sequences",
            "16",
            "--sequence-length",
            "28",
            "--replay-samples",
            "32",
            "--seed",
            "11",
        ]
    ) == 0

    with (output / "summary.json").open(encoding="utf-8") as handle:
        summary = json.load(handle)
    assert summary["all_generators_recovered"] is True
    assert summary["maximum_replay_sampling_mutation"] == 0.0
    for name in (
        "replay_generators.csv",
        "replay_sequences.csv",
        "replay_fits.csv",
        "replay_trials.csv",
    ):
        assert (output / name).is_file()


def test_main_dispatches_replay_benchmark(tmp_path: Path) -> None:
    output = tmp_path / "dispatch"
    assert main(
        [
            "replay-benchmark",
            "--output",
            str(output),
            "--sequences",
            "16",
            "--sequence-length",
            "28",
            "--replay-samples",
            "32",
            "--seed",
            "11",
        ]
    ) == 0
    assert (output / "summary.json").is_file()
