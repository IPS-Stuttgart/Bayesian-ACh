import json
from pathlib import Path

from bayesian_ach.cli import main


def test_measurement_cli(tmp_path: Path) -> None:
    output = tmp_path / "measurement"
    assert (
        main(
            [
                "measurement-benchmark",
                "--output",
                str(output),
                "--subjects",
                "3",
                "--sessions-per-subject",
                "3",
                "--train-sessions-per-subject",
                "2",
                "--calibration-length",
                "40",
                "--task-length",
                "80",
                "--seed",
                "5",
            ]
        )
        == 0
    )

    with (output / "summary.json").open(encoding="utf-8") as handle:
        summary = json.load(handle)
    assert summary["all_generators_recovered"] is True
    assert (output / "measurement_generators.csv").is_file()
    assert (output / "measurement_fits.csv").is_file()
    assert (output / "measurement_kernel_posterior.csv").is_file()
    assert (output / "measurement_samples.csv").is_file()
