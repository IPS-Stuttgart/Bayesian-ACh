from bayesian_ach.measurement_benchmark import (
    MEASUREMENT_CANDIDATE_NAMES,
    MeasurementBenchmarkConfig,
    run_measurement_benchmark,
)


def test_measurement_benchmark_recovers_all_generators() -> None:
    result = run_measurement_benchmark(
        MeasurementBenchmarkConfig(
            n_subjects=3,
            sessions_per_subject=4,
            train_sessions_per_subject=3,
            calibration_length=112,
            task_length=96,
            seed=7,
        )
    )
    assert result.summary["all_generators_recovered"] is True
    assert result.summary["recovery_count"] == len(MEASUREMENT_CANDIDATE_NAMES)
    assert result.summary["calibration_map_timescales_match_truth"] is True
    assert result.summary["minimum_evidence_margin"] > 0.0
    assert result.summary["median_nuisance_coefficient_mae"] < 0.08
    assert result.summary["median_subject_signal_correlation"] > 0.95
