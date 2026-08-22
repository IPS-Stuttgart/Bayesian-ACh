from bayesian_ach.replay_benchmark import (
    ReplayBenchmarkConfig,
    run_replay_benchmark,
)


def test_replay_benchmark_recovers_all_four_generators() -> None:
    result = run_replay_benchmark(
        ReplayBenchmarkConfig(
            n_sequences=48,
            sequence_length=40,
            replay_samples=64,
            seed=9,
        )
    )

    assert result.summary["all_generators_recovered"] is True
    assert result.summary["recovery_count"] == 4
    assert result.summary["minimum_evidence_margin"] > 0.0
    assert result.summary["maximum_replay_sampling_mutation"] == 0.0
    assert result.summary["maximum_absolute_candidate_correlation"] < 0.8
