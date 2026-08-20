from bayesian_ach.attribution import (
    MECHANISM_NAMES,
    ObservationAttributionConfig,
    run_observation_attribution,
)


def test_default_small_benchmark_recovers_all_three_mechanisms() -> None:
    result = run_observation_attribution(
        ObservationAttributionConfig(
            n_sequences_per_class=8,
            pre_length=12,
            post_length=18,
            seed=17,
        )
    )

    assert len(result.sequences) == 24
    assert result.summary["balanced_accuracy"] == 1.0
    assert result.summary["all_sequences_correct"] is True
    assert {sequence.true_mechanism for sequence in result.sequences} == set(MECHANISM_NAMES)
    assert min(sequence.evidence_margin for sequence in result.sequences) > 1.0


def test_benchmark_is_deterministic() -> None:
    config = ObservationAttributionConfig(
        n_sequences_per_class=2,
        pre_length=8,
        post_length=10,
        seed=31,
    )
    first = run_observation_attribution(config)
    second = run_observation_attribution(config)
    assert first.sequences == second.sequences
    assert first.summary == second.summary


def test_identifiability_warning_for_observationally_null_fault() -> None:
    result = run_observation_attribution(
        ObservationAttributionConfig(
            n_sequences_per_class=1,
            pre_length=6,
            post_length=8,
            fault_similarity=1.0,
            seed=5,
        )
    )
    warnings = result.summary["identifiability"]["warnings"]
    assert any("visual fault" in warning for warning in warnings)
