from __future__ import annotations

import numpy as np

from bayesian_ach.optimal_design import (
    DesignConfig,
    balanced_schedule,
    condition_signals,
    design_diagnostics,
    evaluate_schedule,
    fractional_factorial_conditions,
    greedy_maximin_schedule,
    matched_confidence_conditions,
    min_pairwise_symmetric_kl,
    novelty_only_conditions,
    schedule_counts,
)


def test_matched_confidence_dissociates_gain_from_surprise() -> None:
    common_low, common_high, rare_low, rare_high = matched_confidence_conditions()
    common_low_signals = condition_signals(common_low)
    common_high_signals = condition_signals(common_high)
    rare_low_signals = condition_signals(rare_low)
    rare_high_signals = condition_signals(rare_high)

    assert common_low_signals["predictive_surprise"] == common_high_signals[
        "predictive_surprise"
    ]
    assert rare_low_signals["predictive_surprise"] == rare_high_signals[
        "predictive_surprise"
    ]
    assert common_low_signals["bayesian_gain"] > common_high_signals["bayesian_gain"]
    assert rare_low_signals["rational_update"] > rare_high_signals["rational_update"]


def test_fractional_factorial_restores_candidate_rank() -> None:
    novelty = design_diagnostics(novelty_only_conditions())
    factorial = design_diagnostics(fractional_factorial_conditions())

    assert int(novelty["rank"]) < 6
    assert int(factorial["rank"]) == 6
    assert float(factorial["maximum_absolute_correlation"]) < 0.95


def test_greedy_schedule_is_deterministic_and_respects_repeat_limit() -> None:
    conditions = fractional_factorial_conditions()
    first = greedy_maximin_schedule(
        conditions,
        12,
        DesignConfig(),
        max_repeats_per_condition=2,
    )
    second = greedy_maximin_schedule(
        conditions,
        12,
        DesignConfig(),
        max_repeats_per_condition=2,
    )

    assert first == second
    assert len(first) == 12
    assert max(schedule_counts(first).values()) <= 2


def test_prior_predictive_factorial_design_improves_identifiability() -> None:
    config = DesignConfig()
    reference = fractional_factorial_conditions()
    novelty = balanced_schedule(novelty_only_conditions(), 16)
    factorial = balanced_schedule(reference, 16)

    novelty_separation = min_pairwise_symmetric_kl(
        novelty,
        config,
        reference_conditions=reference,
    )
    factorial_separation = min_pairwise_symmetric_kl(
        factorial,
        config,
        reference_conditions=reference,
    )
    assert factorial_separation > novelty_separation
    assert factorial_separation > 0.0

    evaluation = evaluate_schedule(
        "fractional_factorial",
        factorial,
        config,
        reference_conditions=reference,
        information_samples=48,
        recovery_replicates=48,
        seed=17,
    )
    assert np.isfinite(evaluation.expected_information_gain_nats)
    assert evaluation.expected_information_gain_nats > 0.0
    assert 0.0 <= evaluation.recovery_accuracy <= 1.0
    assert np.isfinite(evaluation.mean_posterior_entropy_nats)
