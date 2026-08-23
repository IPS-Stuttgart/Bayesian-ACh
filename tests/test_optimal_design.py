from __future__ import annotations

import numpy as np

from bayesian_ach.design_benchmark import DesignBenchmarkConfig, run_design_benchmark
from bayesian_ach.optimal_design import (
    DesignConfig,
    condition_signals,
    design_diagnostics,
    fractional_factorial_conditions,
    greedy_maximin_schedule,
    matched_confidence_conditions,
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


def test_quick_design_benchmark_improves_identifiability() -> None:
    result = run_design_benchmark(
        config=DesignBenchmarkConfig(
            trial_budgets=(8, 16),
            target_trial_budget=16,
            subject_grid=(1, 2),
            information_samples=48,
            recovery_replicates=48,
            group_replicates=32,
            seed=17,
        )
    )
    acceptance = result.summary["acceptance"]

    assert acceptance["factorial_library_full_rank"]
    assert acceptance["optimal_improves_worst_pair_separation"]
    assert acceptance["optimal_improves_recovery"]
    assert acceptance["optimal_reduces_entropy"]
    information = result.summary["target_metrics"]["robust_optimal"][
        "expected_information_gain_nats"
    ]
    assert np.isfinite(information)
