import numpy as np
import pytest

from bayesian_ach.design import (
    DESIGN_CANDIDATE_NAMES,
    TransitionDesignGridConfig,
    coupled_novelty_design,
    design_diagnostics,
    generate_transition_design_grid,
    optimize_maximin_design,
    pairwise_residual_matrix,
    uniform_factorial_design,
)
from bayesian_ach.design_benchmark import DesignBenchmarkConfig, run_design_benchmark


def test_design_grid_is_finite_and_dissociates_all_candidates() -> None:
    rows, raw, standardized = generate_transition_design_grid()
    assert len(rows) == 240
    assert raw.shape == standardized.shape == (240, len(DESIGN_CANDIDATE_NAMES))
    assert np.all(np.isfinite(standardized))
    np.testing.assert_allclose(np.mean(standardized, axis=0), 0.0, atol=1e-12)
    np.testing.assert_allclose(np.std(standardized, axis=0), 1.0, atol=1e-12)


def test_maximin_design_improves_worst_pair_over_coupled_novelty() -> None:
    rows, _, standardized = generate_transition_design_grid()
    budget = 36
    novelty = coupled_novelty_design(rows, budget)
    optimized = optimize_maximin_design(standardized, budget, exchange_passes=2)
    novelty_diagnostics = design_diagnostics(standardized, novelty)

    assert int(np.sum(optimized.counts)) == budget
    assert optimized.diagnostics.minimum_pairwise_residual_variance > (
        10.0 * novelty_diagnostics.minimum_pairwise_residual_variance
    )
    assert optimized.diagnostics.maximum_absolute_correlation < (
        novelty_diagnostics.maximum_absolute_correlation
    )


def test_optimizer_is_deterministic_and_pairwise_matrix_is_directional() -> None:
    _, _, standardized = generate_transition_design_grid()
    first = optimize_maximin_design(standardized, 24, exchange_passes=1)
    second = optimize_maximin_design(standardized, 24, exchange_passes=1)
    np.testing.assert_array_equal(first.counts, second.counts)
    geometry = pairwise_residual_matrix(standardized, first.counts)
    np.testing.assert_allclose(np.diag(geometry), 0.0)
    assert np.min(geometry[~np.eye(geometry.shape[0], dtype=bool)]) > 0.0


def test_equal_budget_recovery_beats_coupled_novelty() -> None:
    result = run_design_benchmark(
        DesignBenchmarkConfig(
            budget=36,
            replicates_per_generator=40,
            seed=13,
        )
    )
    minimum = result.summary["minimum_recovery_rate"]
    mean = result.summary["mean_recovery_rate"]
    assert minimum["maximin_optimized"] > minimum["coupled_novelty"]
    assert mean["maximin_optimized"] > mean["coupled_novelty"]
    assert result.summary["optimized_over_novelty_residual_ratio"] > 10.0


def test_invalid_budget_and_allocation_are_rejected() -> None:
    _, _, standardized = generate_transition_design_grid(
        TransitionDesignGridConfig()
    )
    with pytest.raises(ValueError):
        optimize_maximin_design(standardized, 4)
    with pytest.raises(ValueError):
        design_diagnostics(standardized, np.ones(4, dtype=np.int64))
    with pytest.raises(ValueError):
        uniform_factorial_design(0, 10)
