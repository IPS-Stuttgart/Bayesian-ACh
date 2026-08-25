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
    profiled_gaussian_log_score_gap,
    uniform_factorial_design,
)
from bayesian_ach.design_benchmark import DesignBenchmarkConfig, run_design_benchmark
from bayesian_ach.design_recovery import _fit_and_score


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


def test_profiled_gaussian_gap_and_default_trial_targets() -> None:
    rows, _, standardized = generate_transition_design_grid()
    budget = 60
    optimized = optimize_maximin_design(standardized, budget)
    diagnostics = {
        "maximin_optimized": optimized.diagnostics,
        "uniform_factorial": design_diagnostics(
            standardized,
            uniform_factorial_design(len(rows), budget, seed=7),
        ),
        "coupled_novelty": design_diagnostics(
            standardized,
            coupled_novelty_design(rows, budget),
        ),
    }

    assert profiled_gaussian_log_score_gap(
        0.25,
        effect_size=1.0,
        noise_std=1.0,
    ) == pytest.approx(0.5 * np.log1p(0.25))
    assert {
        name: value.trials_for_expected_log_score_gap_target
        for name, value in diagnostics.items()
    } == {
        "maximin_optimized": 45,
        "uniform_factorial": 93,
        "coupled_novelty": 1113,
    }
    assert optimized.diagnostics.expected_log_bf_per_trial == (
        optimized.diagnostics.expected_profiled_log_score_gap_per_trial
    )
    legacy = design_diagnostics(
        standardized,
        optimized.counts,
        target_log_bf=5.0,
    )
    assert legacy == optimized.diagnostics
    assert "expected_log_bf_per_trial" not in legacy.as_dict()


def test_affine_reparameterization_preserves_geometry_and_profiled_scores() -> None:
    _, _, standardized = generate_transition_design_grid()
    shifts = np.linspace(-4.0, 3.0, standardized.shape[1])
    scales = np.array([0.5, -1.5, 2.0, -0.75, 3.0, -2.5])
    transformed = standardized * scales + shifts
    transformed = (transformed - transformed.mean(axis=0)) / transformed.std(axis=0)
    baseline = optimize_maximin_design(standardized, 24, exchange_passes=1)
    affine = optimize_maximin_design(transformed, 24, exchange_passes=1)

    np.testing.assert_array_equal(affine.counts, baseline.counts)
    np.testing.assert_allclose(
        pairwise_residual_matrix(transformed, affine.counts),
        pairwise_residual_matrix(standardized, baseline.counts),
        atol=1e-12,
    )

    trial_signals = standardized[np.repeat(np.arange(len(baseline.counts)), baseline.counts)]
    response = 0.7 * trial_signals[:, 2] + np.linspace(-0.2, 0.2, len(trial_signals))
    train = np.arange(0, 16, dtype=np.int64)
    test = np.arange(16, len(trial_signals), dtype=np.int64)
    winner, margin = _fit_and_score(trial_signals, response, train, test)
    affine_winner, affine_margin = _fit_and_score(
        trial_signals * scales + shifts,
        response,
        train,
        test,
    )
    assert affine_winner == winner
    assert affine_margin == pytest.approx(margin, abs=1e-10)

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
