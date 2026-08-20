from __future__ import annotations

import itertools

import numpy as np

from bayesian_ach.smoothing import FiniteStateSmoother


def _exhaustive_posterior(
    initial: np.ndarray,
    transitions: np.ndarray,
    likelihoods: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray]:
    n_times, n_states = likelihoods.shape
    paths = list(itertools.product(range(n_states), repeat=n_times))
    weights = np.empty(len(paths), dtype=float)
    for path_index, path in enumerate(paths):
        weight = initial[path[0]] * likelihoods[0, path[0]]
        for time_index in range(n_times - 1):
            weight *= transitions[time_index, path[time_index], path[time_index + 1]]
            weight *= likelihoods[time_index + 1, path[time_index + 1]]
        weights[path_index] = weight

    evidence = float(np.sum(weights))
    weights /= evidence
    state_posterior = np.zeros((n_times, n_states), dtype=float)
    pair_posterior = np.zeros((n_times - 1, n_states, n_states), dtype=float)
    for path, weight in zip(paths, weights, strict=True):
        for time_index, state in enumerate(path):
            state_posterior[time_index, state] += weight
        for time_index in range(n_times - 1):
            pair_posterior[time_index, path[time_index], path[time_index + 1]] += weight
    return evidence, state_posterior, pair_posterior


def test_smoother_matches_exhaustive_trajectory_enumeration() -> None:
    initial = np.array([0.6, 0.4])
    transitions = np.array(
        [
            [[0.8, 0.2], [0.3, 0.7]],
            [[0.55, 0.45], [0.15, 0.85]],
            [[0.9, 0.1], [0.25, 0.75]],
        ]
    )
    likelihoods = np.array(
        [[0.7, 0.2], [0.1, 0.8], [0.6, 0.4], [0.3, 0.9]]
    )
    evidence, state_posterior, pair_posterior = _exhaustive_posterior(
        initial,
        transitions,
        likelihoods,
    )

    result = FiniteStateSmoother(initial, transitions, likelihoods).run()

    np.testing.assert_allclose(np.exp(result.log_evidence), evidence, atol=1e-14)
    np.testing.assert_allclose(
        result.smoothed_probabilities,
        state_posterior,
        atol=1e-14,
    )
    np.testing.assert_allclose(
        result.smoothed_pair_probabilities,
        pair_posterior,
        atol=1e-14,
    )
    assert result.state_smoothing_kl[-1] < 1e-14
    assert result.transition_smoothing_kl[-1] < 1e-14


def test_future_observation_changes_smoothing_not_prefix_filtering() -> None:
    initial = np.array([0.5, 0.5])
    transition = np.array([[0.95, 0.05], [0.05, 0.95]])
    common_prefix = np.array([[0.8, 0.2], [1.0, 1.0], [1.0, 1.0]])
    favor_first = np.vstack((common_prefix, [0.99, 0.01]))
    favor_second = np.vstack((common_prefix, [0.01, 0.99]))

    first = FiniteStateSmoother(initial, transition, favor_first).run()
    second = FiniteStateSmoother(initial, transition, favor_second).run()

    np.testing.assert_allclose(
        first.filtered_probabilities[:3],
        second.filtered_probabilities[:3],
        atol=1e-14,
    )
    assert (
        np.max(
            np.abs(
                first.smoothed_probabilities[:3]
                - second.smoothed_probabilities[:3]
            )
        )
        > 0.2
    )


def test_missing_observation_has_unit_online_evidence() -> None:
    initial = np.array([0.5, 0.5])
    transition = np.array([[0.85, 0.15], [0.2, 0.8]])
    likelihoods = np.array([[0.8, 0.2], [1.0, 1.0], [0.2, 0.8]])

    result = FiniteStateSmoother(initial, transition, likelihoods).run()

    assert result.online_predictive_probabilities[1] == 1.0
    assert result.online_surprise[1] == 0.0


def test_ffbs_samples_posterior_without_mutating_it() -> None:
    initial = np.array([0.5, 0.5])
    transition = np.array([[0.85, 0.15], [0.2, 0.8]])
    likelihoods = np.array(
        [[0.8, 0.2], [1.0, 1.0], [0.2, 0.8], [0.7, 0.3]]
    )
    smoother = FiniteStateSmoother(initial, transition, likelihoods)
    result = smoother.run()
    before = result.smoothed_probabilities.copy()

    samples = smoother.sample_smoothed_trajectories(
        result,
        n_samples=20_000,
        seed=3,
    )

    np.testing.assert_array_equal(result.smoothed_probabilities, before)
    empirical = np.vstack(
        [
            np.bincount(samples[:, time_index], minlength=2) / samples.shape[0]
            for time_index in range(samples.shape[1])
        ]
    )
    np.testing.assert_allclose(
        empirical,
        result.smoothed_probabilities,
        atol=0.015,
    )


def test_expected_count_revision_is_zero_without_future_information() -> None:
    initial = np.array([0.5, 0.5])
    transition = np.array([[0.7, 0.3], [0.4, 0.6]])
    likelihoods = np.ones((2, 2), dtype=float)

    result = FiniteStateSmoother(initial, transition, likelihoods).run()

    np.testing.assert_allclose(
        result.expected_transition_counts(smoothed=True),
        result.expected_transition_counts(smoothed=False),
        atol=1e-14,
    )
