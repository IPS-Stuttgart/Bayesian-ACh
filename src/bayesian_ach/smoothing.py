"""Exact finite-state filtering, smoothing, and posterior replay sampling."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

_FLOAT_TINY = float(np.finfo(float).tiny)


def _probability_vector(values: ArrayLike, *, name: str) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional; got shape {array.shape}")
    if array.size < 2:
        raise ValueError(f"{name} must contain at least two states")
    if not np.all(np.isfinite(array)) or np.any(array < 0.0):
        raise ValueError(f"{name} must contain finite, non-negative values")
    total = float(np.sum(array))
    if total <= 0.0:
        raise ValueError(f"{name} must have positive total mass")
    return np.asarray(array / total, dtype=np.float64)


def _transition_sequence(values: ArrayLike, n_states: int) -> NDArray[np.float64]:
    transitions = np.asarray(values, dtype=float)
    if transitions.ndim == 2:
        transitions = transitions[None, :, :]
    if transitions.ndim != 3 or transitions.shape[1:] != (n_states, n_states):
        raise ValueError(
            "transitions must have shape (time-1, state, state) or (state, state); "
            f"got {transitions.shape}"
        )
    if not np.all(np.isfinite(transitions)) or np.any(transitions < 0.0):
        raise ValueError("transitions must contain finite, non-negative values")
    row_sums = np.sum(transitions, axis=2, keepdims=True)
    if np.any(row_sums <= 0.0):
        raise ValueError("every transition row must have positive mass")
    return np.asarray(transitions / row_sums, dtype=np.float64)


def _likelihood_matrix(values: ArrayLike, n_states: int) -> NDArray[np.float64]:
    likelihoods = np.asarray(values, dtype=float)
    if likelihoods.ndim != 2 or likelihoods.shape[1] != n_states:
        raise ValueError(
            f"likelihoods must have shape (time, {n_states}); got {likelihoods.shape}"
        )
    if likelihoods.shape[0] < 2:
        raise ValueError("likelihoods must contain at least two time points")
    if not np.all(np.isfinite(likelihoods)) or np.any(likelihoods < 0.0):
        raise ValueError("likelihoods must contain finite, non-negative values")
    if np.any(np.sum(likelihoods, axis=1) <= 0.0):
        raise ValueError("every likelihood row must have positive mass")
    return np.asarray(likelihoods, dtype=np.float64)


def _categorical_kl(
    posterior: NDArray[np.float64],
    prior: NDArray[np.float64],
) -> float:
    posterior_flat = np.ravel(posterior)
    prior_flat = np.ravel(prior)
    positive = posterior_flat > 0.0
    if np.any(prior_flat[positive] <= 0.0):
        return float("inf")
    return float(
        np.sum(
            posterior_flat[positive]
            * np.log(posterior_flat[positive] / prior_flat[positive])
        )
    )


def _entropy(probabilities: NDArray[np.float64]) -> float:
    flat = np.ravel(probabilities)
    positive = flat > 0.0
    return float(-np.sum(flat[positive] * np.log(flat[positive])))


@dataclass(frozen=True, slots=True)
class SmoothingResult:
    """Exact forward-backward posterior and filtering-to-smoothing revisions."""

    log_evidence: float
    online_predictive_probabilities: NDArray[np.float64]
    online_surprise: NDArray[np.float64]
    predicted_probabilities: NDArray[np.float64]
    filtered_probabilities: NDArray[np.float64]
    smoothed_probabilities: NDArray[np.float64]
    backward_messages: NDArray[np.float64]
    filtering_pair_probabilities: NDArray[np.float64]
    smoothed_pair_probabilities: NDArray[np.float64]
    state_smoothing_kl: NDArray[np.float64]
    state_revision_l1: NDArray[np.float64]
    transition_smoothing_kl: NDArray[np.float64]
    transition_revision_l1: NDArray[np.float64]
    filtered_entropy: NDArray[np.float64]
    smoothed_entropy: NDArray[np.float64]

    @property
    def n_times(self) -> int:
        return int(self.filtered_probabilities.shape[0])

    @property
    def n_states(self) -> int:
        return int(self.filtered_probabilities.shape[1])

    def expected_transition_counts(self, *, smoothed: bool = True) -> NDArray[np.float64]:
        """Return expected transition counts under the selected posterior."""

        pairs = (
            self.smoothed_pair_probabilities
            if smoothed
            else self.filtering_pair_probabilities
        )
        return np.asarray(np.sum(pairs, axis=0), dtype=np.float64)


class FiniteStateSmoother:
    """Exact fixed-interval inference for a finite, time-varying Markov model."""

    def __init__(
        self,
        initial_probabilities: ArrayLike,
        transitions: ArrayLike,
        likelihoods: ArrayLike,
    ) -> None:
        initial = _probability_vector(initial_probabilities, name="initial_probabilities")
        likelihood_matrix = _likelihood_matrix(likelihoods, initial.size)
        transition_sequence = _transition_sequence(transitions, initial.size)
        if transition_sequence.shape[0] == 1 and likelihood_matrix.shape[0] > 2:
            transition_sequence = np.repeat(
                transition_sequence,
                likelihood_matrix.shape[0] - 1,
                axis=0,
            )
        expected = likelihood_matrix.shape[0] - 1
        if transition_sequence.shape[0] != expected:
            raise ValueError(
                f"transitions must contain {expected} steps; got {transition_sequence.shape[0]}"
            )
        self.initial_probabilities = initial
        self.transitions = transition_sequence
        self.likelihoods = likelihood_matrix

    def run(self) -> SmoothingResult:
        """Run normalized forward filtering and exact backward smoothing."""

        n_times, n_states = self.likelihoods.shape
        predicted = np.empty((n_times, n_states), dtype=float)
        filtered = np.empty((n_times, n_states), dtype=float)
        scales = np.empty(n_times, dtype=float)

        predicted[0] = self.initial_probabilities
        unnormalized = predicted[0] * self.likelihoods[0]
        scales[0] = float(np.sum(unnormalized))
        if scales[0] <= 0.0 or not np.isfinite(scales[0]):
            raise FloatingPointError("the initial observation has zero predictive probability")
        filtered[0] = unnormalized / scales[0]

        for time_index in range(1, n_times):
            predicted[time_index] = filtered[time_index - 1] @ self.transitions[time_index - 1]
            unnormalized = predicted[time_index] * self.likelihoods[time_index]
            scales[time_index] = float(np.sum(unnormalized))
            if scales[time_index] <= 0.0 or not np.isfinite(scales[time_index]):
                raise FloatingPointError(
                    f"observation {time_index} has zero predictive probability"
                )
            filtered[time_index] = unnormalized / scales[time_index]

        backward = np.ones((n_times, n_states), dtype=float)
        for time_index in range(n_times - 2, -1, -1):
            backward[time_index] = self.transitions[time_index] @ (
                self.likelihoods[time_index + 1] * backward[time_index + 1]
            )
            backward[time_index] /= scales[time_index + 1]

        smoothed = filtered * backward
        smoothed /= np.sum(smoothed, axis=1, keepdims=True)

        n_transitions = n_times - 1
        filtering_pairs = np.empty((n_transitions, n_states, n_states), dtype=float)
        smoothed_pairs = np.empty_like(filtering_pairs)
        transition_kl = np.empty(n_transitions, dtype=float)
        transition_l1 = np.empty(n_transitions, dtype=float)

        for time_index in range(n_transitions):
            online_pair = (
                filtered[time_index, :, None]
                * self.transitions[time_index]
                * self.likelihoods[time_index + 1, None, :]
            )
            online_pair /= float(np.sum(online_pair))
            filtering_pairs[time_index] = online_pair

            smooth_pair = online_pair * backward[time_index + 1, None, :]
            smooth_pair /= float(np.sum(smooth_pair))
            smoothed_pairs[time_index] = smooth_pair
            transition_kl[time_index] = _categorical_kl(smooth_pair, online_pair)
            transition_l1[time_index] = float(np.sum(np.abs(smooth_pair - online_pair)))

        state_kl = np.asarray(
            [
                _categorical_kl(smoothed[index], filtered[index])
                for index in range(n_times)
            ],
            dtype=np.float64,
        )
        state_l1 = np.sum(np.abs(smoothed - filtered), axis=1)
        filtered_entropy = np.asarray(
            [_entropy(row) for row in filtered],
            dtype=np.float64,
        )
        smoothed_entropy = np.asarray(
            [_entropy(row) for row in smoothed],
            dtype=np.float64,
        )

        return SmoothingResult(
            log_evidence=float(np.sum(np.log(np.maximum(scales, _FLOAT_TINY)))),
            online_predictive_probabilities=np.asarray(scales, dtype=np.float64),
            online_surprise=np.asarray(-np.log(np.maximum(scales, _FLOAT_TINY)), dtype=np.float64),
            predicted_probabilities=np.asarray(predicted, dtype=np.float64),
            filtered_probabilities=np.asarray(filtered, dtype=np.float64),
            smoothed_probabilities=np.asarray(smoothed, dtype=np.float64),
            backward_messages=np.asarray(backward, dtype=np.float64),
            filtering_pair_probabilities=np.asarray(filtering_pairs, dtype=np.float64),
            smoothed_pair_probabilities=np.asarray(smoothed_pairs, dtype=np.float64),
            state_smoothing_kl=state_kl,
            state_revision_l1=np.asarray(state_l1, dtype=np.float64),
            transition_smoothing_kl=np.asarray(transition_kl, dtype=np.float64),
            transition_revision_l1=np.asarray(transition_l1, dtype=np.float64),
            filtered_entropy=filtered_entropy,
            smoothed_entropy=smoothed_entropy,
        )

    def sample_smoothed_trajectories(
        self,
        result: SmoothingResult,
        *,
        n_samples: int = 1,
        seed: int = 7,
    ) -> NDArray[np.int64]:
        """Sample latent trajectories by forward-filtering backward-sampling.

        Sampling is read-only: it does not alter the model, the evidence, or any
        posterior quantity. Treating these self-generated samples as new data
        would double-count the observations that produced the posterior.
        """

        if n_samples < 1:
            raise ValueError("n_samples must be positive")
        dimensions_match = (
            result.n_times == self.likelihoods.shape[0]
            and result.n_states == self.likelihoods.shape[1]
        )
        if not dimensions_match:
            raise ValueError("result dimensions do not match this smoother")
        rng = np.random.default_rng(seed)
        samples = np.empty((n_samples, result.n_times), dtype=np.int64)
        samples[:, -1] = rng.choice(
            result.n_states,
            size=n_samples,
            p=result.smoothed_probabilities[-1],
        )
        for time_index in range(result.n_times - 2, -1, -1):
            next_states = samples[:, time_index + 1]
            for next_state in range(result.n_states):
                selected = next_states == next_state
                count = int(np.sum(selected))
                if count == 0:
                    continue
                probabilities = (
                    result.filtered_probabilities[time_index]
                    * self.transitions[time_index, :, next_state]
                )
                total = float(np.sum(probabilities))
                if total <= 0.0:
                    raise FloatingPointError("backward-sampling conditional has zero mass")
                probabilities /= total
                samples[selected, time_index] = rng.choice(
                    result.n_states,
                    size=count,
                    p=probabilities,
                )
        return samples

    def replay_content_surprise(self, result: SmoothingResult) -> NDArray[np.float64]:
        """Return expected transition surprise of replayed posterior content.

        This is a property of trajectories sampled from the already conditioned
        posterior. It is not an evidence increment and must not be fed back into
        the Bayesian recursion as though it were an external observation.
        """

        transition_log_probability = -np.log(np.maximum(self.transitions, _FLOAT_TINY))
        return np.asarray(
            np.sum(result.smoothed_pair_probabilities * transition_log_probability, axis=(1, 2)),
            dtype=np.float64,
        )
