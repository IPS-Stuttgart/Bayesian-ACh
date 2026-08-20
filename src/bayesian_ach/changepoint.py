"""Exact Bayesian online change-point detection for categorical transitions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True, slots=True)
class ChangePointStep:
    """One BOCPD update with the complete run-length posterior."""

    state: int
    action: int
    next_state: int
    run_length_probabilities: NDArray[np.float64]
    predictive_probability: float
    surprise: float
    change_probability: float
    run_length_kl: float
    run_length_entropy: float
    expected_run_length: float
    map_run_length: int
    n_observations: int


def _transition_prior(
    n_states: int,
    n_actions: int,
    concentration: float | ArrayLike,
    probabilities: ArrayLike | None,
) -> NDArray[np.float64]:
    if n_states < 2:
        raise ValueError("n_states must be at least two")
    if n_actions < 1:
        raise ValueError("n_actions must be positive")

    shape = (n_actions, n_states, n_states)
    if probabilities is None:
        rows = np.full(shape, 1.0 / n_states, dtype=float)
    else:
        array = np.asarray(probabilities, dtype=float)
        try:
            rows = np.broadcast_to(array, shape).astype(float, copy=True)
        except ValueError as exc:
            raise ValueError(
                f"probabilities cannot broadcast to {shape}; got {array.shape}"
            ) from exc
        if not np.all(np.isfinite(rows)) or np.any(rows <= 0.0):
            raise ValueError("probabilities must contain finite, strictly positive entries")
        rows /= np.sum(rows, axis=-1, keepdims=True)

    concentration_array = np.asarray(concentration, dtype=float)
    try:
        concentration_rows = np.broadcast_to(
            concentration_array,
            shape[:-1],
        ).astype(float, copy=True)
    except ValueError as exc:
        raise ValueError(
            f"concentration cannot broadcast to {shape[:-1]}; got {concentration_array.shape}"
        ) from exc
    if not np.all(np.isfinite(concentration_rows)) or np.any(concentration_rows <= 0.0):
        raise ValueError("concentration must contain finite, strictly positive entries")

    return rows * concentration_rows[..., None]


def _entropy(probabilities: NDArray[np.float64]) -> float:
    positive = probabilities > 0.0
    return float(-np.sum(probabilities[positive] * np.log(probabilities[positive])))


def _categorical_kl(
    posterior: NDArray[np.float64],
    prior: NDArray[np.float64],
) -> float:
    positive = posterior > 0.0
    if np.any(prior[positive] <= 0.0):
        return float("inf")
    return float(np.sum(posterior[positive] * np.log(posterior[positive] / prior[positive])))


class DirichletBOCPD:
    """Full run-length Bayesian online change-point detector.

    Every run-length hypothesis retains its own complete Dirichlet sufficient
    statistics over ``(action, current_state, next_state)``. No pruning or
    moment merging is performed, so inference is exact under the constant-hazard
    piecewise-stationary transition model. Memory and update cost grow linearly
    with the number of observations.

    The first observation is defined to start the first segment. Thereafter,
    run length zero means that a new segment started at the current observation.
    """

    def __init__(
        self,
        n_states: int,
        *,
        n_actions: int = 1,
        concentration: float | ArrayLike = 1.0,
        probabilities: ArrayLike | None = None,
        hazard: float = 0.02,
    ) -> None:
        if not np.isfinite(hazard) or not 0.0 < hazard < 1.0:
            raise ValueError("hazard must be finite and strictly between zero and one")
        self.prior_alpha = _transition_prior(
            n_states,
            n_actions,
            concentration,
            probabilities,
        )
        self.hazard = float(hazard)
        self._run_length_probabilities = np.empty(0, dtype=float)
        self._alpha_hypotheses = np.empty((0,) + self.prior_alpha.shape, dtype=float)
        self.n_observations = 0
        self.cumulative_log_evidence = 0.0

    @property
    def n_actions(self) -> int:
        return int(self.prior_alpha.shape[0])

    @property
    def n_states(self) -> int:
        return int(self.prior_alpha.shape[1])

    @property
    def run_length_probabilities(self) -> NDArray[np.float64]:
        return self._run_length_probabilities.copy()

    @property
    def alpha_hypotheses(self) -> NDArray[np.float64]:
        return self._alpha_hypotheses.copy()

    def _validate_transition(self, state: int, action: int, next_state: int) -> None:
        if not 0 <= action < self.n_actions:
            raise IndexError(f"action {action} is outside [0, {self.n_actions})")
        if not 0 <= state < self.n_states:
            raise IndexError(f"state {state} is outside [0, {self.n_states})")
        if not 0 <= next_state < self.n_states:
            raise IndexError(f"next_state {next_state} is outside [0, {self.n_states})")

    def predictive_distribution(
        self,
        state: int,
        *,
        action: int = 0,
    ) -> NDArray[np.float64]:
        """Return the run-length-marginalized next-state prediction."""

        self._validate_transition(state, action, 0)
        prior_row = self.prior_alpha[action, state]
        reset_predictive = prior_row / float(np.sum(prior_row))
        if self.n_observations == 0:
            return reset_predictive

        rows = self._alpha_hypotheses[:, action, state, :]
        predictive_rows = rows / np.sum(rows, axis=1, keepdims=True)
        continuation = self._run_length_probabilities @ predictive_rows
        return self.hazard * reset_predictive + (1.0 - self.hazard) * continuation

    def observe(
        self,
        state: int,
        next_state: int,
        *,
        action: int = 0,
    ) -> ChangePointStep:
        """Assimilate one transition and update all run-length hypotheses."""

        self._validate_transition(state, action, next_state)
        prior_row = self.prior_alpha[action, state]
        reset_predictive = float(prior_row[next_state] / np.sum(prior_row))

        if self.n_observations == 0:
            posterior = np.array([1.0], dtype=float)
            alpha = self.prior_alpha.copy()
            alpha[action, state, next_state] += 1.0
            self._alpha_hypotheses = alpha[None, ...]
            predictive_probability = reset_predictive
            run_length_prior = posterior.copy()
        else:
            rows = self._alpha_hypotheses[:, action, state, :]
            likelihoods = rows[:, next_state] / np.sum(rows, axis=1)
            run_length_prior = np.concatenate(
                (
                    np.array([self.hazard], dtype=float),
                    (1.0 - self.hazard) * self._run_length_probabilities,
                )
            )
            joint = np.empty(self._run_length_probabilities.size + 1, dtype=float)
            joint[0] = self.hazard * reset_predictive
            joint[1:] = (
                (1.0 - self.hazard)
                * self._run_length_probabilities
                * likelihoods
            )
            predictive_probability = float(np.sum(joint))
            if predictive_probability <= 0.0 or not np.isfinite(predictive_probability):
                raise FloatingPointError("BOCPD predictive probability is invalid")
            posterior = joint / predictive_probability

            new_alpha = np.empty(
                (self._alpha_hypotheses.shape[0] + 1,) + self.prior_alpha.shape,
                dtype=float,
            )
            new_alpha[0] = self.prior_alpha
            new_alpha[1:] = self._alpha_hypotheses
            new_alpha[:, action, state, next_state] += 1.0
            self._alpha_hypotheses = new_alpha

        self._run_length_probabilities = posterior
        self.n_observations += 1
        log_evidence = float(np.log(predictive_probability))
        self.cumulative_log_evidence += log_evidence

        run_lengths = np.arange(posterior.size, dtype=float)
        return ChangePointStep(
            state=state,
            action=action,
            next_state=next_state,
            run_length_probabilities=posterior.copy(),
            predictive_probability=predictive_probability,
            surprise=-log_evidence,
            change_probability=float(posterior[0]),
            run_length_kl=_categorical_kl(posterior, run_length_prior),
            run_length_entropy=_entropy(posterior),
            expected_run_length=float(np.dot(run_lengths, posterior)),
            map_run_length=int(np.argmax(posterior)),
            n_observations=self.n_observations,
        )

    def reset(self) -> None:
        """Discard all observations while retaining the configured prior."""

        self._run_length_probabilities = np.empty(0, dtype=float)
        self._alpha_hypotheses = np.empty((0,) + self.prior_alpha.shape, dtype=float)
        self.n_observations = 0
        self.cumulative_log_evidence = 0.0

    def copy(self) -> DirichletBOCPD:
        """Return an independent detector copy."""

        duplicate = object.__new__(DirichletBOCPD)
        duplicate.prior_alpha = self.prior_alpha.copy()
        duplicate.hazard = self.hazard
        duplicate._run_length_probabilities = self._run_length_probabilities.copy()
        duplicate._alpha_hypotheses = self._alpha_hypotheses.copy()
        duplicate.n_observations = self.n_observations
        duplicate.cumulative_log_evidence = self.cumulative_log_evidence
        return duplicate
