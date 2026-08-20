"""Exact filtering over a finite bank of known transition contexts."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True, slots=True)
class ContextStep:
    """One exact HMM filtering step for fixed context-specific kernels."""

    state: int
    action: int
    next_state: int
    prior_context: NDArray[np.float64]
    posterior_context: NDArray[np.float64]
    transition_likelihoods: NDArray[np.float64]
    joint_context_posterior: NDArray[np.float64]
    predictive_probability: float
    surprise: float
    context_kl: float
    posterior_entropy: float
    switch_probability: float
    expected_parameter_update_l2: float
    applied_parameter_update_l2: float
    learned_context: int | None
    map_context: int


def _normalize_probabilities(values: ArrayLike, *, name: str) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional; got shape {array.shape}")
    if not np.all(np.isfinite(array)) or np.any(array < 0.0):
        raise ValueError(f"{name} must contain finite, non-negative values")
    total = float(np.sum(array))
    if total <= 0.0:
        raise ValueError(f"{name} must have positive total mass")
    return array / total


def _row_stochastic(values: ArrayLike, n_contexts: int) -> NDArray[np.float64]:
    matrix = np.asarray(values, dtype=float)
    if matrix.shape != (n_contexts, n_contexts):
        raise ValueError(
            "context_transition must have shape "
            f"({n_contexts}, {n_contexts}); got {matrix.shape}"
        )
    if not np.all(np.isfinite(matrix)) or np.any(matrix < 0.0):
        raise ValueError("context_transition must contain finite, non-negative values")
    row_sums = np.sum(matrix, axis=1)
    if np.any(row_sums <= 0.0):
        raise ValueError("each context_transition row must have positive mass")
    return matrix / row_sums[:, None]


def _transition_alpha(values: ArrayLike) -> NDArray[np.float64]:
    alpha = np.asarray(values, dtype=float)
    if alpha.ndim == 3:
        alpha = alpha[:, None, :, :]
    if alpha.ndim != 4:
        raise ValueError(
            "alpha must have shape (context, action, state, next_state) "
            f"or (context, state, next_state); got {alpha.shape}"
        )
    if alpha.shape[-1] != alpha.shape[-2]:
        raise ValueError("state and next-state dimensions must be equal")
    if alpha.shape[0] < 2 or alpha.shape[1] < 1 or alpha.shape[2] < 2:
        raise ValueError("alpha requires at least two contexts and two states")
    if not np.all(np.isfinite(alpha)) or np.any(alpha <= 0.0):
        raise ValueError("alpha must contain finite, strictly positive values")
    return alpha.copy()


def _categorical_kl(
    posterior: NDArray[np.float64],
    prior: NDArray[np.float64],
) -> float:
    positive = posterior > 0.0
    if np.any(prior[positive] <= 0.0):
        return float("inf")
    return float(np.sum(posterior[positive] * np.log(posterior[positive] / prior[positive])))


def _entropy(probabilities: NDArray[np.float64]) -> float:
    positive = probabilities > 0.0
    return float(-np.sum(probabilities[positive] * np.log(probabilities[positive])))


class SwitchingContextFilter:
    """Exact HMM context filter with optional supervised parameter learning.

    Context inference is exact conditional on the fixed context transition
    matrix and context-specific posterior-predictive transition rows. Parameter
    learning is deliberately separate: an update is applied only when
    ``learn_context`` supplies an externally known context label. This avoids
    silently conflating latent-mode inference with an approximate soft
    parameter update.
    """

    def __init__(
        self,
        alpha: ArrayLike,
        context_transition: ArrayLike,
        *,
        initial_context: ArrayLike | None = None,
    ) -> None:
        transition_alpha = _transition_alpha(alpha)
        n_contexts = int(transition_alpha.shape[0])
        self.alpha = transition_alpha
        self.context_transition = _row_stochastic(context_transition, n_contexts)
        self.context_probabilities = (
            np.full(n_contexts, 1.0 / n_contexts, dtype=float)
            if initial_context is None
            else _normalize_probabilities(initial_context, name="initial_context")
        )
        if self.context_probabilities.size != n_contexts:
            raise ValueError(
                f"initial_context must contain {n_contexts} entries; "
                f"got {self.context_probabilities.size}"
            )
        self.cumulative_log_evidence = 0.0

    @property
    def n_contexts(self) -> int:
        return int(self.alpha.shape[0])

    @property
    def n_actions(self) -> int:
        return int(self.alpha.shape[1])

    @property
    def n_states(self) -> int:
        return int(self.alpha.shape[2])

    def _validate_transition(self, state: int, action: int, next_state: int) -> None:
        if not 0 <= action < self.n_actions:
            raise IndexError(f"action {action} is outside [0, {self.n_actions})")
        if not 0 <= state < self.n_states:
            raise IndexError(f"state {state} is outside [0, {self.n_states})")
        if not 0 <= next_state < self.n_states:
            raise IndexError(f"next_state {next_state} is outside [0, {self.n_states})")

    def predicted_context(self) -> NDArray[np.float64]:
        """Return the context prior after one context-dynamics step."""

        return self.context_probabilities @ self.context_transition

    def context_transition_probabilities(
        self,
        state: int,
        *,
        action: int = 0,
    ) -> NDArray[np.float64]:
        """Return next-state rows for every context."""

        self._validate_transition(state, action, 0)
        rows = self.alpha[:, action, state, :]
        return rows / np.sum(rows, axis=1, keepdims=True)

    def predictive_distribution(
        self,
        state: int,
        *,
        action: int = 0,
    ) -> NDArray[np.float64]:
        """Return the context-marginalized next-state predictive distribution."""

        context_prior = self.predicted_context()
        rows = self.context_transition_probabilities(state, action=action)
        return context_prior @ rows

    def observe(
        self,
        state: int,
        next_state: int,
        *,
        action: int = 0,
        learn_context: int | None = None,
    ) -> ContextStep:
        """Filter the latent context and optionally learn a labelled context.

        Passing ``learn_context=None`` performs context inference only and leaves
        every Dirichlet parameter unchanged. Passing a valid integer performs
        one exact conjugate update for that externally known context after the
        filtering calculation.
        """

        self._validate_transition(state, action, next_state)
        if learn_context is not None and not 0 <= learn_context < self.n_contexts:
            raise IndexError(
                f"learn_context {learn_context} is outside [0, {self.n_contexts})"
            )

        previous = self.context_probabilities.copy()
        joint_prior = previous[:, None] * self.context_transition
        context_prior = np.sum(joint_prior, axis=0)
        rows = self.alpha[:, action, state, :]
        row_totals = np.sum(rows, axis=1)
        likelihoods = rows[:, next_state] / row_totals
        predictive_probability = float(np.dot(context_prior, likelihoods))
        if predictive_probability <= 0.0 or not np.isfinite(predictive_probability):
            raise FloatingPointError("context-marginal predictive probability is invalid")

        joint_posterior = joint_prior * likelihoods[None, :] / predictive_probability
        posterior = np.sum(joint_posterior, axis=0)
        posterior /= float(np.sum(posterior))

        off_diagonal = ~np.eye(self.n_contexts, dtype=bool)
        switch_probability = float(np.sum(joint_posterior[off_diagonal]))
        context_kl = _categorical_kl(posterior, context_prior)
        posterior_entropy = _entropy(posterior)

        probabilities = rows / row_totals[:, None]
        innovations = -probabilities
        innovations[:, next_state] += 1.0
        gains = 1.0 / (row_totals + 1.0)
        update_magnitudes = gains * np.linalg.norm(innovations, axis=1)
        expected_parameter_update_l2 = float(np.dot(posterior, update_magnitudes))

        applied_parameter_update_l2 = 0.0
        if learn_context is not None:
            applied_parameter_update_l2 = float(update_magnitudes[learn_context])
            self.alpha[learn_context, action, state, next_state] += 1.0

        self.context_probabilities = posterior
        surprise = float(-np.log(predictive_probability))
        self.cumulative_log_evidence += float(np.log(predictive_probability))

        return ContextStep(
            state=state,
            action=action,
            next_state=next_state,
            prior_context=context_prior.copy(),
            posterior_context=posterior.copy(),
            transition_likelihoods=likelihoods.copy(),
            joint_context_posterior=joint_posterior.copy(),
            predictive_probability=predictive_probability,
            surprise=surprise,
            context_kl=context_kl,
            posterior_entropy=posterior_entropy,
            switch_probability=switch_probability,
            expected_parameter_update_l2=expected_parameter_update_l2,
            applied_parameter_update_l2=applied_parameter_update_l2,
            learned_context=learn_context,
            map_context=int(np.argmax(posterior)),
        )

    def reset_context(self, probabilities: ArrayLike) -> None:
        """Reset only the context belief, preserving all transition parameters."""

        normalized = _normalize_probabilities(probabilities, name="probabilities")
        if normalized.size != self.n_contexts:
            raise ValueError(
                f"probabilities must contain {self.n_contexts} entries; got {normalized.size}"
            )
        self.context_probabilities = normalized
        self.cumulative_log_evidence = 0.0

    def copy(self) -> SwitchingContextFilter:
        """Return an independent filter copy."""

        duplicate = object.__new__(SwitchingContextFilter)
        duplicate.alpha = self.alpha.copy()
        duplicate.context_transition = self.context_transition.copy()
        duplicate.context_probabilities = self.context_probabilities.copy()
        duplicate.cumulative_log_evidence = self.cumulative_log_evidence
        return duplicate
