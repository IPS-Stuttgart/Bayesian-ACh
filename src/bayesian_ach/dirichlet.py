"""Recursive Dirichlet transition estimator."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from bayesian_ach.signals import TransitionSignals, compute_transition_signals


@dataclass(slots=True)
class DirichletTransitionModel:
    """Action-conditioned categorical transition model with exact updates.

    The final three dimensions of ``alpha`` are
    ``(action, current_state, next_state)``. The model keeps its initialization
    as the reset reference distribution for local change evidence.
    """

    alpha: NDArray[np.float64]
    reset_alpha: NDArray[np.float64]
    hazard: float = 0.02

    def __init__(
        self,
        n_states: int,
        *,
        n_actions: int = 1,
        concentration: float | ArrayLike = 1.0,
        probabilities: ArrayLike | None = None,
        reset_probabilities: ArrayLike | None = None,
        reset_concentration: float = 1.0,
        hazard: float = 0.02,
    ) -> None:
        if n_states < 2:
            raise ValueError("n_states must be at least two")
        if n_actions < 1:
            raise ValueError("n_actions must be positive")
        if not np.isfinite(hazard) or not 0.0 < hazard < 1.0:
            raise ValueError("hazard must be finite and strictly between zero and one")
        if not np.isfinite(reset_concentration) or reset_concentration <= 0.0:
            raise ValueError("reset_concentration must be finite and positive")

        shape = (n_actions, n_states, n_states)
        probability_rows = self._probability_rows(probabilities, shape, name="probabilities")
        concentration_rows = self._concentration_rows(concentration, shape[:-1])
        alpha = probability_rows * concentration_rows[..., None]

        reset_rows = self._probability_rows(
            reset_probabilities,
            shape,
            name="reset_probabilities",
        )
        self.alpha = np.asarray(alpha, dtype=float)
        self.reset_alpha = np.asarray(reset_rows * reset_concentration, dtype=float)
        self.hazard = float(hazard)

    @staticmethod
    def _probability_rows(
        values: ArrayLike | None,
        shape: tuple[int, int, int],
        *,
        name: str,
    ) -> NDArray[np.float64]:
        if values is None:
            return np.full(shape, 1.0 / shape[-1], dtype=float)
        array = np.asarray(values, dtype=float)
        try:
            rows = np.broadcast_to(array, shape).astype(float, copy=True)
        except ValueError as exc:
            raise ValueError(f"{name} cannot broadcast to {shape}; got {array.shape}") from exc
        if not np.all(np.isfinite(rows)) or np.any(rows <= 0.0):
            raise ValueError(f"{name} must contain finite, strictly positive entries")
        row_sums = np.sum(rows, axis=-1, keepdims=True)
        return rows / row_sums

    @staticmethod
    def _concentration_rows(
        values: float | ArrayLike,
        shape: tuple[int, int],
    ) -> NDArray[np.float64]:
        array = np.asarray(values, dtype=float)
        try:
            rows = np.broadcast_to(array, shape).astype(float, copy=True)
        except ValueError as exc:
            raise ValueError(
                f"concentration cannot broadcast to {shape}; got {array.shape}"
            ) from exc
        if not np.all(np.isfinite(rows)) or np.any(rows <= 0.0):
            raise ValueError("concentration must contain finite, strictly positive entries")
        return rows

    @property
    def n_actions(self) -> int:
        return int(self.alpha.shape[0])

    @property
    def n_states(self) -> int:
        return int(self.alpha.shape[1])

    def _validate_indices(self, state: int, action: int, next_state: int | None = None) -> None:
        if not 0 <= action < self.n_actions:
            raise IndexError(f"action {action} is outside [0, {self.n_actions})")
        if not 0 <= state < self.n_states:
            raise IndexError(f"state {state} is outside [0, {self.n_states})")
        if next_state is not None and not 0 <= next_state < self.n_states:
            raise IndexError(f"next_state {next_state} is outside [0, {self.n_states})")

    def predict(self, state: int, *, action: int = 0) -> NDArray[np.float64]:
        """Return the posterior predictive next-state probabilities."""

        self._validate_indices(state, action)
        row = self.alpha[action, state]
        return row / float(np.sum(row))

    def score(self, state: int, next_state: int, *, action: int = 0) -> TransitionSignals:
        """Compute candidate signals without changing the model."""

        self._validate_indices(state, action, next_state)
        return compute_transition_signals(
            self.alpha[action, state],
            next_state,
            reset_alpha=self.reset_alpha[action, state],
            hazard=self.hazard,
        )

    def observe(self, state: int, next_state: int, *, action: int = 0) -> TransitionSignals:
        """Score and assimilate one transition."""

        signals = self.score(state, next_state, action=action)
        self.alpha[action, state, next_state] += 1.0
        return signals

    def copy(self) -> DirichletTransitionModel:
        """Return an independent model copy."""

        duplicate = object.__new__(DirichletTransitionModel)
        duplicate.alpha = self.alpha.copy()
        duplicate.reset_alpha = self.reset_alpha.copy()
        duplicate.hazard = self.hazard
        return duplicate
