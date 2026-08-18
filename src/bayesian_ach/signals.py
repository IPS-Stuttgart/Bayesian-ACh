"""Exact candidate signals for one categorical state transition."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Final

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import digamma, gammaln

_FLOAT_EPS: Final[float] = float(np.finfo(float).tiny)

CANDIDATE_SIGNAL_NAMES: Final[tuple[str, ...]] = (
    "innovation_l2",
    "surprise",
    "gain",
    "update_l2",
    "information_gain",
    "change_probability",
)


@dataclass(frozen=True, slots=True)
class TransitionSignals:
    """Candidate scalar signals computed before a Dirichlet transition update.

    `change_probability` is a local two-model posterior: the probability that
    this observation came from a reset predictive distribution rather than the
    current predictive distribution. It is not a full Bayesian online
    change-point posterior over run length.
    """

    observed_index: int
    predictive_probability: float
    predictive_entropy: float
    concentration: float
    innovation_l2: float
    brier_score: float
    surprise: float
    gain: float
    update_l2: float
    information_gain: float
    change_probability: float

    def as_dict(self) -> dict[str, float | int]:
        """Return a serialization-friendly mapping."""

        return asdict(self)


def _positive_vector(values: ArrayLike, *, name: str) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional; got shape {array.shape}")
    if array.size < 2:
        raise ValueError(f"{name} must contain at least two states")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    if np.any(array <= 0.0):
        raise ValueError(f"{name} entries must be strictly positive")
    return array


def dirichlet_kl(alpha_p: ArrayLike, alpha_q: ArrayLike) -> float:
    """Return ``KL(Dir(alpha_p) || Dir(alpha_q))``.

    The implementation follows the standard closed form and validates both
    parameter vectors before evaluation.
    """

    p = _positive_vector(alpha_p, name="alpha_p")
    q = _positive_vector(alpha_q, name="alpha_q")
    if p.shape != q.shape:
        raise ValueError(f"alpha_p and alpha_q must have equal shape; got {p.shape} and {q.shape}")

    p0 = float(np.sum(p))
    q0 = float(np.sum(q))
    value = (
        gammaln(p0)
        - np.sum(gammaln(p))
        - gammaln(q0)
        + np.sum(gammaln(q))
        + np.sum((p - q) * (digamma(p) - digamma(p0)))
    )
    # Floating-point roundoff can produce tiny negative values near zero.
    return max(0.0, float(value))


def compute_transition_signals(
    alpha: ArrayLike,
    observed_index: int,
    *,
    reset_alpha: ArrayLike | None = None,
    hazard: float = 0.02,
) -> TransitionSignals:
    """Compute exact scalar hypotheses for one categorical transition.

    Parameters
    ----------
    alpha:
        Current Dirichlet parameters for one state-action transition row.
    observed_index:
        Index of the observed next state.
    reset_alpha:
        Dirichlet parameters defining the reset model used by the local change
        comparison. If omitted, a symmetric unit Dirichlet is used.
    hazard:
        Prior probability of a reset on this transition, in the open interval
        ``(0, 1)``.
    """

    current = _positive_vector(alpha, name="alpha")
    if not isinstance(observed_index, (int, np.integer)):
        raise TypeError("observed_index must be an integer")
    observed = int(observed_index)
    if observed < 0 or observed >= current.size:
        raise IndexError(f"observed_index {observed} is outside [0, {current.size})")
    if not np.isfinite(hazard) or not 0.0 < hazard < 1.0:
        raise ValueError("hazard must be finite and strictly between zero and one")

    reset = (
        np.ones_like(current)
        if reset_alpha is None
        else _positive_vector(reset_alpha, name="reset_alpha")
    )
    if reset.shape != current.shape:
        raise ValueError(
            f"reset_alpha must match alpha shape; got {reset.shape} and {current.shape}"
        )

    concentration = float(np.sum(current))
    probabilities = current / concentration
    reset_probabilities = reset / float(np.sum(reset))

    one_hot = np.zeros_like(probabilities)
    one_hot[observed] = 1.0
    innovation = one_hot - probabilities
    innovation_l2 = float(np.linalg.norm(innovation))
    brier_score = float(np.dot(innovation, innovation))
    predictive_probability = float(probabilities[observed])
    surprise = float(-np.log(max(predictive_probability, _FLOAT_EPS)))
    predictive_entropy = float(
        -np.sum(probabilities * np.log(np.maximum(probabilities, _FLOAT_EPS)))
    )

    gain = 1.0 / (concentration + 1.0)
    posterior = current.copy()
    posterior[observed] += 1.0
    posterior_probabilities = posterior / float(np.sum(posterior))
    update_l2 = float(np.linalg.norm(posterior_probabilities - probabilities))
    information_gain = dirichlet_kl(posterior, current)

    stable_evidence = (1.0 - hazard) * predictive_probability
    reset_evidence = hazard * float(reset_probabilities[observed])
    change_probability = reset_evidence / (stable_evidence + reset_evidence)

    return TransitionSignals(
        observed_index=observed,
        predictive_probability=predictive_probability,
        predictive_entropy=predictive_entropy,
        concentration=concentration,
        innovation_l2=innovation_l2,
        brier_score=brier_score,
        surprise=surprise,
        gain=gain,
        update_l2=update_l2,
        information_gain=information_gain,
        change_probability=float(change_probability),
    )
