from itertools import product

import numpy as np
import pytest
from scipy.special import gammaln

from bayesian_ach.changepoint import DirichletBOCPD


def _segment_marginal(sequence: list[int], alpha: np.ndarray) -> float:
    counts = np.bincount(sequence, minlength=alpha.size).astype(float)
    return float(
        np.exp(
            gammaln(alpha.sum())
            - gammaln(alpha.sum() + counts.sum())
            + np.sum(gammaln(alpha + counts) - gammaln(alpha))
        )
    )


def _enumerated_run_length_posterior(
    sequence: list[int],
    alpha: np.ndarray,
    hazard: float,
) -> np.ndarray:
    if len(sequence) == 1:
        return np.array([1.0])
    weights = np.zeros(len(sequence), dtype=float)
    for boundaries in product((0, 1), repeat=len(sequence) - 1):
        starts = [0] + [index + 1 for index, flag in enumerate(boundaries) if flag]
        ends = starts[1:] + [len(sequence)]
        likelihood = 1.0
        for start, end in zip(starts, ends, strict=True):
            likelihood *= _segment_marginal(sequence[start:end], alpha)
        n_changes = sum(boundaries)
        prior = hazard**n_changes * (1.0 - hazard) ** (len(boundaries) - n_changes)
        last_start = starts[-1]
        run_length = len(sequence) - last_start - 1
        weights[run_length] += prior * likelihood
    return weights / weights.sum()


def test_bocpd_matches_exhaustive_segmentation_posterior() -> None:
    hazard = 0.23
    alpha = np.array([1.3, 0.7])
    sequence = [0, 0, 1, 1]
    detector = DirichletBOCPD(
        2,
        concentration=float(alpha.sum()),
        probabilities=alpha / alpha.sum(),
        hazard=hazard,
    )
    step = None
    for observation in sequence:
        step = detector.observe(0, observation)
    assert step is not None

    expected = _enumerated_run_length_posterior(sequence, alpha, hazard)
    np.testing.assert_allclose(step.run_length_probabilities, expected, atol=1e-12)


def test_run_length_posterior_is_normalized_and_grows() -> None:
    detector = DirichletBOCPD(3, hazard=0.05)
    for index in range(12):
        step = detector.observe(0, 1)
        assert step.run_length_probabilities.sum() == pytest.approx(1.0)
        assert step.run_length_probabilities.size == index + 1
    assert step.map_run_length >= 8


def test_abrupt_change_increases_change_probability() -> None:
    detector = DirichletBOCPD(2, concentration=1.0, hazard=0.02)
    stable_probabilities = []
    for _ in range(30):
        stable_probabilities.append(detector.observe(0, 0).change_probability)
    change_step = detector.observe(0, 1)

    assert change_step.change_probability > max(stable_probabilities[5:])
    assert change_step.run_length_kl > 0.0


def test_copy_is_independent() -> None:
    detector = DirichletBOCPD(2)
    detector.observe(0, 0)
    duplicate = detector.copy()
    duplicate.observe(0, 1)
    assert duplicate.n_observations == detector.n_observations + 1
    assert duplicate.run_length_probabilities.size != detector.run_length_probabilities.size
