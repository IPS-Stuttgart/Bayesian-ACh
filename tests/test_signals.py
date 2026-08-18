import numpy as np
import pytest

from bayesian_ach.signals import compute_transition_signals, dirichlet_kl


def test_exact_posterior_mean_update_matches_gain_times_innovation() -> None:
    alpha = np.array([7.0, 2.0, 1.0])
    observed = 1
    signals = compute_transition_signals(alpha, observed)

    probabilities = alpha / alpha.sum()
    posterior = alpha.copy()
    posterior[observed] += 1.0
    exact_update = posterior / posterior.sum() - probabilities
    one_hot = np.eye(alpha.size)[observed]
    expected = signals.gain * (one_hot - probabilities)

    np.testing.assert_allclose(exact_update, expected, atol=1e-14)
    assert signals.update_l2 == pytest.approx(np.linalg.norm(exact_update))


def test_matched_probability_different_confidence() -> None:
    probabilities = np.array([0.7, 0.2, 0.1])
    low = compute_transition_signals(5.0 * probabilities, 1)
    high = compute_transition_signals(100.0 * probabilities, 1)

    assert low.predictive_probability == pytest.approx(high.predictive_probability)
    assert low.surprise == pytest.approx(high.surprise)
    assert low.innovation_l2 == pytest.approx(high.innovation_l2)
    assert low.gain > high.gain
    assert low.update_l2 > high.update_l2
    assert low.information_gain > high.information_gain


def test_change_probability_responds_to_reset_evidence() -> None:
    current = np.array([98.0, 1.0, 1.0])
    reset = np.array([1.0, 8.0, 1.0])
    surprising_for_current = compute_transition_signals(current, 1, reset_alpha=reset, hazard=0.05)
    expected_for_current = compute_transition_signals(current, 0, reset_alpha=reset, hazard=0.05)

    assert surprising_for_current.change_probability > expected_for_current.change_probability


def test_dirichlet_kl_is_zero_for_equal_parameters() -> None:
    alpha = np.array([1.5, 2.0, 4.0])
    assert dirichlet_kl(alpha, alpha) == pytest.approx(0.0, abs=1e-13)


def test_invalid_inputs_are_rejected() -> None:
    with pytest.raises(ValueError):
        compute_transition_signals([1.0, 0.0], 0)
    with pytest.raises(IndexError):
        compute_transition_signals([1.0, 1.0], 2)
    with pytest.raises(ValueError):
        compute_transition_signals([1.0, 1.0], 0, hazard=1.0)
