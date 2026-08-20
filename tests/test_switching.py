import numpy as np
import pytest

from bayesian_ach.switching import SwitchingContextFilter


def _model() -> SwitchingContextFilter:
    probabilities = np.array(
        [
            [[0.9, 0.1], [0.8, 0.2]],
            [[0.2, 0.8], [0.1, 0.9]],
        ],
        dtype=float,
    )
    alpha = 100.0 * probabilities
    return SwitchingContextFilter(
        alpha,
        np.array([[0.9, 0.1], [0.2, 0.8]]),
        initial_context=[0.75, 0.25],
    )


def test_exact_context_update_matches_manual_hmm_filter() -> None:
    model = _model()
    step = model.observe(0, 1)

    previous = np.array([0.75, 0.25])
    dynamics = np.array([[0.9, 0.1], [0.2, 0.8]])
    context_prior = previous @ dynamics
    likelihood = np.array([0.1, 0.8])
    expected_posterior = context_prior * likelihood
    expected_posterior /= expected_posterior.sum()

    np.testing.assert_allclose(step.prior_context, context_prior)
    np.testing.assert_allclose(step.posterior_context, expected_posterior)
    assert step.predictive_probability == pytest.approx(float(context_prior @ likelihood))
    assert step.map_context == 1


def test_inference_does_not_learn_parameters() -> None:
    model = _model()
    before = model.alpha.copy()
    model.observe(0, 1)
    np.testing.assert_array_equal(model.alpha, before)


def test_supervised_learning_updates_only_labelled_context() -> None:
    model = _model()
    before = model.alpha.copy()
    step = model.observe(0, 1, learn_context=1)

    expected = before.copy()
    expected[1, 0, 0, 1] += 1.0
    np.testing.assert_allclose(model.alpha, expected)
    assert step.learned_context == 1
    assert step.applied_parameter_update_l2 > 0.0


def test_switch_probability_uses_joint_context_posterior() -> None:
    model = _model()
    step = model.observe(0, 1)
    off_diagonal = ~np.eye(2, dtype=bool)
    assert step.switch_probability == pytest.approx(
        float(step.joint_context_posterior[off_diagonal].sum())
    )
    assert step.context_kl > 0.0
