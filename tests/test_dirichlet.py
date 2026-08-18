import numpy as np

from bayesian_ach.dirichlet import DirichletTransitionModel


def test_model_updates_only_selected_transition_row() -> None:
    model = DirichletTransitionModel(3, n_actions=2, concentration=6.0)
    before = model.alpha.copy()

    signals = model.observe(1, 2, action=1)

    expected = before.copy()
    expected[1, 1, 2] += 1.0
    np.testing.assert_allclose(model.alpha, expected)
    assert signals.observed_index == 2


def test_predict_returns_normalized_copy() -> None:
    model = DirichletTransitionModel(
        3,
        concentration=10.0,
        probabilities=np.array([0.6, 0.3, 0.1]),
    )
    prediction = model.predict(0)
    np.testing.assert_allclose(prediction, [0.6, 0.3, 0.1])
    prediction[0] = 0.0
    assert model.predict(0)[0] == 0.6


def test_copy_is_independent() -> None:
    model = DirichletTransitionModel(3)
    duplicate = model.copy()
    duplicate.observe(0, 1)
    assert not np.array_equal(model.alpha, duplicate.alpha)
