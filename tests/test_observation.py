import numpy as np
import pytest

from bayesian_ach.observation import MultisensoryContextFilter


def _state_emission(n_states: int, accuracy: float, offset: int = 0) -> np.ndarray:
    emission = np.full((n_states, n_states), (1.0 - accuracy) / (n_states - 1))
    for state in range(n_states):
        emission[state, (state + offset) % n_states] = accuracy
    return emission


def test_one_step_matches_exhaustive_joint_enumeration() -> None:
    transitions = np.array(
        [
            [[[0.85, 0.15], [0.20, 0.80]]],
            [[[0.30, 0.70], [0.75, 0.25]]],
        ]
    )
    context_transition = np.array([[0.8, 0.2], [0.3, 0.7]])
    emission = np.array(
        [
            [
                [[0.90, 0.10], [0.20, 0.80]],
                [[0.75, 0.25], [0.10, 0.90]],
            ],
            [
                [[0.55, 0.45], [0.40, 0.60]],
                [[0.60, 0.40], [0.35, 0.65]],
            ],
        ]
    )
    health_transition = np.array([[[0.9, 0.1], [0.2, 0.8]]])
    model = MultisensoryContextFilter(
        transitions,
        context_transition,
        (emission,),
        health_transition,
        initial_context=np.array([0.6, 0.4]),
        initial_state=np.array([0.7, 0.3]),
        initial_sensor_health=np.array([[0.8, 0.2]]),
    )
    model.initialize((0,))
    previous = model.posterior_joint
    step = model.step((1,))

    prior = np.zeros_like(step.prior_joint)
    unnormalized = np.zeros_like(step.posterior_joint)
    joint_context = np.zeros_like(step.joint_context_posterior)
    joint_health = np.zeros_like(step.joint_health_posterior)
    for previous_context in range(model.n_contexts):
        for previous_state in range(model.n_states):
            for previous_health in range(model.n_health_configurations):
                previous_weight = previous[previous_context, previous_state, previous_health]
                for context in range(model.n_contexts):
                    for state in range(model.n_states):
                        for health in range(model.n_health_configurations):
                            transition_weight = (
                                previous_weight
                                * model.context_transition[previous_context, context]
                                * model.transition_probabilities[context, 0, previous_state, state]
                                * model.joint_health_transition[previous_health, health]
                            )
                            prior[context, state, health] += transition_weight
                            health_value = model.health_configurations[health, 0]
                            likelihood = emission[health_value, context, state, 1]
                            weighted = transition_weight * likelihood
                            unnormalized[context, state, health] += weighted
                            joint_context[previous_context, context] += weighted
                            joint_health[previous_health, health] += weighted

    evidence = float(np.sum(unnormalized))
    np.testing.assert_allclose(step.prior_joint, prior, atol=1e-14)
    np.testing.assert_allclose(step.posterior_joint, unnormalized / evidence, atol=1e-14)
    np.testing.assert_allclose(step.joint_context_posterior, joint_context / evidence, atol=1e-14)
    np.testing.assert_allclose(step.joint_health_posterior, joint_health / evidence, atol=1e-14)
    assert step.predictive_probability == pytest.approx(evidence)


def test_conflicting_visual_observation_is_attributed_to_visual_fault() -> None:
    n_states = 4
    identity = np.full((n_states, n_states), 0.05)
    np.fill_diagonal(identity, 0.85)
    visual_healthy = _state_emission(n_states, 0.94)
    visual_fault = _state_emission(n_states, 0.94, offset=2)
    proprioceptive = _state_emission(n_states, 0.94)
    health_transition = np.array(
        [
            [[0.90, 0.10], [0.05, 0.95]],
            [[1.00, 0.00], [1.00, 0.00]],
        ]
    )
    model = MultisensoryContextFilter(
        np.stack((identity,)),
        np.array([[1.0]]),
        (
            np.stack((visual_healthy, visual_fault)),
            np.stack((proprioceptive, proprioceptive)),
        ),
        health_transition,
        initial_context=np.array([1.0]),
        initial_state=np.full(n_states, 1.0 / n_states),
        initial_sensor_health=np.array([[1.0, 0.0], [1.0, 0.0]]),
    )
    model.initialize((0, 0))
    step = model.step((2, 0))

    assert step.map_state == 0
    assert step.sensor_fault_probabilities[0] > 0.70
    assert step.sensor_fault_probabilities[1] == pytest.approx(0.0)
    assert step.sensor_fault_onset_probabilities[0] > 0.70
    assert step.state_sensor_conflict_js > 0.20


def test_context_dependent_cue_drives_context_belief_without_sensor_fault() -> None:
    transitions = np.array(
        [
            [[0.95, 0.05], [0.05, 0.95]],
            [[0.05, 0.95], [0.95, 0.05]],
        ]
    )
    state_sensor = _state_emission(2, 0.90)
    cue = np.empty((2, 2, 2, 2))
    cue[:, 0, :, :] = np.array([0.90, 0.10])
    cue[:, 1, :, :] = np.array([0.10, 0.90])
    fixed_health = np.array(
        [
            [[1.0, 0.0], [1.0, 0.0]],
            [[1.0, 0.0], [1.0, 0.0]],
        ]
    )
    model = MultisensoryContextFilter(
        transitions,
        np.array([[0.80, 0.20], [0.20, 0.80]]),
        (np.stack((state_sensor, state_sensor)), cue),
        fixed_health,
        initial_context=np.array([1.0, 0.0]),
        initial_state=np.array([1.0, 0.0]),
        initial_sensor_health=np.array([[1.0, 0.0], [1.0, 0.0]]),
    )
    model.initialize((0, 0))
    step = model.step((1, 1))

    assert step.posterior_context[1] > step.prior_context[1]
    assert step.posterior_context[1] > 0.70
    np.testing.assert_allclose(step.sensor_fault_probabilities, 0.0)
    assert step.context_switch_probability > 0.70


def test_missing_modality_is_supported_and_copy_is_independent() -> None:
    transition = np.array([[[0.8, 0.2], [0.2, 0.8]]])
    emission = _state_emission(2, 0.8)
    fixed_health = np.array(
        [
            [[1.0, 0.0], [1.0, 0.0]],
            [[1.0, 0.0], [1.0, 0.0]],
        ]
    )
    model = MultisensoryContextFilter(
        transition,
        np.array([[1.0]]),
        (emission, emission),
        fixed_health,
    )
    initial = model.initialize((0, None))
    assert initial.state_sensor_conflict_js == pytest.approx(0.0)

    duplicate = model.copy()
    duplicate.step((1, 1))
    assert not np.array_equal(model.posterior_joint, duplicate.posterior_joint)
    model.reset()
    assert model.initialized is False
    assert model.time_index == -1
