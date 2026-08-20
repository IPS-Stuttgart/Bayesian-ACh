"""Exact multisensory filtering with latent state, context, and sensor health."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import product

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True, slots=True)
class MultisensoryStep:
    """One exact partial-observation filtering update.

    Sensor health zero denotes the nominal observation model and health one the
    configured fault model. Modality-only posteriors and conflict scores are
    evaluated under nominal health to expose raw cross-sensor disagreement.
    The complete posterior is retained over
    ``(context, state, health_configuration)``.
    """

    time_index: int
    action: int | None
    observations: tuple[int | None, ...]
    prior_joint: NDArray[np.float64]
    posterior_joint: NDArray[np.float64]
    joint_context_posterior: NDArray[np.float64]
    joint_health_posterior: NDArray[np.float64]
    prior_state: NDArray[np.float64]
    posterior_state: NDArray[np.float64]
    prior_context: NDArray[np.float64]
    posterior_context: NDArray[np.float64]
    prior_health: NDArray[np.float64]
    posterior_health: NDArray[np.float64]
    modality_state_posteriors: NDArray[np.float64]
    modality_context_posteriors: NDArray[np.float64]
    predictive_probability: float
    surprise: float
    state_kl: float
    context_kl: float
    health_kl: float
    posterior_state_entropy: float
    posterior_context_entropy: float
    posterior_health_entropy: float
    context_switch_probability: float
    sensor_fault_probabilities: NDArray[np.float64]
    sensor_fault_onset_probabilities: NDArray[np.float64]
    all_sensors_healthy_probability: float
    state_sensor_conflict_js: float
    context_sensor_conflict_js: float
    map_state: int
    map_context: int
    map_health_index: int
    map_health_configuration: tuple[int, ...]


def _normalize_vector(values: ArrayLike, *, name: str, size: int) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=float)
    if array.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},); got {array.shape}")
    if not np.all(np.isfinite(array)) or np.any(array < 0.0):
        raise ValueError(f"{name} must contain finite, non-negative values")
    total = float(np.sum(array))
    if total <= 0.0:
        raise ValueError(f"{name} must have positive total mass")
    return array / total


def _normalize_rows(values: ArrayLike, *, name: str) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(array)) or np.any(array < 0.0):
        raise ValueError(f"{name} must contain finite, non-negative values")
    row_sums = np.sum(array, axis=-1, keepdims=True)
    if np.any(row_sums <= 0.0):
        raise ValueError(f"every {name} row must have positive mass")
    return array / row_sums


def _transition_probabilities(values: ArrayLike) -> NDArray[np.float64]:
    probabilities = np.asarray(values, dtype=float)
    if probabilities.ndim == 3:
        probabilities = probabilities[:, None, :, :]
    if probabilities.ndim != 4:
        raise ValueError(
            "transition_probabilities must have shape (context, action, state, next_state) "
            f"or (context, state, next_state); got {probabilities.shape}"
        )
    if probabilities.shape[-1] != probabilities.shape[-2]:
        raise ValueError("state and next-state dimensions must be equal")
    if probabilities.shape[0] < 1 or probabilities.shape[1] < 1:
        raise ValueError("at least one context and one action are required")
    if probabilities.shape[2] < 2:
        raise ValueError("at least two states are required")
    return _normalize_rows(probabilities, name="transition_probabilities")


def _context_transition(values: ArrayLike, n_contexts: int) -> NDArray[np.float64]:
    matrix = np.asarray(values, dtype=float)
    if matrix.shape != (n_contexts, n_contexts):
        raise ValueError(
            f"context_transition must have shape ({n_contexts}, {n_contexts}); "
            f"got {matrix.shape}"
        )
    return _normalize_rows(matrix, name="context_transition")


def _health_transition(values: ArrayLike, n_sensors: int) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=float)
    if array.shape != (n_sensors, 2, 2):
        raise ValueError(
            f"sensor_health_transition must have shape ({n_sensors}, 2, 2); "
            f"got {array.shape}"
        )
    return _normalize_rows(array, name="sensor_health_transition")


def _initial_sensor_health(
    values: ArrayLike | None,
    n_sensors: int,
) -> NDArray[np.float64]:
    if values is None:
        array = np.zeros((n_sensors, 2), dtype=float)
        array[:, 0] = 1.0
        return array
    array = np.asarray(values, dtype=float)
    if array.shape != (n_sensors, 2):
        raise ValueError(
            f"initial_sensor_health must have shape ({n_sensors}, 2); got {array.shape}"
        )
    return _normalize_rows(array, name="initial_sensor_health")


def _emission_probabilities(
    values: Sequence[ArrayLike],
    *,
    n_contexts: int,
    n_states: int,
) -> tuple[NDArray[np.float64], ...]:
    if len(values) < 1:
        raise ValueError("at least one observation modality is required")
    result: list[NDArray[np.float64]] = []
    for sensor_index, item in enumerate(values):
        model = np.asarray(item, dtype=float)
        if model.ndim == 2:
            if model.shape[0] != n_states:
                raise ValueError(
                    f"emission_probabilities[{sensor_index}] has state dimension "
                    f"{model.shape[0]}, expected {n_states}"
                )
            model = np.stack((model, model))[:, None, :, :]
            model = np.broadcast_to(
                model,
                (2, n_contexts, n_states, model.shape[-1]),
            ).copy()
        elif model.ndim == 3:
            if model.shape[:2] != (2, n_states):
                raise ValueError(
                    f"emission_probabilities[{sensor_index}] with three dimensions must "
                    f"have shape (2, {n_states}, observation); got {model.shape}"
                )
            model = model[:, None, :, :]
            model = np.broadcast_to(
                model,
                (2, n_contexts, n_states, model.shape[-1]),
            ).copy()
        elif model.ndim == 4:
            if model.shape[:3] != (2, n_contexts, n_states):
                raise ValueError(
                    f"emission_probabilities[{sensor_index}] with four dimensions must "
                    f"have shape (2, {n_contexts}, {n_states}, observation); "
                    f"got {model.shape}"
                )
            model = model.copy()
        else:
            raise ValueError(
                f"emission_probabilities[{sensor_index}] must have two, three, or four "
                f"dimensions; got {model.shape}"
            )
        if model.shape[-1] < 1:
            raise ValueError(f"emission_probabilities[{sensor_index}] has no observations")
        result.append(_normalize_rows(model, name=f"emission_probabilities[{sensor_index}]"))
    return tuple(result)


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


def _jensen_shannon(probabilities: NDArray[np.float64]) -> float:
    if probabilities.shape[0] < 2:
        return 0.0
    mean = np.mean(probabilities, axis=0)
    value = _entropy(mean) - float(
        np.mean([_entropy(probability) for probability in probabilities])
    )
    return max(0.0, value)


class MultisensoryContextFilter:
    """Exact finite-state filter for state, context, and independent sensor health.

    The state posterior has shape ``(context, state, 2**n_sensors)``. Each
    sensor has a binary Markov health state. Observation models may depend on
    health, context, and state, which supports both ordinary state sensors and
    explicit context-cue channels.

    Complexity is exponential in the number of sensors because every health
    configuration is represented exactly. This class is intended for small,
    hypothesis-driven sensor sets rather than high-dimensional raw perception.
    """

    transition_probabilities: NDArray[np.float64]
    context_transition: NDArray[np.float64]
    emission_probabilities: tuple[NDArray[np.float64], ...]
    sensor_health_transition: NDArray[np.float64]
    health_configurations: NDArray[np.int64]
    joint_health_transition: NDArray[np.float64]
    initial_joint: NDArray[np.float64]
    _posterior_joint: NDArray[np.float64]

    def __init__(
        self,
        transition_probabilities: ArrayLike,
        context_transition: ArrayLike,
        emission_probabilities: Sequence[ArrayLike],
        sensor_health_transition: ArrayLike,
        *,
        initial_context: ArrayLike | None = None,
        initial_state: ArrayLike | None = None,
        initial_sensor_health: ArrayLike | None = None,
    ) -> None:
        transitions = _transition_probabilities(transition_probabilities)
        n_contexts = int(transitions.shape[0])
        n_states = int(transitions.shape[2])
        emissions = _emission_probabilities(
            emission_probabilities,
            n_contexts=n_contexts,
            n_states=n_states,
        )
        n_sensors = len(emissions)

        self.transition_probabilities = transitions
        self.context_transition = _context_transition(context_transition, n_contexts)
        self.emission_probabilities = emissions
        self.sensor_health_transition = _health_transition(
            sensor_health_transition,
            n_sensors,
        )
        self.health_configurations = np.asarray(
            list(product((0, 1), repeat=n_sensors)),
            dtype=np.int64,
        )
        n_health = int(self.health_configurations.shape[0])
        joint_health = np.ones((n_health, n_health), dtype=float)
        for sensor_index in range(n_sensors):
            health = self.health_configurations[:, sensor_index]
            joint_health *= self.sensor_health_transition[sensor_index][
                health[:, None],
                health[None, :],
            ]
        self.joint_health_transition = joint_health

        context_prior = (
            np.full(n_contexts, 1.0 / n_contexts, dtype=float)
            if initial_context is None
            else _normalize_vector(initial_context, name="initial_context", size=n_contexts)
        )
        state_prior = (
            np.full(n_states, 1.0 / n_states, dtype=float)
            if initial_state is None
            else _normalize_vector(initial_state, name="initial_state", size=n_states)
        )
        sensor_health = _initial_sensor_health(initial_sensor_health, n_sensors)
        health_prior = np.prod(
            sensor_health[
                np.arange(n_sensors, dtype=int)[None, :],
                self.health_configurations,
            ],
            axis=1,
        )
        initial_joint = (
            context_prior[:, None, None]
            * state_prior[None, :, None]
            * health_prior[None, None, :]
        )
        self.initial_joint = initial_joint
        self._posterior_joint = initial_joint.copy()
        self.time_index = -1
        self.initialized = False
        self.cumulative_log_evidence = 0.0

    @property
    def n_contexts(self) -> int:
        return int(self.transition_probabilities.shape[0])

    @property
    def n_actions(self) -> int:
        return int(self.transition_probabilities.shape[1])

    @property
    def n_states(self) -> int:
        return int(self.transition_probabilities.shape[2])

    @property
    def n_sensors(self) -> int:
        return len(self.emission_probabilities)

    @property
    def n_health_configurations(self) -> int:
        return int(self.health_configurations.shape[0])

    @property
    def posterior_joint(self) -> NDArray[np.float64]:
        return self._posterior_joint.copy()

    def _validated_observations(
        self,
        observations: Sequence[int | None],
    ) -> tuple[int | None, ...]:
        if len(observations) != self.n_sensors:
            raise ValueError(
                f"observations must contain {self.n_sensors} entries; got {len(observations)}"
            )
        validated: list[int | None] = []
        for sensor_index, observation in enumerate(observations):
            if observation is None:
                validated.append(None)
                continue
            if not isinstance(observation, (int, np.integer)):
                raise TypeError(
                    f"observation {sensor_index} must be an integer or None; "
                    f"got {type(observation).__name__}"
                )
            value = int(observation)
            n_observations = int(self.emission_probabilities[sensor_index].shape[-1])
            if not 0 <= value < n_observations:
                raise IndexError(
                    f"observation {sensor_index} value {value} is outside "
                    f"[0, {n_observations})"
                )
            validated.append(value)
        return tuple(validated)

    def _observation_likelihoods(
        self,
        observations: tuple[int | None, ...],
    ) -> tuple[NDArray[np.float64], tuple[NDArray[np.float64], ...]]:
        combined = np.ones(
            (self.n_contexts, self.n_states, self.n_health_configurations),
            dtype=float,
        )
        nominal_likelihoods: list[NDArray[np.float64]] = []
        for sensor_index, observation in enumerate(observations):
            if observation is None:
                likelihood = np.ones_like(combined)
                nominal = np.ones_like(combined)
            else:
                health = self.health_configurations[:, sensor_index]
                raw = self.emission_probabilities[sensor_index][
                    health,
                    :,
                    :,
                    observation,
                ]
                likelihood = np.transpose(raw, (1, 2, 0))
                nominal_raw = self.emission_probabilities[sensor_index][
                    0,
                    :,
                    :,
                    observation,
                ]
                nominal = np.broadcast_to(
                    nominal_raw[:, :, None],
                    combined.shape,
                )
            combined *= likelihood
            nominal_likelihoods.append(nominal)
        return combined, tuple(nominal_likelihoods)

    def _summarize(
        self,
        *,
        time_index: int,
        action: int | None,
        observations: tuple[int | None, ...],
        prior_joint: NDArray[np.float64],
        posterior_joint: NDArray[np.float64],
        joint_context_posterior: NDArray[np.float64],
        joint_health_posterior: NDArray[np.float64],
        modality_likelihoods: tuple[NDArray[np.float64], ...],
        predictive_probability: float,
    ) -> MultisensoryStep:
        prior_context = np.sum(prior_joint, axis=(1, 2))
        posterior_context = np.sum(posterior_joint, axis=(1, 2))
        prior_state = np.sum(prior_joint, axis=(0, 2))
        posterior_state = np.sum(posterior_joint, axis=(0, 2))
        prior_health = np.sum(prior_joint, axis=(0, 1))
        posterior_health = np.sum(posterior_joint, axis=(0, 1))

        modality_state: list[NDArray[np.float64]] = []
        modality_context: list[NDArray[np.float64]] = []
        observed_indices: list[int] = []
        for sensor_index, likelihood in enumerate(modality_likelihoods):
            evidence = float(np.sum(prior_joint * likelihood))
            if evidence <= 0.0 or not np.isfinite(evidence):
                raise FloatingPointError(
                    f"modality {sensor_index} has invalid predictive probability"
                )
            modality_posterior = prior_joint * likelihood / evidence
            modality_state.append(np.sum(modality_posterior, axis=(0, 2)))
            modality_context.append(np.sum(modality_posterior, axis=(1, 2)))
            if observations[sensor_index] is not None:
                observed_indices.append(sensor_index)
        modality_state_array = np.stack(modality_state)
        modality_context_array = np.stack(modality_context)
        selected_state = modality_state_array[observed_indices]
        selected_context = modality_context_array[observed_indices]

        sensor_fault_probabilities = posterior_health @ self.health_configurations
        fault_onsets = np.zeros(self.n_sensors, dtype=float)
        for sensor_index in range(self.n_sensors):
            previous = self.health_configurations[:, sensor_index]
            current = self.health_configurations[:, sensor_index]
            onset_mask = (previous[:, None] == 0) & (current[None, :] == 1)
            fault_onsets[sensor_index] = float(np.sum(joint_health_posterior[onset_mask]))

        off_diagonal = ~np.eye(self.n_contexts, dtype=bool)
        context_switch_probability = float(
            np.sum(joint_context_posterior[off_diagonal])
        )
        map_health_index = int(np.argmax(posterior_health))
        map_health_configuration = tuple(
            int(value) for value in self.health_configurations[map_health_index]
        )

        return MultisensoryStep(
            time_index=time_index,
            action=action,
            observations=observations,
            prior_joint=prior_joint.copy(),
            posterior_joint=posterior_joint.copy(),
            joint_context_posterior=joint_context_posterior.copy(),
            joint_health_posterior=joint_health_posterior.copy(),
            prior_state=prior_state.copy(),
            posterior_state=posterior_state.copy(),
            prior_context=prior_context.copy(),
            posterior_context=posterior_context.copy(),
            prior_health=prior_health.copy(),
            posterior_health=posterior_health.copy(),
            modality_state_posteriors=modality_state_array,
            modality_context_posteriors=modality_context_array,
            predictive_probability=predictive_probability,
            surprise=float(-np.log(predictive_probability)),
            state_kl=_categorical_kl(posterior_state, prior_state),
            context_kl=_categorical_kl(posterior_context, prior_context),
            health_kl=_categorical_kl(posterior_health, prior_health),
            posterior_state_entropy=_entropy(posterior_state),
            posterior_context_entropy=_entropy(posterior_context),
            posterior_health_entropy=_entropy(posterior_health),
            context_switch_probability=context_switch_probability,
            sensor_fault_probabilities=np.asarray(
                sensor_fault_probabilities,
                dtype=float,
            ),
            sensor_fault_onset_probabilities=fault_onsets,
            all_sensors_healthy_probability=float(posterior_health[0]),
            state_sensor_conflict_js=_jensen_shannon(selected_state),
            context_sensor_conflict_js=_jensen_shannon(selected_context),
            map_state=int(np.argmax(posterior_state)),
            map_context=int(np.argmax(posterior_context)),
            map_health_index=map_health_index,
            map_health_configuration=map_health_configuration,
        )

    def initialize(self, observations: Sequence[int | None]) -> MultisensoryStep:
        """Assimilate observations at the initial time without a transition."""

        if self.initialized:
            raise RuntimeError("filter is already initialized; call reset() before reinitializing")
        validated = self._validated_observations(observations)
        prior_joint = self._posterior_joint.copy()
        combined, modality_likelihoods = self._observation_likelihoods(validated)
        predictive_probability = float(np.sum(prior_joint * combined))
        if predictive_probability <= 0.0 or not np.isfinite(predictive_probability):
            raise FloatingPointError("initial predictive probability is invalid")
        posterior_joint = prior_joint * combined / predictive_probability
        posterior_context = np.sum(posterior_joint, axis=(1, 2))
        posterior_health = np.sum(posterior_joint, axis=(0, 1))
        joint_context = np.diag(posterior_context)
        joint_health = np.diag(posterior_health)

        self._posterior_joint = posterior_joint
        self.time_index = 0
        self.initialized = True
        self.cumulative_log_evidence = float(np.log(predictive_probability))
        return self._summarize(
            time_index=0,
            action=None,
            observations=validated,
            prior_joint=prior_joint,
            posterior_joint=posterior_joint,
            joint_context_posterior=joint_context,
            joint_health_posterior=joint_health,
            modality_likelihoods=modality_likelihoods,
            predictive_probability=predictive_probability,
        )

    def step(
        self,
        observations: Sequence[int | None],
        *,
        action: int = 0,
    ) -> MultisensoryStep:
        """Predict through state, context, and health dynamics, then assimilate sensors."""

        if not self.initialized:
            raise RuntimeError("call initialize() before step()")
        if not 0 <= action < self.n_actions:
            raise IndexError(f"action {action} is outside [0, {self.n_actions})")
        validated = self._validated_observations(observations)
        previous_joint = self._posterior_joint
        state_transition = self.transition_probabilities[:, action, :, :]
        prior_joint = np.asarray(
            np.einsum(
                "pih,pc,cij,hk->cjk",
                previous_joint,
                self.context_transition,
                state_transition,
                self.joint_health_transition,
                optimize=False,
            ),
            dtype=float,
        )
        combined, modality_likelihoods = self._observation_likelihoods(validated)
        predictive_probability = float(np.sum(prior_joint * combined))
        if predictive_probability <= 0.0 or not np.isfinite(predictive_probability):
            raise FloatingPointError("multisensory predictive probability is invalid")
        posterior_joint = prior_joint * combined / predictive_probability

        joint_context = np.asarray(
            np.einsum(
                "pih,pc,cij,hk,cjk->pc",
                previous_joint,
                self.context_transition,
                state_transition,
                self.joint_health_transition,
                combined,
                optimize=False,
            ),
            dtype=float,
        ) / predictive_probability
        joint_health = np.asarray(
            np.einsum(
                "pih,pc,cij,hk,cjk->hk",
                previous_joint,
                self.context_transition,
                state_transition,
                self.joint_health_transition,
                combined,
                optimize=False,
            ),
            dtype=float,
        ) / predictive_probability

        self._posterior_joint = posterior_joint
        self.time_index += 1
        self.cumulative_log_evidence += float(np.log(predictive_probability))
        return self._summarize(
            time_index=self.time_index,
            action=action,
            observations=validated,
            prior_joint=prior_joint,
            posterior_joint=posterior_joint,
            joint_context_posterior=joint_context,
            joint_health_posterior=joint_health,
            modality_likelihoods=modality_likelihoods,
            predictive_probability=predictive_probability,
        )

    def reset(self) -> None:
        """Return to the configured prior and discard accumulated evidence."""

        self._posterior_joint = self.initial_joint.copy()
        self.time_index = -1
        self.initialized = False
        self.cumulative_log_evidence = 0.0

    def copy(self) -> MultisensoryContextFilter:
        """Return an independent exact filter copy."""

        duplicate = object.__new__(MultisensoryContextFilter)
        duplicate.transition_probabilities = self.transition_probabilities.copy()
        duplicate.context_transition = self.context_transition.copy()
        duplicate.emission_probabilities = tuple(
            model.copy() for model in self.emission_probabilities
        )
        duplicate.sensor_health_transition = self.sensor_health_transition.copy()
        duplicate.health_configurations = self.health_configurations.copy()
        duplicate.joint_health_transition = self.joint_health_transition.copy()
        duplicate.initial_joint = self.initial_joint.copy()
        duplicate._posterior_joint = self._posterior_joint.copy()
        duplicate.time_index = self.time_index
        duplicate.initialized = self.initialized
        duplicate.cumulative_log_evidence = self.cumulative_log_evidence
        return duplicate
