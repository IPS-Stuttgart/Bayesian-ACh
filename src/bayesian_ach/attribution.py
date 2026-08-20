"""Exact partial-observation recovery of sensor faults and world changes."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Final, Literal

import numpy as np
from numpy.typing import NDArray

from bayesian_ach.observation import MultisensoryContextFilter, MultisensoryStep

MechanismKind = Literal[
    "visual_sensor_fault",
    "known_context_switch",
    "structural_transition_change",
]
MECHANISM_NAMES: Final[tuple[MechanismKind, ...]] = (
    "visual_sensor_fault",
    "known_context_switch",
    "structural_transition_change",
)


@dataclass(frozen=True, slots=True)
class ObservationAttributionConfig:
    """Configuration for the exact multisensory attribution benchmark."""

    n_sequences_per_class: int = 36
    pre_length: int = 24
    post_length: int = 40
    n_states: int = 4
    dominant_probability: float = 0.82
    stay_probability: float = 0.10
    visual_accuracy: float = 0.90
    proprioceptive_accuracy: float = 0.78
    cue_accuracy: float = 0.80
    fault_accuracy: float = 0.90
    fault_similarity: float = 0.0
    structural_similarity: float = 0.0
    mechanism_switch_probability: float = 0.03
    fault_recovery_probability: float = 0.01
    seed: int = 7

    def validate(self) -> None:
        if self.n_sequences_per_class < 1:
            raise ValueError("n_sequences_per_class must be positive")
        if self.pre_length < 1 or self.post_length < 1:
            raise ValueError("pre_length and post_length must be positive")
        if self.n_states < 4:
            raise ValueError("n_states must be at least four")
        if not 0.0 < self.dominant_probability < 1.0:
            raise ValueError("dominant_probability must lie in (0, 1)")
        if not 0.0 < self.stay_probability < 1.0:
            raise ValueError("stay_probability must lie in (0, 1)")
        if self.dominant_probability + self.stay_probability >= 1.0:
            raise ValueError("dominant_probability + stay_probability must be below one")
        chance = 1.0 / self.n_states
        for name, value in (
            ("visual_accuracy", self.visual_accuracy),
            ("proprioceptive_accuracy", self.proprioceptive_accuracy),
            ("fault_accuracy", self.fault_accuracy),
        ):
            if not chance < value <= 1.0:
                raise ValueError(f"{name} must lie in ({chance}, 1]")
        if not 0.5 < self.cue_accuracy <= 1.0:
            raise ValueError("cue_accuracy must lie in (0.5, 1]")
        if not 0.0 <= self.fault_similarity <= 1.0:
            raise ValueError("fault_similarity must lie in [0, 1]")
        if not 0.0 <= self.structural_similarity <= 1.0:
            raise ValueError("structural_similarity must lie in [0, 1]")
        if not 0.0 < self.mechanism_switch_probability < 0.5:
            raise ValueError("mechanism_switch_probability must lie in (0, 0.5)")
        if not 0.0 < self.fault_recovery_probability < 1.0:
            raise ValueError("fault_recovery_probability must lie in (0, 1)")


@dataclass(frozen=True, slots=True)
class ObservationSequenceResult:
    """Prequential model-class evidence for one partially observed sequence."""

    sequence_id: int
    true_mechanism: str
    predicted_mechanism: str
    correct: bool
    visual_sensor_fault_log_evidence_post: float
    known_context_switch_log_evidence_post: float
    structural_transition_change_log_evidence_post: float
    evidence_margin: float
    final_visual_sensor_fault_probability: float
    final_known_context_switch_probability: float
    final_structural_transition_change_probability: float
    peak_visual_fault_probability: float
    peak_visual_fault_onset_probability: float
    peak_known_context_probability: float
    peak_structural_context_probability: float
    mean_state_sensor_conflict_js_post: float
    visual_sensor_fault_model_state_accuracy_post: float
    known_context_model_state_accuracy_post: float
    structural_model_state_accuracy_post: float

    def as_dict(self) -> dict[str, float | int | str | bool]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ObservationAttributionResult:
    """Complete exact multisensory attribution benchmark output."""

    sequences: tuple[ObservationSequenceResult, ...]
    trials: tuple[dict[str, Any], ...]
    summary: dict[str, Any]


@dataclass(frozen=True, slots=True)
class _SyntheticObservation:
    transition_index: int
    phase: str
    true_state: int
    true_context: int
    true_visual_fault: int
    observations: tuple[int, int, int]


def _ring_transition_kernels(
    n_states: int,
    *,
    dominant_probability: float,
    stay_probability: float,
    structural_similarity: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    residual = 1.0 - dominant_probability - stay_probability
    offsets = (1, -1, max(2, n_states // 2))
    kernels: list[NDArray[np.float64]] = []
    for offset in offsets:
        kernel = np.zeros((n_states, n_states), dtype=float)
        for state in range(n_states):
            dominant = (state + offset) % n_states
            kernel[state, state] = stay_probability
            kernel[state, dominant] = dominant_probability
            remaining = [
                candidate
                for candidate in range(n_states)
                if candidate not in (state, dominant)
            ]
            for candidate in remaining:
                kernel[state, candidate] = residual / len(remaining)
        kernels.append(kernel)
    known_mixture = 0.5 * (kernels[0] + kernels[1])
    structural = (
        (1.0 - structural_similarity) * kernels[2]
        + structural_similarity * known_mixture
    )
    return kernels[0], kernels[1], structural


def _state_emission(
    n_states: int,
    accuracy: float,
    *,
    offset: int = 0,
) -> NDArray[np.float64]:
    emission = np.full(
        (n_states, n_states),
        (1.0 - accuracy) / (n_states - 1),
        dtype=float,
    )
    for state in range(n_states):
        emission[state, (state + offset) % n_states] = accuracy
    return emission


def _cue_emission(
    n_contexts: int,
    n_states: int,
    accuracy: float,
    labels: tuple[int, ...],
) -> NDArray[np.float64]:
    if len(labels) != n_contexts or any(label not in (0, 1) for label in labels):
        raise ValueError("cue labels must contain one binary label per context")
    emission = np.empty((2, n_contexts, n_states, 2), dtype=float)
    for health in range(2):
        for context, label in enumerate(labels):
            probabilities = np.array([1.0 - accuracy, 1.0 - accuracy], dtype=float)
            probabilities[label] = accuracy
            emission[health, context, :, :] = probabilities
    return emission


def _healthy_health_transition(n_sensors: int) -> NDArray[np.float64]:
    transition = np.zeros((n_sensors, 2, 2), dtype=float)
    transition[:, 0, 0] = 1.0
    transition[:, 1, 0] = 1.0
    return transition


def _model_bank(
    config: ObservationAttributionConfig,
) -> tuple[
    tuple[MultisensoryContextFilter, ...],
    tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]],
    tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]],
]:
    forward, backward, structural = _ring_transition_kernels(
        config.n_states,
        dominant_probability=config.dominant_probability,
        stay_probability=config.stay_probability,
        structural_similarity=config.structural_similarity,
    )
    visual_healthy = _state_emission(config.n_states, config.visual_accuracy)
    visual_shifted = _state_emission(
        config.n_states,
        config.fault_accuracy,
        offset=max(2, config.n_states // 2),
    )
    visual_fault = (
        config.fault_similarity * visual_healthy
        + (1.0 - config.fault_similarity) * visual_shifted
    )
    proprioceptive = _state_emission(
        config.n_states,
        config.proprioceptive_accuracy,
    )

    healthy_transition = _healthy_health_transition(3)
    fault_transition = healthy_transition.copy()
    switch = config.mechanism_switch_probability
    fault_transition[0] = np.array(
        [
            [1.0 - switch, switch],
            [config.fault_recovery_probability, 1.0 - config.fault_recovery_probability],
        ],
        dtype=float,
    )
    initial_health = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])

    fault_model = MultisensoryContextFilter(
        np.stack((forward,)),
        np.array([[1.0]]),
        (
            np.stack((visual_healthy, visual_fault)),
            np.stack((proprioceptive, proprioceptive)),
            _cue_emission(1, config.n_states, config.cue_accuracy, (0,)),
        ),
        fault_transition,
        initial_context=np.array([1.0]),
        initial_sensor_health=initial_health,
    )
    context_transition = np.array(
        [[1.0 - switch, switch], [switch, 1.0 - switch]],
        dtype=float,
    )
    known_context_model = MultisensoryContextFilter(
        np.stack((forward, backward)),
        context_transition,
        (
            np.stack((visual_healthy, visual_healthy)),
            np.stack((proprioceptive, proprioceptive)),
            _cue_emission(2, config.n_states, config.cue_accuracy, (0, 1)),
        ),
        healthy_transition,
        initial_context=np.array([1.0, 0.0]),
        initial_sensor_health=initial_health,
    )
    structural_model = MultisensoryContextFilter(
        np.stack((forward, structural)),
        context_transition,
        (
            np.stack((visual_healthy, visual_healthy)),
            np.stack((proprioceptive, proprioceptive)),
            _cue_emission(2, config.n_states, config.cue_accuracy, (0, 0)),
        ),
        healthy_transition,
        initial_context=np.array([1.0, 0.0]),
        initial_sensor_health=initial_health,
    )
    return (
        (fault_model, known_context_model, structural_model),
        (forward, backward, structural),
        (visual_healthy, visual_fault, proprioceptive),
    )


def _cue_probabilities(label: int, accuracy: float) -> NDArray[np.float64]:
    probabilities = np.array([1.0 - accuracy, 1.0 - accuracy], dtype=float)
    probabilities[label] = accuracy
    return probabilities


def _simulate_sequence(
    kind: MechanismKind,
    config: ObservationAttributionConfig,
    rng: np.random.Generator,
) -> list[_SyntheticObservation]:
    _, kernels, emissions = _model_bank(config)
    forward, backward, structural = kernels
    visual_healthy, visual_fault, proprioceptive = emissions
    state = int(rng.integers(config.n_states))
    initial_observations = (
        int(rng.choice(config.n_states, p=visual_healthy[state])),
        int(rng.choice(config.n_states, p=proprioceptive[state])),
        int(rng.choice(2, p=_cue_probabilities(0, config.cue_accuracy))),
    )
    rows = [
        _SyntheticObservation(
            transition_index=-1,
            phase="initial",
            true_state=state,
            true_context=0,
            true_visual_fault=0,
            observations=initial_observations,
        )
    ]

    total_length = config.pre_length + config.post_length
    for transition_index in range(total_length):
        in_post = transition_index >= config.pre_length
        phase = "post" if in_post else "pre"
        kernel = forward
        visual_emission = visual_healthy
        cue_label = 0
        true_context = 0
        true_fault = 0
        if in_post and kind == "visual_sensor_fault":
            visual_emission = visual_fault
            true_fault = 1
        elif in_post and kind == "known_context_switch":
            kernel = backward
            cue_label = 1
            true_context = 1
        elif in_post and kind == "structural_transition_change":
            kernel = structural
            true_context = -1

        state = int(rng.choice(config.n_states, p=kernel[state]))
        observations = (
            int(rng.choice(config.n_states, p=visual_emission[state])),
            int(rng.choice(config.n_states, p=proprioceptive[state])),
            int(rng.choice(2, p=_cue_probabilities(cue_label, config.cue_accuracy))),
        )
        rows.append(
            _SyntheticObservation(
                transition_index=transition_index,
                phase=phase,
                true_state=state,
                true_context=true_context,
                true_visual_fault=true_fault,
                observations=observations,
            )
        )
    return rows


def _softmax(log_values: NDArray[np.float64]) -> NDArray[np.float64]:
    shifted = log_values - float(np.max(log_values))
    weights = np.exp(shifted)
    return weights / float(np.sum(weights))


def _step_columns(prefix: str, step: MultisensoryStep) -> dict[str, Any]:
    row: dict[str, Any] = {
        f"{prefix}_predictive_probability": step.predictive_probability,
        f"{prefix}_surprise": step.surprise,
        f"{prefix}_map_state": step.map_state,
        f"{prefix}_map_context": step.map_context,
        f"{prefix}_state_kl": step.state_kl,
        f"{prefix}_context_kl": step.context_kl,
        f"{prefix}_health_kl": step.health_kl,
        f"{prefix}_context_switch_probability": step.context_switch_probability,
        f"{prefix}_state_sensor_conflict_js": step.state_sensor_conflict_js,
        f"{prefix}_context_sensor_conflict_js": step.context_sensor_conflict_js,
    }
    for sensor_index, probability in enumerate(step.sensor_fault_probabilities):
        row[f"{prefix}_sensor_{sensor_index}_fault_probability"] = float(probability)
    for sensor_index, probability in enumerate(step.sensor_fault_onset_probabilities):
        row[f"{prefix}_sensor_{sensor_index}_fault_onset_probability"] = float(probability)
    for context_index, probability in enumerate(step.posterior_context):
        row[f"{prefix}_context_{context_index}_probability"] = float(probability)
    return row


def _evaluate_sequence(
    sequence_id: int,
    kind: MechanismKind,
    synthetic: list[_SyntheticObservation],
    config: ObservationAttributionConfig,
) -> tuple[ObservationSequenceResult, list[dict[str, Any]]]:
    models, _, _ = _model_bank(config)
    initial = synthetic[0]
    initial_steps = [model.initialize(initial.observations) for model in models]
    trial_rows: list[dict[str, Any]] = []
    initial_row: dict[str, Any] = {
        "sequence_id": sequence_id,
        "transition_index": -1,
        "phase": "initial",
        "true_mechanism": kind,
        "true_state": initial.true_state,
        "true_context": initial.true_context,
        "true_visual_fault": initial.true_visual_fault,
        "visual_observation": initial.observations[0],
        "proprioceptive_observation": initial.observations[1],
        "cue_observation": initial.observations[2],
        "posterior_visual_sensor_fault": 1.0 / 3.0,
        "posterior_known_context_switch": 1.0 / 3.0,
        "posterior_structural_transition_change": 1.0 / 3.0,
    }
    for model_name, step in zip(MECHANISM_NAMES, initial_steps, strict=True):
        initial_row.update(_step_columns(model_name, step))
    trial_rows.append(initial_row)

    post_log_evidence = np.zeros(len(MECHANISM_NAMES), dtype=float)
    peak_visual_fault = 0.0
    peak_visual_fault_onset = 0.0
    peak_known_context = 0.0
    peak_structural_context = 0.0
    conflicts: list[float] = []
    state_correct: list[list[bool]] = [[], [], []]

    for synthetic_step in synthetic[1:]:
        steps = [model.step(synthetic_step.observations) for model in models]
        if synthetic_step.phase == "post":
            log_increments = np.log(
                np.asarray(
                    [step.predictive_probability for step in steps],
                    dtype=float,
                )
            )
            post_log_evidence += log_increments
            model_probabilities = _softmax(post_log_evidence)
            peak_visual_fault = max(
                peak_visual_fault,
                float(steps[0].sensor_fault_probabilities[0]),
            )
            peak_visual_fault_onset = max(
                peak_visual_fault_onset,
                float(steps[0].sensor_fault_onset_probabilities[0]),
            )
            peak_known_context = max(
                peak_known_context,
                float(steps[1].posterior_context[1]),
            )
            peak_structural_context = max(
                peak_structural_context,
                float(steps[2].posterior_context[1]),
            )
            conflicts.append(steps[0].state_sensor_conflict_js)
            for model_index, step in enumerate(steps):
                state_correct[model_index].append(step.map_state == synthetic_step.true_state)
        else:
            model_probabilities = np.full(len(MECHANISM_NAMES), 1.0 / 3.0)

        row: dict[str, Any] = {
            "sequence_id": sequence_id,
            "transition_index": synthetic_step.transition_index,
            "phase": synthetic_step.phase,
            "true_mechanism": kind,
            "true_state": synthetic_step.true_state,
            "true_context": synthetic_step.true_context,
            "true_visual_fault": synthetic_step.true_visual_fault,
            "visual_observation": synthetic_step.observations[0],
            "proprioceptive_observation": synthetic_step.observations[1],
            "cue_observation": synthetic_step.observations[2],
            "posterior_visual_sensor_fault": float(model_probabilities[0]),
            "posterior_known_context_switch": float(model_probabilities[1]),
            "posterior_structural_transition_change": float(model_probabilities[2]),
        }
        for model_name, step in zip(MECHANISM_NAMES, steps, strict=True):
            row.update(_step_columns(model_name, step))
        trial_rows.append(row)

    predicted_index = int(np.argmax(post_log_evidence))
    predicted = MECHANISM_NAMES[predicted_index]
    sorted_evidence = np.sort(post_log_evidence)
    final_probabilities = _softmax(post_log_evidence)
    result = ObservationSequenceResult(
        sequence_id=sequence_id,
        true_mechanism=kind,
        predicted_mechanism=predicted,
        correct=predicted == kind,
        visual_sensor_fault_log_evidence_post=float(post_log_evidence[0]),
        known_context_switch_log_evidence_post=float(post_log_evidence[1]),
        structural_transition_change_log_evidence_post=float(post_log_evidence[2]),
        evidence_margin=float(sorted_evidence[-1] - sorted_evidence[-2]),
        final_visual_sensor_fault_probability=float(final_probabilities[0]),
        final_known_context_switch_probability=float(final_probabilities[1]),
        final_structural_transition_change_probability=float(final_probabilities[2]),
        peak_visual_fault_probability=peak_visual_fault,
        peak_visual_fault_onset_probability=peak_visual_fault_onset,
        peak_known_context_probability=peak_known_context,
        peak_structural_context_probability=peak_structural_context,
        mean_state_sensor_conflict_js_post=float(np.mean(conflicts)),
        visual_sensor_fault_model_state_accuracy_post=float(np.mean(state_correct[0])),
        known_context_model_state_accuracy_post=float(np.mean(state_correct[1])),
        structural_model_state_accuracy_post=float(np.mean(state_correct[2])),
    )
    return result, trial_rows


def _mean_total_variation(
    first: NDArray[np.float64],
    second: NDArray[np.float64],
) -> float:
    return float(np.mean(0.5 * np.sum(np.abs(first - second), axis=-1)))


def run_observation_attribution(
    config: ObservationAttributionConfig,
) -> ObservationAttributionResult:
    """Recover sensor fault, known-context, and structural-change mechanisms.

    All three exact model classes receive identical observations. True latent
    states are used only for post-hoc decoding diagnostics and never enter any
    filter update or model-evidence calculation.
    """

    config.validate()
    rng = np.random.default_rng(config.seed)
    sequences: list[ObservationSequenceResult] = []
    trials: list[dict[str, Any]] = []
    sequence_id = 0
    for kind in MECHANISM_NAMES:
        for _ in range(config.n_sequences_per_class):
            synthetic = _simulate_sequence(kind, config, rng)
            result, rows = _evaluate_sequence(sequence_id, kind, synthetic, config)
            sequences.append(result)
            trials.extend(rows)
            sequence_id += 1

    per_class_accuracy: dict[str, float] = {}
    median_evidence_margin: dict[str, float] = {}
    median_true_model_probability: dict[str, float] = {}
    confusion: dict[str, dict[str, int]] = {}
    for kind in MECHANISM_NAMES:
        selected = [sequence for sequence in sequences if sequence.true_mechanism == kind]
        per_class_accuracy[kind] = float(np.mean([sequence.correct for sequence in selected]))
        median_evidence_margin[kind] = float(
            np.median([sequence.evidence_margin for sequence in selected])
        )
        probability_field = {
            "visual_sensor_fault": "final_visual_sensor_fault_probability",
            "known_context_switch": "final_known_context_switch_probability",
            "structural_transition_change": (
                "final_structural_transition_change_probability"
            ),
        }[kind]
        median_true_model_probability[kind] = float(
            np.median([getattr(sequence, probability_field) for sequence in selected])
        )
        confusion[kind] = {
            predicted: sum(
                sequence.predicted_mechanism == predicted for sequence in selected
            )
            for predicted in MECHANISM_NAMES
        }

    _, kernels, emissions = _model_bank(config)
    forward, backward, structural = kernels
    visual_healthy, visual_fault, _ = emissions
    visual_fault_separation = _mean_total_variation(visual_healthy, visual_fault)
    structural_separation = min(
        _mean_total_variation(structural, forward),
        _mean_total_variation(structural, backward),
    )
    cue_separation = abs(2.0 * config.cue_accuracy - 1.0)
    warnings: list[str] = []
    if visual_fault_separation <= 1e-12:
        warnings.append("configured visual fault is observationally identical to healthy vision")
    if structural_separation <= 1e-12:
        warnings.append("configured structural kernel is identical to a known transition kernel")
    if cue_separation <= 1e-12:
        warnings.append("context cue is uninformative")

    balanced_accuracy = float(np.mean(list(per_class_accuracy.values())))
    summary: dict[str, Any] = {
        "experiment": "partial_observation_causal_attribution",
        "config": asdict(config),
        "n_sequences": len(sequences),
        "per_class_accuracy": per_class_accuracy,
        "balanced_accuracy": balanced_accuracy,
        "confusion_matrix": confusion,
        "median_evidence_margin": median_evidence_margin,
        "median_true_model_probability": median_true_model_probability,
        "all_sequences_correct": all(sequence.correct for sequence in sequences),
        "identifiability": {
            "mean_visual_fault_total_variation": visual_fault_separation,
            "minimum_structural_to_known_total_variation": structural_separation,
            "context_cue_total_variation": cue_separation,
            "warnings": warnings,
        },
        "interpretation": (
            "Prequential observation evidence distinguishes a persistent visual sensor fault, "
            "retrieval of a known transition context, and a preregistered structural-transition "
            "alternative without using true latent states."
        ),
        "scope": (
            "The structural model is exact for the specified candidate transition kernel. "
            "Learning an arbitrary unseen kernel under hidden-state uncertainty remains a "
            "separate open-set inference problem."
        ),
    }
    return ObservationAttributionResult(
        sequences=tuple(sequences),
        trials=tuple(trials),
        summary=summary,
    )
