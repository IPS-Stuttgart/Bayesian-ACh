"""Synthetic candidate recovery through ACh release and sensor dynamics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from bayesian_ach.measurement import convolve_by_session, double_exponential_kernel
from bayesian_ach.measurement_benchmark import (
    MEASUREMENT_CANDIDATE_NAMES,
    MEASUREMENT_NUISANCE_NAMES,
    MeasurementBenchmarkConfig,
)
from bayesian_ach.observation import MultisensoryContextFilter, MultisensoryStep


@dataclass(frozen=True, slots=True)
class _LatentMeasurementDesign:
    candidate_events: NDArray[np.float64]
    nuisance: NDArray[np.float64]
    calibration_event: NDArray[np.float64]
    subject_ids: NDArray[np.int64]
    session_ids: NDArray[np.int64]
    train_mask: NDArray[np.bool_]
    calibration_mask: NDArray[np.bool_]
    task_mask: NDArray[np.bool_]
    baseline_mask: NDArray[np.bool_]


@dataclass(frozen=True, slots=True)
class _CommonMeasurementComponents:
    common_trace: NDArray[np.float64]
    subject_amplitudes: NDArray[np.float64]


def _ring_kernel(
    n_states: int,
    offset: int,
    *,
    dominant_probability: float = 0.82,
    stay_probability: float = 0.10,
) -> NDArray[np.float64]:
    residual = 1.0 - dominant_probability - stay_probability
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
    return kernel


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
    emission = np.empty((2, n_contexts, n_states, 2), dtype=float)
    for health in range(2):
        for context, label in enumerate(labels):
            probabilities = np.array([1.0 - accuracy, 1.0 - accuracy], dtype=float)
            probabilities[label] = accuracy
            emission[health, context, :, :] = probabilities
    return emission


def _unified_filter(
    n_states: int,
) -> tuple[
    MultisensoryContextFilter,
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    forward = _ring_kernel(n_states, 1)
    backward = _ring_kernel(n_states, -1)
    structural = _ring_kernel(n_states, max(2, n_states // 2))
    transitions = np.stack((forward, backward, structural))
    context_transition = np.full((3, 3), 0.02, dtype=float)
    np.fill_diagonal(context_transition, 0.96)

    visual_healthy = _state_emission(n_states, 0.90)
    visual_fault = _state_emission(
        n_states,
        0.90,
        offset=max(2, n_states // 2),
    )
    visual = np.empty((2, 3, n_states, n_states), dtype=float)
    visual[0] = visual_healthy
    visual[1] = visual_fault
    proprioceptive = _state_emission(n_states, 0.78)
    proprioceptive_models = np.stack((proprioceptive, proprioceptive))
    cue = _cue_emission(3, n_states, 0.82, (0, 1, 0))

    health_transition = np.zeros((3, 2, 2), dtype=float)
    health_transition[0] = np.array([[0.97, 0.03], [0.08, 0.92]])
    health_transition[1] = np.array([[1.00, 0.00], [1.00, 0.00]])
    health_transition[2] = np.array([[1.00, 0.00], [1.00, 0.00]])
    initial_health = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
    filter_ = MultisensoryContextFilter(
        transitions,
        context_transition,
        (visual, proprioceptive_models, cue),
        health_transition,
        initial_context=np.array([1.0, 0.0, 0.0]),
        initial_sensor_health=initial_health,
    )
    return filter_, transitions, visual_healthy, visual_fault, proprioceptive


def _candidate_row(step: MultisensoryStep) -> NDArray[np.float64]:
    return np.asarray(
        [
            step.surprise,
            step.state_kl,
            step.context_kl,
            step.health_kl,
            step.context_switch_probability,
            step.sensor_fault_onset_probabilities[0],
            step.state_sensor_conflict_js,
        ],
        dtype=float,
    )


def _task_schedule(
    length: int,
    rng: np.random.Generator,
) -> NDArray[np.int64]:
    block_length = max(8, length // 8)
    available = length - 3 * block_length
    gap = max(1, available // 4)
    starts = (gap, 2 * gap + block_length, 3 * gap + 2 * block_length)
    labels = np.zeros(length, dtype=np.int64)
    for start, mechanism in zip(starts, rng.permutation((1, 2, 3)), strict=True):
        stop = min(length, start + block_length)
        labels[start:stop] = int(mechanism)
    return labels


def _simulate_belief_candidates(
    length: int,
    n_states: int,
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    filter_, transitions, visual_healthy, visual_fault, proprioceptive = _unified_filter(
        n_states
    )
    schedule = _task_schedule(length, rng)
    state = int(rng.integers(n_states))
    initial_cue = np.array([0.82, 0.18], dtype=float)
    initial_observations = (
        int(rng.choice(n_states, p=visual_healthy[state])),
        int(rng.choice(n_states, p=proprioceptive[state])),
        int(rng.choice(2, p=initial_cue)),
    )
    rows = np.zeros((length, len(MEASUREMENT_CANDIDATE_NAMES)), dtype=float)
    rows[0] = _candidate_row(filter_.initialize(initial_observations))

    for time_index in range(1, length):
        mechanism = int(schedule[time_index])
        context = 0
        visual_is_faulty = False
        cue_label = 0
        if mechanism == 1:
            visual_is_faulty = True
        elif mechanism == 2:
            context = 1
            cue_label = 1
        elif mechanism == 3:
            context = 2

        state = int(rng.choice(n_states, p=transitions[context, state]))
        visual_model = visual_fault if visual_is_faulty else visual_healthy
        cue_probabilities = np.array([0.18, 0.18], dtype=float)
        cue_probabilities[cue_label] = 0.82
        observations = (
            int(rng.choice(n_states, p=visual_model[state])),
            int(rng.choice(n_states, p=proprioceptive[state])),
            int(rng.choice(2, p=cue_probabilities)),
        )
        rows[time_index] = _candidate_row(filter_.step(observations))
    return rows


def _calibration_pulses(length: int) -> NDArray[np.float64]:
    baseline_length = max(8, length // 6)
    reference_positions = np.asarray((12, 19, 30, 36, 49, 58, 71), dtype=int)
    positions = np.unique(reference_positions * length // 80)
    positions = positions[(positions >= baseline_length) & (positions < length - 3)]
    pulses = np.zeros(length, dtype=float)
    pulses[positions] = 1.0
    return pulses


def _latent_design(config: MeasurementBenchmarkConfig) -> _LatentMeasurementDesign:
    rng = np.random.default_rng(config.seed)
    n_sessions = config.n_subjects * config.sessions_per_subject
    samples_per_session = config.calibration_length + config.task_length
    n_samples = n_sessions * samples_per_session
    subject_ids = np.repeat(
        np.arange(config.n_subjects, dtype=np.int64),
        config.sessions_per_subject * samples_per_session,
    )
    session_ids = np.repeat(
        np.arange(n_sessions, dtype=np.int64),
        samples_per_session,
    )
    candidate_events = np.zeros(
        (n_samples, len(MEASUREMENT_CANDIDATE_NAMES)),
        dtype=float,
    )
    nuisance = np.zeros((n_samples, len(MEASUREMENT_NUISANCE_NAMES)), dtype=float)
    calibration_event = np.zeros(n_samples, dtype=float)
    calibration_mask = np.zeros(n_samples, dtype=bool)
    task_mask = np.zeros(n_samples, dtype=bool)
    baseline_mask = np.zeros(n_samples, dtype=bool)
    train_sessions = np.zeros(n_sessions, dtype=bool)
    baseline_length = max(8, config.calibration_length // 6)

    for subject in range(config.n_subjects):
        sessions = np.arange(
            subject * config.sessions_per_subject,
            (subject + 1) * config.sessions_per_subject,
        )
        order = rng.permutation(sessions)
        train_sessions[order[: config.train_sessions_per_subject]] = True

    pulses = _calibration_pulses(config.calibration_length)
    for session in range(n_sessions):
        indices = np.flatnonzero(session_ids == session)
        calibration_indices = indices[: config.calibration_length]
        task_indices = indices[config.calibration_length :]
        calibration_mask[calibration_indices] = True
        task_mask[task_indices] = True
        baseline_mask[calibration_indices[:baseline_length]] = True
        calibration_event[calibration_indices] = pulses

        candidates = _simulate_belief_candidates(
            config.task_length,
            4,
            rng,
        )
        candidate_events[task_indices] = candidates

        movement = np.zeros(samples_per_session, dtype=float)
        pupil = np.zeros(samples_per_session, dtype=float)
        theta = np.zeros(samples_per_session, dtype=float)
        engagement = np.zeros(samples_per_session, dtype=float)
        for time_index in range(1, samples_per_session):
            movement[time_index] = 0.92 * movement[time_index - 1] + rng.normal(0.0, 0.30)
            pupil[time_index] = 0.97 * pupil[time_index - 1] + rng.normal(0.0, 0.15)
            theta[time_index] = 0.85 * theta[time_index - 1] + rng.normal(0.0, 0.25)
            engagement[time_index] = (
                0.995 * engagement[time_index - 1] + rng.normal(0.0, 0.05)
            )
        movement[config.calibration_length :] += (
            0.20 * candidates[:, 0] + 0.15 * candidates[:, 4]
        )
        pupil[config.calibration_length :] += (
            0.30 * candidates[:, 0] + 0.20 * candidates[:, 3]
        )
        theta[config.calibration_length :] += 0.25 * candidates[:, 2]
        engagement[config.calibration_length :] += (
            0.10
            * np.cumsum(candidates[:, 5] - float(np.mean(candidates[:, 5])))
            / config.task_length
        )
        acceleration = np.concatenate((np.zeros(1), np.diff(movement)))
        nuisance[indices] = np.column_stack(
            (movement, acceleration, pupil, theta, engagement)
        )

    train_mask = train_sessions[session_ids]
    return _LatentMeasurementDesign(
        candidate_events=candidate_events,
        nuisance=nuisance,
        calibration_event=calibration_event,
        subject_ids=subject_ids,
        session_ids=session_ids,
        train_mask=train_mask,
        calibration_mask=calibration_mask,
        task_mask=task_mask,
        baseline_mask=baseline_mask,
    )


def _common_measurement_components(
    design: _LatentMeasurementDesign,
    config: MeasurementBenchmarkConfig,
) -> _CommonMeasurementComponents:
    rng = np.random.default_rng(config.seed + 7919)
    kernel = double_exponential_kernel(
        config.dt,
        config.tau_rise,
        config.tau_decay,
    )
    calibration = convolve_by_session(
        design.calibration_event,
        design.session_ids,
        kernel,
    )
    subject_amplitudes = config.phasic_amplitude + rng.normal(
        0.0,
        config.subject_amplitude_std,
        size=config.n_subjects,
    )
    n_sessions = config.n_subjects * config.sessions_per_subject
    subject_baselines = rng.normal(
        0.0,
        config.subject_baseline_std,
        size=config.n_subjects,
    )
    session_baselines = rng.normal(
        0.0,
        config.session_baseline_std,
        size=n_sessions,
    )
    rise_decay = float(np.exp(-config.dt / config.tau_rise))
    slow_decay = float(np.exp(-config.dt / config.tau_decay))
    unscaled_kernel = np.exp(
        -np.arange(kernel.size, dtype=float) * config.dt / config.tau_decay
    ) - np.exp(
        -np.arange(kernel.size, dtype=float) * config.dt / config.tau_rise
    )
    sensor_scale = 1.0 / float(np.max(unscaled_kernel))
    tonic_sensor = np.zeros(design.subject_ids.size, dtype=float)
    initial_release_std = (
        config.tonic_innovation_std / np.sqrt(1.0 - config.tonic_rho**2)
        if config.tonic_rho > 0.0
        else config.tonic_innovation_std
    )
    for session in range(n_sessions):
        indices = np.flatnonzero(design.session_ids == session)
        tonic_release = rng.normal(0.0, initial_release_std)
        rise_state = 0.0
        decay_state = 0.0
        for index in indices:
            rise_state = rise_decay * rise_state + tonic_release
            decay_state = slow_decay * decay_state + tonic_release
            tonic_sensor[index] = sensor_scale * (decay_state - rise_state)
            tonic_release = (
                config.tonic_rho * tonic_release
                + rng.normal(0.0, config.tonic_innovation_std)
            )
    nuisance_coefficients = np.asarray(config.nuisance_coefficients, dtype=float)
    common_trace = (
        subject_baselines[design.subject_ids]
        + session_baselines[design.session_ids]
        + config.calibration_amplitude * calibration
        + design.nuisance @ nuisance_coefficients
        + tonic_sensor
    )
    return _CommonMeasurementComponents(
        common_trace=np.asarray(common_trace, dtype=float),
        subject_amplitudes=np.asarray(subject_amplitudes, dtype=float),
    )


def _observed_trace(
    design: _LatentMeasurementDesign,
    generator_index: int,
    config: MeasurementBenchmarkConfig,
    common: _CommonMeasurementComponents,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    kernel = double_exponential_kernel(
        config.dt,
        config.tau_rise,
        config.tau_decay,
    )
    task_values = design.candidate_events[:, generator_index]
    task_scale = max(
        float(np.std(task_values[design.task_mask & design.train_mask])),
        1e-12,
    )
    phasic = convolve_by_session(
        task_values / task_scale,
        design.session_ids,
        kernel,
    )
    true_raw_subject_coefficients = common.subject_amplitudes / task_scale
    observed = common.common_trace + common.subject_amplitudes[design.subject_ids] * phasic
    return np.asarray(observed, dtype=float), true_raw_subject_coefficients


def _safe_correlation(first: NDArray[np.float64], second: NDArray[np.float64]) -> float:
    if first.size < 2 or float(np.std(first)) <= 0.0 or float(np.std(second)) <= 0.0:
        return float("nan")
    return float(np.corrcoef(first, second)[0, 1])
