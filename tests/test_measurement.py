import numpy as np
import pytest

from bayesian_ach.measurement import (
    MeasurementDataset,
    MeasurementFitConfig,
    MeasurementGridPoint,
    convolve_by_session,
    double_exponential_kernel,
    fit_measurement_models,
    tonic_sensor_ar_coefficients,
)


def test_double_exponential_kernel_and_session_boundaries() -> None:
    kernel = double_exponential_kernel(0.1, 0.3, 1.2)
    assert kernel[0] == pytest.approx(0.0)
    assert np.max(kernel) == pytest.approx(1.0)
    assert np.all(kernel >= 0.0)

    events = np.array([1.0, 0.0, 0.0, 1.0, 0.0, 0.0])
    sessions = np.array([0, 0, 0, 1, 1, 1])
    filtered = convolve_by_session(events, sessions, kernel)
    np.testing.assert_allclose(filtered[:3], filtered[3:])


def test_conditional_ar3_matches_filtered_tonic_release() -> None:
    dt = 0.2
    point = MeasurementGridPoint(0.4, 1.6, 0.97)
    first, second, third = tonic_sensor_ar_coefficients(point, dt)
    rise_decay = np.exp(-dt / point.tau_rise)
    slow_decay = np.exp(-dt / point.tau_decay)
    kernel = double_exponential_kernel(dt, point.tau_rise, point.tau_decay)
    unscaled = (
        np.exp(-np.arange(kernel.size) * dt / point.tau_decay)
        - np.exp(-np.arange(kernel.size) * dt / point.tau_rise)
    )
    scale = 1.0 / np.max(unscaled)

    rng = np.random.default_rng(3)
    release = 0.0
    rise_state = 0.0
    decay_state = 0.0
    sensor = []
    innovations = []
    for _ in range(1000):
        rise_state = rise_decay * rise_state + release
        decay_state = slow_decay * decay_state + release
        sensor.append(scale * (decay_state - rise_state))
        innovation = rng.normal(0.0, 0.2)
        innovations.append(innovation)
        release = point.tonic_rho * release + innovation

    values = np.asarray(sensor)
    whitened = (
        values[3:]
        - first * values[2:-1]
        + second * values[1:-2]
        - third * values[:-3]
    )
    correlation = np.corrcoef(whitened, np.asarray(innovations)[1:-2])[0, 1]
    assert correlation > 0.999999


def _small_dataset(test_offset: float = 0.0) -> MeasurementDataset:
    rng = np.random.default_rng(9)
    n_sessions = 4
    length = 48
    sessions = np.repeat(np.arange(n_sessions), length)
    subjects = np.repeat(np.array([0, 0, 1, 1]), length)
    train_sessions = np.array([True, False, True, False])
    train = train_sessions[sessions]
    calibration = np.zeros(sessions.size, dtype=bool)
    task = np.zeros(sessions.size, dtype=bool)
    baseline = np.zeros(sessions.size, dtype=bool)
    calibration_event = np.zeros(sessions.size)
    candidates = np.zeros((sessions.size, 2))
    nuisance = rng.normal(size=(sessions.size, 1))
    kernel = double_exponential_kernel(0.2, 0.4, 1.6)

    for session in range(n_sessions):
        indices = np.flatnonzero(sessions == session)
        calibration[indices[:24]] = True
        task[indices[24:]] = True
        baseline[indices[:8]] = True
        calibration_event[indices[[10, 16, 21]]] = 1.0
        candidates[indices[24:], 0] = rng.gamma(1.5, 0.5, size=24)
        candidates[indices[24:], 1] = rng.gamma(1.5, 0.5, size=24)

    signal = convolve_by_session(candidates[:, 0], sessions, kernel)
    calibration_trace = convolve_by_session(calibration_event, sessions, kernel)
    observed = signal + 1.5 * calibration_trace + 0.2 * nuisance[:, 0]
    observed += rng.normal(0.0, 0.05, size=observed.size)
    observed[(~train) & task] += test_offset
    return MeasurementDataset(
        observed=observed,
        calibration_event=calibration_event,
        candidate_events=candidates,
        nuisance=nuisance,
        subject_ids=subjects,
        session_ids=sessions,
        train_mask=train,
        calibration_mask=calibration,
        task_mask=task,
        baseline_mask=baseline,
        candidate_names=("candidate_a", "candidate_b"),
        nuisance_names=("movement",),
    )


def test_calibration_posterior_does_not_use_held_out_task() -> None:
    grid = (
        MeasurementGridPoint(0.25, 1.0, 0.92),
        MeasurementGridPoint(0.4, 1.6, 0.97),
    )
    config = MeasurementFitConfig(dt=0.2, grid=grid)
    original = fit_measurement_models(_small_dataset(), config)
    changed = fit_measurement_models(_small_dataset(test_offset=100.0), config)
    np.testing.assert_allclose(
        original.calibration.log_weights,
        changed.calibration.log_weights,
        atol=0.0,
        rtol=0.0,
    )
