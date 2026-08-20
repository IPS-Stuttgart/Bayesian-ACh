"""Training-only calibration and held-out ACh measurement-model fitting."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.special import logsumexp

from bayesian_ach.measurement import (
    CalibrationPosterior,
    MeasurementCandidateFit,
    MeasurementDataset,
    MeasurementFitConfig,
    MeasurementGridPoint,
    MeasurementRecoveryResult,
    convolve_by_session,
    double_exponential_kernel,
    tonic_sensor_ar_coefficients,
)


@dataclass(frozen=True, slots=True)
class _DesignStats:
    signal_mean: float
    signal_std: float
    nuisance_mean: NDArray[np.float64]
    nuisance_std: NDArray[np.float64]
    subject_levels: NDArray[np.int64]


@dataclass(frozen=True, slots=True)
class _GridRegression:
    coefficients: NDArray[np.float64]
    stats: _DesignStats
    innovation_variance: float
    train_log_likelihood: float


def _baseline_correct(
    values: NDArray[np.float64],
    session_ids: NDArray[np.int64],
    baseline_mask: NDArray[np.bool_],
) -> NDArray[np.float64]:
    result = values.copy()
    for session in np.unique(session_ids):
        session_mask = session_ids == session
        selected = session_mask & baseline_mask
        if not np.any(selected):
            raise ValueError(f"session {int(session)} has no baseline samples")
        baseline = np.mean(values[selected], axis=0)
        result[session_mask] -= baseline
    return result


def _build_design(
    signal: NDArray[np.float64],
    nuisance: NDArray[np.float64],
    subjects: NDArray[np.int64],
    *,
    subject_penalty: float,
    stats: _DesignStats | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], _DesignStats]:
    if stats is None:
        signal_mean = float(np.mean(signal))
        signal_std = float(np.std(signal))
        nuisance_mean = np.mean(nuisance, axis=0)
        nuisance_std = np.std(nuisance, axis=0)
        subject_levels = np.unique(subjects)
        stats = _DesignStats(
            signal_mean=signal_mean,
            signal_std=max(signal_std, 1e-12),
            nuisance_mean=np.asarray(nuisance_mean, dtype=float),
            nuisance_std=np.maximum(np.asarray(nuisance_std, dtype=float), 1e-12),
            subject_levels=np.asarray(subject_levels, dtype=np.int64),
        )
    unknown = sorted(
        set(int(value) for value in subjects)
        - set(int(value) for value in stats.subject_levels)
    )
    if unknown:
        raise ValueError(f"subjects were not represented in training data: {unknown}")

    standardized_signal = (signal - stats.signal_mean) / stats.signal_std
    standardized_nuisance = (nuisance - stats.nuisance_mean) / stats.nuisance_std
    n_samples = signal.size
    n_subjects = stats.subject_levels.size
    subject_intercepts = np.zeros((n_samples, n_subjects), dtype=float)
    subject_slopes = np.zeros((n_samples, n_subjects), dtype=float)
    level_to_column = {int(level): index for index, level in enumerate(stats.subject_levels)}
    for row_index, subject in enumerate(subjects):
        column = level_to_column[int(subject)]
        subject_intercepts[row_index, column] = 1.0
        subject_slopes[row_index, column] = standardized_signal[row_index]

    design = np.column_stack(
        (
            np.ones(n_samples, dtype=float),
            standardized_signal,
            standardized_nuisance,
            subject_intercepts,
            subject_slopes,
        )
    )
    penalty = np.zeros(design.shape[1], dtype=float)
    fixed_count = 2 + nuisance.shape[1]
    penalty[fixed_count:] = subject_penalty
    return design, penalty, stats


def _innovations_transform(
    values: NDArray[np.float64],
    session_ids: NDArray[np.int64],
    point: MeasurementGridPoint,
    *,
    dt: float,
) -> NDArray[np.float64]:
    matrix = values[:, None] if values.ndim == 1 else values
    if matrix.ndim != 2 or matrix.shape[0] != session_ids.size:
        raise ValueError("values must have one row per session identifier")
    first, second, third = tonic_sensor_ar_coefficients(point, dt)
    transformed_sessions: list[NDArray[np.float64]] = []
    for session in np.unique(session_ids):
        indices = np.flatnonzero(session_ids == session)
        if indices.size <= 3:
            raise ValueError("every fitted session segment must contain at least four samples")
        session_values = matrix[indices]
        transformed_sessions.append(
            session_values[3:]
            - first * session_values[2:-1]
            + second * session_values[1:-2]
            - third * session_values[:-3]
        )
    return np.concatenate(transformed_sessions, axis=0)


def _conditional_sample_count(session_ids: NDArray[np.int64]) -> int:
    return int(
        sum(
            max(0, np.sum(session_ids == session) - 3)
            for session in np.unique(session_ids)
        )
    )


def _ridge_solution(
    response: NDArray[np.float64],
    design: NDArray[np.float64],
    penalty: NDArray[np.float64],
) -> NDArray[np.float64]:
    precision = design.T @ design + np.diag(penalty)
    target = design.T @ response
    try:
        return np.asarray(np.linalg.solve(precision, target), dtype=float)
    except np.linalg.LinAlgError:
        return np.asarray(np.linalg.lstsq(precision, target, rcond=None)[0], dtype=float)


def _innovation_log_likelihood(
    transformed_residual: NDArray[np.float64],
    innovation_variance_scale: float,
) -> float:
    if innovation_variance_scale <= 0.0 or not np.isfinite(innovation_variance_scale):
        raise ValueError("innovation_variance_scale must be finite and positive")
    return float(
        -0.5
        * (
            transformed_residual.size
            * np.log(2.0 * np.pi * innovation_variance_scale)
            + float(np.dot(transformed_residual, transformed_residual))
            / innovation_variance_scale
        )
    )


def _fit_regression(
    response: NDArray[np.float64],
    signal: NDArray[np.float64],
    nuisance: NDArray[np.float64],
    subjects: NDArray[np.int64],
    sessions: NDArray[np.int64],
    *,
    point: MeasurementGridPoint,
    dt: float,
    subject_penalty: float,
    variance_floor: float,
) -> _GridRegression:
    design, penalty, stats = _build_design(
        signal,
        nuisance,
        subjects,
        subject_penalty=subject_penalty,
    )
    transformed = _innovations_transform(
        np.column_stack((response, design)),
        sessions,
        point,
        dt=dt,
    )
    whitened_response = transformed[:, 0]
    whitened_design = transformed[:, 1:]
    coefficients = _ridge_solution(whitened_response, whitened_design, penalty)
    transformed_residual = whitened_response - whitened_design @ coefficients
    innovation_variance = max(
        float(np.mean(transformed_residual**2)),
        variance_floor,
    )
    train_log_likelihood = _innovation_log_likelihood(
        transformed_residual,
        innovation_variance,
    )
    return _GridRegression(
        coefficients=coefficients,
        stats=stats,
        innovation_variance=innovation_variance,
        train_log_likelihood=train_log_likelihood,
    )


def _evaluate_regression(
    regression: _GridRegression,
    response: NDArray[np.float64],
    signal: NDArray[np.float64],
    nuisance: NDArray[np.float64],
    subjects: NDArray[np.int64],
    sessions: NDArray[np.int64],
    *,
    point: MeasurementGridPoint,
    dt: float,
    subject_penalty: float,
) -> tuple[float, float, NDArray[np.float64]]:
    design, _, _ = _build_design(
        signal,
        nuisance,
        subjects,
        subject_penalty=subject_penalty,
        stats=regression.stats,
    )
    transformed = _innovations_transform(
        np.column_stack((response, design)),
        sessions,
        point,
        dt=dt,
    )
    transformed_residual = transformed[:, 0] - transformed[:, 1:] @ regression.coefficients
    log_likelihood = _innovation_log_likelihood(
        transformed_residual,
        regression.innovation_variance,
    )
    denominator = float(
        np.sum((transformed[:, 0] - float(np.mean(transformed[:, 0]))) ** 2)
    )
    r2 = (
        float("nan")
        if denominator <= 0.0
        else 1.0 - float(np.sum(transformed_residual**2)) / denominator
    )
    prediction = design @ regression.coefficients
    return log_likelihood, r2, prediction


def _calibration_posterior(
    dataset: MeasurementDataset,
    config: MeasurementFitConfig,
    response: NDArray[np.float64],
    nuisance: NDArray[np.float64],
) -> CalibrationPosterior:
    mask = dataset.train_mask & dataset.calibration_mask
    log_likelihoods: list[float] = []
    for point in config.grid:
        kernel = double_exponential_kernel(
            config.dt,
            point.tau_rise,
            point.tau_decay,
        )
        signal = convolve_by_session(
            dataset.calibration_event,
            dataset.session_ids,
            kernel,
        )
        regression = _fit_regression(
            response[mask],
            signal[mask],
            nuisance[mask],
            dataset.subject_ids[mask],
            dataset.session_ids[mask],
            point=point,
            dt=config.dt,
            subject_penalty=config.subject_penalty,
            variance_floor=config.variance_floor,
        )
        log_likelihoods.append(regression.train_log_likelihood)
    values = np.asarray(log_likelihoods, dtype=float)
    log_weights = values - float(logsumexp(values))
    weights = np.exp(log_weights)
    map_index = int(np.argmax(log_weights))
    weighted_rho = float(
        np.dot(weights, np.asarray([point.tonic_rho for point in config.grid]))
    )
    weighted_timescale = (
        0.0
        if weighted_rho == 0.0
        else float(-config.dt / np.log(weighted_rho))
    )
    return CalibrationPosterior(
        points=config.grid,
        log_weights=log_weights,
        training_log_likelihoods=values,
        map_index=map_index,
        weighted_tau_rise=float(
            np.dot(weights, np.asarray([point.tau_rise for point in config.grid]))
        ),
        weighted_tau_decay=float(
            np.dot(weights, np.asarray([point.tau_decay for point in config.grid]))
        ),
        weighted_tonic_rho=weighted_rho,
        weighted_tonic_timescale=weighted_timescale,
        effective_grid_size=float(1.0 / np.sum(weights**2)),
    )


def fit_measurement_models(
    dataset: MeasurementDataset,
    config: MeasurementFitConfig | None = None,
) -> MeasurementRecoveryResult:
    """Fit all event candidates and rank them by held-out predictive likelihood.

    The sensor-kernel and tonic-release posterior is learned exclusively from the
    training calibration block. Candidate coefficients are learned on training
    task samples, and final scores use held-out task samples only.
    """

    selected_config = MeasurementFitConfig() if config is None else config
    selected_config.validate()
    dataset.validate()
    response = _baseline_correct(
        np.asarray(dataset.observed, dtype=float),
        np.asarray(dataset.session_ids, dtype=np.int64),
        np.asarray(dataset.baseline_mask, dtype=bool),
    )
    nuisance = _baseline_correct(
        np.asarray(dataset.nuisance, dtype=float),
        np.asarray(dataset.session_ids, dtype=np.int64),
        np.asarray(dataset.baseline_mask, dtype=bool),
    )
    calibration = _calibration_posterior(
        dataset,
        selected_config,
        response,
        nuisance,
    )
    train_mask = dataset.train_mask & dataset.task_mask
    test_mask = (~dataset.train_mask) & dataset.task_mask
    grid_test_log_likelihoods = np.empty(
        (dataset.n_candidates, len(selected_config.grid)),
        dtype=float,
    )
    unsorted_fits: list[MeasurementCandidateFit] = []

    for candidate_index, candidate_name in enumerate(dataset.candidate_names):
        event_train = np.asarray(dataset.candidate_events[:, candidate_index], dtype=float)
        grid_regressions: list[_GridRegression] = []
        grid_r2: list[float] = []
        for grid_index, point in enumerate(selected_config.grid):
            kernel = double_exponential_kernel(
                selected_config.dt,
                point.tau_rise,
                point.tau_decay,
            )
            signal = convolve_by_session(event_train, dataset.session_ids, kernel)
            regression = _fit_regression(
                response[train_mask],
                signal[train_mask],
                nuisance[train_mask],
                dataset.subject_ids[train_mask],
                dataset.session_ids[train_mask],
                point=point,
                dt=selected_config.dt,
                subject_penalty=selected_config.subject_penalty,
                variance_floor=selected_config.variance_floor,
            )
            test_log_likelihood, test_r2, _ = _evaluate_regression(
                regression,
                response[test_mask],
                signal[test_mask],
                nuisance[test_mask],
                dataset.subject_ids[test_mask],
                dataset.session_ids[test_mask],
                point=point,
                dt=selected_config.dt,
                subject_penalty=selected_config.subject_penalty,
            )
            grid_test_log_likelihoods[candidate_index, grid_index] = test_log_likelihood
            grid_regressions.append(regression)
            grid_r2.append(test_r2)

        marginal = float(
            logsumexp(
                calibration.log_weights
                + grid_test_log_likelihoods[candidate_index]
            )
        )
        map_index = calibration.map_index
        map_regression = grid_regressions[map_index]
        coefficients = map_regression.coefficients
        stats = map_regression.stats
        fixed_count = 2 + dataset.n_nuisance
        n_subjects = stats.subject_levels.size
        subject_deviations = coefficients[
            fixed_count + n_subjects : fixed_count + 2 * n_subjects
        ]
        subject_signal = (coefficients[1] + subject_deviations) / stats.signal_std
        nuisance_coefficients = (
            coefficients[2 : 2 + dataset.n_nuisance] / stats.nuisance_std
        )
        unsorted_fits.append(
            MeasurementCandidateFit(
                candidate=candidate_name,
                marginal_test_log_likelihood=marginal,
                test_mean_log_likelihood=(
                    marginal / _conditional_sample_count(dataset.session_ids[test_mask])
                ),
                map_test_log_likelihood=float(
                    grid_test_log_likelihoods[candidate_index, map_index]
                ),
                map_test_innovation_r2=float(grid_r2[map_index]),
                global_signal_coefficient=float(coefficients[1] / stats.signal_std),
                nuisance_coefficients=np.asarray(nuisance_coefficients, dtype=float),
                subject_signal_coefficients=np.asarray(subject_signal, dtype=float),
                residual_innovation_std=float(
                    np.sqrt(map_regression.innovation_variance)
                ),
                n_train=_conditional_sample_count(dataset.session_ids[train_mask]),
                n_test=_conditional_sample_count(dataset.session_ids[test_mask]),
            )
        )

    fits = tuple(
        sorted(
            unsorted_fits,
            key=lambda fit: fit.marginal_test_log_likelihood,
            reverse=True,
        )
    )
    return MeasurementRecoveryResult(
        calibration=calibration,
        fits=fits,
        candidate_names=dataset.candidate_names,
        nuisance_names=dataset.nuisance_names,
        grid_test_log_likelihoods=grid_test_log_likelihoods,
    )
