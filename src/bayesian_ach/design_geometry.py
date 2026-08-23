"""Finite-design covariance and expected evidence geometry."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class DesignDiagnostics:
    """Geometry of one integer trial allocation."""

    trial_count: int
    support_size: int
    minimum_candidate_variance: float
    maximum_absolute_correlation: float
    minimum_pairwise_residual_variance: float
    covariance_condition_number: float
    covariance_log_determinant: float
    expected_log_bf_per_trial: float
    trials_for_expected_log_bf_target: int

    def as_dict(self) -> dict[str, float | int]:
        return asdict(self)


def moments_from_counts(
    signals: NDArray[np.float64],
    counts: NDArray[np.int64],
) -> tuple[int, NDArray[np.float64], NDArray[np.float64]]:
    """Return trial count, first moments, and uncentred second moments."""

    trial_count = int(np.sum(counts))
    if trial_count < 1:
        width = signals.shape[1]
        return 0, np.zeros(width), np.zeros((width, width))
    weights = counts.astype(float)
    return (
        trial_count,
        weights @ signals,
        np.einsum("i,ij,ik->jk", weights, signals, signals),
    )


def covariance_from_moments(
    trial_count: int,
    sums: NDArray[np.float64],
    second: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Return the population covariance represented by sufficient moments."""

    if trial_count < 1:
        return np.zeros_like(second)
    mean = sums / trial_count
    covariance = second / trial_count - np.outer(mean, mean)
    return 0.5 * (covariance + covariance.T)


def pairwise_residuals_from_covariance(
    covariance: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Return ordered generator-versus-alternative projection residuals."""

    variance = np.maximum(np.diag(covariance), 0.0)
    result = np.zeros_like(covariance)
    for generator in range(covariance.shape[0]):
        for alternative in range(covariance.shape[1]):
            if generator == alternative:
                continue
            denominator = float(variance[alternative])
            if denominator > 1e-15:
                result[generator, alternative] = max(
                    0.0,
                    float(variance[generator])
                    - float(covariance[generator, alternative]) ** 2 / denominator,
                )
    return result


def diagnostics_from_covariance(
    covariance: NDArray[np.float64],
    *,
    trial_count: int,
    support_size: int,
    effect_size: float,
    noise_std: float,
    target_log_bf: float,
) -> DesignDiagnostics:
    """Summarize identifiability and Gaussian expected-evidence geometry."""

    if trial_count < 1:
        return DesignDiagnostics(
            0, 0, 0.0, 1.0, 0.0, math.inf, -math.inf, 0.0, 2**31 - 1
        )
    if noise_std <= 0.0:
        raise ValueError("noise_std must be positive")
    variance = np.maximum(np.diag(covariance), 0.0)
    residuals = pairwise_residuals_from_covariance(covariance)
    off_diagonal = ~np.eye(covariance.shape[0], dtype=bool)
    minimum_residual = float(np.min(residuals[off_diagonal]))
    denominator = np.sqrt(np.outer(variance, variance))
    correlations = np.divide(
        covariance,
        denominator,
        out=np.zeros_like(covariance),
        where=denominator > 1e-15,
    )
    maximum_correlation = float(np.max(np.abs(correlations[off_diagonal])))
    eigenvalues = np.maximum(np.linalg.eigvalsh(covariance), 1e-12)
    expected_per_trial = effect_size**2 * minimum_residual / (2.0 * noise_std**2)
    required = (
        2**31 - 1
        if expected_per_trial <= 1e-15
        else int(math.ceil(target_log_bf / expected_per_trial))
    )
    return DesignDiagnostics(
        trial_count=trial_count,
        support_size=support_size,
        minimum_candidate_variance=float(np.min(variance)),
        maximum_absolute_correlation=maximum_correlation,
        minimum_pairwise_residual_variance=minimum_residual,
        covariance_condition_number=float(np.max(eigenvalues) / np.min(eigenvalues)),
        covariance_log_determinant=float(np.sum(np.log(eigenvalues))),
        expected_log_bf_per_trial=expected_per_trial,
        trials_for_expected_log_bf_target=required,
    )


def design_diagnostics(
    standardized_signals: NDArray[np.float64],
    counts: NDArray[np.int64],
    *,
    effect_size: float = 1.0,
    noise_std: float = 1.0,
    target_log_bf: float = 5.0,
) -> DesignDiagnostics:
    """Evaluate one allocation without expanding individual trials."""

    signals = np.asarray(standardized_signals, dtype=float)
    allocation = np.asarray(counts, dtype=np.int64)
    if signals.ndim != 2 or signals.shape[1] < 2 or not np.all(np.isfinite(signals)):
        raise ValueError("standardized_signals must be a finite two-dimensional matrix")
    if allocation.shape != (signals.shape[0],) or np.any(allocation < 0):
        raise ValueError("counts must be a non-negative vector matching the grid")
    trial_count, sums, second = moments_from_counts(signals, allocation)
    covariance = covariance_from_moments(trial_count, sums, second)
    return diagnostics_from_covariance(
        covariance,
        trial_count=trial_count,
        support_size=int(np.sum(allocation > 0)),
        effect_size=effect_size,
        noise_std=noise_std,
        target_log_bf=target_log_bf,
    )


def pairwise_residual_matrix(
    standardized_signals: NDArray[np.float64],
    counts: NDArray[np.int64],
) -> NDArray[np.float64]:
    """Return ordered generator-versus-alternative residual variances."""

    trial_count, sums, second = moments_from_counts(standardized_signals, counts)
    return pairwise_residuals_from_covariance(
        covariance_from_moments(trial_count, sums, second)
    )
