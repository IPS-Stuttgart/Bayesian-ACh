"""Held-out model recovery for competing scalar ACh hypotheses."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from bayesian_ach.signals import CANDIDATE_SIGNAL_NAMES


@dataclass(frozen=True, slots=True)
class CandidateFit:
    """Held-out statistics for one univariate candidate model."""

    candidate: str
    intercept: float
    slope: float
    train_feature_mean: float
    train_feature_std: float
    residual_std: float
    test_log_likelihood: float
    test_mean_log_likelihood: float
    test_r2: float
    n_train: int
    n_test: int

    def as_dict(self) -> dict[str, float | int | str]:
        return asdict(self)


def _feature(rows: Sequence[Mapping[str, Any]], name: str) -> NDArray[np.float64]:
    try:
        values = np.asarray([float(row[name]) for row in rows], dtype=float)
    except KeyError as exc:
        raise KeyError(f"candidate feature {name!r} is missing from at least one row") from exc
    if values.ndim != 1 or not np.all(np.isfinite(values)):
        raise ValueError(f"candidate feature {name!r} must be a finite one-dimensional vector")
    return values


def generate_synthetic_ach(
    rows: Sequence[Mapping[str, Any]],
    generator: str,
    *,
    effect_size: float = 1.0,
    noise_std: float = 0.25,
    baseline: float = 0.0,
    seed: int = 7,
) -> NDArray[np.float64]:
    """Generate synthetic ACh amplitudes from one candidate regressor.

    The generating feature is standardized over the supplied design, making
    ``effect_size`` comparable across candidate units.
    """

    if generator not in CANDIDATE_SIGNAL_NAMES:
        raise ValueError(f"unknown generator {generator!r}; choose from {CANDIDATE_SIGNAL_NAMES}")
    if not np.isfinite(effect_size):
        raise ValueError("effect_size must be finite")
    if not np.isfinite(noise_std) or noise_std <= 0.0:
        raise ValueError("noise_std must be finite and positive")
    if not rows:
        raise ValueError("rows must not be empty")

    x = _feature(rows, generator)
    scale = float(np.std(x))
    if scale <= 0.0:
        raise ValueError(f"generator {generator!r} is constant in this design")
    z = (x - float(np.mean(x))) / scale
    rng = np.random.default_rng(seed)
    noise = rng.normal(loc=0.0, scale=noise_std, size=len(rows))
    return np.asarray(baseline + effect_size * z + noise, dtype=float)


def _split_indices(
    n_rows: int,
    test_fraction: float,
    seed: int,
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    if n_rows < 4:
        raise ValueError("at least four rows are required for held-out model fitting")
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must lie strictly between zero and one")
    rng = np.random.default_rng(seed)
    order = np.asarray(rng.permutation(n_rows), dtype=np.int64)
    n_test = min(n_rows - 2, max(2, int(round(test_fraction * n_rows))))
    return order[n_test:], order[:n_test]


def fit_candidate_models(
    rows: Sequence[Mapping[str, Any]],
    ach: Sequence[float] | NDArray[np.float64],
    *,
    candidates: Iterable[str] = CANDIDATE_SIGNAL_NAMES,
    test_fraction: float = 0.25,
    seed: int = 19,
) -> list[CandidateFit]:
    """Fit each scalar candidate by train-only standardization and OLS.

    Models are ranked by Gaussian held-out log likelihood. The residual variance
    is estimated on the training data with a conservative floor.
    """

    y = np.asarray(ach, dtype=float)
    if y.ndim != 1 or y.size != len(rows):
        raise ValueError(f"ach must have shape ({len(rows)},); got {y.shape}")
    if not np.all(np.isfinite(y)):
        raise ValueError("ach must contain only finite values")

    train_idx, test_idx = _split_indices(len(rows), test_fraction, seed)
    y_train = y[train_idx]
    y_test = y[test_idx]
    fits: list[CandidateFit] = []

    for candidate in candidates:
        if candidate not in CANDIDATE_SIGNAL_NAMES:
            raise ValueError(f"unknown candidate {candidate!r}")
        x = _feature(rows, candidate)
        x_train_raw = x[train_idx]
        x_test_raw = x[test_idx]
        mean = float(np.mean(x_train_raw))
        std = float(np.std(x_train_raw))
        if std <= 1e-15:
            raise ValueError(f"candidate {candidate!r} is constant in the training split")
        x_train = (x_train_raw - mean) / std
        x_test = (x_test_raw - mean) / std

        design_train = np.column_stack((np.ones_like(x_train), x_train))
        coefficients, _, _, _ = np.linalg.lstsq(design_train, y_train, rcond=None)
        intercept = float(coefficients[0])
        slope = float(coefficients[1])

        train_prediction = intercept + slope * x_train
        train_residual = y_train - train_prediction
        residual_std = max(float(np.sqrt(np.mean(train_residual**2))), 1e-8)

        test_prediction = intercept + slope * x_test
        residual = y_test - test_prediction
        variance = residual_std**2
        log_likelihood_terms = -0.5 * (
            np.log(2.0 * np.pi * variance) + residual**2 / variance
        )
        test_log_likelihood = float(np.sum(log_likelihood_terms))
        denominator = float(np.sum((y_test - float(np.mean(y_test))) ** 2))
        test_r2 = (
            float("nan")
            if denominator <= 0.0
            else 1.0 - float(np.sum(residual**2)) / denominator
        )

        fits.append(
            CandidateFit(
                candidate=candidate,
                intercept=intercept,
                slope=slope,
                train_feature_mean=mean,
                train_feature_std=std,
                residual_std=residual_std,
                test_log_likelihood=test_log_likelihood,
                test_mean_log_likelihood=test_log_likelihood / len(test_idx),
                test_r2=test_r2,
                n_train=len(train_idx),
                n_test=len(test_idx),
            )
        )

    return sorted(fits, key=lambda fit: fit.test_log_likelihood, reverse=True)
