"""Held-out recovery utilities for comparing finite trial allocations."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from numpy.typing import NDArray

from bayesian_ach.design_grid import DESIGN_CANDIDATE_NAMES


@dataclass(frozen=True, slots=True)
class DesignRecoveryRow:
    """Recovery summary for one design and one generating candidate."""

    design: str
    generator: str
    replicate_count: int
    correct_count: int
    recovery_rate: float
    median_log_evidence_margin: float
    minimum_log_evidence_margin: float

    def as_dict(self) -> dict[str, float | int | str]:
        return asdict(self)


def recover_design(
    name: str,
    signals: NDArray[np.float64],
    counts: NDArray[np.int64],
    *,
    replicates: int,
    test_fraction: float,
    effect_size: float,
    noise_std: float,
    seed: int,
) -> list[DesignRecoveryRow]:
    """Recover each generating candidate using held-out Gaussian likelihood."""

    indices = np.repeat(np.arange(counts.size), counts)
    trials = signals[indices]
    n_trials = trials.shape[0]
    n_test = max(2, min(n_trials - 2, int(round(test_fraction * n_trials))))
    rng = np.random.default_rng(seed)
    rows: list[DesignRecoveryRow] = []
    for generator_index, generator in enumerate(DESIGN_CANDIDATE_NAMES):
        correct = 0
        margins: list[float] = []
        for _ in range(replicates):
            order = rng.permutation(n_trials)
            test = np.asarray(order[:n_test], dtype=np.int64)
            train = np.asarray(order[n_test:], dtype=np.int64)
            response = (
                effect_size * trials[:, generator_index]
                + rng.normal(0.0, noise_std, size=n_trials)
            )
            winner, margin = _fit_and_score(trials, response, train, test)
            correct += int(winner == generator_index)
            margins.append(margin)
        rows.append(
            DesignRecoveryRow(
                design=name,
                generator=generator,
                replicate_count=replicates,
                correct_count=correct,
                recovery_rate=correct / replicates,
                median_log_evidence_margin=float(np.median(margins)),
                minimum_log_evidence_margin=float(np.min(margins)),
            )
        )
    return rows


def _fit_and_score(
    signals: NDArray[np.float64],
    response: NDArray[np.float64],
    train: NDArray[np.int64],
    test: NDArray[np.int64],
) -> tuple[int, float]:
    scores: list[float] = []
    for candidate in range(signals.shape[1]):
        x_train = signals[train, candidate]
        design = np.column_stack((np.ones(x_train.size), x_train))
        coefficients, _, _, _ = np.linalg.lstsq(design, response[train], rcond=None)
        training_residual = response[train] - design @ coefficients
        variance = max(float(np.mean(training_residual**2)), 1e-8)
        prediction = coefficients[0] + coefficients[1] * signals[test, candidate]
        residual = response[test] - prediction
        scores.append(
            float(
                np.sum(
                    -0.5
                    * (
                        np.log(2.0 * np.pi * variance)
                        + residual**2 / variance
                    )
                )
            )
        )
    order = np.argsort(np.asarray(scores))[::-1]
    return int(order[0]), float(scores[order[0]] - scores[order[1]])
