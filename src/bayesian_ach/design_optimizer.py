"""Deterministic integer maximin allocation for finite experimental grids."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from bayesian_ach.design_geometry import (
    DesignDiagnostics,
    covariance_from_moments,
    diagnostics_from_covariance,
)


@dataclass(frozen=True, slots=True)
class OptimizedDesign:
    """An integer allocation, diagnostics, and complete optimization trace."""

    counts: NDArray[np.int64]
    diagnostics: DesignDiagnostics
    optimization_trace: tuple[dict[str, float | int], ...]


def _score(diagnostics: DesignDiagnostics) -> tuple[float, float, float, float]:
    return (
        diagnostics.minimum_pairwise_residual_variance,
        diagnostics.minimum_candidate_variance,
        diagnostics.covariance_log_determinant,
        -diagnostics.maximum_absolute_correlation,
    )


def optimize_maximin_design(
    standardized_signals: NDArray[np.float64],
    budget: int,
    *,
    max_point_fraction: float = 0.15,
    exchange_passes: int = 3,
    effect_size: float = 1.0,
    noise_std: float = 1.0,
    target_log_score_gap: float = 5.0,
    target_log_bf: float | None = None,
) -> OptimizedDesign:
    """Allocate trials greedily, then refine by deterministic one-for-one swaps."""

    signals = np.asarray(standardized_signals, dtype=float)
    if signals.ndim != 2 or signals.shape[1] < 2 or not np.all(np.isfinite(signals)):
        raise ValueError("standardized_signals must be a finite two-dimensional matrix")
    if budget < signals.shape[1] + 1:
        raise ValueError("budget must exceed the number of candidate signals")
    if not 0.0 < max_point_fraction <= 1.0:
        raise ValueError("max_point_fraction must lie in (0, 1]")
    maximum_count = max(1, int(math.ceil(max_point_fraction * budget)))
    counts = np.zeros(signals.shape[0], dtype=np.int64)
    sums = np.zeros(signals.shape[1], dtype=float)
    second = np.zeros((signals.shape[1], signals.shape[1]), dtype=float)
    outer = np.einsum("ij,ik->ijk", signals, signals)
    trace: list[dict[str, float | int]] = []

    def evaluate(
        n: int,
        candidate_sums: NDArray[np.float64],
        candidate_second: NDArray[np.float64],
        support_size: int,
    ) -> DesignDiagnostics:
        return diagnostics_from_covariance(
            covariance_from_moments(n, candidate_sums, candidate_second),
            trial_count=n,
            support_size=support_size,
            effect_size=effect_size,
            noise_std=noise_std,
            target_log_score_gap=target_log_score_gap,
            target_log_bf=target_log_bf,
        )

    for step in range(budget):
        best_index = -1
        best: DesignDiagnostics | None = None
        best_score: tuple[float, float, float, float] | None = None
        for index in range(signals.shape[0]):
            if counts[index] >= maximum_count:
                continue
            candidate = evaluate(
                step + 1,
                sums + signals[index],
                second + outer[index],
                int(np.sum(counts > 0)) + int(counts[index] == 0),
            )
            if best_score is None or _score(candidate) > best_score:
                best_index, best, best_score = index, candidate, _score(candidate)
        if best_index < 0 or best is None:
            raise RuntimeError("no feasible point remains under the allocation cap")
        counts[best_index] += 1
        sums += signals[best_index]
        second += outer[best_index]
        trace.append(_trace_row(step + 1, best_index, -1, best))

    for _ in range(exchange_passes):
        current = evaluate(budget, sums, second, int(np.sum(counts > 0)))
        best_move: tuple[int, int] | None = None
        best = current
        for remove in np.flatnonzero(counts > 0):
            for add in range(signals.shape[0]):
                if add == remove or counts[add] >= maximum_count:
                    continue
                support = int(np.sum(counts > 0))
                support -= int(counts[remove] == 1)
                support += int(counts[add] == 0)
                candidate = evaluate(
                    budget,
                    sums - signals[remove] + signals[add],
                    second - outer[remove] + outer[add],
                    support,
                )
                if _score(candidate) > _score(best):
                    best_move, best = (int(remove), add), candidate
        if best_move is None:
            break
        remove, add = best_move
        counts[remove] -= 1
        counts[add] += 1
        sums += signals[add] - signals[remove]
        second += outer[add] - outer[remove]
        trace.append(_trace_row(len(trace) + 1, add, remove, best))

    final = evaluate(budget, sums, second, int(np.sum(counts > 0)))
    return OptimizedDesign(counts, final, tuple(trace))


def _trace_row(
    step: int,
    added: int,
    removed: int,
    diagnostics: DesignDiagnostics,
) -> dict[str, float | int]:
    return {
        "step": step,
        "point_id": added,
        "removed_point_id": removed,
        "minimum_pairwise_residual_variance": (
            diagnostics.minimum_pairwise_residual_variance
        ),
        "maximum_absolute_correlation": diagnostics.maximum_absolute_correlation,
        "support_size": diagnostics.support_size,
    }
