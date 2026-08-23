"""Feasible transition-condition grids and fixed baseline allocations."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray

from bayesian_ach.signals import CANDIDATE_SIGNAL_NAMES, compute_transition_signals

DESIGN_CANDIDATE_NAMES: Final[tuple[str, ...]] = CANDIDATE_SIGNAL_NAMES


@dataclass(frozen=True, slots=True)
class TransitionDesignGridConfig:
    """Finite set of experimentally controllable transition conditions."""

    observed_probabilities: tuple[float, ...] = (0.05, 0.15, 0.35, 0.65, 0.90)
    residual_shapes: tuple[float, ...] = (0.50, 0.90)
    concentrations: tuple[float, ...] = (2.0, 8.0, 32.0, 128.0)
    reset_observed_probabilities: tuple[float, ...] = (0.05, 0.50, 0.95)
    hazards: tuple[float, ...] = (0.01, 0.15)
    n_states: int = 3

    def validate(self) -> None:
        if self.n_states != 3:
            raise ValueError("the transparent grid currently requires n_states=3")
        for name, values in (
            ("observed_probabilities", self.observed_probabilities),
            ("reset_observed_probabilities", self.reset_observed_probabilities),
        ):
            if not values or any(not 0.0 < value < 1.0 for value in values):
                raise ValueError(f"{name} must contain probabilities in (0, 1)")
        if not self.residual_shapes or any(
            not 0.5 <= value < 1.0 for value in self.residual_shapes
        ):
            raise ValueError("residual_shapes must lie in [0.5, 1)")
        if not self.concentrations or any(value <= 0.0 for value in self.concentrations):
            raise ValueError("concentrations must be positive")
        if not self.hazards or any(not 0.0 < value < 1.0 for value in self.hazards):
            raise ValueError("hazards must lie in (0, 1)")


def _probability_vector(observed: float, shape: float) -> NDArray[np.float64]:
    remaining = 1.0 - observed
    return np.asarray(
        [observed, remaining * shape, remaining * (1.0 - shape)],
        dtype=float,
    )


def generate_transition_design_grid(
    config: TransitionDesignGridConfig | None = None,
) -> tuple[tuple[dict[str, Any], ...], NDArray[np.float64], NDArray[np.float64]]:
    """Return feasible conditions and raw/globally standardized candidates."""

    config = TransitionDesignGridConfig() if config is None else config
    config.validate()
    rows: list[dict[str, Any]] = []
    candidate_rows: list[list[float]] = []
    for observed in config.observed_probabilities:
        for shape in config.residual_shapes:
            probabilities = _probability_vector(observed, shape)
            for concentration in config.concentrations:
                alpha = concentration * probabilities
                for reset_observed in config.reset_observed_probabilities:
                    reset_alpha = _probability_vector(reset_observed, 0.5)
                    for hazard in config.hazards:
                        signals = compute_transition_signals(
                            alpha,
                            0,
                            reset_alpha=reset_alpha,
                            hazard=hazard,
                        )
                        values = [
                            float(getattr(signals, name))
                            for name in DESIGN_CANDIDATE_NAMES
                        ]
                        row: dict[str, Any] = {
                            "point_id": len(rows),
                            "observed_probability": observed,
                            "residual_shape": shape,
                            "concentration": concentration,
                            "reset_observed_probability": reset_observed,
                            "hazard": hazard,
                        }
                        row.update(dict(zip(DESIGN_CANDIDATE_NAMES, values, strict=True)))
                        rows.append(row)
                        candidate_rows.append(values)
    raw = np.asarray(candidate_rows, dtype=float)
    scales = np.std(raw, axis=0)
    if np.any(scales <= 1e-12):
        raise RuntimeError("a candidate is constant over the feasible design grid")
    standardized = (raw - np.mean(raw, axis=0)) / scales
    return tuple(rows), raw, standardized


def coupled_novelty_design(
    rows: tuple[dict[str, Any], ...],
    budget: int,
) -> NDArray[np.int64]:
    """Allocate a conventional schedule whose novelty variables co-vary."""

    if budget < 2:
        raise ValueError("budget must be at least two")
    parameters = np.asarray(
        [
            [
                float(row["observed_probability"]),
                math.log(float(row["concentration"])),
                float(row["reset_observed_probability"]),
                float(row["hazard"]),
                float(row["residual_shape"]),
            ]
            for row in rows
        ]
    )
    ranges = np.ptp(parameters, axis=0)
    scales = np.where(ranges > 1e-12, ranges, 1.0)
    counts = np.zeros(len(rows), dtype=np.int64)
    for step in range(budget):
        novelty = step / max(budget - 1, 1)
        target = np.asarray(
            [
                0.90 - 0.85 * novelty,
                math.log(200.0) * (1.0 - novelty) + math.log(2.0) * novelty,
                0.05 + 0.90 * novelty,
                0.005 * (1.0 - novelty) + 0.15 * novelty,
                0.50 + 0.45 * novelty,
            ]
        )
        distance = np.sum(((parameters - target) / scales) ** 2, axis=1)
        counts[int(np.argmin(distance))] += 1
    return counts


def uniform_factorial_design(
    point_count: int,
    budget: int,
    *,
    seed: int = 7,
) -> NDArray[np.int64]:
    """Allocate a budget uniformly over a seeded permutation of grid points."""

    if point_count < 1 or budget < 1:
        raise ValueError("point_count and budget must be positive")
    order = np.random.default_rng(seed).permutation(point_count)
    counts = np.zeros(point_count, dtype=np.int64)
    for step in range(budget):
        counts[int(order[step % point_count])] += 1
    return counts
