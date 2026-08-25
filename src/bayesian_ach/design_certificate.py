"""Certified cutting-plane maximin allocation over a finite design grid."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import Bounds, LinearConstraint, milp

from bayesian_ach.design_geometry import pairwise_residual_matrix
from bayesian_ach.design_optimizer import optimize_maximin_design


@dataclass(frozen=True, slots=True)
class CertifiedDesignConfig:
    """Configuration for continuous or exact-integer maximin certification."""

    budget: int = 60
    max_point_fraction: float = 0.15
    integer: bool = True
    absolute_gap_tolerance: float = 1.0e-8
    relative_gap_tolerance: float = 1.0e-7
    cut_violation_tolerance: float = 1.0e-9
    max_iterations: int = 100
    master_time_limit_s: float = 120.0
    master_mip_relative_gap: float = 1.0e-9

    def validate(self, point_count: int, candidate_count: int) -> None:
        if self.budget < candidate_count + 1:
            raise ValueError("budget must exceed the number of candidate signals")
        if point_count < 1:
            raise ValueError("point_count must be positive")
        if not 0.0 < self.max_point_fraction <= 1.0:
            raise ValueError("max_point_fraction must lie in (0, 1]")
        if self.absolute_gap_tolerance < 0.0:
            raise ValueError("absolute_gap_tolerance must be nonnegative")
        if self.relative_gap_tolerance < 0.0:
            raise ValueError("relative_gap_tolerance must be nonnegative")
        if self.cut_violation_tolerance < 0.0:
            raise ValueError("cut_violation_tolerance must be nonnegative")
        if self.max_iterations < 1:
            raise ValueError("max_iterations must be positive")
        if self.master_time_limit_s <= 0.0:
            raise ValueError("master_time_limit_s must be positive")
        if self.master_mip_relative_gap < 0.0:
            raise ValueError("master_mip_relative_gap must be nonnegative")
        maximum_count = max(1, int(math.ceil(self.max_point_fraction * self.budget)))
        if point_count * maximum_count < self.budget:
            raise ValueError("allocation cap makes the requested budget infeasible")


@dataclass(frozen=True, slots=True)
class MaximinCertificate:
    """Certified objective bounds and the best feasible allocation found."""

    allocation: NDArray[np.float64]
    integer: bool
    certified: bool
    lower_bound: float
    upper_bound: float
    absolute_gap: float
    relative_gap: float
    heuristic_lower_bound: float
    iterations: int
    cut_count: int
    last_master_status: int
    last_master_message: str
    trace: tuple[dict[str, Any], ...]


@dataclass(frozen=True, slots=True)
class _PairCut:
    generator: int
    alternative: int
    intercept: float
    slope: float
    losses: NDArray[np.float64]
    residual: float


def _pairwise_oracle(
    signals: NDArray[np.float64],
    allocation: NDArray[np.float64],
) -> tuple[float, tuple[_PairCut, ...]]:
    total = float(np.sum(allocation))
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("allocation must have positive finite mass")
    weights = allocation / total
    root_weight = np.sqrt(weights)
    cuts: list[_PairCut] = []
    minimum = math.inf
    for generator in range(signals.shape[1]):
        response = signals[:, generator]
        for alternative in range(signals.shape[1]):
            if generator == alternative:
                continue
            design = np.column_stack((np.ones(signals.shape[0]), signals[:, alternative]))
            weighted_design = design * root_weight[:, None]
            weighted_response = response * root_weight
            coefficients, _, _, _ = np.linalg.lstsq(
                weighted_design,
                weighted_response,
                rcond=None,
            )
            residual_vector = response - design @ coefficients
            losses = np.asarray(residual_vector**2, dtype=float)
            residual = float(weights @ losses)
            minimum = min(minimum, residual)
            cuts.append(
                _PairCut(
                    generator=generator,
                    alternative=alternative,
                    intercept=float(coefficients[0]),
                    slope=float(coefficients[1]),
                    losses=losses,
                    residual=residual,
                )
            )
    return float(minimum), tuple(cuts)


def _validate_initial_allocation(
    allocation: NDArray[np.float64],
    *,
    point_count: int,
    config: CertifiedDesignConfig,
    maximum_count: int,
) -> NDArray[np.float64]:
    result = np.asarray(allocation, dtype=float)
    if result.shape != (point_count,) or not np.all(np.isfinite(result)):
        raise ValueError("initial_allocation must be a finite vector matching the grid")
    if np.any(result < -1.0e-9) or np.any(result > maximum_count + 1.0e-9):
        raise ValueError("initial_allocation violates its bounds")
    if not math.isclose(float(np.sum(result)), config.budget, abs_tol=1.0e-7):
        raise ValueError("initial_allocation must sum to budget")
    if config.integer and not np.allclose(result, np.rint(result), atol=1.0e-9):
        raise ValueError("integer certification requires integer initial_allocation")
    return np.rint(result) if config.integer else result


def _master_upper_bound(result: Any, *, integer: bool, incumbent: float) -> float:
    if integer:
        dual = getattr(result, "mip_dual_bound", None)
        if dual is not None and np.isfinite(float(dual)):
            return max(incumbent, float(-float(dual)))
    if bool(result.success):
        return incumbent
    return math.inf


def certify_maximin_design(
    standardized_signals: NDArray[np.float64],
    config: CertifiedDesignConfig | None = None,
    *,
    initial_allocation: NDArray[np.float64] | None = None,
) -> MaximinCertificate:
    """Certify continuous or integer maximin residual geometry with OLS cuts.

    Each ordered residual is the infimum, over an intercept and slope, of a
    loss linear in the allocation weights. A finite set of such losses defines
    a master LP/MILP upper bound. Weighted least squares at each master
    solution supplies a separating cut and a feasible lower bound.
    """

    signals = np.asarray(standardized_signals, dtype=float)
    if signals.ndim != 2 or signals.shape[1] < 2 or not np.all(np.isfinite(signals)):
        raise ValueError("standardized_signals must be a finite two-dimensional matrix")
    config = CertifiedDesignConfig() if config is None else config
    config.validate(signals.shape[0], signals.shape[1])
    maximum_count = max(1, int(math.ceil(config.max_point_fraction * config.budget)))

    if initial_allocation is None:
        heuristic = optimize_maximin_design(
            signals,
            config.budget,
            max_point_fraction=config.max_point_fraction,
        )
        initial = np.asarray(heuristic.counts, dtype=float)
    else:
        initial = np.asarray(initial_allocation, dtype=float)
    initial = _validate_initial_allocation(
        initial,
        point_count=signals.shape[0],
        config=config,
        maximum_count=maximum_count,
    )
    best_lower, initial_cuts = _pairwise_oracle(signals, initial)
    best_allocation = initial.copy()
    heuristic_lower = best_lower

    cuts: list[_PairCut] = []
    cut_keys: set[tuple[int, int, str, str]] = set()

    def add_cut(cut: _PairCut) -> bool:
        key = (
            cut.generator,
            cut.alternative,
            cut.intercept.hex(),
            cut.slope.hex(),
        )
        if key in cut_keys:
            return False
        cut_keys.add(key)
        cuts.append(cut)
        return True

    for cut in initial_cuts:
        add_cut(cut)

    variable_count = signals.shape[0] + 1
    lower_bounds = np.zeros(variable_count, dtype=float)
    upper_bounds = np.full(variable_count, float(maximum_count), dtype=float)
    upper_bounds[-1] = float(np.max(signals**2))
    objective = np.zeros(variable_count, dtype=float)
    objective[-1] = -1.0
    integrality = np.zeros(variable_count, dtype=np.int32)
    if config.integer:
        integrality[:-1] = 1

    global_upper = float(upper_bounds[-1])
    trace: list[dict[str, Any]] = []
    last_status = -1
    last_message = "master was not run"
    certified = False

    for iteration in range(1, config.max_iterations + 1):
        rows = np.zeros((len(cuts) + 1, variable_count), dtype=float)
        lower = np.full(len(cuts) + 1, -np.inf, dtype=float)
        upper = np.zeros(len(cuts) + 1, dtype=float)
        rows[0, :-1] = 1.0
        lower[0] = float(config.budget)
        upper[0] = float(config.budget)
        for row_index, cut in enumerate(cuts, start=1):
            rows[row_index, :-1] = -cut.losses
            rows[row_index, -1] = float(config.budget)

        options: dict[str, float | bool] = {
            "presolve": True,
            "time_limit": float(config.master_time_limit_s),
        }
        if config.integer:
            options["mip_rel_gap"] = float(config.master_mip_relative_gap)
        result = milp(
            objective,
            integrality=integrality,
            bounds=Bounds(lower_bounds, upper_bounds),
            constraints=LinearConstraint(rows, lower, upper),
            options=options,
        )
        last_status = int(result.status)
        last_message = str(result.message)
        if result.x is None:
            trace.append(
                {
                    "iteration": iteration,
                    "status": last_status,
                    "message": last_message,
                    "cut_count": len(cuts),
                    "master_incumbent": None,
                    "master_upper_bound": global_upper,
                    "best_lower_bound": best_lower,
                    "absolute_gap": max(0.0, global_upper - best_lower),
                    "violated_pair_count": None,
                    "new_cut_count": 0,
                }
            )
            break

        master_incumbent = float(result.x[-1])
        master_upper = _master_upper_bound(
            result,
            integer=config.integer,
            incumbent=master_incumbent,
        )
        global_upper = min(global_upper, master_upper)
        candidate = np.asarray(result.x[:-1], dtype=float)
        if config.integer:
            candidate = np.rint(candidate)
        candidate_lower, oracle_cuts = _pairwise_oracle(signals, candidate)
        if candidate_lower > best_lower:
            best_lower = candidate_lower
            best_allocation = candidate.copy()

        absolute_gap = max(0.0, global_upper - best_lower)
        relative_gap = absolute_gap / max(1.0, abs(best_lower))
        violated = [
            cut
            for cut in oracle_cuts
            if master_incumbent > cut.residual + config.cut_violation_tolerance
        ]
        new_cut_count = sum(int(add_cut(cut)) for cut in violated)
        trace.append(
            {
                "iteration": iteration,
                "status": last_status,
                "message": last_message,
                "cut_count": len(cuts),
                "master_incumbent": master_incumbent,
                "master_upper_bound": master_upper,
                "global_upper_bound": global_upper,
                "candidate_lower_bound": candidate_lower,
                "best_lower_bound": best_lower,
                "absolute_gap": absolute_gap,
                "relative_gap": relative_gap,
                "violated_pair_count": len(violated),
                "new_cut_count": new_cut_count,
            }
        )

        tolerance = config.absolute_gap_tolerance + (
            config.relative_gap_tolerance * max(1.0, abs(best_lower))
        )
        if np.isfinite(global_upper) and absolute_gap <= tolerance:
            certified = True
            break
        if not violated or new_cut_count == 0:
            break

    absolute_gap = max(0.0, global_upper - best_lower)
    relative_gap = absolute_gap / max(1.0, abs(best_lower))
    return MaximinCertificate(
        allocation=np.asarray(best_allocation, dtype=float),
        integer=config.integer,
        certified=certified,
        lower_bound=float(best_lower),
        upper_bound=float(global_upper),
        absolute_gap=float(absolute_gap),
        relative_gap=float(relative_gap),
        heuristic_lower_bound=float(heuristic_lower),
        iterations=len(trace),
        cut_count=len(cuts),
        last_master_status=last_status,
        last_master_message=last_message,
        trace=tuple(trace),
    )


def certificate_matches_geometry(
    signals: NDArray[np.float64],
    certificate: MaximinCertificate,
    *,
    tolerance: float = 1.0e-9,
) -> bool:
    """Return whether the frozen lower bound matches direct residual geometry."""

    direct = pairwise_residual_matrix(signals, certificate.allocation)
    off_diagonal = ~np.eye(direct.shape[0], dtype=bool)
    return math.isclose(
        certificate.lower_bound,
        float(np.min(direct[off_diagonal])),
        rel_tol=tolerance,
        abs_tol=tolerance,
    )
