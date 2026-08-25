"""Finite-sample abstention stress tests for maximin design evidence."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from itertools import combinations
from statistics import NormalDist
from typing import Any

import numpy as np
from numpy.typing import NDArray

from bayesian_ach.design_geometry import design_diagnostics
from bayesian_ach.design_grid import (
    DESIGN_CANDIDATE_NAMES,
    coupled_novelty_design,
    generate_transition_design_grid,
    uniform_factorial_design,
)
from bayesian_ach.design_optimizer import optimize_maximin_design

STRESS_DESIGNS = (
    "coupled_novelty",
    "uniform_factorial",
    "maximin_optimized",
)


@dataclass(frozen=True, slots=True)
class DesignStressConfig:
    """Frozen simulation and calibration settings for the sensitivity artifact."""

    fixed_budgets: tuple[int, ...] = (60,)
    budget_factors: tuple[float, ...] = ()
    calibration_replicates: int = 100
    calibration_audit_replicates: int = 100
    evaluation_replicates: int = 200
    test_fraction: float = 0.35
    effect_size: float = 1.0
    noise_std: float = 1.0
    target_log_score_gap: float = 5.0
    alpha: float = 0.05
    confidence_level: float = 0.95
    ridge_lambdas: tuple[float, ...] = (0.0, 0.01, 0.1, 1.0)
    inner_folds: int = 3
    max_point_fraction: float = 0.15
    allocation_seed: int = 7
    threshold_seed: int = 104729
    calibration_audit_seed: int = 130363
    evaluation_seed: int = 155921

    def validate(self) -> None:
        if not self.fixed_budgets and not self.budget_factors:
            raise ValueError("at least one fixed budget or N_eff factor is required")
        if (
            any(value < len(DESIGN_CANDIDATE_NAMES) + 1 for value in self.fixed_budgets)
            or len(set(self.fixed_budgets)) != len(self.fixed_budgets)
        ):
            raise ValueError("fixed_budgets must be unique and exceed candidate count")
        if (
            any(
                not math.isfinite(value) or value <= 0.0
                for value in self.budget_factors
            )
            or len(set(self.budget_factors)) != len(self.budget_factors)
        ):
            raise ValueError("budget_factors must be finite, positive, and unique")
        for name, value in (
            ("calibration_replicates", self.calibration_replicates),
            ("calibration_audit_replicates", self.calibration_audit_replicates),
            ("evaluation_replicates", self.evaluation_replicates),
        ):
            if value < 20:
                raise ValueError(f"{name} must be at least 20")
        if not 0.0 < self.test_fraction < 1.0:
            raise ValueError("test_fraction must lie in (0, 1)")
        if (
            not math.isfinite(self.effect_size)
            or not math.isfinite(self.noise_std)
            or self.effect_size <= 0.0
            or self.noise_std <= 0.0
        ):
            raise ValueError("effect_size and noise_std must be finite and positive")
        if (
            not math.isfinite(self.target_log_score_gap)
            or self.target_log_score_gap <= 0.0
        ):
            raise ValueError("target_log_score_gap must be finite and positive")
        if not 0.0 < self.alpha < 0.5:
            raise ValueError("alpha must lie in (0, 0.5)")
        if not 0.0 < self.confidence_level < 1.0:
            raise ValueError("confidence_level must lie in (0, 1)")
        if not self.ridge_lambdas or any(
            not math.isfinite(value) or value < 0.0
            for value in self.ridge_lambdas
        ):
            raise ValueError("ridge_lambdas must be finite and nonnegative")
        if self.inner_folds < 2:
            raise ValueError("inner_folds must be at least two")
        if not 0.0 < self.max_point_fraction <= 1.0:
            raise ValueError("max_point_fraction must lie in (0, 1]")
        seeds = {
            self.threshold_seed,
            self.calibration_audit_seed,
            self.evaluation_seed,
        }
        if len(seeds) != 3:
            raise ValueError("threshold, audit, and evaluation seeds must be distinct")


@dataclass(frozen=True, slots=True)
class DesignStressResult:
    """Complete descriptive stress-test tables and summary."""

    summary: dict[str, Any]
    thresholds: tuple[dict[str, Any], ...]
    calibration: tuple[dict[str, Any], ...]
    pure_recovery: tuple[dict[str, Any], ...]
    null_evaluation: tuple[dict[str, Any], ...]
    mixture_evaluation: tuple[dict[str, Any], ...]
    out_of_span_evaluation: tuple[dict[str, Any], ...]
    allocations: tuple[dict[str, Any], ...]


@dataclass(frozen=True, slots=True)
class _Scores:
    winner: int
    runner: int
    best_pure_score: float
    runner_score: float
    null_score: float
    flexible_score: float
    ridge_lambda: float

    @property
    def pure_over_null(self) -> float:
        return self.best_pure_score - self.null_score

    @property
    def winner_over_runner(self) -> float:
        return self.best_pure_score - self.runner_score

    @property
    def flexible_over_pure(self) -> float:
        return self.flexible_score - self.best_pure_score


@dataclass(frozen=True, slots=True)
class _Thresholds:
    pure_over_null: float
    winner_over_runner: float
    flexible_over_pure: float


def _fit_variance(residual: NDArray[np.float64]) -> float:
    return max(float(np.mean(residual**2)), 1.0e-8)


def _gaussian_score(residual: NDArray[np.float64], variance: float) -> float:
    return float(
        np.sum(
            -0.5
            * (
                math.log(2.0 * math.pi * variance)
                + residual**2 / variance
            )
        )
    )


def _fit_univariate(
    predictor: NDArray[np.float64],
    response: NDArray[np.float64],
    train: NDArray[np.int64],
    test: NDArray[np.int64],
) -> float:
    design = np.column_stack((np.ones(train.size), predictor[train]))
    coefficients, _, _, _ = np.linalg.lstsq(
        design,
        response[train],
        rcond=None,
    )
    training_residual = response[train] - design @ coefficients
    variance = _fit_variance(training_residual)
    prediction = coefficients[0] + coefficients[1] * predictor[test]
    return _gaussian_score(response[test] - prediction, variance)


def _fit_null(
    response: NDArray[np.float64],
    train: NDArray[np.int64],
    test: NDArray[np.int64],
) -> float:
    mean = float(np.mean(response[train]))
    variance = _fit_variance(response[train] - mean)
    return _gaussian_score(response[test] - mean, variance)


def _ridge_coefficients(
    signals: NDArray[np.float64],
    response: NDArray[np.float64],
    indices: NDArray[np.int64],
    ridge_lambda: float,
) -> NDArray[np.float64]:
    design = np.column_stack((np.ones(indices.size), signals[indices]))
    penalty = np.eye(design.shape[1], dtype=float) * ridge_lambda
    penalty[0, 0] = 0.0
    coefficients, _, _, _ = np.linalg.lstsq(
        design.T @ design + penalty,
        design.T @ response[indices],
        rcond=None,
    )
    return np.asarray(coefficients, dtype=float)


def _score_flexible_at_lambda(
    signals: NDArray[np.float64],
    response: NDArray[np.float64],
    train: NDArray[np.int64],
    test: NDArray[np.int64],
    ridge_lambda: float,
) -> float:
    coefficients = _ridge_coefficients(
        signals,
        response,
        train,
        ridge_lambda,
    )
    train_design = np.column_stack((np.ones(train.size), signals[train]))
    variance = _fit_variance(response[train] - train_design @ coefficients)
    test_design = np.column_stack((np.ones(test.size), signals[test]))
    return _gaussian_score(response[test] - test_design @ coefficients, variance)


def _select_ridge_lambda(
    signals: NDArray[np.float64],
    response: NDArray[np.float64],
    train: NDArray[np.int64],
    *,
    lambdas: tuple[float, ...],
    folds: int,
    rng: np.random.Generator,
) -> float:
    shuffled = np.asarray(rng.permutation(train), dtype=np.int64)
    split = [
        np.asarray(values, dtype=np.int64)
        for values in np.array_split(shuffled, min(folds, train.size))
        if values.size > 0
    ]
    best_lambda = float(min(lambdas))
    best_score = -math.inf
    for ridge_lambda in sorted(lambdas):
        score = 0.0
        for fold_index, validation in enumerate(split):
            inner_train = np.concatenate(
                [values for index, values in enumerate(split) if index != fold_index]
            )
            score += _score_flexible_at_lambda(
                signals,
                response,
                inner_train,
                validation,
                float(ridge_lambda),
            )
        if score > best_score + 1.0e-12:
            best_score = score
            best_lambda = float(ridge_lambda)
    return best_lambda


def _fit_and_score(
    signals: NDArray[np.float64],
    response: NDArray[np.float64],
    train: NDArray[np.int64],
    test: NDArray[np.int64],
    *,
    config: DesignStressConfig,
    rng: np.random.Generator,
) -> _Scores:
    pure = np.asarray(
        [
            _fit_univariate(signals[:, candidate], response, train, test)
            for candidate in range(signals.shape[1])
        ],
        dtype=float,
    )
    order = np.lexsort((np.arange(pure.size), -pure))
    winner = int(order[0])
    runner = int(order[1])
    ridge_lambda = _select_ridge_lambda(
        signals,
        response,
        train,
        lambdas=config.ridge_lambdas,
        folds=config.inner_folds,
        rng=rng,
    )
    return _Scores(
        winner=winner,
        runner=runner,
        best_pure_score=float(pure[winner]),
        runner_score=float(pure[runner]),
        null_score=_fit_null(response, train, test),
        flexible_score=_score_flexible_at_lambda(
            signals,
            response,
            train,
            test,
            ridge_lambda,
        ),
        ridge_lambda=ridge_lambda,
    )


def _simulate_scores(
    signals: NDArray[np.float64],
    generator: NDArray[np.float64],
    *,
    config: DesignStressConfig,
    rng: np.random.Generator,
) -> _Scores:
    response = (
        config.effect_size * generator
        + rng.normal(0.0, config.noise_std, size=generator.size)
    )
    order = np.asarray(rng.permutation(generator.size), dtype=np.int64)
    n_test = max(
        2,
        min(
            generator.size - config.inner_folds - 2,
            int(round(config.test_fraction * generator.size)),
        ),
    )
    test = order[:n_test]
    train = order[n_test:]
    return _fit_and_score(
        signals,
        response,
        train,
        test,
        config=config,
        rng=rng,
    )


def _upper_conformal_quantile(
    values: list[float],
    *,
    alpha: float,
) -> float:
    ordered = np.sort(np.asarray(values, dtype=float))
    rank = int(math.ceil((ordered.size + 1) * (1.0 - alpha)))
    if rank > ordered.size:
        return math.inf
    return float(ordered[max(0, rank - 1)])


def _wilson_interval(
    successes: int,
    total: int,
    confidence_level: float,
) -> tuple[float, float]:
    if total < 1:
        raise ValueError("total must be positive")
    z = NormalDist().inv_cdf(0.5 + confidence_level / 2.0)
    proportion = successes / total
    denominator = 1.0 + z**2 / total
    centre = (proportion + z**2 / (2.0 * total)) / denominator
    radius = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / total
            + z**2 / (4.0 * total**2)
        )
        / denominator
    )
    return max(0.0, centre - radius), min(1.0, centre + radius)


def _call(scores: _Scores, thresholds: _Thresholds) -> tuple[int | None, str]:
    if scores.pure_over_null <= thresholds.pure_over_null:
        return None, "null_not_rejected"
    if scores.winner_over_runner <= thresholds.winner_over_runner:
        return None, "pure_ambiguity"
    if scores.flexible_over_pure > thresholds.flexible_over_pure:
        return None, "flexible_model_better"
    return scores.winner, "pure_call"


def _rng(seed: int, *keys: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([seed, *keys]))


def _design_counts(
    design: str,
    rows: tuple[dict[str, Any], ...],
    signals: NDArray[np.float64],
    budget: int,
    config: DesignStressConfig,
) -> NDArray[np.int64]:
    if design == "coupled_novelty":
        return coupled_novelty_design(rows, budget)
    if design == "uniform_factorial":
        return uniform_factorial_design(
            len(rows),
            budget,
            seed=config.allocation_seed,
        )
    if design == "maximin_optimized":
        return optimize_maximin_design(
            signals,
            budget,
            max_point_fraction=config.max_point_fraction,
        ).counts
    raise ValueError(f"unknown design: {design}")


def _validated_override(
    value: NDArray[np.int64],
    *,
    point_count: int,
    budget: int,
    maximum_count: int,
) -> NDArray[np.int64]:
    counts = np.asarray(value, dtype=np.int64)
    if counts.shape != (point_count,) or np.any(counts < 0):
        raise ValueError("allocation override must be a nonnegative grid-sized vector")
    if int(np.sum(counts)) != budget:
        raise ValueError("allocation override must sum to its declared budget")
    if np.any(counts > maximum_count):
        raise ValueError("allocation override exceeds the per-cell cap")
    return counts


def _base_targets(
    rows: tuple[dict[str, Any], ...],
    signals: NDArray[np.float64],
    config: DesignStressConfig,
) -> dict[str, int]:
    targets: dict[str, int] = {}
    for design in STRESS_DESIGNS:
        counts = _design_counts(design, rows, signals, 60, config)
        targets[design] = design_diagnostics(
            signals,
            counts,
            effect_size=config.effect_size,
            noise_std=config.noise_std,
            target_log_score_gap=config.target_log_score_gap,
        ).trials_for_expected_log_score_gap_target
    return targets


def _calibrate_thresholds(
    signals: NDArray[np.float64],
    *,
    config: DesignStressConfig,
    design_index: int,
    budget: int,
) -> _Thresholds:
    null_over_baseline: list[float] = []
    null_ambiguity: list[float] = []
    null_generator = np.zeros(signals.shape[0], dtype=float)
    for replicate in range(config.calibration_replicates):
        scores = _simulate_scores(
            signals,
            null_generator,
            config=config,
            rng=_rng(
                config.threshold_seed,
                design_index,
                budget,
                0,
                replicate,
            ),
        )
        null_over_baseline.append(scores.pure_over_null)
        null_ambiguity.append(scores.winner_over_runner)

    flexible_thresholds = []
    for generator in range(signals.shape[1]):
        values = []
        for replicate in range(config.calibration_replicates):
            scores = _simulate_scores(
                signals,
                signals[:, generator],
                config=config,
                rng=_rng(
                    config.threshold_seed,
                    design_index,
                    budget,
                    1,
                    generator,
                    replicate,
                ),
            )
            values.append(scores.flexible_over_pure)
        flexible_thresholds.append(
            _upper_conformal_quantile(values, alpha=config.alpha)
        )
    return _Thresholds(
        pure_over_null=_upper_conformal_quantile(
            null_over_baseline,
            alpha=config.alpha,
        ),
        winner_over_runner=_upper_conformal_quantile(
            null_ambiguity,
            alpha=config.alpha,
        ),
        flexible_over_pure=max(0.0, max(flexible_thresholds)),
    )


def _budget_schedule(factor: float | None) -> str:
    return "fixed_budget" if factor is None else "n_eff_factor"


def _calibration_audit_rows(
    design: str,
    design_index: int,
    factor: float | None,
    budget: int,
    signals: NDArray[np.float64],
    thresholds: _Thresholds,
    config: DesignStressConfig,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    null_false_calls = 0
    for replicate in range(config.calibration_audit_replicates):
        scores = _simulate_scores(
            signals,
            np.zeros(signals.shape[0], dtype=float),
            config=config,
            rng=_rng(
                config.calibration_audit_seed,
                design_index,
                budget,
                0,
                replicate,
            ),
        )
        call, _ = _call(scores, thresholds)
        null_false_calls += int(call is not None)
    lower, upper = _wilson_interval(
        null_false_calls,
        config.calibration_audit_replicates,
        config.confidence_level,
    )
    rows.append(
        {
            "design": design,
            "budget_schedule": _budget_schedule(factor),
            "budget_factor": factor,
            "budget": budget,
            "scenario": "null",
            "generator": "null",
            "replicates": config.calibration_audit_replicates,
            "correct_pure_calls": 0,
            "wrong_pure_calls": null_false_calls,
            "abstentions": config.calibration_audit_replicates - null_false_calls,
            "rate": null_false_calls / config.calibration_audit_replicates,
            "wilson_lower": lower,
            "wilson_upper": upper,
        }
    )
    for generator, name in enumerate(DESIGN_CANDIDATE_NAMES):
        correct = 0
        wrong = 0
        abstain = 0
        for replicate in range(config.calibration_audit_replicates):
            scores = _simulate_scores(
                signals,
                signals[:, generator],
                config=config,
                rng=_rng(
                    config.calibration_audit_seed,
                    design_index,
                    budget,
                    1,
                    generator,
                    replicate,
                ),
            )
            call, _ = _call(scores, thresholds)
            correct += int(call == generator)
            wrong += int(call is not None and call != generator)
            abstain += int(call is None)
        lower, upper = _wilson_interval(
            correct,
            config.calibration_audit_replicates,
            config.confidence_level,
        )
        rows.append(
            {
                "design": design,
                "budget_schedule": _budget_schedule(factor),
                "budget_factor": factor,
                "budget": budget,
                "scenario": "matched_pure",
                "generator": name,
                "replicates": config.calibration_audit_replicates,
                "correct_pure_calls": correct,
                "wrong_pure_calls": wrong,
                "abstentions": abstain,
                "rate": correct / config.calibration_audit_replicates,
                "wilson_lower": lower,
                "wilson_upper": upper,
            }
        )
    return rows


def _evaluation_rows(
    design: str,
    design_index: int,
    factor: float | None,
    budget: int,
    trial_signals: NDArray[np.float64],
    full_grid_signals: NDArray[np.float64],
    trial_indices: NDArray[np.int64],
    thresholds: _Thresholds,
    config: DesignStressConfig,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    pure_rows: list[dict[str, Any]] = []
    for generator, name in enumerate(DESIGN_CANDIDATE_NAMES):
        correct = 0
        wrong = 0
        abstain = 0
        raw_correct = 0
        reasons: dict[str, int] = {}
        for replicate in range(config.evaluation_replicates):
            scores = _simulate_scores(
                trial_signals,
                trial_signals[:, generator],
                config=config,
                rng=_rng(
                    config.evaluation_seed,
                    design_index,
                    budget,
                    0,
                    generator,
                    replicate,
                ),
            )
            call, reason = _call(scores, thresholds)
            raw_correct += int(scores.winner == generator)
            correct += int(call == generator)
            wrong += int(call is not None and call != generator)
            abstain += int(call is None)
            reasons[reason] = reasons.get(reason, 0) + 1
        lower, upper = _wilson_interval(
            correct,
            config.evaluation_replicates,
            config.confidence_level,
        )
        pure_rows.append(
            {
                "design": design,
                "budget_schedule": _budget_schedule(factor),
                "budget_factor": factor,
                "budget": budget,
                "generator": name,
                "replicates": config.evaluation_replicates,
                "correct_pure_calls": correct,
                "wrong_pure_calls": wrong,
                "abstentions": abstain,
                "correct_call_rate": correct / config.evaluation_replicates,
                "wilson_lower": lower,
                "wilson_upper": upper,
                "raw_closed_set_winner_rate": (
                    raw_correct / config.evaluation_replicates
                ),
                "abstention_reasons": dict(sorted(reasons.items())),
            }
        )

    null_false = 0
    null_reasons: dict[str, int] = {}
    for replicate in range(config.evaluation_replicates):
        scores = _simulate_scores(
            trial_signals,
            np.zeros(trial_signals.shape[0], dtype=float),
            config=config,
            rng=_rng(
                config.evaluation_seed,
                design_index,
                budget,
                1,
                replicate,
            ),
        )
        call, reason = _call(scores, thresholds)
        null_false += int(call is not None)
        null_reasons[reason] = null_reasons.get(reason, 0) + 1
    lower, upper = _wilson_interval(
        null_false,
        config.evaluation_replicates,
        config.confidence_level,
    )
    null_rows = [
        {
            "design": design,
            "budget_schedule": _budget_schedule(factor),
            "budget_factor": factor,
            "budget": budget,
            "replicates": config.evaluation_replicates,
            "false_pure_calls": null_false,
            "abstentions": config.evaluation_replicates - null_false,
            "false_pure_call_rate": null_false / config.evaluation_replicates,
            "wilson_lower": lower,
            "wilson_upper": upper,
            "abstention_reasons": dict(sorted(null_reasons.items())),
        }
    ]

    mixture_rows: list[dict[str, Any]] = []
    for first, second in combinations(range(full_grid_signals.shape[1]), 2):
        mixture_grid = 0.5 * (
            full_grid_signals[:, first] + full_grid_signals[:, second]
        )
        scale = float(np.std(mixture_grid))
        if scale <= 1.0e-12:
            raise RuntimeError("a 50/50 mixture is constant on the full grid")
        mixture = mixture_grid[trial_indices] / scale
        false_calls = 0
        constituent_calls = 0
        mixture_reasons: dict[str, int] = {}
        for replicate in range(config.evaluation_replicates):
            scores = _simulate_scores(
                trial_signals,
                mixture,
                config=config,
                rng=_rng(
                    config.evaluation_seed,
                    design_index,
                    budget,
                    2,
                    first,
                    second,
                    replicate,
                ),
            )
            call, reason = _call(scores, thresholds)
            false_calls += int(call is not None)
            constituent_calls += int(call in {first, second})
            mixture_reasons[reason] = mixture_reasons.get(reason, 0) + 1
        lower, upper = _wilson_interval(
            false_calls,
            config.evaluation_replicates,
            config.confidence_level,
        )
        mixture_rows.append(
            {
                "design": design,
                "budget_schedule": _budget_schedule(factor),
                "budget_factor": factor,
                "budget": budget,
                "first_candidate": DESIGN_CANDIDATE_NAMES[first],
                "second_candidate": DESIGN_CANDIDATE_NAMES[second],
                "mixture_definition": (
                    "equal coefficients, then unit-SD scaling on full grid"
                ),
                "replicates": config.evaluation_replicates,
                "false_pure_calls": false_calls,
                "constituent_pure_calls": constituent_calls,
                "abstentions": config.evaluation_replicates - false_calls,
                "false_pure_call_rate": (
                    false_calls / config.evaluation_replicates
                ),
                "wilson_lower": lower,
                "wilson_upper": upper,
                "abstention_reasons": dict(sorted(mixture_reasons.items())),
            }
        )
    return pure_rows, null_rows, mixture_rows


def _out_of_span_probe(
    full_grid_signals: NDArray[np.float64],
) -> tuple[NDArray[np.float64], float, float]:
    """Return a fixed nonlinear surprise probe orthogonal to the linear span."""

    surprise_index = DESIGN_CANDIDATE_NAMES.index("surprise")
    raw = np.tanh(full_grid_signals[:, surprise_index])
    design = np.column_stack(
        (np.ones(full_grid_signals.shape[0]), full_grid_signals)
    )
    coefficients, _, _, _ = np.linalg.lstsq(design, raw, rcond=None)
    residual = raw - design @ coefficients
    scale = float(np.std(residual))
    if scale <= 1.0e-12:
        raise RuntimeError("the nonlinear open-set probe is numerically degenerate")
    probe = residual / scale
    maximum_inner_product = float(
        np.max(np.abs(design.T @ probe)) / full_grid_signals.shape[0]
    )
    return np.asarray(probe, dtype=float), scale, maximum_inner_product


def _out_of_span_rows(
    design: str,
    design_index: int,
    factor: float | None,
    budget: int,
    trial_signals: NDArray[np.float64],
    full_grid_signals: NDArray[np.float64],
    trial_indices: NDArray[np.int64],
    thresholds: _Thresholds,
    config: DesignStressConfig,
) -> list[dict[str, Any]]:
    probe, residual_scale, maximum_inner_product = _out_of_span_probe(
        full_grid_signals
    )
    generator = probe[trial_indices]
    false_calls = 0
    reasons: dict[str, int] = {}
    pure_call_counts = {name: 0 for name in DESIGN_CANDIDATE_NAMES}
    for replicate in range(config.evaluation_replicates):
        scores = _simulate_scores(
            trial_signals,
            generator,
            config=config,
            rng=_rng(
                config.evaluation_seed,
                design_index,
                budget,
                3,
                replicate,
            ),
        )
        call, reason = _call(scores, thresholds)
        false_calls += int(call is not None)
        if call is not None:
            pure_call_counts[DESIGN_CANDIDATE_NAMES[call]] += 1
        reasons[reason] = reasons.get(reason, 0) + 1
    lower, upper = _wilson_interval(
        false_calls,
        config.evaluation_replicates,
        config.confidence_level,
    )
    return [
        {
            "design": design,
            "budget_schedule": _budget_schedule(factor),
            "budget_factor": factor,
            "budget": budget,
            "probe": "full_grid_orthogonalized_tanh_surprise",
            "probe_definition": (
                "unit-SD residual of tanh(standardized surprise) after full-grid "
                "OLS projection on intercept plus all six standardized candidates"
            ),
            "full_grid_prestandardization_residual_sd": residual_scale,
            "full_grid_maximum_absolute_mean_inner_product": (
                maximum_inner_product
            ),
            "replicates": config.evaluation_replicates,
            "false_pure_calls": false_calls,
            "abstentions": config.evaluation_replicates - false_calls,
            "false_pure_call_rate": false_calls / config.evaluation_replicates,
            "wilson_lower": lower,
            "wilson_upper": upper,
            "pure_call_counts": dict(sorted(pure_call_counts.items())),
            "abstention_reasons": dict(sorted(reasons.items())),
        }
    ]


def run_design_stress(
    config: DesignStressConfig | None = None,
    *,
    allocation_overrides: Mapping[
        tuple[str, int],
        NDArray[np.int64],
    ]
    | None = None,
) -> DesignStressResult:
    """Run a bounded post-freeze pure/null/mixture sensitivity analysis."""

    config = DesignStressConfig() if config is None else config
    config.validate()
    grid_rows, _, standardized = generate_transition_design_grid()
    base_targets = _base_targets(grid_rows, standardized, config)
    overrides = {} if allocation_overrides is None else dict(allocation_overrides)
    used_overrides: set[tuple[str, int]] = set()

    threshold_rows: list[dict[str, Any]] = []
    calibration_rows: list[dict[str, Any]] = []
    pure_rows: list[dict[str, Any]] = []
    null_rows: list[dict[str, Any]] = []
    mixture_rows: list[dict[str, Any]] = []
    out_of_span_rows: list[dict[str, Any]] = []
    allocation_rows: list[dict[str, Any]] = []

    schedules_per_design = len(config.fixed_budgets) + len(config.budget_factors)
    for design_index, design in enumerate(STRESS_DESIGNS):
        schedules = [
            ("fixed_budget", None, budget)
            for budget in config.fixed_budgets
        ] + [
            (
                "n_eff_factor",
                factor,
                int(math.ceil(factor * base_targets[design])),
            )
            for factor in config.budget_factors
        ]
        budgets = [budget for _, _, budget in schedules]
        if len(set(budgets)) != len(budgets):
            raise ValueError(
                f"stress schedules duplicate a budget for design {design}"
            )
        for budget_schedule, factor, budget in schedules:
            maximum_count = max(
                1,
                int(math.ceil(config.max_point_fraction * budget)),
            )
            key = (design, budget)
            if key in overrides:
                counts = _validated_override(
                    overrides[key],
                    point_count=len(grid_rows),
                    budget=budget,
                    maximum_count=maximum_count,
                )
                allocation_source = "frozen_override"
                used_overrides.add(key)
            else:
                counts = _design_counts(
                    design,
                    grid_rows,
                    standardized,
                    budget,
                    config,
                )
                allocation_source = (
                    "deterministic_greedy_exchange"
                    if design == "maximin_optimized"
                    else "declared_baseline_constructor"
                )
            indices = np.repeat(np.arange(counts.size), counts)
            trial_signals = standardized[indices]
            diagnostics = design_diagnostics(
                standardized,
                counts,
                effect_size=config.effect_size,
                noise_std=config.noise_std,
                target_log_score_gap=config.target_log_score_gap,
            )
            for point_index in np.flatnonzero(counts):
                allocation_rows.append(
                    {
                        "design": design,
                        "budget_schedule": budget_schedule,
                        "budget_factor": factor,
                        "budget": budget,
                        "allocation_source": allocation_source,
                        "point_id": int(point_index),
                        "count": int(counts[point_index]),
                    }
                )

            thresholds = _calibrate_thresholds(
                trial_signals,
                config=config,
                design_index=design_index,
                budget=budget,
            )
            threshold_rows.append(
                {
                    "design": design,
                    "budget_schedule": budget_schedule,
                    "budget_factor": factor,
                    "budget": budget,
                    "allocation_source": allocation_source,
                    "base_n_eff_index_from_n60_geometry": base_targets[design],
                    "realized_minimum_pairwise_residual": (
                        diagnostics.minimum_pairwise_residual_variance
                    ),
                    "realized_population_n_eff_index": (
                        diagnostics.trials_for_expected_log_score_gap_target
                    ),
                    "pure_over_null_threshold": thresholds.pure_over_null,
                    "winner_over_runner_threshold": thresholds.winner_over_runner,
                    "flexible_over_pure_threshold": thresholds.flexible_over_pure,
                }
            )
            calibration_rows.extend(
                _calibration_audit_rows(
                    design,
                    design_index,
                    factor,
                    budget,
                    trial_signals,
                    thresholds,
                    config,
                )
            )
            pure, null, mixture = _evaluation_rows(
                design,
                design_index,
                factor,
                budget,
                trial_signals,
                standardized,
                indices,
                thresholds,
                config,
            )
            pure_rows.extend(pure)
            null_rows.extend(null)
            mixture_rows.extend(mixture)
            out_of_span_rows.extend(
                _out_of_span_rows(
                    design,
                    design_index,
                    factor,
                    budget,
                    trial_signals,
                    standardized,
                    indices,
                    thresholds,
                    config,
                )
            )

    unused_overrides = set(overrides) - used_overrides
    if unused_overrides:
        raise ValueError(
            "allocation overrides did not match a stress budget: "
            f"{sorted(unused_overrides)}"
        )

    minimum_pure_lower = min(float(row["wilson_lower"]) for row in pure_rows)
    maximum_null_upper = max(float(row["wilson_upper"]) for row in null_rows)
    maximum_mixture_upper = max(
        float(row["wilson_upper"]) for row in mixture_rows
    )
    maximum_out_of_span_upper = max(
        float(row["wilson_upper"]) for row in out_of_span_rows
    )
    summary = {
        "schema_version": 1,
        "experiment": "post_freeze_design_abstention_sensitivity",
        "config": asdict(config),
        "candidate_names": list(DESIGN_CANDIDATE_NAMES),
        "design_names": list(STRESS_DESIGNS),
        "base_n_eff_indices_from_n60_geometry": base_targets,
        "minimum_matched_pure_wilson_lower": minimum_pure_lower,
        "maximum_null_false_pure_wilson_upper": maximum_null_upper,
        "maximum_mixture_false_pure_wilson_upper": maximum_mixture_upper,
        "maximum_out_of_span_false_pure_wilson_upper": maximum_out_of_span_upper,
        "threshold_rule": {
            "pure_signal": (
                "best pure minus intercept-only null exceeds a familywise "
                "null-calibrated upper threshold"
            ),
            "ambiguity": (
                "best pure minus runner exceeds a separately null-calibrated "
                "upper threshold"
            ),
            "adequacy": (
                "all-six train-only-selected ridge model does not beat the "
                "best pure by more than the worst-candidate pure-calibrated "
                "upper threshold"
            ),
        },
        "scope": (
            "This immutable post-freeze sensitivity probes finite-training "
            "plug-in variability, a no-signal null, standardized equal "
            "50/50 in-span mixtures, and one residualized nonlinear probe. "
            "It does not establish robustness to arbitrary out-of-span "
            "biology, other nonlinear mixtures, serial "
            "dependence, sensor dynamics, subject hierarchy, or sequential "
            "protocol feasibility. A mixture pure call is counted as false "
            "even when it names a constituent."
        ),
        "technical_gates": {
            "calibration_and_evaluation_seeds_disjoint": len(
                {
                    config.threshold_seed,
                    config.calibration_audit_seed,
                    config.evaluation_seed,
                }
            )
            == 3,
            "all_fifteen_mixtures_per_design_budget": len(mixture_rows)
            == len(STRESS_DESIGNS) * schedules_per_design * 15,
            "one_out_of_span_probe_per_design_budget": len(out_of_span_rows)
            == len(STRESS_DESIGNS) * schedules_per_design,
            "all_thresholds_finite": all(
                np.isfinite(
                    [
                        row["pure_over_null_threshold"],
                        row["winner_over_runner_threshold"],
                        row["flexible_over_pure_threshold"],
                    ]
                ).all()
                for row in threshold_rows
            ),
        },
    }
    return DesignStressResult(
        summary=summary,
        thresholds=tuple(threshold_rows),
        calibration=tuple(calibration_rows),
        pure_recovery=tuple(pure_rows),
        null_evaluation=tuple(null_rows),
        mixture_evaluation=tuple(mixture_rows),
        out_of_span_evaluation=tuple(out_of_span_rows),
        allocations=tuple(allocation_rows),
    )
