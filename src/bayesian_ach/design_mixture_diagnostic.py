"""Prospectively locked post-failure mixture-aware design diagnostic."""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import combinations
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import nnls

from bayesian_ach.design_grid import (
    DESIGN_CANDIDATE_NAMES,
    generate_transition_design_grid,
)
from bayesian_ach.design_stress import (
    _out_of_span_probe,
    _upper_conformal_quantile,
    _wilson_interval,
)

_PAIR_INDICES = tuple(combinations(range(len(DESIGN_CANDIDATE_NAMES)), 2))


@dataclass(frozen=True, slots=True)
class MixtureDiagnosticConfig:
    """Settings fixed before the post-failure evaluation is generated."""

    calibration_replicates: int = 200
    calibration_audit_replicates: int = 200
    evaluation_replicates: int = 200
    folds: int = 3
    effect_size: float = 1.0
    noise_std: float = 1.0
    alpha: float = 0.05
    confidence_level: float = 0.95
    minimum_pure_retention_wilson_lower: float = 0.70
    minimum_rejection_power_wilson_lower: float = 0.70
    threshold_seed: int = 196613
    calibration_audit_seed: int = 262147
    evaluation_seed: int = 324949
    design: str = "maximin_optimized"
    budget: int = 60

    def validate(self) -> None:
        if min(
            self.calibration_replicates,
            self.calibration_audit_replicates,
            self.evaluation_replicates,
        ) < 20:
            raise ValueError("all replicate counts must be at least 20")
        if self.folds < 2:
            raise ValueError("folds must be at least two")
        if (
            not math.isfinite(self.effect_size)
            or not math.isfinite(self.noise_std)
            or self.effect_size <= 0.0
            or self.noise_std <= 0.0
        ):
            raise ValueError("effect size and noise standard deviation must be positive")
        if not 0.0 < self.alpha < 0.5:
            raise ValueError("alpha must lie in (0, 0.5)")
        if not 0.0 < self.confidence_level < 1.0:
            raise ValueError("confidence level must lie in (0, 1)")
        if not 0.0 <= self.minimum_pure_retention_wilson_lower <= 1.0:
            raise ValueError("minimum pure-retention lower bound must lie in [0, 1]")
        if not 0.0 <= self.minimum_rejection_power_wilson_lower <= 1.0:
            raise ValueError("minimum rejection-power lower bound must lie in [0, 1]")
        if len(
            {
                self.threshold_seed,
                self.calibration_audit_seed,
                self.evaluation_seed,
            }
        ) != 3:
            raise ValueError("calibration, audit, and evaluation seeds must be distinct")
        if self.design != "maximin_optimized" or self.budget != 60:
            raise ValueError("the locked diagnostic is restricted to maximin N=60")


@dataclass(frozen=True, slots=True)
class CrossFitScores:
    """Cross-fitted held-out scores and pure-model residual diagnostics."""

    pure_scores: NDArray[np.float64]
    null_score: float
    composite_score: float
    composite_pair: tuple[int, int]
    pure_residual_ratios: NDArray[np.float64]

    @property
    def winner(self) -> int:
        order = np.lexsort(
            (np.arange(self.pure_scores.size), -self.pure_scores)
        )
        return int(order[0])

    @property
    def runner(self) -> int:
        order = np.lexsort(
            (np.arange(self.pure_scores.size), -self.pure_scores)
        )
        return int(order[1])

    @property
    def pure_over_null(self) -> float:
        return float(self.pure_scores[self.winner] - self.null_score)

    @property
    def winner_over_runner(self) -> float:
        return float(
            self.pure_scores[self.winner] - self.pure_scores[self.runner]
        )

    @property
    def composite_over_winner(self) -> float:
        return float(self.composite_score - self.pure_scores[self.winner])


@dataclass(frozen=True, slots=True)
class CandidateThreshold:
    """Candidate-specific composite and lack-of-fit thresholds."""

    composite_over_pure: float
    residual_ratio: float


@dataclass(frozen=True, slots=True)
class DiagnosticThresholds:
    """Familywise null/ambiguity and candidate-specific adequacy thresholds."""

    pure_over_null: float
    winner_over_runner: float
    candidates: tuple[CandidateThreshold, ...]


@dataclass(frozen=True, slots=True)
class MixtureDiagnosticResult:
    """Tables produced by the locked post-failure diagnostic."""

    summary: dict[str, Any]
    thresholds: tuple[dict[str, Any], ...]
    calibration_audit: tuple[dict[str, Any], ...]
    pure_evaluation: tuple[dict[str, Any], ...]
    null_evaluation: tuple[dict[str, Any], ...]
    mixture_evaluation: tuple[dict[str, Any], ...]
    out_of_span_evaluation: tuple[dict[str, Any], ...]
    geometry: tuple[dict[str, Any], ...]


def _rng(seed: int, *keys: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([seed, *keys]))


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


def _pure_fold(
    predictor: NDArray[np.float64],
    response: NDArray[np.float64],
    train: NDArray[np.int64],
    validation: NDArray[np.int64],
) -> tuple[float, float]:
    train_design = np.column_stack((np.ones(train.size), predictor[train]))
    coefficients, _, _, _ = np.linalg.lstsq(
        train_design,
        response[train],
        rcond=None,
    )
    train_residual = response[train] - train_design @ coefficients
    variance = _fit_variance(train_residual)
    prediction = coefficients[0] + coefficients[1] * predictor[validation]
    validation_residual = response[validation] - prediction
    return (
        _gaussian_score(validation_residual, variance),
        float(np.sum(validation_residual**2 / variance)),
    )


def _null_fold(
    response: NDArray[np.float64],
    train: NDArray[np.int64],
    validation: NDArray[np.int64],
) -> float:
    mean = float(np.mean(response[train]))
    variance = _fit_variance(response[train] - mean)
    return _gaussian_score(response[validation] - mean, variance)


def _nonnegative_pair_fold(
    predictors: NDArray[np.float64],
    response: NDArray[np.float64],
    train: NDArray[np.int64],
    validation: NDArray[np.int64],
) -> float:
    train_predictors = predictors[train]
    predictor_mean = np.mean(train_predictors, axis=0)
    response_mean = float(np.mean(response[train]))
    coefficients, _ = nnls(
        train_predictors - predictor_mean,
        response[train] - response_mean,
    )
    intercept = response_mean - float(predictor_mean @ coefficients)
    train_prediction = intercept + train_predictors @ coefficients
    variance = _fit_variance(response[train] - train_prediction)
    validation_prediction = intercept + predictors[validation] @ coefficients
    return _gaussian_score(
        response[validation] - validation_prediction,
        variance,
    )


def crossfit_scores(
    signals: NDArray[np.float64],
    response: NDArray[np.float64],
    *,
    folds: int,
    rng: np.random.Generator,
) -> CrossFitScores:
    """Score all pure models and all pairwise nonnegative cones out of fold."""

    if signals.ndim != 2 or signals.shape[1] != len(DESIGN_CANDIDATE_NAMES):
        raise ValueError("signals must have one column per declared candidate")
    if response.shape != (signals.shape[0],):
        raise ValueError("response must match the signal rows")
    if signals.shape[0] < 2 * folds:
        raise ValueError("cross-fitting requires at least two observations per fold")
    split = tuple(
        np.asarray(values, dtype=np.int64)
        for values in np.array_split(
            np.asarray(rng.permutation(signals.shape[0]), dtype=np.int64),
            folds,
        )
    )
    pure_scores = np.zeros(signals.shape[1], dtype=float)
    residual_sums = np.zeros(signals.shape[1], dtype=float)
    pair_scores = np.zeros(len(_PAIR_INDICES), dtype=float)
    null_score = 0.0
    for fold_index, validation in enumerate(split):
        train = np.concatenate(
            [values for index, values in enumerate(split) if index != fold_index]
        )
        null_score += _null_fold(response, train, validation)
        for candidate in range(signals.shape[1]):
            score, residual_sum = _pure_fold(
                signals[:, candidate],
                response,
                train,
                validation,
            )
            pure_scores[candidate] += score
            residual_sums[candidate] += residual_sum
        for pair_index, pair in enumerate(_PAIR_INDICES):
            pair_scores[pair_index] += _nonnegative_pair_fold(
                signals[:, pair],
                response,
                train,
                validation,
            )
    best_pair_index = int(np.argmax(pair_scores))
    return CrossFitScores(
        pure_scores=np.asarray(pure_scores, dtype=float),
        null_score=float(null_score),
        composite_score=float(pair_scores[best_pair_index]),
        composite_pair=_PAIR_INDICES[best_pair_index],
        pure_residual_ratios=np.asarray(
            residual_sums / signals.shape[0],
            dtype=float,
        ),
    )


def _simulate(
    signals: NDArray[np.float64],
    generator: NDArray[np.float64],
    *,
    config: MixtureDiagnosticConfig,
    rng: np.random.Generator,
) -> CrossFitScores:
    response = (
        config.effect_size * generator
        + rng.normal(0.0, config.noise_std, size=generator.size)
    )
    return crossfit_scores(
        signals,
        np.asarray(response, dtype=float),
        folds=config.folds,
        rng=rng,
    )


def _calibrate(
    signals: NDArray[np.float64],
    *,
    config: MixtureDiagnosticConfig,
) -> DiagnosticThresholds:
    pure_over_null: list[float] = []
    winner_over_runner: list[float] = []
    null = np.zeros(signals.shape[0], dtype=float)
    for replicate in range(config.calibration_replicates):
        scores = _simulate(
            signals,
            null,
            config=config,
            rng=_rng(config.threshold_seed, 0, replicate),
        )
        pure_over_null.append(scores.pure_over_null)
        winner_over_runner.append(scores.winner_over_runner)

    candidate_thresholds: list[CandidateThreshold] = []
    for candidate in range(signals.shape[1]):
        composite_gaps: list[float] = []
        residual_ratios: list[float] = []
        for replicate in range(config.calibration_replicates):
            scores = _simulate(
                signals,
                signals[:, candidate],
                config=config,
                rng=_rng(
                    config.threshold_seed,
                    1,
                    candidate,
                    replicate,
                ),
            )
            composite_gaps.append(
                scores.composite_score - scores.pure_scores[candidate]
            )
            residual_ratios.append(scores.pure_residual_ratios[candidate])
        candidate_thresholds.append(
            CandidateThreshold(
                composite_over_pure=max(
                    0.0,
                    _upper_conformal_quantile(
                        composite_gaps,
                        alpha=config.alpha,
                    ),
                ),
                residual_ratio=_upper_conformal_quantile(
                    residual_ratios,
                    alpha=config.alpha,
                ),
            )
        )
    return DiagnosticThresholds(
        pure_over_null=_upper_conformal_quantile(
            pure_over_null,
            alpha=config.alpha,
        ),
        winner_over_runner=_upper_conformal_quantile(
            winner_over_runner,
            alpha=config.alpha,
        ),
        candidates=tuple(candidate_thresholds),
    )


def _call(
    scores: CrossFitScores,
    thresholds: DiagnosticThresholds,
    enabled: tuple[bool, ...],
) -> tuple[int | None, str]:
    winner = scores.winner
    if not enabled[winner]:
        return None, "candidate_underpowered"
    if scores.pure_over_null <= thresholds.pure_over_null:
        return None, "null_not_rejected"
    if scores.winner_over_runner <= thresholds.winner_over_runner:
        return None, "pure_ambiguity"
    candidate = thresholds.candidates[winner]
    if scores.composite_over_winner > candidate.composite_over_pure:
        return None, "pairwise_composite_better"
    if scores.pure_residual_ratios[winner] > candidate.residual_ratio:
        return None, "residual_lack_of_fit"
    return winner, "pure_call"


def _audit_power(
    signals: NDArray[np.float64],
    full_signals: NDArray[np.float64],
    indices: NDArray[np.int64],
    thresholds: DiagnosticThresholds,
    *,
    config: MixtureDiagnosticConfig,
) -> tuple[
    tuple[bool, ...],
    dict[tuple[int, int], bool],
    bool,
    bool,
    list[dict[str, Any]],
]:
    rows: list[dict[str, Any]] = []
    candidate_enabled: list[bool] = []
    all_enabled = tuple(True for _ in DESIGN_CANDIDATE_NAMES)
    for candidate, name in enumerate(DESIGN_CANDIDATE_NAMES):
        correct = 0
        wrong = 0
        pure_reasons: dict[str, int] = {}
        for replicate in range(config.calibration_audit_replicates):
            scores = _simulate(
                signals,
                signals[:, candidate],
                config=config,
                rng=_rng(
                    config.calibration_audit_seed,
                    0,
                    candidate,
                    replicate,
                ),
            )
            call, reason = _call(scores, thresholds, all_enabled)
            correct += int(call == candidate)
            wrong += int(call is not None and call != candidate)
            pure_reasons[reason] = pure_reasons.get(reason, 0) + 1
        lower, upper = _wilson_interval(
            correct,
            config.calibration_audit_replicates,
            config.confidence_level,
        )
        is_enabled = lower >= config.minimum_pure_retention_wilson_lower
        candidate_enabled.append(is_enabled)
        rows.append(
            {
                "scenario": "matched_pure",
                "candidate": name,
                "audit_measure": "correct_pure_retention_rate",
                "replicates": config.calibration_audit_replicates,
                "successes": correct,
                "wrong_pure_calls": wrong,
                "abstentions": (
                    config.calibration_audit_replicates - correct - wrong
                ),
                "rate": correct / config.calibration_audit_replicates,
                "wilson_lower": lower,
                "wilson_upper": upper,
                "minimum_wilson_lower": (
                    config.minimum_pure_retention_wilson_lower
                ),
                "contrast_enabled": is_enabled,
                "reasons": dict(sorted(pure_reasons.items())),
            }
        )

    enabled_tuple = tuple(candidate_enabled)
    null_abstentions = 0
    null_reasons: dict[str, int] = {}
    null_generator = np.zeros(signals.shape[0], dtype=float)
    for replicate in range(config.calibration_audit_replicates):
        scores = _simulate(
            signals,
            null_generator,
            config=config,
            rng=_rng(config.calibration_audit_seed, 1, replicate),
        )
        call, reason = _call(scores, thresholds, enabled_tuple)
        null_abstentions += int(call is None)
        null_reasons[reason] = null_reasons.get(reason, 0) + 1
    null_lower, null_upper = _wilson_interval(
        null_abstentions,
        config.calibration_audit_replicates,
        config.confidence_level,
    )
    null_enabled = (
        null_lower >= config.minimum_rejection_power_wilson_lower
    )
    rows.append(
        {
            "scenario": "null",
            "candidate": "null",
            "audit_measure": "correct_abstention_rate",
            "replicates": config.calibration_audit_replicates,
            "successes": null_abstentions,
            "wrong_pure_calls": (
                config.calibration_audit_replicates - null_abstentions
            ),
            "abstentions": null_abstentions,
            "rate": null_abstentions / config.calibration_audit_replicates,
            "wilson_lower": null_lower,
            "wilson_upper": null_upper,
            "minimum_wilson_lower": (
                config.minimum_rejection_power_wilson_lower
            ),
            "contrast_enabled": null_enabled,
            "reasons": dict(sorted(null_reasons.items())),
        }
    )

    pair_enabled: dict[tuple[int, int], bool] = {}
    for pair_index, pair in enumerate(_PAIR_INDICES):
        generator = _scaled_mixture(full_signals, *pair)[indices]
        abstentions = 0
        pair_reasons: dict[str, int] = {}
        for replicate in range(config.calibration_audit_replicates):
            scores = _simulate(
                signals,
                generator,
                config=config,
                rng=_rng(
                    config.calibration_audit_seed,
                    2,
                    pair_index,
                    replicate,
                ),
            )
            call, reason = _call(scores, thresholds, enabled_tuple)
            abstentions += int(call is None)
            pair_reasons[reason] = pair_reasons.get(reason, 0) + 1
        lower, upper = _wilson_interval(
            abstentions,
            config.calibration_audit_replicates,
            config.confidence_level,
        )
        is_enabled = lower >= config.minimum_rejection_power_wilson_lower
        pair_enabled[pair] = is_enabled
        rows.append(
            {
                "scenario": "fifty_fifty_mixture",
                "candidate": (
                    f"{DESIGN_CANDIDATE_NAMES[pair[0]]}+"
                    f"{DESIGN_CANDIDATE_NAMES[pair[1]]}"
                ),
                "audit_measure": "correct_abstention_rate",
                "replicates": config.calibration_audit_replicates,
                "successes": abstentions,
                "wrong_pure_calls": (
                    config.calibration_audit_replicates - abstentions
                ),
                "abstentions": abstentions,
                "rate": abstentions / config.calibration_audit_replicates,
                "wilson_lower": lower,
                "wilson_upper": upper,
                "minimum_wilson_lower": (
                    config.minimum_rejection_power_wilson_lower
                ),
                "contrast_enabled": is_enabled,
                "reasons": dict(sorted(pair_reasons.items())),
            }
        )

    probe, _, _ = _out_of_span_probe(full_signals)
    out_abstentions = 0
    out_reasons: dict[str, int] = {}
    for replicate in range(config.calibration_audit_replicates):
        scores = _simulate(
            signals,
            probe[indices],
            config=config,
            rng=_rng(config.calibration_audit_seed, 3, replicate),
        )
        call, reason = _call(scores, thresholds, enabled_tuple)
        out_abstentions += int(call is None)
        out_reasons[reason] = out_reasons.get(reason, 0) + 1
    out_lower, out_upper = _wilson_interval(
        out_abstentions,
        config.calibration_audit_replicates,
        config.confidence_level,
    )
    out_enabled = out_lower >= config.minimum_rejection_power_wilson_lower
    rows.append(
        {
            "scenario": "out_of_span_probe",
            "candidate": "full_grid_orthogonalized_tanh_surprise",
            "audit_measure": "correct_abstention_rate",
            "replicates": config.calibration_audit_replicates,
            "successes": out_abstentions,
            "wrong_pure_calls": (
                config.calibration_audit_replicates - out_abstentions
            ),
            "abstentions": out_abstentions,
            "rate": out_abstentions / config.calibration_audit_replicates,
            "wilson_lower": out_lower,
            "wilson_upper": out_upper,
            "minimum_wilson_lower": (
                config.minimum_rejection_power_wilson_lower
            ),
            "contrast_enabled": out_enabled,
            "reasons": dict(sorted(out_reasons.items())),
        }
    )
    return enabled_tuple, pair_enabled, out_enabled, null_enabled, rows


def _evaluate_pure(
    signals: NDArray[np.float64],
    thresholds: DiagnosticThresholds,
    enabled: tuple[bool, ...],
    *,
    config: MixtureDiagnosticConfig,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for candidate, name in enumerate(DESIGN_CANDIDATE_NAMES):
        correct = 0
        wrong = 0
        reasons: dict[str, int] = {}
        for replicate in range(config.evaluation_replicates):
            scores = _simulate(
                signals,
                signals[:, candidate],
                config=config,
                rng=_rng(
                    config.evaluation_seed,
                    0,
                    candidate,
                    replicate,
                ),
            )
            call, reason = _call(scores, thresholds, enabled)
            correct += int(call == candidate)
            wrong += int(call is not None and call != candidate)
            reasons[reason] = reasons.get(reason, 0) + 1
        lower, upper = _wilson_interval(
            correct,
            config.evaluation_replicates,
            config.confidence_level,
        )
        rows.append(
            {
                "candidate": name,
                "replicates": config.evaluation_replicates,
                "correct_pure_calls": correct,
                "wrong_pure_calls": wrong,
                "abstentions": config.evaluation_replicates - correct - wrong,
                "correct_call_rate": correct / config.evaluation_replicates,
                "wilson_lower": lower,
                "wilson_upper": upper,
                "candidate_enabled_from_audit": enabled[candidate],
                "reasons": dict(sorted(reasons.items())),
            }
        )
    return rows


def _evaluate_null(
    signals: NDArray[np.float64],
    thresholds: DiagnosticThresholds,
    enabled: tuple[bool, ...],
    audit_enabled: bool,
    *,
    config: MixtureDiagnosticConfig,
) -> list[dict[str, Any]]:
    false_calls = 0
    reasons: dict[str, int] = {}
    generator = np.zeros(signals.shape[0], dtype=float)
    for replicate in range(config.evaluation_replicates):
        scores = _simulate(
            signals,
            generator,
            config=config,
            rng=_rng(config.evaluation_seed, 1, replicate),
        )
        call, reason = _call(scores, thresholds, enabled)
        false_calls += int(call is not None)
        reasons[reason] = reasons.get(reason, 0) + 1
    lower, upper = _wilson_interval(
        false_calls,
        config.evaluation_replicates,
        config.confidence_level,
    )
    return [
        {
            "replicates": config.evaluation_replicates,
            "false_pure_calls": false_calls,
            "abstentions": config.evaluation_replicates - false_calls,
            "false_pure_call_rate": false_calls / config.evaluation_replicates,
            "wilson_lower": lower,
            "wilson_upper": upper,
            "contrast_enabled_from_audit": audit_enabled,
            "claim_status": (
                "evaluated"
                if audit_enabled
                else "mandatory_abstain_underpowered"
            ),
            "reasons": dict(sorted(reasons.items())),
        }
    ]


def _scaled_mixture(
    full_signals: NDArray[np.float64],
    first: int,
    second: int,
) -> NDArray[np.float64]:
    value = 0.5 * (full_signals[:, first] + full_signals[:, second])
    scale = float(np.std(value))
    if scale <= 1.0e-12:
        raise RuntimeError("mixture is constant on the full grid")
    return np.asarray(value / scale, dtype=float)


def _evaluate_mixtures(
    signals: NDArray[np.float64],
    full_signals: NDArray[np.float64],
    indices: NDArray[np.int64],
    thresholds: DiagnosticThresholds,
    enabled: tuple[bool, ...],
    pair_enabled: dict[tuple[int, int], bool],
    *,
    config: MixtureDiagnosticConfig,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for pair_index, (first, second) in enumerate(_PAIR_INDICES):
        generator = _scaled_mixture(full_signals, first, second)[indices]
        false_calls = 0
        reasons: dict[str, int] = {}
        for replicate in range(config.evaluation_replicates):
            scores = _simulate(
                signals,
                generator,
                config=config,
                rng=_rng(
                    config.evaluation_seed,
                    2,
                    pair_index,
                    replicate,
                ),
            )
            call, reason = _call(scores, thresholds, enabled)
            false_calls += int(call is not None)
            reasons[reason] = reasons.get(reason, 0) + 1
        lower, upper = _wilson_interval(
            false_calls,
            config.evaluation_replicates,
            config.confidence_level,
        )
        rows.append(
            {
                "first_candidate": DESIGN_CANDIDATE_NAMES[first],
                "second_candidate": DESIGN_CANDIDATE_NAMES[second],
                "replicates": config.evaluation_replicates,
                "false_pure_calls": false_calls,
                "abstentions": config.evaluation_replicates - false_calls,
                "false_pure_call_rate": (
                    false_calls / config.evaluation_replicates
                ),
                "wilson_lower": lower,
                "wilson_upper": upper,
                "contrast_enabled_from_audit": pair_enabled[(first, second)],
                "claim_status": (
                    "evaluated"
                    if pair_enabled[(first, second)]
                    else "mandatory_abstain_underpowered"
                ),
                "reasons": dict(sorted(reasons.items())),
            }
        )
    return rows


def _evaluate_out_of_span(
    signals: NDArray[np.float64],
    full_signals: NDArray[np.float64],
    indices: NDArray[np.int64],
    thresholds: DiagnosticThresholds,
    enabled: tuple[bool, ...],
    audit_enabled: bool,
    *,
    config: MixtureDiagnosticConfig,
) -> list[dict[str, Any]]:
    probe, residual_scale, maximum_inner_product = _out_of_span_probe(
        full_signals
    )
    generator = probe[indices]
    false_calls = 0
    reasons: dict[str, int] = {}
    for replicate in range(config.evaluation_replicates):
        scores = _simulate(
            signals,
            generator,
            config=config,
            rng=_rng(config.evaluation_seed, 3, replicate),
        )
        call, reason = _call(scores, thresholds, enabled)
        false_calls += int(call is not None)
        reasons[reason] = reasons.get(reason, 0) + 1
    lower, upper = _wilson_interval(
        false_calls,
        config.evaluation_replicates,
        config.confidence_level,
    )
    return [
        {
            "probe": "full_grid_orthogonalized_tanh_surprise",
            "replicates": config.evaluation_replicates,
            "false_pure_calls": false_calls,
            "abstentions": config.evaluation_replicates - false_calls,
            "false_pure_call_rate": false_calls / config.evaluation_replicates,
            "wilson_lower": lower,
            "wilson_upper": upper,
            "contrast_enabled_from_audit": audit_enabled,
            "claim_status": (
                "evaluated"
                if audit_enabled
                else "mandatory_abstain_underpowered"
            ),
            "full_grid_prestandardization_residual_sd": residual_scale,
            "full_grid_maximum_absolute_mean_inner_product": (
                maximum_inner_product
            ),
            "reasons": dict(sorted(reasons.items())),
        }
    ]


def _affine_residual(
    response: NDArray[np.float64],
    predictors: NDArray[np.float64],
) -> float:
    design = np.column_stack((np.ones(response.size), predictors))
    coefficients, _, _, _ = np.linalg.lstsq(design, response, rcond=None)
    residual = response - design @ coefficients
    return float(np.mean(residual**2))


def _geometry_rows(
    full_signals: NDArray[np.float64],
    indices: NDArray[np.int64],
    *,
    config: MixtureDiagnosticConfig,
) -> list[dict[str, Any]]:
    trial_signals = full_signals[indices]
    rows: list[dict[str, Any]] = []
    for first, second in _PAIR_INDICES:
        generator = _scaled_mixture(full_signals, first, second)[indices]
        pure_residuals = np.asarray(
            [
                _affine_residual(generator, trial_signals[:, [candidate]])
                for candidate in range(trial_signals.shape[1])
            ],
            dtype=float,
        )
        best = int(np.argmin(pure_residuals))
        pair_residual = _affine_residual(
            generator,
            trial_signals[:, [first, second]],
        )
        residual = float(pure_residuals[best])
        oracle_gap = 0.5 * config.budget * math.log1p(
            config.effect_size**2 * residual / config.noise_std**2
        )
        rows.append(
            {
                "first_candidate": DESIGN_CANDIDATE_NAMES[first],
                "second_candidate": DESIGN_CANDIDATE_NAMES[second],
                "best_pure_candidate": DESIGN_CANDIDATE_NAMES[best],
                "best_pure_affine_residual": residual,
                "true_pair_affine_residual": pair_residual,
                "crossfit_oracle_log_score_gap_index": oracle_gap,
                "below_five_nat_power_index": oracle_gap < 5.0,
            }
        )
    return rows


def run_mixture_diagnostic(
    counts: NDArray[np.int64],
    config: MixtureDiagnosticConfig | None = None,
) -> MixtureDiagnosticResult:
    """Run the prospectively configured post-failure diagnostic once."""

    config = MixtureDiagnosticConfig() if config is None else config
    config.validate()
    _, _, full_signals = generate_transition_design_grid()
    counts = np.asarray(counts, dtype=np.int64)
    if (
        counts.shape != (full_signals.shape[0],)
        or np.any(counts < 0)
        or int(np.sum(counts)) != config.budget
    ):
        raise ValueError("counts must be a valid locked N=60 grid allocation")
    indices = np.repeat(np.arange(counts.size), counts)
    signals = np.asarray(full_signals[indices], dtype=float)
    thresholds = _calibrate(signals, config=config)
    (
        enabled,
        pair_enabled,
        out_enabled,
        null_enabled,
        audit_rows,
    ) = _audit_power(
        signals,
        full_signals,
        indices,
        thresholds,
        config=config,
    )
    pure_rows = _evaluate_pure(
        signals,
        thresholds,
        enabled,
        config=config,
    )
    null_rows = _evaluate_null(
        signals,
        thresholds,
        enabled,
        null_enabled,
        config=config,
    )
    mixture_rows = _evaluate_mixtures(
        signals,
        full_signals,
        indices,
        thresholds,
        enabled,
        pair_enabled,
        config=config,
    )
    out_rows = _evaluate_out_of_span(
        signals,
        full_signals,
        indices,
        thresholds,
        enabled,
        out_enabled,
        config=config,
    )
    geometry_rows = _geometry_rows(
        full_signals,
        indices,
        config=config,
    )
    threshold_rows = [
        {
            "candidate": name,
            "alpha": config.alpha,
            "calibration_replicates": config.calibration_replicates,
            "upper_conformal_rank_one_based": int(
                math.ceil(
                    (config.calibration_replicates + 1)
                    * (1.0 - config.alpha)
                )
            ),
            "pure_over_null_familywise": thresholds.pure_over_null,
            "winner_over_runner_familywise": thresholds.winner_over_runner,
            "pairwise_composite_over_pure_familywise": (
                thresholds.candidates[index].composite_over_pure
            ),
            "residual_ratio_candidate_specific": (
                thresholds.candidates[index].residual_ratio
            ),
            "candidate_enabled_from_audit": enabled[index],
        }
        for index, name in enumerate(DESIGN_CANDIDATE_NAMES)
    ]
    summary = {
        "schema_version": 1,
        "experiment": "post_failure_pairwise_cone_abstention_diagnostic",
        "interpretation": (
            "Separately configured sensitivity after the immutable original "
            "stress failure; not a main-paper open-set robustness claim."
        ),
        "design": config.design,
        "budget": config.budget,
        "calibration_rule": {
            "alpha": config.alpha,
            "upper_conformal_quantile": (
                "one-based order statistic ceil((n+1)*(1-alpha))"
            ),
            "rank": int(
                math.ceil(
                    (config.calibration_replicates + 1)
                    * (1.0 - config.alpha)
                )
            ),
            "pure_over_null_family": "maximum over six pure candidates",
            "winner_runner_family": "best-versus-second-best pure gap",
            "composite_family": (
                "maximum held-out score across all 15 nonnegative pairs, "
                "calibrated separately under each matched pure candidate"
            ),
            "residual_gof_family": (
                "candidate-specific cross-fitted residual ratio"
            ),
        },
        "audit_power_definitions": {
            "matched_pure": "correct pure-call retention rate",
            "null": "correct abstention rate",
            "mixture_pair": "correct abstention rate for that exact pair",
            "out_of_span": "correct abstention rate for the fixed probe",
            "minimum_wilson_lower": (
                config.minimum_rejection_power_wilson_lower
            ),
        },
        "candidate_power_enabled": {
            name: enabled[index]
            for index, name in enumerate(DESIGN_CANDIDATE_NAMES)
        },
        "pair_power_enabled": {
            (
                f"{DESIGN_CANDIDATE_NAMES[first]}+"
                f"{DESIGN_CANDIDATE_NAMES[second]}"
            ): pair_enabled[(first, second)]
            for first, second in _PAIR_INDICES
        },
        "null_power_enabled": null_enabled,
        "out_of_span_power_enabled": out_enabled,
        "minimum_matched_pure_wilson_lower": min(
            float(row["wilson_lower"]) for row in pure_rows
        ),
        "maximum_mixture_false_pure_wilson_upper_descriptive_all": max(
            float(row["wilson_upper"]) for row in mixture_rows
        ),
        "maximum_enabled_mixture_false_pure_wilson_upper": max(
            (
                float(row["wilson_upper"])
                for row in mixture_rows
                if bool(row["contrast_enabled_from_audit"])
            ),
            default=None,
        ),
        "mixture_contrasts_enabled_count": sum(pair_enabled.values()),
        "null_false_pure_wilson_upper": float(null_rows[0]["wilson_upper"]),
        "out_of_span_false_pure_wilson_upper": float(
            out_rows[0]["wilson_upper"]
        ),
        "technical_gates": {
            "streams_disjoint": len(
                {
                    config.threshold_seed,
                    config.calibration_audit_seed,
                    config.evaluation_seed,
                }
            )
            == 3,
            "all_fifteen_pairwise_composites": len(_PAIR_INDICES) == 15,
            "three_fold_cross_fitting": config.folds == 3,
            "candidate_and_contrast_power_gates_applied": True,
            "calibration_quantile_rank_fixed_before_evaluation": True,
            "evaluation_not_used_for_thresholds": True,
        },
    }
    return MixtureDiagnosticResult(
        summary=summary,
        thresholds=tuple(threshold_rows),
        calibration_audit=tuple(audit_rows),
        pure_evaluation=tuple(pure_rows),
        null_evaluation=tuple(null_rows),
        mixture_evaluation=tuple(mixture_rows),
        out_of_span_evaluation=tuple(out_rows),
        geometry=tuple(geometry_rows),
    )
