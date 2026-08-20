"""Causal closed-loop triggering and eligibility-trace model recovery."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Final, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

EligibilityFamily = Literal["exponential", "alpha", "boxcar"]
ClosedLoopFamily = Literal[
    "null",
    "latency_independent",
    "exponential",
    "alpha",
    "boxcar",
]
TriggerReason = Literal[
    "triggered",
    "false_trigger",
    "below_threshold",
    "uncertainty_gate",
    "refractory",
    "missed_trigger",
]

ELIGIBILITY_FAMILIES: Final[tuple[EligibilityFamily, ...]] = (
    "exponential",
    "alpha",
    "boxcar",
)


@dataclass(frozen=True, slots=True)
class TriggerPolicyConfig:
    """Causal trigger configuration.

    ``actuation_delay`` is the independently calibrated delay between the
    recorded command and effective hippocampal cholinergic action. The policy
    reports both command and effective times.
    """

    threshold: float = 0.80
    max_uncertainty: float = 0.35
    actuation_delay: float = 0.08
    jitter_std: float = 0.015
    refractory_period: float = 0.35
    missed_trigger_probability: float = 0.04
    false_trigger_probability: float = 0.05

    def validate(self) -> None:
        if not np.isfinite(self.threshold):
            raise ValueError("threshold must be finite")
        if not np.isfinite(self.max_uncertainty) or self.max_uncertainty < 0.0:
            raise ValueError("max_uncertainty must be finite and non-negative")
        for name, value in (
            ("actuation_delay", self.actuation_delay),
            ("jitter_std", self.jitter_std),
            ("refractory_period", self.refractory_period),
        ):
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        for name, value in (
            ("missed_trigger_probability", self.missed_trigger_probability),
            ("false_trigger_probability", self.false_trigger_probability),
        ):
            if not np.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1]")


@dataclass(frozen=True, slots=True)
class TriggerDecision:
    """One causal trigger decision and its recorded timing."""

    event_time: float
    signal: float
    uncertainty: float
    nominal_command_latency: float
    accepted: bool
    reason: str
    command_time: float
    effective_time: float
    actual_command_latency: float

    def as_dict(self) -> dict[str, float | str | bool]:
        return asdict(self)


class CausalTriggerPolicy:
    """Stateful causal trigger with uncertainty, refractory, jitter, and errors."""

    def __init__(
        self,
        config: TriggerPolicyConfig | None = None,
        *,
        seed: int = 7,
    ) -> None:
        resolved = TriggerPolicyConfig() if config is None else config
        resolved.validate()
        self.config = resolved
        self._rng = np.random.default_rng(seed)
        self._last_command_time = float("-inf")

    @property
    def last_command_time(self) -> float:
        return self._last_command_time

    def observe(
        self,
        event_time: float,
        signal: float,
        uncertainty: float,
        *,
        command_latency: float,
    ) -> TriggerDecision:
        """Process one online candidate using information available at that time."""

        for name, value in (
            ("event_time", event_time),
            ("signal", signal),
            ("uncertainty", uncertainty),
            ("command_latency", command_latency),
        ):
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if uncertainty < 0.0:
            raise ValueError("uncertainty must be non-negative")
        if command_latency < 0.0:
            raise ValueError("command_latency must be non-negative")

        accepted = False
        reason: TriggerReason
        if uncertainty > self.config.max_uncertainty:
            reason = "uncertainty_gate"
        elif event_time < self._last_command_time + self.config.refractory_period:
            reason = "refractory"
        elif signal >= self.config.threshold:
            if self._rng.random() < self.config.missed_trigger_probability:
                reason = "missed_trigger"
            else:
                accepted = True
                reason = "triggered"
        elif self._rng.random() < self.config.false_trigger_probability:
            accepted = True
            reason = "false_trigger"
        else:
            reason = "below_threshold"

        if accepted:
            jitter = float(self._rng.normal(0.0, self.config.jitter_std))
            actual_latency = max(0.0, command_latency + jitter)
            command_time = event_time + actual_latency
            effective_time = command_time + self.config.actuation_delay
            self._last_command_time = command_time
        else:
            actual_latency = float("nan")
            command_time = float("nan")
            effective_time = float("nan")

        return TriggerDecision(
            event_time=float(event_time),
            signal=float(signal),
            uncertainty=float(uncertainty),
            nominal_command_latency=float(command_latency),
            accepted=accepted,
            reason=reason,
            command_time=command_time,
            effective_time=effective_time,
            actual_command_latency=actual_latency,
        )

    def reset(self, *, seed: int | None = None) -> None:
        """Reset refractory state and optionally the random generator."""

        self._last_command_time = float("-inf")
        if seed is not None:
            self._rng = np.random.default_rng(seed)


@dataclass(frozen=True, slots=True)
class EligibilityKernelSpec:
    """One normalized causal eligibility-trace family."""

    family: EligibilityFamily
    tau_rise: float = 0.0
    tau_decay: float = 0.0
    width: float = 0.0

    def validate(self) -> None:
        if self.family == "exponential":
            if not np.isfinite(self.tau_decay) or self.tau_decay <= 0.0:
                raise ValueError("exponential tau_decay must be finite and positive")
        elif self.family == "alpha":
            if not 0.0 < self.tau_rise < self.tau_decay:
                raise ValueError("alpha kernel requires 0 < tau_rise < tau_decay")
        elif self.family == "boxcar":
            if not np.isfinite(self.width) or self.width <= 0.0:
                raise ValueError("boxcar width must be finite and positive")
        else:  # pragma: no cover - guarded by typing, retained for runtime inputs
            raise ValueError(f"unknown eligibility family {self.family!r}")

    @property
    def name(self) -> str:
        if self.family == "exponential":
            return f"exponential_tau_{self.tau_decay:.3f}"
        if self.family == "alpha":
            return f"alpha_rise_{self.tau_rise:.3f}_decay_{self.tau_decay:.3f}"
        return f"boxcar_width_{self.width:.3f}"

    def as_dict(self) -> dict[str, float | str]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ClosedLoopModelSpec:
    """Null, latency-independent, or eligibility-gated causal model."""

    family: ClosedLoopFamily
    kernel: EligibilityKernelSpec | None = None

    def validate(self) -> None:
        if self.family in ELIGIBILITY_FAMILIES:
            if self.kernel is None or self.kernel.family != self.family:
                raise ValueError("eligibility models require a matching kernel")
            self.kernel.validate()
        elif self.family in ("null", "latency_independent"):
            if self.kernel is not None:
                raise ValueError("null and latency-independent models cannot carry a kernel")
        else:  # pragma: no cover
            raise ValueError(f"unknown closed-loop family {self.family!r}")

    @property
    def name(self) -> str:
        if self.kernel is None:
            return self.family
        return self.kernel.name


@dataclass(frozen=True, slots=True)
class ClosedLoopDataset:
    """Yoked active-minus-sham outcomes for strict held-out model comparison."""

    pair_difference: NDArray[np.float64]
    actual_command_latency: NDArray[np.float64]
    eligibility_amplitude: NDArray[np.float64]
    train_mask: NDArray[np.bool_]
    subject_ids: NDArray[np.int64]
    session_ids: NDArray[np.int64]
    pair_ids: NDArray[np.int64]
    true_event: NDArray[np.bool_]
    false_trigger: NDArray[np.bool_]

    def validate(self) -> None:
        n = int(np.asarray(self.pair_difference).size)
        if n < 4:
            raise ValueError("at least four yoked pairs are required")
        one_dimensional = (
            ("pair_difference", self.pair_difference),
            ("actual_command_latency", self.actual_command_latency),
            ("eligibility_amplitude", self.eligibility_amplitude),
            ("train_mask", self.train_mask),
            ("subject_ids", self.subject_ids),
            ("session_ids", self.session_ids),
            ("pair_ids", self.pair_ids),
            ("true_event", self.true_event),
            ("false_trigger", self.false_trigger),
        )
        for name, values in one_dimensional:
            array = np.asarray(values)
            if array.ndim != 1 or array.size != n:
                raise ValueError(f"{name} must have shape ({n},); got {array.shape}")
        for name, values in (
            ("pair_difference", self.pair_difference),
            ("actual_command_latency", self.actual_command_latency),
            ("eligibility_amplitude", self.eligibility_amplitude),
        ):
            if not np.all(np.isfinite(values)):
                raise ValueError(f"{name} must contain finite values")
        if np.any(self.actual_command_latency < 0.0):
            raise ValueError("actual_command_latency must be non-negative")
        if np.any(self.eligibility_amplitude < 0.0):
            raise ValueError("eligibility_amplitude must be non-negative")
        train_count = int(np.sum(self.train_mask))
        if train_count < 2 or n - train_count < 2:
            raise ValueError("train_mask must define at least two train and two test pairs")
        if np.unique(self.pair_ids).size != n:
            raise ValueError("pair_ids must be unique")


@dataclass(frozen=True, slots=True)
class ClosedLoopFitConfig:
    """Known-noise Bayesian linear-model comparison for yoked pair differences."""

    difference_noise_std: float
    actuation_delay: float
    coefficient_prior_std: float = 1.5
    claim_log_evidence_threshold: float = 5.0
    model_grid: tuple[ClosedLoopModelSpec, ...] = ()

    def validate(self) -> None:
        if not np.isfinite(self.difference_noise_std) or self.difference_noise_std <= 0.0:
            raise ValueError("difference_noise_std must be finite and positive")
        if not np.isfinite(self.actuation_delay) or self.actuation_delay < 0.0:
            raise ValueError("actuation_delay must be finite and non-negative")
        if not np.isfinite(self.coefficient_prior_std) or self.coefficient_prior_std <= 0.0:
            raise ValueError("coefficient_prior_std must be finite and positive")
        if (
            not np.isfinite(self.claim_log_evidence_threshold)
            or self.claim_log_evidence_threshold < 0.0
        ):
            raise ValueError("claim_log_evidence_threshold must be finite and non-negative")
        for model in self.model_grid:
            model.validate()


@dataclass(frozen=True, slots=True)
class ClosedLoopModelFit:
    """Training posterior and held-out predictive evidence for one causal model."""

    model: str
    family: str
    tau_rise: float
    tau_decay: float
    width: float
    main_effect_mean: float
    main_effect_std: float
    eligibility_effect_mean: float
    eligibility_effect_std: float
    test_log_predictive: float
    test_mean_log_predictive: float
    test_r2: float
    effective_window_start: float
    effective_window_end: float
    command_window_start: float
    command_window_end: float
    n_train: int
    n_test: int

    def as_dict(self) -> dict[str, float | int | str]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ClosedLoopRecoveryResult:
    """Raw held-out scores plus a conservative nested-hypothesis decision."""

    fits: tuple[ClosedLoopModelFit, ...]
    selected_model: str
    selection_reason: str
    decision_margin: float

    @property
    def winner(self) -> ClosedLoopModelFit:
        return next(fit for fit in self.fits if fit.model == self.selected_model)

    @property
    def raw_winner(self) -> ClosedLoopModelFit:
        return self.fits[0]


def eligibility_kernel(
    lags: ArrayLike,
    spec: EligibilityKernelSpec,
) -> NDArray[np.float64]:
    """Evaluate a normalized causal eligibility kernel at event-to-arrival lags."""

    spec.validate()
    lag = np.asarray(lags, dtype=float)
    if not np.all(np.isfinite(lag)):
        raise ValueError("lags must contain finite values")
    response = np.zeros_like(lag, dtype=float)
    causal = lag >= 0.0
    t = lag[causal]
    if spec.family == "exponential":
        response[causal] = np.exp(-t / spec.tau_decay)
    elif spec.family == "alpha":
        peak_time = (
            spec.tau_rise
            * spec.tau_decay
            / (spec.tau_decay - spec.tau_rise)
            * np.log(spec.tau_decay / spec.tau_rise)
        )
        peak = np.exp(-peak_time / spec.tau_decay) - np.exp(
            -peak_time / spec.tau_rise
        )
        response[causal] = (
            np.exp(-t / spec.tau_decay) - np.exp(-t / spec.tau_rise)
        ) / peak
    else:
        response[causal] = (t <= spec.width).astype(float)
    return response


def eligibility_weight(lag: float, spec: EligibilityKernelSpec) -> float:
    """Scalar convenience wrapper around :func:`eligibility_kernel`."""

    return float(eligibility_kernel(np.array([lag], dtype=float), spec)[0])


def causal_window(
    spec: EligibilityKernelSpec,
    *,
    threshold_fraction: float = 0.10,
) -> tuple[float, float]:
    """Return the lags where the normalized kernel exceeds a peak fraction."""

    spec.validate()
    if not 0.0 < threshold_fraction < 1.0:
        raise ValueError("threshold_fraction must lie in (0, 1)")
    if spec.family == "exponential":
        return 0.0, float(-spec.tau_decay * np.log(threshold_fraction))
    if spec.family == "boxcar":
        return 0.0, spec.width
    grid = np.linspace(0.0, 12.0 * spec.tau_decay, 24001)
    response = eligibility_kernel(grid, spec)
    selected = grid[response >= threshold_fraction]
    if selected.size == 0:  # pragma: no cover
        raise FloatingPointError("alpha kernel did not cross the requested threshold")
    return float(selected[0]), float(selected[-1])


def default_closed_loop_model_grid() -> tuple[ClosedLoopModelSpec, ...]:
    """Return the preregistered family and timescale grid."""

    models: list[ClosedLoopModelSpec] = [
        ClosedLoopModelSpec("null"),
        ClosedLoopModelSpec("latency_independent"),
    ]
    for tau in (0.20, 0.35, 0.70, 1.20):
        models.append(
            ClosedLoopModelSpec(
                "exponential",
                EligibilityKernelSpec("exponential", tau_decay=tau),
            )
        )
    for tau_rise, tau_decay in ((0.08, 0.60), (0.16, 0.90), (0.25, 1.40)):
        models.append(
            ClosedLoopModelSpec(
                "alpha",
                EligibilityKernelSpec(
                    "alpha",
                    tau_rise=tau_rise,
                    tau_decay=tau_decay,
                ),
            )
        )
    for width in (0.25, 0.45, 0.75):
        models.append(
            ClosedLoopModelSpec(
                "boxcar",
                EligibilityKernelSpec("boxcar", width=width),
            )
        )
    return tuple(models)


def _design_matrix(
    model: ClosedLoopModelSpec,
    effective_lag: NDArray[np.float64],
    eligibility_amplitude: NDArray[np.float64],
) -> NDArray[np.float64]:
    model.validate()
    if model.family == "null":
        return np.empty((effective_lag.size, 0), dtype=float)
    if model.family == "latency_independent":
        return np.ones((effective_lag.size, 1), dtype=float)
    if model.kernel is None:  # pragma: no cover
        raise AssertionError("validated eligibility model lacks a kernel")
    gated = eligibility_amplitude * eligibility_kernel(effective_lag, model.kernel)
    return np.column_stack((np.ones(effective_lag.size, dtype=float), gated))


def _posterior(
    design: NDArray[np.float64],
    outcome: NDArray[np.float64],
    *,
    noise_variance: float,
    prior_variance: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    p = design.shape[1]
    if p == 0:
        return np.empty(0, dtype=float), np.empty((0, 0), dtype=float)
    precision = np.eye(p, dtype=float) / prior_variance
    precision += design.T @ design / noise_variance
    covariance = np.asarray(np.linalg.inv(precision), dtype=np.float64)
    mean = np.asarray(
        covariance @ (design.T @ outcome / noise_variance),
        dtype=np.float64,
    )
    return mean, covariance


def _joint_predictive_log_score(
    outcome: NDArray[np.float64],
    design: NDArray[np.float64],
    posterior_mean: NDArray[np.float64],
    posterior_covariance: NDArray[np.float64],
    *,
    noise_variance: float,
) -> tuple[float, NDArray[np.float64]]:
    prediction = design @ posterior_mean
    residual = outcome - prediction
    n = outcome.size
    p = design.shape[1]
    if p == 0:
        log_determinant = n * np.log(noise_variance)
        quadratic = float(residual @ residual / noise_variance)
    else:
        chol = np.linalg.cholesky(posterior_covariance)
        low_rank = design @ chol
        small = np.eye(p, dtype=float) + low_rank.T @ low_rank / noise_variance
        sign, small_logdet = np.linalg.slogdet(small)
        if sign <= 0.0:  # pragma: no cover
            raise FloatingPointError("predictive covariance determinant is non-positive")
        log_determinant = n * np.log(noise_variance) + small_logdet
        projected = low_rank.T @ residual / noise_variance
        correction = float(projected @ np.linalg.solve(small, projected))
        quadratic = float(residual @ residual / noise_variance - correction)
    score = -0.5 * (n * np.log(2.0 * np.pi) + log_determinant + quadratic)
    return float(score), prediction


def fit_closed_loop_models(
    dataset: ClosedLoopDataset,
    config: ClosedLoopFitConfig,
) -> ClosedLoopRecoveryResult:
    """Compare delay-sensitive and delay-insensitive causal hypotheses."""

    dataset.validate()
    config.validate()
    models = config.model_grid or default_closed_loop_model_grid()
    for model in models:
        model.validate()
    names = [model.name for model in models]
    if len(set(names)) != len(names):
        raise ValueError("model_grid model names must be unique")
    families = {model.family for model in models}
    if "null" not in families or "latency_independent" not in families:
        raise ValueError("model_grid must include null and latency-independent models")
    if not families.intersection(ELIGIBILITY_FAMILIES):
        raise ValueError("model_grid must include at least one eligibility model")

    y = np.asarray(dataset.pair_difference, dtype=float)
    train = np.asarray(dataset.train_mask, dtype=bool)
    test = ~train
    command_latency = np.asarray(dataset.actual_command_latency, dtype=float)
    amplitude = np.asarray(dataset.eligibility_amplitude, dtype=float)
    effective_lag = command_latency + config.actuation_delay
    noise_variance = config.difference_noise_std**2
    prior_variance = config.coefficient_prior_std**2
    fits: list[ClosedLoopModelFit] = []

    for model in models:
        design = _design_matrix(model, effective_lag, amplitude)
        mean, covariance = _posterior(
            design[train],
            y[train],
            noise_variance=noise_variance,
            prior_variance=prior_variance,
        )
        score, prediction = _joint_predictive_log_score(
            y[test],
            design[test],
            mean,
            covariance,
            noise_variance=noise_variance,
        )
        denominator = float(np.sum((y[test] - float(np.mean(y[test]))) ** 2))
        test_r2 = (
            float("nan")
            if denominator <= 0.0
            else 1.0 - float(np.sum((y[test] - prediction) ** 2)) / denominator
        )

        main_mean = float("nan")
        main_std = float("nan")
        eligibility_mean = float("nan")
        eligibility_std = float("nan")
        if model.family == "latency_independent":
            main_mean = float(mean[0])
            main_std = float(np.sqrt(covariance[0, 0]))
        elif model.family in ELIGIBILITY_FAMILIES:
            main_mean = float(mean[0])
            main_std = float(np.sqrt(covariance[0, 0]))
            eligibility_mean = float(mean[1])
            eligibility_std = float(np.sqrt(covariance[1, 1]))

        tau_rise = 0.0
        tau_decay = 0.0
        width = 0.0
        effective_start = float("nan")
        effective_end = float("nan")
        command_start = float("nan")
        command_end = float("nan")
        if model.kernel is not None:
            tau_rise = model.kernel.tau_rise
            tau_decay = model.kernel.tau_decay
            width = model.kernel.width
            effective_start, effective_end = causal_window(model.kernel)
            command_start = max(0.0, effective_start - config.actuation_delay)
            command_end = max(0.0, effective_end - config.actuation_delay)

        fits.append(
            ClosedLoopModelFit(
                model=model.name,
                family=model.family,
                tau_rise=tau_rise,
                tau_decay=tau_decay,
                width=width,
                main_effect_mean=main_mean,
                main_effect_std=main_std,
                eligibility_effect_mean=eligibility_mean,
                eligibility_effect_std=eligibility_std,
                test_log_predictive=score,
                test_mean_log_predictive=score / int(np.sum(test)),
                test_r2=test_r2,
                effective_window_start=effective_start,
                effective_window_end=effective_end,
                command_window_start=command_start,
                command_window_end=command_end,
                n_train=int(np.sum(train)),
                n_test=int(np.sum(test)),
            )
        )

    fits.sort(key=lambda item: item.test_log_predictive, reverse=True)
    by_family = {fit.family: fit for fit in fits if fit.family in ("null", "latency_independent")}
    null_fit = by_family["null"]
    main_fit = by_family["latency_independent"]
    threshold = config.claim_log_evidence_threshold
    eligibility_fits = [fit for fit in fits if fit.family in ELIGIBILITY_FAMILIES]
    best_eligibility = max(eligibility_fits, key=lambda item: item.test_log_predictive)

    main_gain = main_fit.test_log_predictive - null_fit.test_log_predictive
    if main_gain > threshold:
        selected = main_fit
        reason = "latency-independent effect cleared the preregistered null threshold"
        simpler_score = main_fit.test_log_predictive
        main_boundary_margin = main_gain - threshold
    else:
        selected = null_fit
        reason = "no stimulation effect cleared the preregistered null threshold"
        simpler_score = null_fit.test_log_predictive
        main_boundary_margin = threshold - main_gain

    eligibility_gain = best_eligibility.test_log_predictive - simpler_score
    if eligibility_gain > threshold:
        selected = best_eligibility
        reason = "latency-dependent eligibility effect cleared the simpler-model threshold"
        decision_margin = eligibility_gain - threshold
    elif selected.family == "latency_independent":
        decision_margin = min(
            main_boundary_margin,
            threshold - eligibility_gain,
        )
    else:
        best_alternative_gain = max(
            main_fit.test_log_predictive,
            best_eligibility.test_log_predictive,
        ) - null_fit.test_log_predictive
        decision_margin = threshold - best_alternative_gain

    return ClosedLoopRecoveryResult(
        fits=tuple(fits),
        selected_model=selected.model,
        selection_reason=reason,
        decision_margin=float(decision_margin),
    )
