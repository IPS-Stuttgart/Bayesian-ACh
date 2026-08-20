"""Synthetic closed-loop recovery of eligibility windows and nonspecific effects."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any, Final

import numpy as np

from bayesian_ach.closed_loop import (
    CausalTriggerPolicy,
    ClosedLoopDataset,
    ClosedLoopFitConfig,
    ClosedLoopModelSpec,
    EligibilityKernelSpec,
    TriggerPolicyConfig,
    eligibility_weight,
    fit_closed_loop_models,
)


@dataclass(frozen=True, slots=True)
class ClosedLoopBenchmarkConfig:
    """Configuration for the delay-dependent perturbation benchmark."""

    n_subjects: int = 8
    sessions_per_subject: int = 5
    train_sessions_per_subject: int = 3
    opportunities_per_session: int = 96
    latency_values: tuple[float, ...] = (
        0.00,
        0.04,
        0.08,
        0.12,
        0.20,
        0.35,
        0.55,
        0.80,
        1.20,
        1.60,
    )
    trial_interval: float = 1.40
    cluster_interval: float = 0.18
    cluster_every: int = 12
    true_event_probability: float = 0.72
    event_signal_mean: float = 1.20
    background_signal_mean: float = 0.35
    signal_std: float = 0.18
    candidate_threshold: float = 0.80
    max_uncertainty: float = 0.35
    true_actuation_delay: float = 0.08
    assumed_actuation_delay: float = 0.08
    jitter_std: float = 0.015
    refractory_period: float = 0.35
    missed_trigger_probability: float = 0.04
    false_trigger_probability: float = 0.05
    eligibility_effect_amplitude: float = 1.00
    latency_independent_amplitude: float = 0.65
    arm_noise_std: float = 0.18
    shared_pair_noise_std: float = 0.25
    subject_baseline_std: float = 0.30
    session_baseline_std: float = 0.30
    coefficient_prior_std: float = 1.50
    claim_log_evidence_threshold: float = 5.0
    seed: int = 7

    def validate(self) -> None:
        if self.n_subjects < 2:
            raise ValueError("n_subjects must be at least two")
        if self.sessions_per_subject < 2:
            raise ValueError("sessions_per_subject must be at least two")
        if not 1 <= self.train_sessions_per_subject < self.sessions_per_subject:
            raise ValueError(
                "train_sessions_per_subject must be positive and below sessions_per_subject"
            )
        if self.opportunities_per_session < 16:
            raise ValueError("opportunities_per_session must be at least 16")
        latency_values = np.asarray(self.latency_values, dtype=float)
        if (
            latency_values.size < 4
            or not np.all(np.isfinite(latency_values))
            or np.any(latency_values < 0.0)
        ):
            raise ValueError(
                "latency_values must contain at least four finite non-negative entries"
            )
        if np.unique(latency_values).size < 4:
            raise ValueError("latency_values must contain at least four distinct values")
        if (
            not np.isfinite(self.trial_interval)
            or not np.isfinite(self.cluster_interval)
            or self.trial_interval <= 0.0
            or self.cluster_interval <= 0.0
        ):
            raise ValueError("trial intervals must be finite and positive")
        if self.cluster_every < 2:
            raise ValueError("cluster_every must be at least two")
        for name, value in (
            ("true_event_probability", self.true_event_probability),
            ("missed_trigger_probability", self.missed_trigger_probability),
            ("false_trigger_probability", self.false_trigger_probability),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1]")
        for name, value in (
            ("event_signal_mean", self.event_signal_mean),
            ("background_signal_mean", self.background_signal_mean),
            ("signal_std", self.signal_std),
            ("candidate_threshold", self.candidate_threshold),
            ("max_uncertainty", self.max_uncertainty),
            ("true_actuation_delay", self.true_actuation_delay),
            ("assumed_actuation_delay", self.assumed_actuation_delay),
            ("jitter_std", self.jitter_std),
            ("refractory_period", self.refractory_period),
            ("eligibility_effect_amplitude", self.eligibility_effect_amplitude),
            ("latency_independent_amplitude", self.latency_independent_amplitude),
            ("arm_noise_std", self.arm_noise_std),
            ("shared_pair_noise_std", self.shared_pair_noise_std),
            ("subject_baseline_std", self.subject_baseline_std),
            ("session_baseline_std", self.session_baseline_std),
            ("coefficient_prior_std", self.coefficient_prior_std),
            ("claim_log_evidence_threshold", self.claim_log_evidence_threshold),
        ):
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.arm_noise_std <= 0.0 or self.coefficient_prior_std <= 0.0:
            raise ValueError("arm_noise_std and coefficient_prior_std must be positive")


@dataclass(frozen=True, slots=True)
class ClosedLoopGeneratorResult:
    """Recovery summary for one causal generating mechanism."""

    generator: str
    true_family: str
    winner: str
    raw_winner: str
    winner_family: str
    selection_reason: str
    correct: bool
    family_correct: bool
    evidence_margin: float
    winner_test_r2: float
    winner_tau_rise: float
    winner_tau_decay: float
    winner_width: float
    command_window_start: float
    command_window_end: float
    main_effect_mean: float
    eligibility_effect_mean: float

    def as_dict(self) -> dict[str, float | str | bool]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ClosedLoopBenchmarkResult:
    """Complete schedule, outcome, and model-recovery evidence package."""

    generators: tuple[ClosedLoopGeneratorResult, ...]
    fits: tuple[dict[str, Any], ...]
    pairs: tuple[dict[str, Any], ...]
    opportunities: tuple[dict[str, Any], ...]
    summary: dict[str, Any]


@dataclass(frozen=True, slots=True)
class _AcceptedPair:
    pair_id: int
    subject_id: int
    session_id: int
    train: bool
    event_time: float
    signal: float
    uncertainty: float
    true_event: bool
    false_trigger: bool
    eligibility_amplitude: float
    true_eligibility_amplitude: float
    nominal_command_latency: float
    actual_command_latency: float
    command_time: float
    effective_time: float


NULL_MODEL = ClosedLoopModelSpec("null")
MAIN_MODEL = ClosedLoopModelSpec("latency_independent")
EXPONENTIAL_MODEL = ClosedLoopModelSpec(
    "exponential",
    EligibilityKernelSpec("exponential", tau_decay=0.35),
)
ALPHA_MODEL = ClosedLoopModelSpec(
    "alpha",
    EligibilityKernelSpec("alpha", tau_rise=0.08, tau_decay=0.60),
)
BOXCAR_MODEL = ClosedLoopModelSpec(
    "boxcar",
    EligibilityKernelSpec("boxcar", width=0.45),
)
GENERATOR_MODELS: Final[tuple[ClosedLoopModelSpec, ...]] = (
    NULL_MODEL,
    MAIN_MODEL,
    EXPONENTIAL_MODEL,
    ALPHA_MODEL,
    BOXCAR_MODEL,
)


def _schedule(
    config: ClosedLoopBenchmarkConfig,
) -> tuple[list[_AcceptedPair], list[dict[str, Any]]]:
    rng = np.random.default_rng(config.seed)
    pairs: list[_AcceptedPair] = []
    rows: list[dict[str, Any]] = []
    pair_id = 0
    for subject in range(config.n_subjects):
        for session_within_subject in range(config.sessions_per_subject):
            session_id = subject * config.sessions_per_subject + session_within_subject
            train = session_within_subject < config.train_sessions_per_subject
            policy = CausalTriggerPolicy(
                TriggerPolicyConfig(
                    threshold=config.candidate_threshold,
                    max_uncertainty=config.max_uncertainty,
                    actuation_delay=config.true_actuation_delay,
                    jitter_std=config.jitter_std,
                    refractory_period=config.refractory_period,
                    missed_trigger_probability=config.missed_trigger_probability,
                    false_trigger_probability=config.false_trigger_probability,
                ),
                seed=config.seed + 1009 * (session_id + 1),
            )
            latency_rng = np.random.default_rng(
                config.seed + 104729 * (session_id + 1)
            )
            latency_schedule = np.resize(
                np.asarray(config.latency_values, dtype=float),
                config.opportunities_per_session,
            )
            latency_rng.shuffle(latency_schedule)
            event_time = 0.0
            for opportunity in range(config.opportunities_per_session):
                if opportunity > 0:
                    event_time += (
                        config.cluster_interval
                        if opportunity % config.cluster_every == 0
                        else config.trial_interval
                    )
                true_event = bool(rng.random() < config.true_event_probability)
                if true_event:
                    signal = float(rng.normal(config.event_signal_mean, config.signal_std))
                    uncertainty = float(rng.uniform(0.04, 0.32))
                else:
                    signal = float(rng.normal(config.background_signal_mean, config.signal_std))
                    uncertainty = float(rng.uniform(0.08, 0.55))
                signal = max(0.0, signal)
                nominal_latency = float(latency_schedule[opportunity])
                decision = policy.observe(
                    event_time,
                    signal,
                    uncertainty,
                    command_latency=nominal_latency,
                )
                row: dict[str, Any] = {
                    "subject_id": subject,
                    "session_id": session_id,
                    "session_within_subject": session_within_subject,
                    "split": "train" if train else "test",
                    "opportunity": opportunity,
                    "true_event": int(true_event),
                    **decision.as_dict(),
                }
                rows.append(row)
                if not decision.accepted:
                    continue
                false_trigger = not true_event
                eligibility_amplitude = (
                    0.45 + max(0.0, signal - config.candidate_threshold)
                    if decision.reason == "triggered"
                    else 0.0
                )
                true_eligibility_amplitude = (
                    eligibility_amplitude if true_event else 0.0
                )
                pairs.append(
                    _AcceptedPair(
                        pair_id=pair_id,
                        subject_id=subject,
                        session_id=session_id,
                        train=train,
                        event_time=event_time,
                        signal=signal,
                        uncertainty=uncertainty,
                        true_event=true_event,
                        false_trigger=false_trigger,
                        eligibility_amplitude=eligibility_amplitude,
                        true_eligibility_amplitude=true_eligibility_amplitude,
                        nominal_command_latency=nominal_latency,
                        actual_command_latency=decision.actual_command_latency,
                        command_time=decision.command_time,
                        effective_time=decision.effective_time,
                    )
                )
                pair_id += 1
    if len(pairs) < 20:
        raise RuntimeError("trigger policy produced too few accepted pairs")
    return pairs, rows


def _effects(
    model: ClosedLoopModelSpec,
    pair: _AcceptedPair,
    config: ClosedLoopBenchmarkConfig,
) -> tuple[float, float]:
    if model.family == "null":
        return 0.0, 0.0
    if model.family == "latency_independent":
        return config.latency_independent_amplitude, 0.0
    if model.kernel is None:  # pragma: no cover
        raise AssertionError("eligibility generator lacks a kernel")
    lag = pair.actual_command_latency + config.true_actuation_delay
    gated = pair.true_eligibility_amplitude * eligibility_weight(lag, model.kernel)
    return 0.0, config.eligibility_effect_amplitude * gated


def _generator_data(
    model: ClosedLoopModelSpec,
    accepted: list[_AcceptedPair],
    config: ClosedLoopBenchmarkConfig,
    *,
    generator_index: int,
) -> tuple[ClosedLoopDataset, list[dict[str, Any]]]:
    rng = np.random.default_rng(config.seed + 7919 * (generator_index + 1))
    subject_baselines = rng.normal(0.0, config.subject_baseline_std, config.n_subjects)
    n_sessions = config.n_subjects * config.sessions_per_subject
    session_baselines = rng.normal(0.0, config.session_baseline_std, n_sessions)
    differences = np.empty(len(accepted), dtype=float)
    rows: list[dict[str, Any]] = []

    for index, pair in enumerate(accepted):
        main_effect, eligibility_effect = _effects(model, pair, config)
        shared = (
            subject_baselines[pair.subject_id]
            + session_baselines[pair.session_id]
            + float(rng.normal(0.0, config.shared_pair_noise_std))
        )
        sham = shared + float(rng.normal(0.0, config.arm_noise_std))
        active = (
            shared
            + main_effect
            + eligibility_effect
            + float(rng.normal(0.0, config.arm_noise_std))
        )
        difference = active - sham
        differences[index] = difference
        rows.append(
            {
                "generator": model.name,
                "pair_id": pair.pair_id,
                "subject_id": pair.subject_id,
                "session_id": pair.session_id,
                "split": "train" if pair.train else "test",
                "event_time": pair.event_time,
                "signal": pair.signal,
                "uncertainty": pair.uncertainty,
                "true_event": int(pair.true_event),
                "false_trigger": int(pair.false_trigger),
                "eligibility_amplitude": pair.eligibility_amplitude,
                "true_eligibility_amplitude": pair.true_eligibility_amplitude,
                "nominal_command_latency": pair.nominal_command_latency,
                "actual_command_latency": pair.actual_command_latency,
                "active_command_time": pair.command_time,
                "sham_command_time": pair.command_time,
                "effective_time": pair.effective_time,
                "true_main_effect": main_effect,
                "true_eligibility_effect": eligibility_effect,
                "active_outcome": active,
                "sham_outcome": sham,
                "pair_difference": difference,
            }
        )

    dataset = ClosedLoopDataset(
        pair_difference=differences,
        actual_command_latency=np.asarray(
            [pair.actual_command_latency for pair in accepted], dtype=float
        ),
        eligibility_amplitude=np.asarray(
            [pair.eligibility_amplitude for pair in accepted], dtype=float
        ),
        train_mask=np.asarray([pair.train for pair in accepted], dtype=bool),
        subject_ids=np.asarray([pair.subject_id for pair in accepted], dtype=np.int64),
        session_ids=np.asarray([pair.session_id for pair in accepted], dtype=np.int64),
        pair_ids=np.asarray([pair.pair_id for pair in accepted], dtype=np.int64),
        true_event=np.asarray([pair.true_event for pair in accepted], dtype=bool),
        false_trigger=np.asarray([pair.false_trigger for pair in accepted], dtype=bool),
    )
    return dataset, rows


def run_closed_loop_benchmark(
    config: ClosedLoopBenchmarkConfig,
) -> ClosedLoopBenchmarkResult:
    """Recover eligibility kernels, a nonspecific main effect, and the null."""

    config.validate()
    accepted, opportunity_rows = _schedule(config)
    generator_results: list[ClosedLoopGeneratorResult] = []
    fit_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    difference_noise_std = float(np.sqrt(2.0) * config.arm_noise_std)

    for generator_index, model in enumerate(GENERATOR_MODELS):
        dataset, rows = _generator_data(
            model,
            accepted,
            config,
            generator_index=generator_index,
        )
        pair_rows.extend(rows)
        recovery = fit_closed_loop_models(
            dataset,
            ClosedLoopFitConfig(
                difference_noise_std=difference_noise_std,
                actuation_delay=config.assumed_actuation_delay,
                coefficient_prior_std=config.coefficient_prior_std,
                claim_log_evidence_threshold=config.claim_log_evidence_threshold,
            ),
        )
        winner = recovery.winner
        generator_results.append(
            ClosedLoopGeneratorResult(
                generator=model.name,
                true_family=model.family,
                winner=winner.model,
                raw_winner=recovery.raw_winner.model,
                winner_family=winner.family,
                selection_reason=recovery.selection_reason,
                correct=winner.model == model.name,
                family_correct=winner.family == model.family,
                evidence_margin=recovery.decision_margin,
                winner_test_r2=winner.test_r2,
                winner_tau_rise=winner.tau_rise,
                winner_tau_decay=winner.tau_decay,
                winner_width=winner.width,
                command_window_start=winner.command_window_start,
                command_window_end=winner.command_window_end,
                main_effect_mean=winner.main_effect_mean,
                eligibility_effect_mean=winner.eligibility_effect_mean,
            )
        )
        for rank, fit in enumerate(recovery.fits, start=1):
            fit_rows.append(
                {
                    "generator": model.name,
                    "rank": rank,
                    **fit.as_dict(),
                }
            )

    trigger_counts = Counter(str(row["reason"]) for row in opportunity_rows)
    results = tuple(generator_results)
    recovery_count = sum(result.correct for result in results)
    family_recovery_count = sum(result.family_correct for result in results)
    max_yoked_time_difference = max(
        abs(float(row["active_command_time"]) - float(row["sham_command_time"]))
        for row in pair_rows
    )
    accepted_false = sum(pair.false_trigger for pair in accepted)
    accepted_true = sum(pair.true_event for pair in accepted)
    summary: dict[str, Any] = {
        "experiment": "closed_loop_eligibility_window_recovery",
        "config": asdict(config),
        "generator_names": [model.name for model in GENERATOR_MODELS],
        "recovery_count": recovery_count,
        "generator_count": len(GENERATOR_MODELS),
        "family_recovery_count": family_recovery_count,
        "all_generators_recovered": recovery_count == len(GENERATOR_MODELS),
        "minimum_evidence_margin": float(
            min(result.evidence_margin for result in results)
        ),
        "median_evidence_margin": float(
            np.median([result.evidence_margin for result in results])
        ),
        "accepted_pair_count": len(accepted),
        "accepted_true_event_count": int(accepted_true),
        "accepted_false_trigger_count": int(accepted_false),
        "trigger_reason_counts": dict(sorted(trigger_counts.items())),
        "maximum_active_sham_command_time_difference": max_yoked_time_difference,
        "difference_noise_std_used_for_scoring": difference_noise_std,
        "strict_separation": {
            "trigger_schedule": "candidate signal and uncertainty only; no outcome access",
            "latency_assignment": "balanced and randomized within each session",
            "active_vs_sham": "identical observation stream and recorded command time",
            "model_coefficients": "training sessions only",
            "model_scores": "held-out sessions only",
        },
        "falsification": (
            "Eligibility gating requires a held-out latency-dependent active-minus-sham "
            "interaction. A latency-independent winner supports a nonspecific stimulation "
            "effect; a null winner supports no causal stimulation effect."
        ),
        "delay_identifiability": (
            "The command-to-effective ACh delay is explicit but treated as independently "
            "calibrated. Under a monotone exponential trace, a constant unknown delay is "
            "confounded with effect amplitude; assumed_actuation_delay exposes sensitivity "
            "rather than claiming unsupported joint identification."
        ),
        "scope": (
            "The benchmark uses yoked active/sham pairs and known Gaussian arm noise. "
            "Real studies should estimate noise from preregistered sham repeats and preserve "
            "hardware timestamps, masking, and randomized latency assignment."
        ),
    }
    return ClosedLoopBenchmarkResult(
        generators=results,
        fits=tuple(fit_rows),
        pairs=tuple(pair_rows),
        opportunities=tuple(opportunity_rows),
        summary=summary,
    )
