"""Bayesian-ACh public API."""

from bayesian_ach.attribution import (
    MECHANISM_NAMES,
    ObservationAttributionConfig,
    ObservationAttributionResult,
    ObservationSequenceResult,
    run_observation_attribution,
)
from bayesian_ach.changepoint import ChangePointStep, DirichletBOCPD
from bayesian_ach.dirichlet import DirichletTransitionModel
from bayesian_ach.measurement import (
    CalibrationPosterior,
    MeasurementCandidateFit,
    MeasurementDataset,
    MeasurementFitConfig,
    MeasurementGridPoint,
    MeasurementRecoveryResult,
    convolve_by_session,
    default_measurement_grid,
    double_exponential_kernel,
    fit_measurement_models,
    tonic_sensor_ar_coefficients,
)
from bayesian_ach.measurement_benchmark import (
    MEASUREMENT_CANDIDATE_NAMES,
    MEASUREMENT_NUISANCE_NAMES,
    MeasurementBenchmarkConfig,
    MeasurementBenchmarkResult,
    MeasurementGeneratorResult,
    run_measurement_benchmark,
)
from bayesian_ach.model_recovery import (
    CandidateFit,
    fit_candidate_models,
    generate_synthetic_ach,
)
from bayesian_ach.observation import MultisensoryContextFilter, MultisensoryStep
from bayesian_ach.regime import (
    RegimeRecoveryConfig,
    RegimeRecoveryResult,
    RegimeSequenceResult,
    ring_transition_kernels,
    run_regime_recovery,
)
from bayesian_ach.signals import (
    CANDIDATE_SIGNAL_NAMES,
    TransitionSignals,
    compute_transition_signals,
    dirichlet_kl,
)
from bayesian_ach.simulation import (
    FactorialDesignConfig,
    MatchedConfidenceConfig,
    simulate_factorial_design,
    simulate_matched_confidence,
)
from bayesian_ach.switching import ContextStep, SwitchingContextFilter

__all__ = [
    "CANDIDATE_SIGNAL_NAMES",
    "MEASUREMENT_CANDIDATE_NAMES",
    "MEASUREMENT_NUISANCE_NAMES",
    "MECHANISM_NAMES",
    "CalibrationPosterior",
    "CandidateFit",
    "ChangePointStep",
    "ContextStep",
    "DirichletBOCPD",
    "DirichletTransitionModel",
    "FactorialDesignConfig",
    "MatchedConfidenceConfig",
    "MeasurementBenchmarkConfig",
    "MeasurementBenchmarkResult",
    "MeasurementCandidateFit",
    "MeasurementDataset",
    "MeasurementFitConfig",
    "MeasurementGeneratorResult",
    "MeasurementGridPoint",
    "MeasurementRecoveryResult",
    "MultisensoryContextFilter",
    "MultisensoryStep",
    "ObservationAttributionConfig",
    "ObservationAttributionResult",
    "ObservationSequenceResult",
    "RegimeRecoveryConfig",
    "RegimeRecoveryResult",
    "RegimeSequenceResult",
    "SwitchingContextFilter",
    "TransitionSignals",
    "compute_transition_signals",
    "convolve_by_session",
    "default_measurement_grid",
    "dirichlet_kl",
    "double_exponential_kernel",
    "fit_candidate_models",
    "fit_measurement_models",
    "generate_synthetic_ach",
    "ring_transition_kernels",
    "run_measurement_benchmark",
    "run_observation_attribution",
    "run_regime_recovery",
    "simulate_factorial_design",
    "simulate_matched_confidence",
    "tonic_sensor_ar_coefficients",
]

__version__ = "0.4.0"
