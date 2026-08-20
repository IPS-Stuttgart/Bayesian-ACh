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
    "MECHANISM_NAMES",
    "CandidateFit",
    "ChangePointStep",
    "ContextStep",
    "DirichletBOCPD",
    "DirichletTransitionModel",
    "FactorialDesignConfig",
    "MatchedConfidenceConfig",
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
    "dirichlet_kl",
    "fit_candidate_models",
    "generate_synthetic_ach",
    "ring_transition_kernels",
    "run_observation_attribution",
    "run_regime_recovery",
    "simulate_factorial_design",
    "simulate_matched_confidence",
]

__version__ = "0.3.0"
