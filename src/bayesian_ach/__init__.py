"""Bayesian-ACh public API."""

from bayesian_ach.dirichlet import DirichletTransitionModel
from bayesian_ach.model_recovery import (
    CandidateFit,
    fit_candidate_models,
    generate_synthetic_ach,
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

__all__ = [
    "CANDIDATE_SIGNAL_NAMES",
    "CandidateFit",
    "DirichletTransitionModel",
    "FactorialDesignConfig",
    "MatchedConfidenceConfig",
    "TransitionSignals",
    "compute_transition_signals",
    "dirichlet_kl",
    "fit_candidate_models",
    "generate_synthetic_ach",
    "simulate_factorial_design",
    "simulate_matched_confidence",
]

__version__ = "0.1.0"
