"""Public prospective experimental-design API."""

from bayesian_ach.design_certificate import (
    CertifiedDesignConfig,
    MaximinCertificate,
    certificate_matches_geometry,
    certify_maximin_design,
)
from bayesian_ach.design_geometry import (
    DesignDiagnostics,
    design_diagnostics,
    pairwise_residual_matrix,
    profiled_gaussian_log_score_gap,
)
from bayesian_ach.design_grid import (
    DESIGN_CANDIDATE_NAMES,
    TransitionDesignGridConfig,
    coupled_novelty_design,
    generate_transition_design_grid,
    uniform_factorial_design,
)
from bayesian_ach.design_optimizer import OptimizedDesign, optimize_maximin_design

__all__ = [
    "CertifiedDesignConfig",
    "DESIGN_CANDIDATE_NAMES",
    "DesignDiagnostics",
    "MaximinCertificate",
    "OptimizedDesign",
    "TransitionDesignGridConfig",
    "certificate_matches_geometry",
    "certify_maximin_design",
    "coupled_novelty_design",
    "design_diagnostics",
    "generate_transition_design_grid",
    "optimize_maximin_design",
    "pairwise_residual_matrix",
    "profiled_gaussian_log_score_gap",
    "uniform_factorial_design",
]
