from bayesian_ach.model_recovery import fit_candidate_models, generate_synthetic_ach
from bayesian_ach.signals import CANDIDATE_SIGNAL_NAMES
from bayesian_ach.simulation import FactorialDesignConfig, simulate_factorial_design


def test_factorial_design_recovers_each_generating_candidate() -> None:
    rows = simulate_factorial_design(FactorialDesignConfig(n_trials=3000, seed=41))

    for index, generator in enumerate(CANDIDATE_SIGNAL_NAMES):
        ach = generate_synthetic_ach(
            rows,
            generator,
            noise_std=0.12,
            seed=100 + index,
        )
        fits = fit_candidate_models(rows, ach, seed=200 + index)
        assert fits[0].candidate == generator
        assert fits[0].test_r2 > 0.95
