import pytest

from bayesian_ach.simulation import (
    FactorialDesignConfig,
    MatchedConfidenceConfig,
    simulate_factorial_design,
    simulate_matched_confidence,
)


def test_matched_confidence_pairs_match_raw_quantities() -> None:
    config = MatchedConfidenceConfig(n_pairs=20, seed=13)
    rows = simulate_matched_confidence(config)
    assert len(rows) == 40

    for pair_id in range(config.n_pairs):
        low, high = rows[2 * pair_id], rows[2 * pair_id + 1]
        assert low["pair_id"] == pair_id
        assert high["pair_id"] == pair_id
        assert low["observed_index"] == high["observed_index"]
        assert low["surprise"] == pytest.approx(high["surprise"])
        assert low["innovation_l2"] == pytest.approx(high["innovation_l2"])
        assert low["update_l2"] > high["update_l2"]


def test_factorial_design_is_deterministic_and_contains_both_sources() -> None:
    config = FactorialDesignConfig(n_trials=200, seed=23)
    first = simulate_factorial_design(config)
    second = simulate_factorial_design(config)
    assert first == second
    assert {row["condition"] for row in first} == {"stable", "reset"}
