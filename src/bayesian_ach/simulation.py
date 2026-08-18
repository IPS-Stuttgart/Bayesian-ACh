"""Synthetic experimental designs that dissociate candidate ACh signals."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from bayesian_ach.signals import compute_transition_signals

Trial = dict[str, Any]


@dataclass(frozen=True, slots=True)
class MatchedConfidenceConfig:
    """Configuration for the exact matched-prediction confidence experiment."""

    n_pairs: int = 512
    n_states: int = 4
    low_concentration: float = 4.0
    high_concentration: float = 128.0
    probability_concentration: float = 0.8
    hazard: float = 0.02
    seed: int = 7

    def validate(self) -> None:
        if self.n_pairs < 1:
            raise ValueError("n_pairs must be positive")
        if self.n_states < 2:
            raise ValueError("n_states must be at least two")
        if not 0.0 < self.low_concentration < self.high_concentration:
            raise ValueError("require 0 < low_concentration < high_concentration")
        if self.probability_concentration <= 0.0:
            raise ValueError("probability_concentration must be positive")
        if not 0.0 < self.hazard < 1.0:
            raise ValueError("hazard must lie in (0, 1)")


@dataclass(frozen=True, slots=True)
class FactorialDesignConfig:
    """Configuration for broad candidate-signal model recovery."""

    n_trials: int = 4096
    n_states: int = 5
    min_concentration: float = 2.0
    max_concentration: float = 256.0
    probability_concentration: float = 0.6
    reset_probability_concentration: float = 2.0
    change_rate: float = 0.20
    min_hazard: float = 0.005
    max_hazard: float = 0.20
    seed: int = 7

    def validate(self) -> None:
        if self.n_trials < 2:
            raise ValueError("n_trials must be at least two")
        if self.n_states < 2:
            raise ValueError("n_states must be at least two")
        if not 0.0 < self.min_concentration < self.max_concentration:
            raise ValueError("require 0 < min_concentration < max_concentration")
        if self.probability_concentration <= 0.0:
            raise ValueError("probability_concentration must be positive")
        if self.reset_probability_concentration <= 0.0:
            raise ValueError("reset_probability_concentration must be positive")
        if not 0.0 <= self.change_rate <= 1.0:
            raise ValueError("change_rate must lie in [0, 1]")
        if not 0.0 < self.min_hazard < self.max_hazard < 1.0:
            raise ValueError("require 0 < min_hazard < max_hazard < 1")


def _sample_probabilities(
    rng: np.random.Generator,
    n_states: int,
    concentration: float,
) -> NDArray[np.float64]:
    return np.asarray(rng.dirichlet(np.full(n_states, concentration)), dtype=float)


def simulate_matched_confidence(config: MatchedConfidenceConfig) -> list[Trial]:
    """Generate paired trials with identical predictions and observations.

    Within each pair, only Dirichlet concentration differs. Consequently,
    predictive probability, surprise, and raw innovation are matched exactly,
    whereas Bayesian gain and posterior-update quantities differ.
    """

    config.validate()
    rng = np.random.default_rng(config.seed)
    reset_alpha = np.ones(config.n_states, dtype=float)
    rows: list[Trial] = []

    for pair_id in range(config.n_pairs):
        probabilities = _sample_probabilities(
            rng,
            config.n_states,
            config.probability_concentration,
        )
        observed = int(rng.choice(config.n_states, p=probabilities))
        for condition, concentration in (
            ("low", config.low_concentration),
            ("high", config.high_concentration),
        ):
            signals = compute_transition_signals(
                concentration * probabilities,
                observed,
                reset_alpha=reset_alpha,
                hazard=config.hazard,
            )
            row: Trial = {
                "trial_id": len(rows),
                "pair_id": pair_id,
                "condition": condition,
                "is_change": 0,
                "observed_index": observed,
            }
            row.update(signals.as_dict())
            rows.append(row)

    return rows


def simulate_factorial_design(config: FactorialDesignConfig) -> list[Trial]:
    """Generate a broad design for recovery of competing scalar hypotheses.

    Predictive distributions, confidence, reset distributions, hazard, and the
    source of the observation vary independently enough to avoid relying on a
    single contrast. Observations are drawn either from the current predictive
    model or from a reset model.
    """

    config.validate()
    rng = np.random.default_rng(config.seed)
    log_min = np.log(config.min_concentration)
    log_max = np.log(config.max_concentration)
    log_hazard_min = np.log(config.min_hazard)
    log_hazard_max = np.log(config.max_hazard)
    rows: list[Trial] = []

    for trial_id in range(config.n_trials):
        probabilities = _sample_probabilities(
            rng,
            config.n_states,
            config.probability_concentration,
        )
        reset_probabilities = _sample_probabilities(
            rng,
            config.n_states,
            config.reset_probability_concentration,
        )
        concentration = float(np.exp(rng.uniform(log_min, log_max)))
        reset_concentration = float(np.exp(rng.uniform(log_min, log_max)))
        hazard = float(np.exp(rng.uniform(log_hazard_min, log_hazard_max)))
        is_change = bool(rng.random() < config.change_rate)
        source_probabilities = reset_probabilities if is_change else probabilities
        observed = int(rng.choice(config.n_states, p=source_probabilities))

        signals = compute_transition_signals(
            concentration * probabilities,
            observed,
            reset_alpha=reset_concentration * reset_probabilities,
            hazard=hazard,
        )
        row: Trial = {
            "trial_id": trial_id,
            "pair_id": -1,
            "condition": "reset" if is_change else "stable",
            "is_change": int(is_change),
            "observed_index": observed,
            "hazard": hazard,
            "reset_concentration": reset_concentration,
        }
        row.update(signals.as_dict())
        rows.append(row)

    return rows
