"""Synthetic recovery of known context switches versus structural resets."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Final, Literal

import numpy as np
from numpy.typing import NDArray

from bayesian_ach.changepoint import DirichletBOCPD
from bayesian_ach.switching import SwitchingContextFilter

RegimeKind = Literal["known_context_switch", "structural_reset"]
_REGIME_KINDS: Final[tuple[RegimeKind, ...]] = (
    "known_context_switch",
    "structural_reset",
)


@dataclass(frozen=True, slots=True)
class RegimeRecoveryConfig:
    """Configuration for the model-class recovery benchmark."""

    n_sequences_per_class: int = 48
    pre_length: int = 64
    post_length: int = 96
    n_states: int = 4
    dominant_probability: float = 0.82
    stay_probability: float = 0.10
    novel_similarity: float = 0.0
    known_concentration: float = 256.0
    reset_concentration: float = 1.0
    context_switch_probability: float = 0.02
    change_hazard: float = 0.02
    seed: int = 7

    def validate(self) -> None:
        if self.n_sequences_per_class < 1:
            raise ValueError("n_sequences_per_class must be positive")
        if self.pre_length < 1 or self.post_length < 1:
            raise ValueError("pre_length and post_length must be positive")
        if self.n_states < 4:
            raise ValueError("n_states must be at least four for three distinct ring kernels")
        if not 0.0 < self.dominant_probability < 1.0:
            raise ValueError("dominant_probability must lie in (0, 1)")
        if not 0.0 < self.stay_probability < 1.0:
            raise ValueError("stay_probability must lie in (0, 1)")
        if self.dominant_probability + self.stay_probability >= 1.0:
            raise ValueError("dominant_probability + stay_probability must be below one")
        if not 0.0 <= self.novel_similarity <= 1.0:
            raise ValueError("novel_similarity must lie in [0, 1]")
        if self.known_concentration <= 0.0 or self.reset_concentration <= 0.0:
            raise ValueError("Dirichlet concentrations must be positive")
        if not 0.0 < self.context_switch_probability < 0.5:
            raise ValueError("context_switch_probability must lie in (0, 0.5)")
        if not 0.0 < self.change_hazard < 1.0:
            raise ValueError("change_hazard must lie in (0, 1)")


@dataclass(frozen=True, slots=True)
class RegimeSequenceResult:
    """Prequential evidence comparison for one simulated sequence."""

    sequence_id: int
    true_regime: str
    predicted_regime: str
    correct: bool
    context_log_evidence_post: float
    changepoint_log_evidence_post: float
    context_minus_changepoint: float
    peak_context_switch_probability: float
    peak_change_probability: float
    mean_context_kl_post: float
    mean_run_length_kl_post: float

    def as_dict(self) -> dict[str, float | int | str | bool]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class RegimeRecoveryResult:
    """Complete benchmark output."""

    sequences: tuple[RegimeSequenceResult, ...]
    trials: tuple[dict[str, Any], ...]
    summary: dict[str, Any]


def ring_transition_kernels(
    n_states: int,
    *,
    dominant_probability: float = 0.82,
    stay_probability: float = 0.10,
    novel_similarity: float = 0.0,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Return forward, backward, and novel jump transition kernels."""

    if n_states < 4:
        raise ValueError("n_states must be at least four")
    if not 0.0 <= novel_similarity <= 1.0:
        raise ValueError("novel_similarity must lie in [0, 1]")
    residual = 1.0 - dominant_probability - stay_probability
    if dominant_probability <= 0.0 or stay_probability <= 0.0 or residual <= 0.0:
        raise ValueError("dominant and stay probabilities must be positive and sum below one")

    offsets = (1, -1, max(2, n_states // 2))
    kernels: list[NDArray[np.float64]] = []
    for offset in offsets:
        kernel = np.zeros((n_states, n_states), dtype=float)
        for state in range(n_states):
            dominant = (state + offset) % n_states
            kernel[state, state] = stay_probability
            kernel[state, dominant] = dominant_probability
            remaining = [
                candidate
                for candidate in range(n_states)
                if candidate not in (state, dominant)
            ]
            for candidate in remaining:
                kernel[state, candidate] = residual / len(remaining)
        kernels.append(kernel)
    known_mixture = 0.5 * (kernels[0] + kernels[1])
    novel = (1.0 - novel_similarity) * kernels[2] + novel_similarity * known_mixture
    return kernels[0], kernels[1], novel


def _simulate_sequence(
    kind: RegimeKind,
    config: RegimeRecoveryConfig,
    rng: np.random.Generator,
) -> list[tuple[int, int, int, int]]:
    forward, backward, novel = ring_transition_kernels(
        config.n_states,
        dominant_probability=config.dominant_probability,
        stay_probability=config.stay_probability,
        novel_similarity=config.novel_similarity,
    )
    post_kernel = backward if kind == "known_context_switch" else novel
    state = int(rng.integers(config.n_states))
    transitions: list[tuple[int, int, int, int]] = []
    total_length = config.pre_length + config.post_length
    for trial_index in range(total_length):
        in_post = trial_index >= config.pre_length
        kernel = post_kernel if in_post else forward
        next_state = int(rng.choice(config.n_states, p=kernel[state]))
        true_context = 0 if not in_post else (1 if kind == "known_context_switch" else -1)
        transitions.append((trial_index, state, next_state, true_context))
        state = next_state
    return transitions


def _evaluate_sequence(
    sequence_id: int,
    kind: RegimeKind,
    transitions: list[tuple[int, int, int, int]],
    config: RegimeRecoveryConfig,
) -> tuple[RegimeSequenceResult, list[dict[str, Any]]]:
    forward, backward, _ = ring_transition_kernels(
        config.n_states,
        dominant_probability=config.dominant_probability,
        stay_probability=config.stay_probability,
        novel_similarity=config.novel_similarity,
    )
    alpha = config.known_concentration * np.stack((forward, backward))[:, None, :, :]
    switch = config.context_switch_probability
    context_model = SwitchingContextFilter(
        alpha,
        np.array([[1.0 - switch, switch], [switch, 1.0 - switch]]),
        initial_context=np.array([1.0, 0.0]),
    )
    changepoint_model = DirichletBOCPD(
        config.n_states,
        concentration=config.reset_concentration,
        hazard=config.change_hazard,
    )

    context_post_log = 0.0
    changepoint_post_log = 0.0
    trial_rows: list[dict[str, Any]] = []
    peak_context_switch = 0.0
    peak_change = 0.0
    context_kls: list[float] = []
    run_length_kls: list[float] = []

    for trial_index, state, next_state, true_context in transitions:
        context_step = context_model.observe(state, next_state)
        changepoint_step = changepoint_model.observe(state, next_state)
        context_log = float(np.log(context_step.predictive_probability))
        changepoint_log = float(np.log(changepoint_step.predictive_probability))
        phase = "post" if trial_index >= config.pre_length else "pre"
        if phase == "post":
            context_post_log += context_log
            changepoint_post_log += changepoint_log
            peak_context_switch = max(peak_context_switch, context_step.switch_probability)
            peak_change = max(peak_change, changepoint_step.change_probability)
            context_kls.append(context_step.context_kl)
            run_length_kls.append(changepoint_step.run_length_kl)

        trial_rows.append(
            {
                "sequence_id": sequence_id,
                "trial_index": trial_index,
                "phase": phase,
                "true_regime": kind,
                "true_context": true_context,
                "state": state,
                "next_state": next_state,
                "context_predictive_probability": context_step.predictive_probability,
                "changepoint_predictive_probability": changepoint_step.predictive_probability,
                "context_log_evidence": context_log,
                "changepoint_log_evidence": changepoint_log,
                "context_minus_changepoint": context_log - changepoint_log,
                "context_posterior_0": float(context_step.posterior_context[0]),
                "context_posterior_1": float(context_step.posterior_context[1]),
                "context_kl": context_step.context_kl,
                "context_switch_probability": context_step.switch_probability,
                "expected_parameter_update_l2": context_step.expected_parameter_update_l2,
                "change_probability": changepoint_step.change_probability,
                "run_length_kl": changepoint_step.run_length_kl,
                "expected_run_length": changepoint_step.expected_run_length,
                "map_run_length": changepoint_step.map_run_length,
            }
        )

    margin = context_post_log - changepoint_post_log
    predicted: RegimeKind = (
        "known_context_switch" if margin > 0.0 else "structural_reset"
    )
    result = RegimeSequenceResult(
        sequence_id=sequence_id,
        true_regime=kind,
        predicted_regime=predicted,
        correct=predicted == kind,
        context_log_evidence_post=context_post_log,
        changepoint_log_evidence_post=changepoint_post_log,
        context_minus_changepoint=margin,
        peak_context_switch_probability=peak_context_switch,
        peak_change_probability=peak_change,
        mean_context_kl_post=float(np.mean(context_kls)),
        mean_run_length_kl_post=float(np.mean(run_length_kls)),
    )
    return result, trial_rows


def run_regime_recovery(config: RegimeRecoveryConfig) -> RegimeRecoveryResult:
    """Recover known context switches versus genuinely novel transition regimes."""

    config.validate()
    rng = np.random.default_rng(config.seed)
    sequences: list[RegimeSequenceResult] = []
    trials: list[dict[str, Any]] = []
    sequence_id = 0

    for kind in _REGIME_KINDS:
        for _ in range(config.n_sequences_per_class):
            transitions = _simulate_sequence(kind, config, rng)
            result, rows = _evaluate_sequence(sequence_id, kind, transitions, config)
            sequences.append(result)
            trials.extend(rows)
            sequence_id += 1

    per_class_accuracy: dict[str, float] = {}
    median_margin: dict[str, float] = {}
    for kind in _REGIME_KINDS:
        selected = [result for result in sequences if result.true_regime == kind]
        per_class_accuracy[kind] = float(np.mean([result.correct for result in selected]))
        median_margin[kind] = float(
            np.median([result.context_minus_changepoint for result in selected])
        )
    balanced_accuracy = float(np.mean(list(per_class_accuracy.values())))

    summary: dict[str, Any] = {
        "experiment": "known_context_switch_vs_structural_reset",
        "config": asdict(config),
        "n_sequences": len(sequences),
        "per_class_accuracy": per_class_accuracy,
        "balanced_accuracy": balanced_accuracy,
        "median_context_minus_changepoint": median_margin,
        "all_sequences_correct": all(result.correct for result in sequences),
        "interpretation": (
            "Positive evidence margins favor retrieval of a known transition context; "
            "negative margins favor a newly learned piecewise-stationary regime."
        ),
    }
    return RegimeRecoveryResult(
        sequences=tuple(sequences),
        trials=tuple(trials),
        summary=summary,
    )
