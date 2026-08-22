"""Synthetic recovery of filtering, smoothing, and posterior-replay hypotheses."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray

from bayesian_ach.smoothing import FiniteStateSmoother

REPLAY_CANDIDATE_NAMES: Final[tuple[str, ...]] = (
    "online_surprise",
    "state_smoothing_kl",
    "transition_smoothing_kl",
    "replay_content_surprise",
)


@dataclass(frozen=True, slots=True)
class ReplayBenchmarkConfig:
    """Configuration for exact filtering-to-smoothing model recovery."""

    n_sequences: int = 96
    sequence_length: int = 64
    n_states: int = 6
    train_fraction: float = 0.70
    effect_size: float = 0.75
    noise_std: float = 0.50
    replay_samples: int = 128
    seed: int = 7

    def validate(self) -> None:
        if self.n_sequences < 8:
            raise ValueError("n_sequences must be at least eight")
        if self.sequence_length < 24:
            raise ValueError("sequence_length must be at least 24")
        if self.n_states < 4:
            raise ValueError("n_states must be at least four")
        if not 0.2 <= self.train_fraction <= 0.9:
            raise ValueError("train_fraction must lie in [0.2, 0.9]")
        if not np.isfinite(self.effect_size) or self.effect_size <= 0.0:
            raise ValueError("effect_size must be finite and positive")
        if not np.isfinite(self.noise_std) or self.noise_std <= 0.0:
            raise ValueError("noise_std must be finite and positive")
        if self.replay_samples < 32:
            raise ValueError("replay_samples must be at least 32")


@dataclass(frozen=True, slots=True)
class ReplayGeneratorResult:
    """Held-out recovery result for one synthetic event generator."""

    generator: str
    winner: str
    runner_up: str
    correct: bool
    evidence_margin: float
    winner_test_log_likelihood: float
    winner_test_r2: float
    n_train_sequences: int
    n_test_sequences: int

    def as_dict(self) -> dict[str, str | bool | float | int]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ReplaySequenceResult:
    """Exact posterior and posterior-sampling diagnostics for one sequence."""

    sequence_id: int
    log_evidence: float
    missing_start: int
    missing_end: int
    landmark_time: int
    mean_online_surprise: float
    mean_state_smoothing_kl: float
    mean_transition_smoothing_kl: float
    mean_replay_content_surprise: float
    transition_count_revision_l1: float
    replay_sampling_mutation_max_abs: float
    replay_state_marginal_mae: float

    def as_dict(self) -> dict[str, float | int]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ReplayBenchmarkResult:
    """Complete exact smoothing and replay model-recovery package."""

    generators: tuple[ReplayGeneratorResult, ...]
    sequences: tuple[ReplaySequenceResult, ...]
    fits: tuple[dict[str, Any], ...]
    trials: tuple[dict[str, Any], ...]
    summary: dict[str, Any]


def _ring_kernel(
    n_states: int,
    dominant_probability: float,
    stay_probability: float,
    offset: int,
) -> NDArray[np.float64]:
    residual = 1.0 - dominant_probability - stay_probability
    kernel = np.full(
        (n_states, n_states),
        residual / (n_states - 2),
        dtype=float,
    )
    for state in range(n_states):
        kernel[state, state] = stay_probability
        kernel[state, (state + offset) % n_states] = dominant_probability
    return kernel


def _emission_matrix(n_states: int, accuracy: float) -> NDArray[np.float64]:
    emission = np.full(
        (n_states, n_states),
        (1.0 - accuracy) / (n_states - 1),
        dtype=float,
    )
    np.fill_diagonal(emission, accuracy)
    return emission


def _maximum_result_difference(
    before: tuple[NDArray[np.float64], ...],
    after: tuple[NDArray[np.float64], ...],
) -> float:
    return max(
        float(np.max(np.abs(first - second)))
        for first, second in zip(before, after, strict=True)
    )


def _simulate_exact_design(
    config: ReplayBenchmarkConfig,
) -> tuple[list[dict[str, Any]], list[ReplaySequenceResult]]:
    rng = np.random.default_rng(config.seed)
    trial_rows: list[dict[str, Any]] = []
    sequence_rows: list[ReplaySequenceResult] = []

    for sequence_id in range(config.n_sequences):
        dominant = float(rng.uniform(0.52, 0.90))
        stay = float(rng.uniform(0.03, min(0.25, 0.97 - dominant)))
        offset = int(rng.choice(np.array([-2, -1, 1, 2], dtype=int)))
        transition = _ring_kernel(config.n_states, dominant, stay, offset)
        initial = np.full(config.n_states, 1.0 / config.n_states, dtype=float)

        n_times = config.sequence_length + 1
        missing_start = int(rng.integers(7, max(8, config.sequence_length // 2)))
        missing_length = int(rng.integers(5, max(6, config.sequence_length // 5)))
        missing_end = min(config.sequence_length - 7, missing_start + missing_length)
        landmark_time = min(
            config.sequence_length,
            missing_end + int(rng.integers(2, 8)),
        )
        base_accuracy = float(rng.uniform(0.72, 0.95))
        low_accuracy = float(rng.uniform(1.0 / config.n_states + 0.01, 0.42))

        latent_states = np.empty(n_times, dtype=np.int64)
        observations = np.full(n_times, -1, dtype=np.int64)
        likelihoods = np.empty((n_times, config.n_states), dtype=float)
        latent_states[0] = int(rng.integers(config.n_states))

        for time_index in range(n_times):
            if time_index > 0:
                latent_states[time_index] = int(
                    rng.choice(
                        config.n_states,
                        p=transition[latent_states[time_index - 1]],
                    )
                )
            if missing_start <= time_index <= missing_end:
                # Missing data contribute a likelihood of one, not an artificial
                # uniform-observation surprise of log(number of states).
                likelihoods[time_index] = 1.0
                continue

            accuracy = base_accuracy
            if rng.random() < 0.12:
                accuracy = low_accuracy
            if time_index == landmark_time:
                accuracy = float(rng.uniform(0.985, 0.9995))
            emission = _emission_matrix(config.n_states, accuracy)
            observation = int(
                rng.choice(
                    config.n_states,
                    p=emission[latent_states[time_index]],
                )
            )
            observations[time_index] = observation
            likelihoods[time_index] = emission[:, observation]

        smoother = FiniteStateSmoother(initial, transition, likelihoods)
        posterior = smoother.run()
        replay_surprise = smoother.replay_content_surprise(posterior)

        tracked_before = (
            posterior.filtered_probabilities.copy(),
            posterior.smoothed_probabilities.copy(),
            posterior.smoothed_pair_probabilities.copy(),
            posterior.state_smoothing_kl.copy(),
            posterior.transition_smoothing_kl.copy(),
        )
        samples = smoother.sample_smoothed_trajectories(
            posterior,
            n_samples=config.replay_samples,
            seed=config.seed + 104729 * (sequence_id + 1),
        )
        tracked_after = (
            posterior.filtered_probabilities,
            posterior.smoothed_probabilities,
            posterior.smoothed_pair_probabilities,
            posterior.state_smoothing_kl,
            posterior.transition_smoothing_kl,
        )
        mutation = _maximum_result_difference(tracked_before, tracked_after)

        empirical_state = np.zeros_like(posterior.smoothed_probabilities)
        for time_index in range(n_times):
            empirical_state[time_index] = np.bincount(
                samples[:, time_index],
                minlength=config.n_states,
            ) / config.replay_samples
        sampling_mae = float(
            np.mean(np.abs(empirical_state - posterior.smoothed_probabilities))
        )

        filtered_counts = posterior.expected_transition_counts(smoothed=False)
        smoothed_counts = posterior.expected_transition_counts(smoothed=True)
        count_revision = float(np.sum(np.abs(smoothed_counts - filtered_counts)))

        sequence_rows.append(
            ReplaySequenceResult(
                sequence_id=sequence_id,
                log_evidence=posterior.log_evidence,
                missing_start=missing_start,
                missing_end=missing_end,
                landmark_time=landmark_time,
                mean_online_surprise=float(np.mean(posterior.online_surprise[1:])),
                mean_state_smoothing_kl=float(
                    np.mean(posterior.state_smoothing_kl[:-1])
                ),
                mean_transition_smoothing_kl=float(
                    np.mean(posterior.transition_smoothing_kl)
                ),
                mean_replay_content_surprise=float(np.mean(replay_surprise)),
                transition_count_revision_l1=count_revision,
                replay_sampling_mutation_max_abs=mutation,
                replay_state_marginal_mae=sampling_mae,
            )
        )

        for transition_index in range(config.sequence_length):
            trial_rows.append(
                {
                    "sequence_id": sequence_id,
                    "transition_index": transition_index,
                    "latent_state": int(latent_states[transition_index]),
                    "latent_next_state": int(latent_states[transition_index + 1]),
                    "observation": int(observations[transition_index + 1]),
                    "observation_missing": int(
                        observations[transition_index + 1] < 0
                    ),
                    "in_missing_window": int(
                        missing_start <= transition_index + 1 <= missing_end
                    ),
                    "before_landmark": int(transition_index + 1 < landmark_time),
                    "online_surprise": float(
                        posterior.online_surprise[transition_index + 1]
                    ),
                    "state_smoothing_kl": float(
                        posterior.state_smoothing_kl[transition_index]
                    ),
                    "state_revision_l1": float(
                        posterior.state_revision_l1[transition_index]
                    ),
                    "transition_smoothing_kl": float(
                        posterior.transition_smoothing_kl[transition_index]
                    ),
                    "transition_revision_l1": float(
                        posterior.transition_revision_l1[transition_index]
                    ),
                    "replay_content_surprise": float(
                        replay_surprise[transition_index]
                    ),
                }
            )

    return trial_rows, sequence_rows


def _split_sequences(
    n_sequences: int,
    train_fraction: float,
    seed: int,
) -> tuple[set[int], set[int]]:
    rng = np.random.default_rng(seed)
    order = np.asarray(rng.permutation(n_sequences), dtype=np.int64)
    n_train = min(n_sequences - 2, max(2, int(round(train_fraction * n_sequences))))
    return set(map(int, order[:n_train])), set(map(int, order[n_train:]))


def _feature_matrix(
    rows: list[dict[str, Any]],
) -> NDArray[np.float64]:
    return np.asarray(
        [[float(row[name]) for name in REPLAY_CANDIDATE_NAMES] for row in rows],
        dtype=float,
    )


def _fit_candidate(
    candidate_index: int,
    features: NDArray[np.float64],
    response: NDArray[np.float64],
    train_mask: NDArray[np.bool_],
    test_mask: NDArray[np.bool_],
) -> dict[str, float | str | int]:
    train_feature = features[train_mask, candidate_index]
    test_feature = features[test_mask, candidate_index]
    mean = float(np.mean(train_feature))
    standard_deviation = float(np.std(train_feature))
    if standard_deviation <= 1e-15:
        raise ValueError(
            f"candidate {REPLAY_CANDIDATE_NAMES[candidate_index]!r} is constant in training"
        )
    train_z = (train_feature - mean) / standard_deviation
    test_z = (test_feature - mean) / standard_deviation
    design = np.column_stack((np.ones_like(train_z), train_z))
    coefficients, _, _, _ = np.linalg.lstsq(design, response[train_mask], rcond=None)
    prediction_train = coefficients[0] + coefficients[1] * train_z
    residual_train = response[train_mask] - prediction_train
    residual_std = max(float(np.sqrt(np.mean(residual_train**2))), 1e-8)
    prediction_test = coefficients[0] + coefficients[1] * test_z
    residual_test = response[test_mask] - prediction_test
    variance = residual_std**2
    log_likelihood = float(
        np.sum(
            -0.5
            * (
                np.log(2.0 * np.pi * variance)
                + residual_test**2 / variance
            )
        )
    )
    centered = response[test_mask] - float(np.mean(response[test_mask]))
    denominator = float(np.sum(centered**2))
    test_r2 = (
        float("nan")
        if denominator <= 0.0
        else 1.0 - float(np.sum(residual_test**2)) / denominator
    )
    return {
        "candidate": REPLAY_CANDIDATE_NAMES[candidate_index],
        "intercept": float(coefficients[0]),
        "slope": float(coefficients[1]),
        "train_feature_mean": mean,
        "train_feature_std": standard_deviation,
        "residual_std": residual_std,
        "test_log_likelihood": log_likelihood,
        "test_mean_log_likelihood": log_likelihood / int(np.sum(test_mask)),
        "test_r2": test_r2,
        "n_train": int(np.sum(train_mask)),
        "n_test": int(np.sum(test_mask)),
    }


def run_replay_benchmark(config: ReplayBenchmarkConfig) -> ReplayBenchmarkResult:
    """Recover distinct online, smoothing-correction, and replay-content signals."""

    config.validate()
    trial_rows, sequence_rows = _simulate_exact_design(config)
    features = _feature_matrix(trial_rows)
    sequence_ids = np.asarray(
        [int(row["sequence_id"]) for row in trial_rows],
        dtype=np.int64,
    )
    train_sequences, test_sequences = _split_sequences(
        config.n_sequences,
        config.train_fraction,
        config.seed + 17,
    )
    train_mask = np.asarray(
        [int(sequence_id) in train_sequences for sequence_id in sequence_ids],
        dtype=bool,
    )
    test_mask = np.asarray(
        [int(sequence_id) in test_sequences for sequence_id in sequence_ids],
        dtype=bool,
    )

    generator_results: list[ReplayGeneratorResult] = []
    fit_rows: list[dict[str, Any]] = []
    for generator_index, generator in enumerate(REPLAY_CANDIDATE_NAMES):
        generator_feature = features[:, generator_index]
        scale = float(np.std(generator_feature))
        if scale <= 0.0:
            raise ValueError(f"generator {generator!r} is constant")
        standardized = (
            generator_feature - float(np.mean(generator_feature))
        ) / scale
        rng = np.random.default_rng(config.seed + 7919 * (generator_index + 1))
        response = (
            config.effect_size * standardized
            + rng.normal(0.0, config.noise_std, standardized.size)
        )

        fits = [
            _fit_candidate(
                candidate_index,
                features,
                response,
                train_mask,
                test_mask,
            )
            for candidate_index in range(len(REPLAY_CANDIDATE_NAMES))
        ]
        fits.sort(key=lambda row: float(row["test_log_likelihood"]), reverse=True)
        winner = str(fits[0]["candidate"])
        runner_up = str(fits[1]["candidate"])
        margin = float(fits[0]["test_log_likelihood"]) - float(
            fits[1]["test_log_likelihood"]
        )
        generator_results.append(
            ReplayGeneratorResult(
                generator=generator,
                winner=winner,
                runner_up=runner_up,
                correct=winner == generator,
                evidence_margin=margin,
                winner_test_log_likelihood=float(fits[0]["test_log_likelihood"]),
                winner_test_r2=float(fits[0]["test_r2"]),
                n_train_sequences=len(train_sequences),
                n_test_sequences=len(test_sequences),
            )
        )
        for rank, fit in enumerate(fits, start=1):
            fit_rows.append(
                {
                    "generator": generator,
                    "rank": rank,
                    **fit,
                }
            )

    candidate_correlation = np.corrcoef(features, rowvar=False)
    off_diagonal = np.abs(
        candidate_correlation[~np.eye(candidate_correlation.shape[0], dtype=bool)]
    )
    recovery_count = sum(result.correct for result in generator_results)
    margins = [result.evidence_margin for result in generator_results]
    sampling_mutations = [
        result.replay_sampling_mutation_max_abs for result in sequence_rows
    ]
    sampling_errors = [result.replay_state_marginal_mae for result in sequence_rows]
    summary: dict[str, Any] = {
        "experiment": "filtering_smoothing_and_posterior_replay_recovery",
        "config": asdict(config),
        "candidate_names": list(REPLAY_CANDIDATE_NAMES),
        "recovery_count": recovery_count,
        "candidate_count": len(REPLAY_CANDIDATE_NAMES),
        "all_generators_recovered": recovery_count == len(REPLAY_CANDIDATE_NAMES),
        "minimum_evidence_margin": float(np.min(margins)),
        "median_evidence_margin": float(np.median(margins)),
        "maximum_absolute_candidate_correlation": float(np.max(off_diagonal)),
        "candidate_correlation": candidate_correlation.tolist(),
        "maximum_replay_sampling_mutation": float(np.max(sampling_mutations)),
        "median_replay_state_marginal_mae": float(np.median(sampling_errors)),
        "maximum_replay_state_marginal_mae": float(np.max(sampling_errors)),
        "n_train_sequences": len(train_sequences),
        "n_test_sequences": len(test_sequences),
        "strict_separation": {
            "filtering": "conditions only on observations available online",
            "smoothing": "conditions on the fixed complete observation interval",
            "posterior_replay": "read-only FFBS samples from the smoothed posterior",
            "model_scores": "held-out whole sequences only",
        },
        "interpretation": (
            "Filtering-to-smoothing KL measures how future external evidence revises a past "
            "state or transition belief. Replay-content surprise describes posterior samples "
            "after conditioning and is not a new evidence increment."
        ),
        "scope": (
            "The benchmark assumes a known finite-state generative model. Expected smoothed "
            "transition counts are posterior diagnostics or EM sufficient statistics; treating "
            "them, or sampled replay paths, as independent observations would double-count the "
            "measurement interval."
        ),
    }
    return ReplayBenchmarkResult(
        generators=tuple(generator_results),
        sequences=tuple(sequence_rows),
        fits=tuple(fit_rows),
        trials=tuple(trial_rows),
        summary=summary,
    )
