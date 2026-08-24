"""Emission-level injection recovery for spatial replay model comparison."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from numpy.typing import NDArray
from scipy.special import logsumexp

from bayesian_ach.replay_spatial import (
    SPATIAL_CANDIDATE_NAMES,
    SpatialComparisonConfig,
    SpatialRecoveryGate,
    SpatialRecoveryRecord,
    SpatialReplayComparison,
    SpatialReplayDataset,
    compare_spatial_replay_candidates,
)


@dataclass(frozen=True, slots=True)
class SpatialInjectionRecoveryConfig:
    """Frozen settings for end-to-end raw-emission recovery checks."""

    injection_temperature: float = 4.0
    spatial_sigma: float = 0.35
    emission_noise_sd: float = 0.02
    mixtures: tuple[tuple[str, str], ...] = (
        ("smoothing_revision", "td_error"),
    )
    seed: int = 701

    def validate(self) -> None:
        if not np.isfinite(self.injection_temperature) or self.injection_temperature <= 0:
            raise ValueError("injection_temperature must be finite and positive")
        if not np.isfinite(self.spatial_sigma) or self.spatial_sigma <= 0:
            raise ValueError("spatial_sigma must be finite and positive")
        if not np.isfinite(self.emission_noise_sd) or self.emission_noise_sd < 0:
            raise ValueError("emission_noise_sd must be finite and nonnegative")
        for first, second in self.mixtures:
            if (
                first not in SPATIAL_CANDIDATE_NAMES
                or second not in SPATIAL_CANDIDATE_NAMES
                or first == second
            ):
                raise ValueError("mixtures must contain two distinct registered candidates")


def _standardized_fields(
    dataset: SpatialReplayDataset,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    base = np.asarray(dataset.nuisance_base, dtype=float).copy()
    base /= base.sum(axis=1, keepdims=True)
    fields = np.asarray(dataset.candidate_fields, dtype=float)
    mean = np.sum(base[:, None, :] * fields, axis=2, keepdims=True)
    centered = fields - mean
    variance = np.sum(base[:, None, :] * centered**2, axis=2)
    if np.any(variance <= 1e-14):
        raise ValueError("recovery requires every candidate field to be nonconstant")
    standardized = centered / np.sqrt(variance)[:, :, None]
    support = np.broadcast_to(dataset.active_spatial_mask[:, None, :], fields.shape)
    standardized[~support] = 0.0
    return base, standardized


def inject_spatial_replay_emissions(
    dataset: SpatialReplayDataset,
    generators: tuple[str, ...],
    config: SpatialInjectionRecoveryConfig | None = None,
    *,
    seed: int | None = None,
) -> SpatialReplayDataset:
    """Replace replay emissions with draws from one candidate or a 50/50 mixture.

    Candidate priors generate latent grid bins. A Gaussian likelihood around the
    sampled bin is then written in the same max-shifted raw-log-emission domain
    as the real decoder. The candidate fields themselves are not altered.
    """

    dataset.validate()
    config = SpatialInjectionRecoveryConfig() if config is None else config
    config.validate()
    if len(generators) not in (1, 2):
        raise ValueError("generators must contain one candidate or a 50/50 pair")
    if any(name not in dataset.candidate_names for name in generators):
        raise ValueError("all generators must be registered candidates")
    generator_indices = [dataset.candidate_names.index(name) for name in generators]
    for index in generator_indices:
        if not np.all(dataset.candidate_available[:, index]):
            raise ValueError("recovery generator is unavailable for at least one event")

    base, standardized = _standardized_fields(dataset)
    rng = np.random.default_rng(config.seed if seed is None else seed)
    emissions = np.full_like(dataset.log_emissions, -np.inf, dtype=float)
    for event_index in range(dataset.n_events):
        active = np.asarray(dataset.active_spatial_mask[event_index], dtype=bool)
        active_indices = np.flatnonzero(active)
        coordinates = dataset.spatial_coordinates[event_index, active]
        for time_index in np.flatnonzero(dataset.time_mask[event_index]):
            if len(generator_indices) == 1:
                generator_position = 0
            else:
                generator_position = (event_index + int(time_index)) % 2
            candidate_index = generator_indices[generator_position]
            log_prior = np.log(base[event_index, active])
            log_prior += (
                config.injection_temperature
                * standardized[event_index, candidate_index, active]
            )
            probabilities = np.exp(log_prior - logsumexp(log_prior))
            sampled_position = int(rng.choice(len(active_indices), p=probabilities))
            squared_distance = np.sum(
                (coordinates - coordinates[sampled_position]) ** 2,
                axis=1,
            )
            row = -0.5 * squared_distance / config.spatial_sigma**2
            if config.emission_noise_sd > 0:
                row += rng.normal(0.0, config.emission_noise_sd, size=row.shape)
            row -= np.max(row)
            emissions[event_index, time_index, active] = row

    return replace(
        dataset,
        log_emissions=np.asarray(emissions, dtype=np.float64),
        log_emission_offsets=np.zeros_like(dataset.log_emission_offsets, dtype=float),
    )


def _comparison_for_split(
    dataset: SpatialReplayDataset,
    comparison_config: SpatialComparisonConfig,
    split_unit: str,
) -> SpatialReplayComparison:
    if split_unit == "leave_one_rat_out":
        grouped = dataset
    elif split_unit == "leave_one_session_out":
        compound = np.asarray(
            [
                f"{rat}|{session}"
                for rat, session in zip(
                    np.asarray(dataset.rat_ids, dtype=str),
                    np.asarray(dataset.session_ids, dtype=str),
                    strict=True,
                )
            ],
            dtype=str,
        )
        grouped = replace(dataset, rat_ids=compound, session_ids=compound)
    else:
        raise ValueError("split_unit must be leave_one_rat_out or leave_one_session_out")
    return compare_spatial_replay_candidates(grouped, comparison_config)


def _fixed_candidate_contrast(
    result: SpatialReplayComparison,
    candidate: str,
    comparison_config: SpatialComparisonConfig,
    *,
    seed: int,
) -> tuple[float, float, bool]:
    candidate_index = result.candidate_names.index(candidate)
    alternatives = [
        index
        for index in range(len(result.candidate_names))
        if index != candidate_index
    ]
    paired = (
        result.rat_scores[:, [candidate_index]]
        - result.rat_scores[:, alternatives]
    )
    if paired.shape[0] < 2:
        return float("nan"), float("nan"), False
    observed = paired.mean(axis=0)
    rng = np.random.default_rng(seed)
    indices = rng.integers(
        0,
        paired.shape[0],
        size=(comparison_config.bootstrap_replicates, paired.shape[0]),
    )
    bootstrap = paired[indices].mean(axis=1)
    shortfall = np.max(observed[None, :] - bootstrap, axis=1)
    critical = float(
        np.quantile(shortfall, comparison_config.simultaneous_confidence_level)
    )
    lower = observed - critical
    return float(np.min(observed)), float(np.min(lower)), bool(np.all(lower > 0.0))


def run_spatial_recovery_checks(
    dataset: SpatialReplayDataset,
    comparison_config: SpatialComparisonConfig | None = None,
    injection_config: SpatialInjectionRecoveryConfig | None = None,
) -> SpatialRecoveryGate:
    """Compute LOAO/LOSO pure recovery and registered-mixture abstention."""

    dataset.validate()
    comparison_config = (
        SpatialComparisonConfig()
        if comparison_config is None
        else comparison_config
    )
    comparison_config.validate()
    injection_config = (
        SpatialInjectionRecoveryConfig()
        if injection_config is None
        else injection_config
    )
    injection_config.validate()

    pure_records: list[SpatialRecoveryRecord] = []
    mixture_records: list[SpatialRecoveryRecord] = []
    split_units = ("leave_one_rat_out", "leave_one_session_out")
    for generator_index, generator in enumerate(dataset.candidate_names):
        injected = inject_spatial_replay_emissions(
            dataset,
            (generator,),
            injection_config,
            seed=injection_config.seed + 1009 * generator_index,
        )
        for split_index, split_unit in enumerate(split_units):
            result = _comparison_for_split(
                injected,
                comparison_config,
                split_unit,
            )
            margin, lower, decisive = _fixed_candidate_contrast(
                result,
                generator,
                comparison_config,
                seed=injection_config.seed + 1009 * generator_index + split_index,
            )
            pure_records.append(
                SpatialRecoveryRecord(
                    generator=generator,
                    split_unit=split_unit,
                    selected_candidate=result.winner,
                    selected_margin=margin,
                    selected_margin_lower=lower,
                    decisive=decisive,
                    n_held_out_groups=len(result.rat_ids),
                )
            )

    mixture_names: list[str] = []
    for mixture_index, mixture in enumerate(injection_config.mixtures):
        mixture_name = "+".join(mixture)
        mixture_names.append(mixture_name)
        injected = inject_spatial_replay_emissions(
            dataset,
            mixture,
            injection_config,
            seed=injection_config.seed + 100_003 + 1009 * mixture_index,
        )
        for split_index, split_unit in enumerate(split_units):
            result = _comparison_for_split(
                injected,
                comparison_config,
                split_unit,
            )
            margin, lower, decisive = _fixed_candidate_contrast(
                result,
                "smoothing_revision",
                comparison_config,
                seed=(
                    injection_config.seed
                    + 100_003
                    + 1009 * mixture_index
                    + split_index
                ),
            )
            mixture_records.append(
                SpatialRecoveryRecord(
                    generator=mixture_name,
                    split_unit=split_unit,
                    selected_candidate=result.winner,
                    selected_margin=margin,
                    selected_margin_lower=lower,
                    decisive=decisive,
                    n_held_out_groups=len(result.rat_ids),
                )
            )

    return SpatialRecoveryGate(
        pure_records=tuple(pure_records),
        mixture_records=tuple(mixture_records),
        required_mixtures=tuple(mixture_names),
    )


__all__ = [
    "SpatialInjectionRecoveryConfig",
    "inject_spatial_replay_emissions",
    "run_spatial_recovery_checks",
]
