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
    spatial_common_candidate_mask,
    subset_spatial_replay_dataset,
)


@dataclass(frozen=True, slots=True)
class SpatialInjectionRecoveryConfig:
    """Frozen settings for post-decoder raw-emission scoring recovery."""

    injection_temperature: float = 4.0
    spatial_sigma_multipliers: tuple[float, ...] = (0.5, 1.0, 2.0)
    coordinate_units: str = "cm"
    emission_noise_sd_nats: float = 0.02
    mixtures: tuple[tuple[str, str], ...] = (
        ("smoothing_revision", "td_error"),
        ("smoothing_revision", "prospective"),
        ("smoothing_revision", "recency"),
        ("smoothing_revision", "posterior_content"),
    )
    seed: int = 701

    def validate(self) -> None:
        if not np.isfinite(self.injection_temperature) or self.injection_temperature <= 0:
            raise ValueError("injection_temperature must be finite and positive")
        multipliers = np.asarray(self.spatial_sigma_multipliers, dtype=float)
        if (
            multipliers.ndim != 1
            or multipliers.size < 1
            or not np.all(np.isfinite(multipliers))
            or np.any(multipliers <= 0.0)
            or len(set(float(value) for value in multipliers)) != multipliers.size
        ):
            raise ValueError(
                "spatial_sigma_multipliers must be unique finite positive values"
            )
        if self.coordinate_units != "cm":
            raise ValueError("spatial coordinates and decoder point spread must use cm")
        if (
            not np.isfinite(self.emission_noise_sd_nats)
            or self.emission_noise_sd_nats < 0
        ):
            raise ValueError("emission_noise_sd_nats must be finite and nonnegative")
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
    spatial_sigma_multiplier: float = 1.0,
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
    if len(generators) not in (0, 1, 2):
        raise ValueError(
            "generators must be empty for the nuisance-base null, "
            "one candidate, or a 50/50 pair"
        )
    if any(name not in dataset.candidate_names for name in generators):
        raise ValueError("all generators must be registered candidates")
    multiplier = float(spatial_sigma_multiplier)
    if not np.isfinite(multiplier) or multiplier <= 0.0:
        raise ValueError("spatial_sigma_multiplier must be finite and positive")
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
            candidate_index: int | None
            if len(generator_indices) == 0:
                candidate_index = None
            elif len(generator_indices) == 1:
                candidate_index = generator_indices[0]
            else:
                generator_position = (event_index + int(time_index)) % 2
                candidate_index = generator_indices[generator_position]
            log_prior = np.log(base[event_index, active])
            if candidate_index is not None:
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
            spatial_sigma_cm = (
                dataset.decoder_point_spread_cm[event_index] * multiplier
            )
            row = -0.5 * squared_distance / spatial_sigma_cm**2
            if config.emission_noise_sd_nats > 0:
                row += rng.normal(
                    0.0,
                    config.emission_noise_sd_nats,
                    size=row.shape,
                )
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
    """Compute post-decoder LOAO/LOSO scoring recovery on the common cohort.\n\n    This tests Gaussian raw-emission score discrimination at the empirical\n    decoder point spread. It is not end-to-end spike/place-field decoder\n    recovery. Original event IDs and exclusions are retained in the gate.\n    """

    dataset.validate()
    source_event_ids = tuple(dataset.event_ids)
    common_mask = spatial_common_candidate_mask(dataset)
    excluded_event_ids = tuple(
        event_id
        for event_id, keep in zip(source_event_ids, common_mask, strict=True)
        if not bool(keep)
    )
    dataset = subset_spatial_replay_dataset(dataset, common_mask)
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
    null_records: list[SpatialRecoveryRecord] = []
    split_units = ("leave_one_rat_out", "leave_one_session_out")
    for generator_index, generator in enumerate(dataset.candidate_names):
        for multiplier_index, multiplier in enumerate(
            injection_config.spatial_sigma_multipliers
        ):
            injected = inject_spatial_replay_emissions(
                dataset,
                (generator,),
                injection_config,
                spatial_sigma_multiplier=float(multiplier),
                seed=(
                    injection_config.seed
                    + 10_007 * generator_index
                    + 101 * multiplier_index
                ),
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
                    seed=(
                        injection_config.seed
                        + 10_007 * generator_index
                        + 101 * multiplier_index
                        + split_index
                    ),
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
                        spatial_sigma_multiplier=float(multiplier),
                    )
                )

    mixture_names: list[str] = []
    for mixture_index, mixture in enumerate(injection_config.mixtures):
        mixture_name = "+".join(mixture)
        mixture_names.append(mixture_name)
        for multiplier_index, multiplier in enumerate(
            injection_config.spatial_sigma_multipliers
        ):
            injected = inject_spatial_replay_emissions(
                dataset,
                mixture,
                injection_config,
                spatial_sigma_multiplier=float(multiplier),
                seed=(
                    injection_config.seed
                    + 100_003
                    + 10_007 * mixture_index
                    + 101 * multiplier_index
                ),
            )
            for split_index, split_unit in enumerate(split_units):
                result = _comparison_for_split(
                    injected,
                    comparison_config,
                    split_unit,
                )
                summaries = [
                    (
                        candidate,
                        *_fixed_candidate_contrast(
                            result,
                            candidate,
                            comparison_config,
                            seed=(
                                injection_config.seed
                                + 100_003
                                + 10_007 * mixture_index
                                + 101 * multiplier_index
                                + split_index
                                + 1_000_003 * candidate_index
                            ),
                        ),
                    )
                    for candidate_index, candidate in enumerate(
                        result.candidate_names
                    )
                ]
                selected = max(summaries, key=lambda values: values[2])
                mixture_records.append(
                    SpatialRecoveryRecord(
                        generator=mixture_name,
                        split_unit=split_unit,
                        selected_candidate=selected[0],
                        selected_margin=selected[1],
                        selected_margin_lower=selected[2],
                        decisive=any(values[3] for values in summaries),
                        n_held_out_groups=len(result.rat_ids),
                        spatial_sigma_multiplier=float(multiplier),
                    )
                )

    for multiplier_index, multiplier in enumerate(
        injection_config.spatial_sigma_multipliers
    ):
        injected = inject_spatial_replay_emissions(
            dataset,
            (),
            injection_config,
            spatial_sigma_multiplier=float(multiplier),
            seed=injection_config.seed + 900_001 + 101 * multiplier_index,
        )
        for split_index, split_unit in enumerate(split_units):
            result = _comparison_for_split(
                injected,
                comparison_config,
                split_unit,
            )
            summaries = [
                (
                    candidate,
                    *_fixed_candidate_contrast(
                        result,
                        candidate,
                        comparison_config,
                        seed=(
                            injection_config.seed
                            + 900_001
                            + 101 * multiplier_index
                            + split_index
                            + 1_000_003 * candidate_index
                        ),
                    ),
                )
                for candidate_index, candidate in enumerate(
                    result.candidate_names[:-1]
                )
            ]
            selected = max(summaries, key=lambda values: values[2])
            null_records.append(
                SpatialRecoveryRecord(
                    generator="null",
                    split_unit=split_unit,
                    selected_candidate=selected[0],
                    selected_margin=selected[1],
                    selected_margin_lower=selected[2],
                    decisive=any(values[3] for values in summaries),
                    n_held_out_groups=len(result.rat_ids),
                    spatial_sigma_multiplier=float(multiplier),
                )
            )

    return SpatialRecoveryGate(
        pure_records=tuple(pure_records),
        mixture_records=tuple(mixture_records),
        null_records=tuple(null_records),
        source_event_count=len(source_event_ids),
        common_event_count=dataset.n_events,
        excluded_event_ids=excluded_event_ids,
        required_mixtures=tuple(mixture_names),
        required_sigma_multipliers=tuple(
            float(value) for value in injection_config.spatial_sigma_multipliers
        ),
    )


__all__ = [
    "SpatialInjectionRecoveryConfig",
    "inject_spatial_replay_emissions",
    "run_spatial_recovery_checks",
]
