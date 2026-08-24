"""Leakage-safe spatial test of replay as filtering-to-smoothing revision."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import logsumexp

REPLAY_SPATIAL_SCHEMA_VERSION: Final[str] = "bayesian-ach.replay-spatial.v2"
SPATIAL_CANDIDATE_NAMES: Final[tuple[str, ...]] = (
    "smoothing_revision",
    "online_surprise",
    "posterior_content",
    "current_location",
    "recency",
    "prospective",
    "td_error",
)
NULL_CANDIDATE_NAME: Final[str] = "null"
_FLOAT_TOL = 1e-10


def _probability_rows(values: ArrayLike, *, name: str) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=float)
    if array.ndim != 2 or array.shape[0] < 1 or array.shape[1] < 2:
        raise ValueError(f"{name} must have shape (positive snippets, at least two states)")
    if not np.all(np.isfinite(array)) or np.any(array < 0.0):
        raise ValueError(f"{name} must contain finite nonnegative values")
    totals = array.sum(axis=1)
    if np.any(totals <= 0.0):
        raise ValueError(f"every {name} row must contain positive mass")
    return np.asarray(array / totals[:, None], dtype=np.float64)


def _categorical_kl_rows(
    posterior: NDArray[np.float64],
    prior: NDArray[np.float64],
) -> NDArray[np.float64]:
    positive = posterior > 0.0
    if np.any(positive & (prior <= 0.0)):
        raise ValueError("smoothing posterior has mass outside filtering support")
    terms = np.zeros_like(posterior)
    terms[positive] = posterior[positive] * np.log(
        posterior[positive] / prior[positive]
    )
    return np.asarray(terms.sum(axis=1), dtype=np.float64)


@dataclass(frozen=True, slots=True)
class SignedRevisionField:
    """Pre-replay spatial field assembled from historical smoothing revisions."""

    signed_field: NDArray[np.float64]
    per_snippet_kl: NDArray[np.float64]
    snippet_weights: NDArray[np.float64]
    total_weight: float
    identifiable: bool


def build_signed_revision_field(
    filtered_probabilities: ArrayLike,
    smoothed_probabilities: ArrayLike,
    snippet_end_s: ArrayLike,
    *,
    event_start_s: float,
    state_to_spatial: ArrayLike | None = None,
    recency_tau_s: float = 30.0,
    minimum_total_weight: float = 1e-10,
) -> SignedRevisionField:
    """Build a signed spatial revision field from strictly historical snippets.

    Each state-wise difference (smoothed minus filtered) is projected onto the
    spatial grid and weighted by KL(smoothed || filtered). An exponential age
    weight is optional through recency_tau_s. Every snippet must end no later
    than the replay event; replay emissions and later outcomes are forbidden.
    """

    filtered = _probability_rows(filtered_probabilities, name="filtered_probabilities")
    smoothed = _probability_rows(smoothed_probabilities, name="smoothed_probabilities")
    if smoothed.shape != filtered.shape:
        raise ValueError("filtered and smoothed probabilities must have matching shapes")
    ends = np.asarray(snippet_end_s, dtype=float)
    if ends.shape != (filtered.shape[0],) or not np.all(np.isfinite(ends)):
        raise ValueError("snippet_end_s must contain one finite time per snippet")
    event_start = float(event_start_s)
    if not np.isfinite(event_start):
        raise ValueError("event_start_s must be finite")
    if np.any(ends > event_start + _FLOAT_TOL):
        raise ValueError("all smoothing snippets must end before replay starts")
    tau = float(recency_tau_s)
    if not np.isfinite(tau) or tau <= 0.0:
        raise ValueError("recency_tau_s must be finite and positive")
    threshold = float(minimum_total_weight)
    if not np.isfinite(threshold) or threshold < 0.0:
        raise ValueError("minimum_total_weight must be finite and nonnegative")

    n_snippets, n_states = filtered.shape
    if state_to_spatial is None:
        mapping = np.broadcast_to(
            np.eye(n_states, dtype=float)[None, :, :],
            (n_snippets, n_states, n_states),
        ).copy()
    else:
        raw_mapping = np.asarray(state_to_spatial, dtype=float)
        if raw_mapping.ndim == 2:
            if raw_mapping.shape[0] != n_states or raw_mapping.shape[1] < 2:
                raise ValueError(
                    "state_to_spatial must have shape (state, at least two spatial bins)"
                )
            mapping = np.broadcast_to(
                raw_mapping[None, :, :],
                (n_snippets, *raw_mapping.shape),
            ).copy()
        elif raw_mapping.ndim == 3:
            if (
                raw_mapping.shape[0] != n_snippets
                or raw_mapping.shape[1] != n_states
                or raw_mapping.shape[2] < 2
            ):
                raise ValueError(
                    "time-varying state_to_spatial must have shape "
                    "(snippet, state, at least two spatial bins)"
                )
            mapping = raw_mapping.copy()
        else:
            raise ValueError("state_to_spatial must be two- or three-dimensional")
        if not np.all(np.isfinite(mapping)) or np.any(mapping < 0.0):
            raise ValueError("state_to_spatial must contain finite nonnegative values")
        row_mass = mapping.sum(axis=2)
        if np.any(row_mass <= 0.0):
            raise ValueError("every state_to_spatial row must contain positive mass")
        mapping /= row_mass[:, :, None]

    kl = _categorical_kl_rows(smoothed, filtered)
    age = np.maximum(event_start - ends, 0.0)
    weights = kl * np.exp(-age / tau)
    signed_snippets = np.einsum(
        "hs,hsb->hb",
        smoothed - filtered,
        mapping,
    )
    signed_field = np.sum(weights[:, None] * signed_snippets, axis=0)
    total_weight = float(weights.sum())
    return SignedRevisionField(
        signed_field=np.asarray(signed_field, dtype=np.float64),
        per_snippet_kl=kl,
        snippet_weights=np.asarray(weights, dtype=np.float64),
        total_weight=total_weight,
        identifiable=bool(total_weight > threshold),
    )


@dataclass(frozen=True, slots=True)
class SpatialReplayDataset:
    """Frozen predictor-only artifact used for spatial replay comparison.

    Log emissions are stored after a per-event/time additive offset has been
    removed. log_emission_offsets restores that offset. Candidate fields and
    their source cutoffs are frozen before the replay event. Later behavioral
    outcomes are intentionally absent from this object.
    """

    event_ids: tuple[str, ...]
    rat_ids: NDArray[np.str_]
    session_ids: NDArray[np.str_]
    event_start_s: NDArray[np.float64]
    event_end_s: NDArray[np.float64]
    history_cutoff_s: NDArray[np.float64]
    decoder_training_cutoff_s: NDArray[np.float64]
    field_available_s: NDArray[np.float64]
    log_emissions: NDArray[np.float64]
    log_emission_offsets: NDArray[np.float64]
    time_mask: NDArray[np.bool_]
    active_spatial_mask: NDArray[np.bool_]
    spatial_coordinates: NDArray[np.float64]
    decoder_point_spread_cm: NDArray[np.float64]
    nuisance_base: NDArray[np.float64]
    candidate_fields: NDArray[np.float64]
    candidate_available: NDArray[np.bool_]
    candidate_names: tuple[str, ...] = SPATIAL_CANDIDATE_NAMES
    well_masses: NDArray[np.float64] | None = None
    well_ids: tuple[str, ...] = ()

    @property
    def n_events(self) -> int:
        return len(self.event_ids)

    @property
    def n_time(self) -> int:
        return int(self.log_emissions.shape[1])

    @property
    def n_spatial_bins(self) -> int:
        return int(self.log_emissions.shape[2])

    def validate(self) -> None:
        """Reject shape, provenance, support, and future-leakage violations."""

        n_events = self.n_events
        if n_events < 1 or len(set(self.event_ids)) != n_events:
            raise ValueError("event_ids must be nonempty and unique")
        if self.candidate_names != SPATIAL_CANDIDATE_NAMES:
            raise ValueError(
                "candidate_names must equal the frozen spatial candidate registry"
            )
        for name in self.event_ids:
            if not str(name):
                raise ValueError("event_ids must not contain empty values")

        rats = np.asarray(self.rat_ids, dtype=str)
        sessions = np.asarray(self.session_ids, dtype=str)
        if rats.shape != (n_events,) or sessions.shape != (n_events,):
            raise ValueError("rat_ids and session_ids must contain one value per event")
        if np.any(rats == "") or np.any(sessions == ""):
            raise ValueError("rat_ids and session_ids must not contain empty values")

        starts = np.asarray(self.event_start_s, dtype=float)
        ends = np.asarray(self.event_end_s, dtype=float)
        history = np.asarray(self.history_cutoff_s, dtype=float)
        decoder = np.asarray(self.decoder_training_cutoff_s, dtype=float)
        for values, name in (
            (starts, "event_start_s"),
            (ends, "event_end_s"),
            (history, "history_cutoff_s"),
            (decoder, "decoder_training_cutoff_s"),
        ):
            if values.shape != (n_events,) or not np.all(np.isfinite(values)):
                raise ValueError(f"{name} must contain one finite value per event")
        if np.any(ends <= starts):
            raise ValueError("each event must end after it starts")
        if np.any(history > starts + _FLOAT_TOL):
            raise ValueError("history_cutoff_s must not extend into replay")
        if np.any(decoder > starts + _FLOAT_TOL):
            raise ValueError("decoder training must not use observations after replay starts")

        emissions = np.asarray(self.log_emissions, dtype=float)
        if emissions.ndim != 3 or emissions.shape[0] != n_events:
            raise ValueError("log_emissions must have shape (event, time, spatial_bin)")
        if emissions.shape[2] < 2:
            raise ValueError("log_emissions must contain at least two spatial bins")

        fields = np.asarray(self.candidate_fields, dtype=float)
        expected_fields = (
            n_events,
            len(self.candidate_names),
            self.n_spatial_bins,
        )
        if fields.shape != expected_fields or not np.all(np.isfinite(fields)):
            raise ValueError(
                "candidate_fields must be finite with shape "
                "(event, candidate, spatial_bin)"
            )
        field_times = np.asarray(self.field_available_s, dtype=float)
        if field_times.shape != (n_events, len(self.candidate_names)):
            raise ValueError("field_available_s must have shape (event, candidate)")
        if not np.all(np.isfinite(field_times)):
            raise ValueError("field_available_s must be finite")
        if np.any(field_times > starts[:, None] + _FLOAT_TOL):
            raise ValueError("candidate fields must use only evidence available before replay")
        available = np.asarray(self.candidate_available)
        if available.dtype != np.bool_ or available.shape != field_times.shape:
            raise ValueError("candidate_available must be boolean with shape (event, candidate)")

        mask = np.asarray(self.time_mask)
        if mask.dtype != np.bool_ or mask.shape != emissions.shape[:2]:
            raise ValueError("time_mask must be boolean with shape (event, time)")
        if np.any(mask.sum(axis=1) < 1):
            raise ValueError("every event must contain at least one valid emission row")
        spatial = np.asarray(self.active_spatial_mask)
        if spatial.dtype != np.bool_ or spatial.shape != (
            n_events,
            emissions.shape[2],
        ):
            raise ValueError(
                "active_spatial_mask must be boolean with shape (event, spatial_bin)"
            )
        if np.any(spatial.sum(axis=1) < 2):
            raise ValueError("every event must contain at least two active spatial bins")
        coordinates = np.asarray(self.spatial_coordinates, dtype=float)
        if coordinates.shape != (n_events, emissions.shape[2], 2):
            raise ValueError(
                "spatial_coordinates must have shape (event, spatial_bin, xy)"
            )
        if not np.all(np.isfinite(coordinates[spatial])):
            raise ValueError("active spatial coordinates must be finite")
        if np.any(np.isfinite(coordinates[~spatial])):
            raise ValueError("inactive spatial coordinates must be NaN")
        point_spread = np.asarray(self.decoder_point_spread_cm, dtype=float)
        if (
            point_spread.shape != (n_events,)
            or not np.all(np.isfinite(point_spread))
            or np.any(point_spread <= 0.0)
        ):
            raise ValueError(
                "decoder_point_spread_cm must contain one finite positive value per event"
            )
        if np.any(np.isnan(emissions)) or np.any(emissions == np.inf):
            raise ValueError("log_emissions must not contain NaN or positive infinity")
        offsets = np.asarray(self.log_emission_offsets, dtype=float)
        if offsets.shape != mask.shape or not np.all(np.isfinite(offsets)):
            raise ValueError(
                "log_emission_offsets must be finite with shape (event, time)"
            )

        for event_index in range(n_events):
            active = spatial[event_index]
            for time_index in range(emissions.shape[1]):
                row = emissions[event_index, time_index]
                if mask[event_index, time_index]:
                    finite = np.isfinite(row) & active
                    if not np.any(finite):
                        raise ValueError("every valid emission row needs finite active support")
                    if np.any(np.isfinite(row[~active])):
                        raise ValueError("emissions outside active spatial support must be -inf")
                    row_max = float(np.max(row[finite]))
                    if abs(row_max) > 1e-8:
                        raise ValueError(
                            "each valid log-emission row must be max-shifted to zero"
                        )
                elif np.any(np.isfinite(row)):
                    raise ValueError("padded emission rows must contain only -inf")

        base = np.asarray(self.nuisance_base, dtype=float)
        if base.shape != spatial.shape or not np.all(np.isfinite(base)):
            raise ValueError("nuisance_base must be finite with shape (event, spatial_bin)")
        if np.any(base < 0.0) or np.any(base[~spatial] != 0.0):
            raise ValueError("nuisance_base must be nonnegative and zero off active support")
        if np.any(base.sum(axis=1) <= 0.0):
            raise ValueError("every nuisance_base row must contain positive mass")
        if np.any(fields[~np.broadcast_to(spatial[:, None, :], fields.shape)] != 0.0):
            raise ValueError("candidate_fields must be zero off active spatial support")

        if self.well_masses is None:
            if self.well_ids:
                raise ValueError("well_ids require well_masses")
        else:
            well_mass = np.asarray(self.well_masses, dtype=float)
            if (
                well_mass.ndim != 2
                or well_mass.shape[0] != n_events
                or well_mass.shape[1] != len(self.well_ids)
                or len(set(self.well_ids)) != len(self.well_ids)
            ):
                raise ValueError("well_masses and well_ids have incompatible shapes")
            if not np.all(np.isfinite(well_mass)) or np.any(well_mass < 0.0):
                raise ValueError("well_masses must contain finite nonnegative values")
            if not np.allclose(
                well_mass.sum(axis=1),
                1.0,
                rtol=0.0,
                atol=1e-8,
            ):
                raise ValueError("every well_masses row must sum to one")


@dataclass(frozen=True, slots=True)
class SpatialComparisonConfig:
    """Predeclared grouped scoring and abstention thresholds."""

    temperatures: tuple[float, ...] = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
    minimum_rats: int = 3
    bootstrap_replicates: int = 5000
    maximum_field_correlation: float = 0.98
    simultaneous_confidence_level: float = 0.95
    seed: int = 7

    def validate(self) -> None:
        temperatures = np.asarray(self.temperatures, dtype=float)
        if temperatures.ndim != 1 or temperatures.size < 2:
            raise ValueError("temperatures must contain at least two values")
        if not np.all(np.isfinite(temperatures)) or np.any(temperatures < 0.0):
            raise ValueError("temperatures must be finite and nonnegative")
        if 0.0 not in self.temperatures:
            raise ValueError("temperatures must include the null temperature zero")
        if self.minimum_rats < 2:
            raise ValueError("minimum_rats must be at least two")
        if self.bootstrap_replicates < 100:
            raise ValueError("bootstrap_replicates must be at least 100")
        if not 0.0 < self.maximum_field_correlation <= 1.0:
            raise ValueError("maximum_field_correlation must lie in (0, 1]")
        if not 0.5 < self.simultaneous_confidence_level < 1.0:
            raise ValueError("simultaneous_confidence_level must lie in (0.5, 1)")


@dataclass(frozen=True, slots=True)
class SpatialRecoveryRecord:
    """One computed emission-injection recovery result."""

    generator: str
    split_unit: str
    selected_candidate: str
    selected_margin: float
    selected_margin_lower: float
    decisive: bool
    n_held_out_groups: int
    spatial_sigma_multiplier: float


@dataclass(frozen=True, slots=True)
class SpatialRecoveryGate:
    """Computed recovery evidence required before a biological claim is made.

    Records are produced by run_spatial_recovery_checks. There are no
    user-asserted pass flags: every pure generator must be decisively recovered
    under both held-out-animal and held-out-session calibration, while every
    registered 50/50 mixture must trigger uncertainty abstention under both.
    """

    pure_records: tuple[SpatialRecoveryRecord, ...]
    mixture_records: tuple[SpatialRecoveryRecord, ...]
    required_mixtures: tuple[str, ...] = (
        "smoothing_revision+td_error",
        "smoothing_revision+prospective",
        "smoothing_revision+recency",
        "smoothing_revision+posterior_content",
    )
    required_sigma_multipliers: tuple[float, ...] = (0.5, 1.0, 2.0)

    @property
    def passed(self) -> bool:
        required_splits = ("leave_one_rat_out", "leave_one_session_out")
        required_cells = {
            (split_unit, float(multiplier))
            for split_unit in required_splits
            for multiplier in self.required_sigma_multipliers
        }
        for generator in SPATIAL_CANDIDATE_NAMES:
            matches = [
                record
                for record in self.pure_records
                if record.generator == generator
            ]
            observed_cells = {
                (record.split_unit, float(record.spatial_sigma_multiplier))
                for record in matches
            }
            if (
                observed_cells != required_cells
                or len(matches) != len(required_cells)
                or any(
                    record.selected_candidate != generator or not record.decisive
                    for record in matches
                )
            ):
                return False
        for mixture in self.required_mixtures:
            matches = [
                record
                for record in self.mixture_records
                if record.generator == mixture
            ]
            observed_cells = {
                (record.split_unit, float(record.spatial_sigma_multiplier))
                for record in matches
            }
            if (
                observed_cells != required_cells
                or len(matches) != len(required_cells)
                or any(record.decisive for record in matches)
            ):
                return False
        return True


@dataclass(frozen=True, slots=True)
class SpatialCandidateFold:
    candidate: str
    held_out_rat: str
    temperature: float
    mean_log_score_per_bin: float
    n_events: int
    n_sessions: int


@dataclass(frozen=True, slots=True)
class SpatialTargetContrast:
    """Prespecified animal-level smoothing contrast with simultaneous coverage."""

    alternative: str
    mean_margin: float
    simultaneous_lower_bound: float


@dataclass(frozen=True, slots=True)
class SpatialReplayComparison:
    """LORO comparison with confirmatory, prespecified smoothing contrasts.

    winner and winner_margin_ci are descriptive because their identities are
    selected on the same held-out scores. The status decision uses only the
    prespecified smoothing-revision contrasts against every alternative and
    their joint bootstrap lower bounds.
    """

    candidate_names: tuple[str, ...]
    folds: tuple[SpatialCandidateFold, ...]
    rat_ids: tuple[str, ...]
    rat_scores: NDArray[np.float64]
    winner: str
    runner_up: str
    winner_margin: float
    winner_margin_ci: tuple[float, float]
    target_candidate: str
    target_contrasts: tuple[SpatialTargetContrast, ...]
    simultaneous_confidence_level: float
    maximum_field_correlation: float
    common_event_count: int
    status: str
    abstention_reasons: tuple[str, ...]


def _normalized_base(dataset: SpatialReplayDataset) -> NDArray[np.float64]:
    base = np.asarray(dataset.nuisance_base, dtype=float).copy()
    base /= base.sum(axis=1, keepdims=True)
    return base


def _standardized_fields(
    dataset: SpatialReplayDataset,
    base: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    fields = np.asarray(dataset.candidate_fields, dtype=float)
    mean = np.sum(base[:, None, :] * fields, axis=2, keepdims=True)
    centered = fields - mean
    variance = np.sum(base[:, None, :] * centered**2, axis=2)
    usable = variance > 1e-14
    scale = np.sqrt(np.maximum(variance, 1e-14))
    standardized = centered / scale[:, :, None]
    standardized[~np.broadcast_to(dataset.active_spatial_mask[:, None, :], fields.shape)] = 0.0
    return standardized, usable


def _event_scores_for_temperature(
    dataset: SpatialReplayDataset,
    base: NDArray[np.float64],
    standardized: NDArray[np.float64],
    candidate_index: int | None,
    temperature: float,
) -> NDArray[np.float64]:
    scores = np.empty(dataset.n_events, dtype=float)
    for event_index in range(dataset.n_events):
        active = dataset.active_spatial_mask[event_index]
        log_prior = np.full(dataset.n_spatial_bins, -np.inf, dtype=float)
        active_base = base[event_index, active]
        log_prior[active] = np.log(active_base)
        if candidate_index is not None:
            log_prior[active] += (
                float(temperature) * standardized[event_index, candidate_index, active]
            )
            log_prior[active] -= logsumexp(log_prior[active])
        total = 0.0
        count = 0
        for time_index in np.flatnonzero(dataset.time_mask[event_index]):
            total += float(
                logsumexp(
                    dataset.log_emissions[event_index, time_index, active]
                    + log_prior[active]
                )
                + dataset.log_emission_offsets[event_index, time_index]
            )
            count += 1
        scores[event_index] = total / count
    return scores


def _equal_rat_session_mean(
    values: NDArray[np.float64],
    selected: NDArray[np.bool_],
    rats: NDArray[np.str_],
    sessions: NDArray[np.str_],
) -> float:
    rat_means: list[float] = []
    for rat in sorted(set(rats[selected])):
        rat_mask = selected & (rats == rat)
        session_means = [
            float(np.mean(values[rat_mask & (sessions == session)]))
            for session in sorted(set(sessions[rat_mask]))
        ]
        if session_means:
            rat_means.append(float(np.mean(session_means)))
    return float(np.mean(rat_means)) if rat_means else -np.inf


def _field_correlation(
    standardized: NDArray[np.float64],
    common: NDArray[np.bool_],
    active: NDArray[np.bool_],
) -> float:
    vectors: list[NDArray[np.float64]] = []
    for candidate_index in range(standardized.shape[1]):
        values = np.concatenate(
            [
                standardized[event_index, candidate_index, active[event_index]]
                for event_index in np.flatnonzero(common)
            ]
        )
        vectors.append(values)
    matrix = np.asarray(vectors, dtype=float)
    nonconstant = np.std(matrix, axis=1) > 1e-12
    if int(np.sum(nonconstant)) < 2:
        return 1.0
    correlation = np.corrcoef(matrix[nonconstant])
    off_diagonal = np.abs(
        correlation[~np.eye(correlation.shape[0], dtype=bool)]
    )
    return float(np.max(off_diagonal)) if off_diagonal.size else 0.0


def compare_spatial_replay_candidates(
    dataset: SpatialReplayDataset,
    config: SpatialComparisonConfig | None = None,
    *,
    recovery_gate: SpatialRecoveryGate | None = None,
) -> SpatialReplayComparison:
    """Score raw replay emissions with LORO tuning and animal-level abstention."""

    dataset.validate()
    config = SpatialComparisonConfig() if config is None else config
    config.validate()
    rats = np.asarray(dataset.rat_ids, dtype=str)
    sessions = np.asarray(dataset.session_ids, dtype=str)
    base = _normalized_base(dataset)
    standardized, field_usable = _standardized_fields(dataset, base)
    common = np.all(np.asarray(dataset.candidate_available, dtype=bool), axis=1)
    common &= np.all(field_usable, axis=1)
    unique_rats = tuple(sorted(set(rats[common])))
    all_names = (*dataset.candidate_names, NULL_CANDIDATE_NAME)

    temperature_scores: dict[tuple[int, float], NDArray[np.float64]] = {}
    for candidate_index in range(len(dataset.candidate_names)):
        for temperature in config.temperatures:
            temperature_scores[candidate_index, float(temperature)] = (
                _event_scores_for_temperature(
                    dataset,
                    base,
                    standardized,
                    candidate_index,
                    float(temperature),
                )
            )
    null_scores = _event_scores_for_temperature(
        dataset,
        base,
        standardized,
        None,
        0.0,
    )

    folds: list[SpatialCandidateFold] = []
    rat_score_rows: list[list[float]] = []
    retained_rats: list[str] = []
    for held_out_rat in unique_rats:
        train = common & (rats != held_out_rat)
        test = common & (rats == held_out_rat)
        if not np.any(train) or not np.any(test):
            continue
        row: list[float] = []
        for candidate_index, candidate in enumerate(dataset.candidate_names):
            objectives = [
                _equal_rat_session_mean(
                    temperature_scores[candidate_index, float(temperature)],
                    train,
                    rats,
                    sessions,
                )
                for temperature in config.temperatures
            ]
            best_index = int(np.argmax(objectives))
            temperature = float(config.temperatures[best_index])
            score = temperature_scores[candidate_index, temperature]
            held_score = _equal_rat_session_mean(score, test, rats, sessions)
            row.append(held_score)
            folds.append(
                SpatialCandidateFold(
                    candidate=candidate,
                    held_out_rat=held_out_rat,
                    temperature=temperature,
                    mean_log_score_per_bin=held_score,
                    n_events=int(np.sum(test)),
                    n_sessions=len(set(sessions[test])),
                )
            )
        held_null = _equal_rat_session_mean(null_scores, test, rats, sessions)
        row.append(held_null)
        folds.append(
            SpatialCandidateFold(
                candidate=NULL_CANDIDATE_NAME,
                held_out_rat=held_out_rat,
                temperature=0.0,
                mean_log_score_per_bin=held_null,
                n_events=int(np.sum(test)),
                n_sessions=len(set(sessions[test])),
            )
        )
        retained_rats.append(held_out_rat)
        rat_score_rows.append(row)

    target_candidate = "smoothing_revision"
    target_index = all_names.index(target_candidate)
    alternative_indices = [
        index for index in range(len(all_names)) if index != target_index
    ]
    ci: tuple[float, float]
    if rat_score_rows:
        rat_scores = np.asarray(rat_score_rows, dtype=float)
        mean_scores = rat_scores.mean(axis=0)
        order = np.argsort(mean_scores)[::-1]
        winner_index = int(order[0])
        runner_index = int(order[1])
        winner = all_names[winner_index]
        runner_up = all_names[runner_index]
        rat_differences = rat_scores[:, winner_index] - rat_scores[:, runner_index]
        winner_margin = float(np.mean(rat_differences))
        rng = np.random.default_rng(config.seed)
        descriptive_draws = rng.choice(
            rat_differences,
            size=(config.bootstrap_replicates, len(rat_differences)),
            replace=True,
        ).mean(axis=1)
        tail = (1.0 - config.simultaneous_confidence_level) / 2.0
        quantiles = np.quantile(
            descriptive_draws,
            [tail, 1.0 - tail],
        )
        ci = (float(quantiles[0]), float(quantiles[1]))

        paired = (
            rat_scores[:, [target_index]]
            - rat_scores[:, alternative_indices]
        )
        observed_margins = paired.mean(axis=0)
        bootstrap_indices = rng.integers(
            0,
            paired.shape[0],
            size=(config.bootstrap_replicates, paired.shape[0]),
        )
        bootstrap_margins = paired[bootstrap_indices].mean(axis=1)
        maximum_shortfall = np.max(
            observed_margins[None, :] - bootstrap_margins,
            axis=1,
        )
        critical_value = float(
            np.quantile(
                maximum_shortfall,
                config.simultaneous_confidence_level,
            )
        )
        simultaneous_lower = observed_margins - critical_value
        target_contrasts = tuple(
            SpatialTargetContrast(
                alternative=all_names[alternative_index],
                mean_margin=float(observed_margins[position]),
                simultaneous_lower_bound=float(simultaneous_lower[position]),
            )
            for position, alternative_index in enumerate(alternative_indices)
        )
    else:
        rat_scores = np.empty((0, len(all_names)), dtype=float)
        winner = NULL_CANDIDATE_NAME
        runner_up = dataset.candidate_names[0]
        winner_margin = float("nan")
        ci = (float("nan"), float("nan"))
        target_contrasts = tuple(
            SpatialTargetContrast(
                alternative=all_names[index],
                mean_margin=float("nan"),
                simultaneous_lower_bound=float("nan"),
            )
            for index in alternative_indices
        )

    maximum_correlation = (
        _field_correlation(standardized, common, dataset.active_spatial_mask)
        if np.any(common)
        else 1.0
    )
    reasons: list[str] = []
    if len(retained_rats) < config.minimum_rats:
        reasons.append("too_few_independent_rats")
    if recovery_gate is None:
        reasons.append("recovery_gate_missing")
    elif not recovery_gate.passed:
        reasons.append("recovery_gate_failed")
    if maximum_correlation > config.maximum_field_correlation:
        reasons.append("candidate_fields_collinear")
    target_lower = np.asarray(
        [contrast.simultaneous_lower_bound for contrast in target_contrasts],
        dtype=float,
    )
    if np.any(~np.isfinite(target_lower)) or np.any(target_lower <= 0.0):
        reasons.append("smoothing_revision_contrast_uncertain")
    status = "identified" if not reasons else "abstain"

    return SpatialReplayComparison(
        candidate_names=all_names,
        folds=tuple(folds),
        rat_ids=tuple(retained_rats),
        rat_scores=rat_scores,
        winner=winner,
        runner_up=runner_up,
        winner_margin=winner_margin,
        winner_margin_ci=ci,
        target_candidate=target_candidate,
        target_contrasts=target_contrasts,
        simultaneous_confidence_level=config.simultaneous_confidence_level,
        maximum_field_correlation=maximum_correlation,
        common_event_count=int(np.sum(common)),
        status=status,
        abstention_reasons=tuple(reasons),
    )


__all__ = [
    "NULL_CANDIDATE_NAME",
    "REPLAY_SPATIAL_SCHEMA_VERSION",
    "SPATIAL_CANDIDATE_NAMES",
    "SignedRevisionField",
    "SpatialCandidateFold",
    "SpatialComparisonConfig",
    "SpatialRecoveryGate",
    "SpatialRecoveryRecord",
    "SpatialReplayComparison",
    "SpatialTargetContrast",
    "SpatialReplayDataset",
    "build_signed_revision_field",
    "compare_spatial_replay_candidates",
]
