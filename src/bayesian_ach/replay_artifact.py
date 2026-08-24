"""Versioned predictor/outcome artifact contract for real replay analyses."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final

import numpy as np
from numpy.typing import NDArray

from bayesian_ach.replay_spatial import (
    REPLAY_SPATIAL_SCHEMA_VERSION,
    SpatialReplayDataset,
)

LATER_OUTCOME_SCHEMA_VERSION: Final[str] = "bayesian-ach.replay-later-outcome.v1"
HIPPO_TRACE_SCHEMA_VERSION: Final[str] = (
    "hipporeplayimm.first-order-smoothing-trace.v1"
)
HIPPO_TRANSITION_CONVENTION: Final[str] = (
    "column-stochastic: transition[destination, source] = "
    "P(x_t=destination | x_(t-1)=source)"
)
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class ReplaySpatialManifest:
    """Provenance that can be frozen before any later outcome is joined."""

    producer_repository: str
    producer_commit: str
    dataset_id: str
    dataset_sha256: str
    dataset_manifest_file_sha256: str
    dataset_verifier_report_file: str
    dataset_verifier_report_sha256: str
    dataset_verified_file_count: int
    dataset_verified_total_bytes: int
    dataset_verified_session_count: int
    dataset_verified_file_records_sha256: str
    route_manifest_file_sha256: str
    route_producer_commit: str
    route_producer_clean_worktree: bool
    route_parameters_sha256: str
    route_segments_sha256: str
    route_points_sha256: str
    cohort_sha256: str
    event_audit_sha256: str
    event_selection_parameters_sha256: str
    behavior_field_parameters_sha256: str
    decoder_parameters_sha256: str
    trace_schema_version: str = HIPPO_TRACE_SCHEMA_VERSION
    transition_convention: str = HIPPO_TRANSITION_CONVENTION
    candidate_evidence_cutoff: str = "strict_pre_replay"
    likelihood_domain: str = "max_shifted_log_emission_plus_offset"
    decoder_training_schedule: str = "event_specific_prefix_refit"
    decoder_point_spread_schedule: str = "pre_event_temporal_holdout_run_68pct"
    event_selection_schedule: str = "lfp_raw_peak_power_top_n_per_session"
    event_selection_time_scope: str = "full_session_offline_rank"
    dataset_verification_schedule: str = (
        "locked_full_tree_path_size_sha256_no_extra_files"
    )
    route_smoothing_scope: str = "within_completed_fill_interval"
    spatial_coordinate_units: str = "cm"
    well_mass_source: str = "raw_log_emission_posterior"
    behavior_latent_state: str = "compact_destination_well"
    behavior_observation_schedule: str = "tracked_position_and_well_visits_pre_replay"
    state_to_spatial_mapping: str = "pre_replay_route_kernel"
    replay_feedback_used: bool = False
    outcomes_in_predictor: bool = False
    producer_clean_worktree: bool = True
    schema_version: str = REPLAY_SPATIAL_SCHEMA_VERSION

    def validate(self) -> None:
        if self.schema_version != REPLAY_SPATIAL_SCHEMA_VERSION:
            raise ValueError("unsupported replay spatial schema version")
        if self.trace_schema_version != HIPPO_TRACE_SCHEMA_VERSION:
            raise ValueError("unsupported HippoReplayDynamics trace schema")
        if self.transition_convention != HIPPO_TRANSITION_CONVENTION:
            raise ValueError("transition convention does not match the trace contract")
        if not self.producer_repository or not self.dataset_id:
            raise ValueError("producer_repository and dataset_id are required")
        if _COMMIT_PATTERN.fullmatch(self.producer_commit) is None:
            raise ValueError("producer_commit must be a lowercase 40-character commit SHA")
        if _SHA256_PATTERN.fullmatch(self.dataset_sha256) is None:
            raise ValueError("dataset_sha256 must be a lowercase SHA-256 digest")
        for value, name in (
            (
                self.dataset_manifest_file_sha256,
                "dataset_manifest_file_sha256",
            ),
            (
                self.dataset_verifier_report_sha256,
                "dataset_verifier_report_sha256",
            ),
            (
                self.dataset_verified_file_records_sha256,
                "dataset_verified_file_records_sha256",
            ),
            (
                self.route_manifest_file_sha256,
                "route_manifest_file_sha256",
            ),
            (self.route_parameters_sha256, "route_parameters_sha256"),
            (self.route_segments_sha256, "route_segments_sha256"),
            (self.route_points_sha256, "route_points_sha256"),
            (self.cohort_sha256, "cohort_sha256"),
            (self.event_audit_sha256, "event_audit_sha256"),
            (
                self.event_selection_parameters_sha256,
                "event_selection_parameters_sha256",
            ),
            (
                self.behavior_field_parameters_sha256,
                "behavior_field_parameters_sha256",
            ),
            (self.decoder_parameters_sha256, "decoder_parameters_sha256"),
        ):
            if _SHA256_PATTERN.fullmatch(value) is None:
                raise ValueError(f"{name} must be a lowercase SHA-256 digest")
        if _COMMIT_PATTERN.fullmatch(self.route_producer_commit) is None:
            raise ValueError(
                "route_producer_commit must be a lowercase 40-character commit SHA"
            )
        if not self.route_producer_clean_worktree:
            raise ValueError("route producer must run from a clean committed worktree")
        if self.dataset_verifier_report_file != (
            "replay_spatial_dataset_verification.json"
        ):
            raise ValueError("dataset verifier report file is not the frozen name")
        if (
            self.dataset_verified_file_count < 1
            or self.dataset_verified_total_bytes < 1
            or self.dataset_verified_session_count < 1
        ):
            raise ValueError("dataset verification counts must be positive")
        report = {
            "schema_version": "hipporeplayimm.pf-dataset-verification.v1",
            "status": "pass",
            "dataset_sha256": self.dataset_sha256,
            "dataset_manifest_file_sha256": self.dataset_manifest_file_sha256,
            "verified_file_count": self.dataset_verified_file_count,
            "verified_total_bytes": self.dataset_verified_total_bytes,
            "verified_session_count": self.dataset_verified_session_count,
            "verified_file_records_sha256": (
                self.dataset_verified_file_records_sha256
            ),
            "missing_files": [],
            "extra_files": [],
        }
        report_sha256 = hashlib.sha256(
            (json.dumps(report, indent=2, sort_keys=True) + "\n").encode("utf-8")
        ).hexdigest()
        if report_sha256 != self.dataset_verifier_report_sha256:
            raise ValueError("dataset verifier report digest does not match its content")
        if self.candidate_evidence_cutoff != "strict_pre_replay":
            raise ValueError("candidate evidence must be frozen strictly before replay")
        if self.likelihood_domain != "max_shifted_log_emission_plus_offset":
            raise ValueError("raw replay scores require shifted log emissions and offsets")
        if self.decoder_training_schedule != "event_specific_prefix_refit":
            raise ValueError("decoder training must use an event-specific prefix refit")
        if (
            self.decoder_point_spread_schedule
            != "pre_event_temporal_holdout_run_68pct"
        ):
            raise ValueError("decoder point spread must use the frozen prefix holdout")
        if self.event_selection_schedule != "lfp_raw_peak_power_top_n_per_session":
            raise ValueError("event selection must use raw LFP power only")
        if self.event_selection_time_scope != "full_session_offline_rank":
            raise ValueError("event selection time scope must be the frozen offline rank")
        if self.dataset_verification_schedule != (
            "locked_full_tree_path_size_sha256_no_extra_files"
        ):
            raise ValueError("dataset verification must check the entire locked tree")
        if self.route_smoothing_scope != "within_completed_fill_interval":
            raise ValueError("route smoothing must not cross completed-route boundaries")
        if self.spatial_coordinate_units != "cm":
            raise ValueError("spatial coordinates and point spread must use cm")
        if self.well_mass_source != "raw_log_emission_posterior":
            raise ValueError("well masses must be derived from the raw posterior")
        if self.behavior_latent_state != "compact_destination_well":
            raise ValueError("behavioral smoothing must use the compact well state")
        if (
            self.behavior_observation_schedule
            != "tracked_position_and_well_visits_pre_replay"
        ):
            raise ValueError("behavior observation schedule is not the frozen schedule")
        if self.state_to_spatial_mapping != "pre_replay_route_kernel":
            raise ValueError("state-to-spatial mapping is not the frozen mapping")
        if self.replay_feedback_used:
            raise ValueError("decoded replay must not be fed back as a new observation")
        if self.outcomes_in_predictor:
            raise ValueError("later outcomes must not be present in the predictor artifact")
        if not self.producer_clean_worktree:
            raise ValueError("producer must run from a clean committed worktree")


@dataclass(frozen=True, slots=True)
class FrozenPredictorArtifact:
    dataset: SpatialReplayDataset
    manifest: ReplaySpatialManifest
    predictor_sha256: str


def write_spatial_predictor_artifact(
    directory: Path | str,
    dataset: SpatialReplayDataset,
    manifest: ReplaySpatialManifest,
) -> FrozenPredictorArtifact:
    """Write a predictor-only NPZ and a hash-binding JSON manifest."""

    dataset.validate()
    manifest.validate()
    output = Path(directory)
    output.mkdir(parents=True, exist_ok=True)
    predictor_path = output / "replay_spatial_predictors.npz"
    well_masses = (
        np.empty((dataset.n_events, 0), dtype=float)
        if dataset.well_masses is None
        else np.asarray(dataset.well_masses, dtype=float)
    )
    np.savez_compressed(
        predictor_path,
        event_ids=np.asarray(dataset.event_ids, dtype=str),
        rat_ids=np.asarray(dataset.rat_ids, dtype=str),
        session_ids=np.asarray(dataset.session_ids, dtype=str),
        event_start_s=np.asarray(dataset.event_start_s, dtype=float),
        event_end_s=np.asarray(dataset.event_end_s, dtype=float),
        history_cutoff_s=np.asarray(dataset.history_cutoff_s, dtype=float),
        decoder_training_cutoff_s=np.asarray(
            dataset.decoder_training_cutoff_s,
            dtype=float,
        ),
        field_available_s=np.asarray(dataset.field_available_s, dtype=float),
        log_emissions=np.asarray(dataset.log_emissions, dtype=float),
        log_emission_offsets=np.asarray(dataset.log_emission_offsets, dtype=float),
        time_mask=np.asarray(dataset.time_mask, dtype=bool),
        active_spatial_mask=np.asarray(dataset.active_spatial_mask, dtype=bool),
        spatial_coordinates=np.asarray(dataset.spatial_coordinates, dtype=float),
        decoder_point_spread_cm=np.asarray(
            dataset.decoder_point_spread_cm,
            dtype=float,
        ),
        nuisance_base=np.asarray(dataset.nuisance_base, dtype=float),
        candidate_fields=np.asarray(dataset.candidate_fields, dtype=float),
        candidate_available=np.asarray(dataset.candidate_available, dtype=bool),
        candidate_names=np.asarray(dataset.candidate_names, dtype=str),
        well_masses=well_masses,
        well_ids=np.asarray(dataset.well_ids, dtype=str),
    )
    predictor_sha256 = _sha256(predictor_path)
    manifest_payload = {
        **asdict(manifest),
        "predictor_file": predictor_path.name,
        "predictor_sha256": predictor_sha256,
    }
    manifest_path = output / "replay_spatial_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return FrozenPredictorArtifact(dataset, manifest, predictor_sha256)


def load_spatial_predictor_artifact(
    directory: Path | str,
) -> FrozenPredictorArtifact:
    """Load, hash-check, and validate a predictor-only artifact."""

    source = Path(directory)
    manifest_payload = json.loads(
        (source / "replay_spatial_manifest.json").read_text(encoding="utf-8")
    )
    predictor_name = str(manifest_payload.pop("predictor_file"))
    expected_sha256 = str(manifest_payload.pop("predictor_sha256"))
    if _SHA256_PATTERN.fullmatch(expected_sha256) is None:
        raise ValueError("predictor_sha256 is not a valid SHA-256 digest")
    predictor_path = source / predictor_name
    observed_sha256 = _sha256(predictor_path)
    if observed_sha256 != expected_sha256:
        raise ValueError("predictor artifact SHA-256 does not match its manifest")
    manifest = ReplaySpatialManifest(**manifest_payload)
    manifest.validate()

    with np.load(predictor_path, allow_pickle=False) as arrays:
        well_mass = np.asarray(arrays["well_masses"], dtype=float)
        dataset = SpatialReplayDataset(
            event_ids=tuple(str(value) for value in arrays["event_ids"]),
            rat_ids=np.asarray(arrays["rat_ids"], dtype=str),
            session_ids=np.asarray(arrays["session_ids"], dtype=str),
            event_start_s=np.asarray(arrays["event_start_s"], dtype=float),
            event_end_s=np.asarray(arrays["event_end_s"], dtype=float),
            history_cutoff_s=np.asarray(arrays["history_cutoff_s"], dtype=float),
            decoder_training_cutoff_s=np.asarray(
                arrays["decoder_training_cutoff_s"],
                dtype=float,
            ),
            field_available_s=np.asarray(arrays["field_available_s"], dtype=float),
            log_emissions=np.asarray(arrays["log_emissions"], dtype=float),
            log_emission_offsets=np.asarray(
                arrays["log_emission_offsets"],
                dtype=float,
            ),
            time_mask=np.asarray(arrays["time_mask"], dtype=bool),
            active_spatial_mask=np.asarray(
                arrays["active_spatial_mask"],
                dtype=bool,
            ),
            spatial_coordinates=np.asarray(
                arrays["spatial_coordinates"],
                dtype=float,
            ),
            decoder_point_spread_cm=np.asarray(
                arrays["decoder_point_spread_cm"],
                dtype=float,
            ),
            nuisance_base=np.asarray(arrays["nuisance_base"], dtype=float),
            candidate_fields=np.asarray(arrays["candidate_fields"], dtype=float),
            candidate_available=np.asarray(
                arrays["candidate_available"],
                dtype=bool,
            ),
            candidate_names=tuple(str(value) for value in arrays["candidate_names"]),
            well_masses=None if well_mass.shape[1] == 0 else well_mass,
            well_ids=tuple(str(value) for value in arrays["well_ids"]),
        )
    dataset.validate()
    return FrozenPredictorArtifact(dataset, manifest, observed_sha256)


@dataclass(frozen=True, slots=True)
class LaterOutcomeTable:
    """Behavior observed after replay and stored outside the predictor artifact."""

    event_ids: tuple[str, ...]
    outcome_time_s: NDArray[np.float64]
    next_well_ids: tuple[str, ...]
    valid: NDArray[np.bool_]

    def validate(self, predictors: SpatialReplayDataset) -> None:
        n_rows = len(self.event_ids)
        if n_rows < 1 or len(set(self.event_ids)) != n_rows:
            raise ValueError("outcome event_ids must be nonempty and unique")
        times = np.asarray(self.outcome_time_s, dtype=float)
        valid = np.asarray(self.valid)
        if times.shape != (n_rows,) or not np.all(np.isfinite(times)):
            raise ValueError("outcome_time_s must contain one finite value per row")
        if valid.dtype != np.bool_ or valid.shape != (n_rows,):
            raise ValueError("outcome valid flags must be boolean")
        if len(self.next_well_ids) != n_rows:
            raise ValueError("next_well_ids must contain one value per outcome")
        event_lookup = {
            event_id: index for index, event_id in enumerate(predictors.event_ids)
        }
        for row, event_id in enumerate(self.event_ids):
            if event_id not in event_lookup:
                raise ValueError(f"outcome references unknown event {event_id!r}")
            predictor_index = event_lookup[event_id]
            if times[row] <= predictors.event_end_s[predictor_index]:
                raise ValueError("later outcomes must occur strictly after replay ends")
            if valid[row]:
                if not self.next_well_ids[row]:
                    raise ValueError("valid outcomes require a next well")
                if (
                    predictors.well_ids
                    and self.next_well_ids[row] not in predictors.well_ids
                ):
                    raise ValueError("valid outcome well is absent from predictor support")


@dataclass(frozen=True, slots=True)
class FrozenOutcomeArtifact:
    outcomes: LaterOutcomeTable
    predictor_sha256: str
    outcome_sha256: str


def write_later_outcome_artifact(
    directory: Path | str,
    predictors: FrozenPredictorArtifact,
    outcomes: LaterOutcomeTable,
) -> FrozenOutcomeArtifact:
    """Write outcomes separately and bind them to an already frozen predictor."""

    outcomes.validate(predictors.dataset)
    output = Path(directory)
    output.mkdir(parents=True, exist_ok=True)
    outcome_path = output / "replay_later_outcomes.npz"
    np.savez_compressed(
        outcome_path,
        event_ids=np.asarray(outcomes.event_ids, dtype=str),
        outcome_time_s=np.asarray(outcomes.outcome_time_s, dtype=float),
        next_well_ids=np.asarray(outcomes.next_well_ids, dtype=str),
        valid=np.asarray(outcomes.valid, dtype=bool),
    )
    outcome_sha256 = _sha256(outcome_path)
    payload = {
        "schema_version": LATER_OUTCOME_SCHEMA_VERSION,
        "outcome_file": outcome_path.name,
        "outcome_sha256": outcome_sha256,
        "predictor_sha256": predictors.predictor_sha256,
    }
    (output / "replay_later_outcome_manifest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return FrozenOutcomeArtifact(
        outcomes,
        predictors.predictor_sha256,
        outcome_sha256,
    )


def load_later_outcome_artifact(
    directory: Path | str,
    predictors: FrozenPredictorArtifact,
) -> FrozenOutcomeArtifact:
    """Load outcomes only after verifying their frozen predictor binding."""

    source = Path(directory)
    payload = json.loads(
        (source / "replay_later_outcome_manifest.json").read_text(encoding="utf-8")
    )
    if payload.get("schema_version") != LATER_OUTCOME_SCHEMA_VERSION:
        raise ValueError("unsupported later-outcome schema version")
    if payload.get("predictor_sha256") != predictors.predictor_sha256:
        raise ValueError("outcomes are not bound to this predictor artifact")
    outcome_path = source / str(payload["outcome_file"])
    observed_sha256 = _sha256(outcome_path)
    if observed_sha256 != payload.get("outcome_sha256"):
        raise ValueError("outcome artifact SHA-256 does not match its manifest")
    with np.load(outcome_path, allow_pickle=False) as arrays:
        outcomes = LaterOutcomeTable(
            event_ids=tuple(str(value) for value in arrays["event_ids"]),
            outcome_time_s=np.asarray(arrays["outcome_time_s"], dtype=float),
            next_well_ids=tuple(str(value) for value in arrays["next_well_ids"]),
            valid=np.asarray(arrays["valid"], dtype=bool),
        )
    outcomes.validate(predictors.dataset)
    return FrozenOutcomeArtifact(
        outcomes,
        predictors.predictor_sha256,
        observed_sha256,
    )


__all__ = [
    "FrozenOutcomeArtifact",
    "FrozenPredictorArtifact",
    "HIPPO_TRACE_SCHEMA_VERSION",
    "HIPPO_TRANSITION_CONVENTION",
    "LATER_OUTCOME_SCHEMA_VERSION",
    "LaterOutcomeTable",
    "ReplaySpatialManifest",
    "load_later_outcome_artifact",
    "load_spatial_predictor_artifact",
    "write_later_outcome_artifact",
    "write_spatial_predictor_artifact",
]
