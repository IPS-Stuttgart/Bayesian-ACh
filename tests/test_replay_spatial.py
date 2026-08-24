from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from bayesian_ach.replay_artifact import (
    LaterOutcomeTable,
    ReplaySpatialManifest,
    load_later_outcome_artifact,
    load_spatial_predictor_artifact,
    write_later_outcome_artifact,
    write_spatial_predictor_artifact,
)
from bayesian_ach.replay_recovery import (
    SpatialInjectionRecoveryConfig,
    run_spatial_recovery_checks,
)
from bayesian_ach.replay_spatial import (
    SPATIAL_CANDIDATE_NAMES,
    SpatialComparisonConfig,
    SpatialRecoveryGate,
    SpatialRecoveryRecord,
    SpatialReplayDataset,
    build_signed_revision_field,
    compare_spatial_replay_candidates,
)


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values, axis=-1, keepdims=True)
    probabilities = np.exp(shifted)
    return probabilities / probabilities.sum(axis=-1, keepdims=True)


def _dataset(*, signal: bool = True, seed: int = 17) -> SpatialReplayDataset:
    rng = np.random.default_rng(seed)
    n_rats = 4
    events_per_rat = 8
    n_events = n_rats * events_per_rat
    n_time = 5
    n_bins = 12
    n_candidates = len(SPATIAL_CANDIDATE_NAMES)
    rats = np.repeat([f"Rat{index}" for index in range(n_rats)], events_per_rat)
    sessions = np.asarray(
        [
            f"{rat}/Session{1 + (event_index % 2)}"
            for event_index, rat in enumerate(rats)
        ],
        dtype=str,
    )
    starts = 100.0 + np.arange(n_events, dtype=float) * 10.0
    fields = rng.normal(size=(n_events, n_candidates, n_bins))
    base = rng.uniform(0.5, 1.5, size=(n_events, n_bins))
    base /= base.sum(axis=1, keepdims=True)

    revision = fields[:, 0]
    revision -= np.sum(base * revision, axis=1, keepdims=True)
    revision /= np.sqrt(
        np.sum(base * revision**2, axis=1, keepdims=True)
    )
    log_emissions = np.empty((n_events, n_time, n_bins), dtype=float)
    for event_index in range(n_events):
        for time_index in range(n_time):
            row = rng.normal(0.0, 0.12, n_bins)
            if signal:
                row += 3.0 * revision[event_index]
            row -= np.max(row)
            log_emissions[event_index, time_index] = row

    well_mass = _softmax(rng.normal(size=(n_events, 3)))
    grid_x, grid_y = np.meshgrid(np.arange(4, dtype=float), np.arange(3, dtype=float))
    coordinates = np.column_stack((grid_x.ravel(), grid_y.ravel()))
    return SpatialReplayDataset(
        event_ids=tuple(f"event-{index:03d}" for index in range(n_events)),
        rat_ids=np.asarray(rats, dtype=str),
        session_ids=sessions,
        event_start_s=starts,
        event_end_s=starts + 0.12,
        history_cutoff_s=starts - 0.01,
        decoder_training_cutoff_s=starts - 1.0,
        field_available_s=np.repeat(
            (starts - 0.01)[:, None],
            n_candidates,
            axis=1,
        ),
        log_emissions=log_emissions,
        log_emission_offsets=rng.normal(size=(n_events, n_time)),
        time_mask=np.ones((n_events, n_time), dtype=bool),
        active_spatial_mask=np.ones((n_events, n_bins), dtype=bool),
        spatial_coordinates=np.broadcast_to(
            coordinates[None, :, :],
            (n_events, n_bins, 2),
        ).copy(),
        decoder_point_spread_cm=np.full(n_events, 0.5, dtype=float),
        nuisance_base=base,
        candidate_fields=fields,
        candidate_available=np.ones((n_events, n_candidates), dtype=bool),
        well_masses=well_mass,
        well_ids=("well-0", "well-1", "well-2"),
    )


def _passing_gate() -> SpatialRecoveryGate:
    pure = tuple(
        SpatialRecoveryRecord(
            generator=generator,
            split_unit=split_unit,
            selected_candidate=generator,
            selected_margin=1.0,
            selected_margin_lower=0.5,
            decisive=True,
            n_held_out_groups=4,
            spatial_sigma_multiplier=1.0,
        )
        for generator in SPATIAL_CANDIDATE_NAMES
        for split_unit in ("leave_one_rat_out", "leave_one_session_out")
    )
    mixture = tuple(
        SpatialRecoveryRecord(
            generator="smoothing_revision+td_error",
            split_unit=split_unit,
            selected_candidate="smoothing_revision",
            selected_margin=0.0,
            selected_margin_lower=-0.1,
            decisive=False,
            n_held_out_groups=4,
            spatial_sigma_multiplier=1.0,
        )
        for split_unit in ("leave_one_rat_out", "leave_one_session_out")
    )
    return SpatialRecoveryGate(
        pure_records=pure,
        mixture_records=mixture,
        required_mixtures=("smoothing_revision+td_error",),
        required_sigma_multipliers=(1.0,),
    )


def _config() -> SpatialComparisonConfig:
    return SpatialComparisonConfig(
        temperatures=(0.0, 0.5, 1.0, 2.0, 3.0, 4.0),
        bootstrap_replicates=1000,
        maximum_field_correlation=0.999,
        seed=11,
    )


def test_signed_revision_field_is_kl_weighted_signed_and_pre_replay() -> None:
    filtered = np.array([[0.8, 0.2], [0.5, 0.5]], dtype=float)
    smoothed = np.array([[0.6, 0.4], [0.2, 0.8]], dtype=float)
    ends = np.array([5.0, 9.0], dtype=float)

    mapping = np.array(
        [
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        ]
    )
    result = build_signed_revision_field(
        filtered,
        smoothed,
        ends,
        event_start_s=10.0,
        state_to_spatial=mapping,
        recency_tau_s=4.0,
    )

    kl = np.sum(smoothed * np.log(smoothed / filtered), axis=1)
    weights = kl * np.exp(-(10.0 - ends) / 4.0)
    projected = np.einsum("hs,hsb->hb", smoothed - filtered, mapping)
    expected = np.sum(weights[:, None] * projected, axis=0)
    np.testing.assert_allclose(result.per_snippet_kl, kl, atol=1e-14)
    np.testing.assert_allclose(result.snippet_weights, weights, atol=1e-14)
    np.testing.assert_allclose(result.signed_field, expected, atol=1e-14)
    np.testing.assert_allclose(result.signed_field.sum(), 0.0, atol=1e-14)
    assert result.identifiable is True

    with pytest.raises(ValueError, match="end before replay"):
        build_signed_revision_field(
            filtered,
            smoothed,
            np.array([5.0, 10.1]),
            event_start_s=10.0,
        )


def test_signed_revision_rejects_smoothing_mass_outside_filter_support() -> None:
    with pytest.raises(ValueError, match="outside filtering support"):
        build_signed_revision_field(
            np.array([[1.0, 0.0]]),
            np.array([[0.5, 0.5]]),
            np.array([1.0]),
            event_start_s=2.0,
        )


def test_dataset_rejects_future_candidate_or_decoder_evidence() -> None:
    dataset = _dataset()
    dataset.validate()

    future_field = np.asarray(dataset.field_available_s).copy()
    future_field[0, 0] = dataset.event_start_s[0] + 0.1
    with pytest.raises(ValueError, match="before replay"):
        replace(dataset, field_available_s=future_field).validate()

    future_decoder = np.asarray(dataset.decoder_training_cutoff_s).copy()
    future_decoder[0] = dataset.event_start_s[0] + 0.1
    with pytest.raises(ValueError, match="decoder training"):
        replace(dataset, decoder_training_cutoff_s=future_decoder).validate()


def test_loro_raw_emission_score_recovers_revision_only_with_gate() -> None:
    dataset = _dataset(signal=True)
    missing_gate = compare_spatial_replay_candidates(dataset, _config())
    assert missing_gate.winner == "smoothing_revision"
    assert missing_gate.status == "abstain"
    assert "recovery_gate_missing" in missing_gate.abstention_reasons

    result = compare_spatial_replay_candidates(
        dataset,
        _config(),
        recovery_gate=_passing_gate(),
    )

    assert result.winner == "smoothing_revision"
    assert result.runner_up != result.winner
    assert result.winner_margin > 0.0
    assert result.winner_margin_ci[0] > 0.0
    assert all(
        contrast.simultaneous_lower_bound > 0.0
        for contrast in result.target_contrasts
    )
    assert result.target_candidate == "smoothing_revision"
    assert result.status == "identified"
    assert result.rat_ids == ("Rat0", "Rat1", "Rat2", "Rat3")
    assert result.rat_scores.shape == (4, len(SPATIAL_CANDIDATE_NAMES) + 1)
    assert all(fold.n_sessions == 2 for fold in result.folds)


def test_candidate_differences_are_invariant_to_log_emission_offsets() -> None:
    dataset = _dataset(signal=True)
    baseline = compare_spatial_replay_candidates(
        dataset,
        _config(),
        recovery_gate=_passing_gate(),
    )
    shift = np.linspace(-100.0, 100.0, dataset.n_events)[:, None]
    changed = replace(
        dataset,
        log_emission_offsets=dataset.log_emission_offsets + shift,
    )
    shifted = compare_spatial_replay_candidates(
        changed,
        _config(),
        recovery_gate=_passing_gate(),
    )

    assert shifted.winner == baseline.winner
    assert shifted.runner_up == baseline.runner_up
    np.testing.assert_allclose(
        shifted.rat_scores
        - shifted.rat_scores[:, [-1]],
        baseline.rat_scores
        - baseline.rat_scores[:, [-1]],
        atol=1e-12,
    )
    np.testing.assert_allclose(
        shifted.winner_margin,
        baseline.winner_margin,
        atol=1e-12,
    )


def test_uninformative_emissions_force_uncertainty_abstention() -> None:
    dataset = _dataset(signal=False)
    zero = np.zeros_like(dataset.log_emissions)
    result = compare_spatial_replay_candidates(
        replace(dataset, log_emissions=zero),
        _config(),
        recovery_gate=_passing_gate(),
    )

    assert result.status == "abstain"
    assert "smoothing_revision_contrast_uncertain" in result.abstention_reasons
    assert result.winner_margin == pytest.approx(0.0, abs=1e-14)


def test_recovery_is_computed_from_loao_and_loso_emission_injections() -> None:
    dataset = _dataset(signal=False)
    gate = run_spatial_recovery_checks(
        dataset,
        _config(),
        SpatialInjectionRecoveryConfig(
            injection_temperature=4.0,
            spatial_sigma_multipliers=(1.0,),
            emission_noise_sd_nats=0.0,
            mixtures=(("smoothing_revision", "td_error"),),
            seed=29,
        ),
    )

    assert len(gate.pure_records) == 2 * len(SPATIAL_CANDIDATE_NAMES)
    assert {record.generator for record in gate.pure_records} == set(
        SPATIAL_CANDIDATE_NAMES
    )
    assert {record.split_unit for record in gate.pure_records} == {
        "leave_one_rat_out",
        "leave_one_session_out",
    }
    assert {record.generator for record in gate.mixture_records} == {
        "smoothing_revision+td_error"
    }
    assert all(record.n_held_out_groups >= 4 for record in gate.pure_records)
    assert all(np.isfinite(record.selected_margin) for record in gate.pure_records)


def test_decisive_td_win_cannot_pass_as_mixture_abstention() -> None:
    passing = _passing_gate()
    decisive_td = tuple(
        SpatialRecoveryRecord(
            generator="smoothing_revision+td_error",
            split_unit=split_unit,
            selected_candidate="td_error",
            selected_margin=0.4,
            selected_margin_lower=0.2,
            decisive=True,
            n_held_out_groups=4,
            spatial_sigma_multiplier=1.0,
        )
        for split_unit in ("leave_one_rat_out", "leave_one_session_out")
    )
    invalid = replace(passing, mixture_records=decisive_td)

    assert passing.passed is True
    assert invalid.passed is False


def test_spatial_coordinates_are_required_and_cannot_leak_off_support() -> None:
    dataset = _dataset()
    missing = np.asarray(dataset.spatial_coordinates).copy()
    missing[0, 0] = np.nan
    with pytest.raises(ValueError, match="active spatial coordinates"):
        replace(dataset, spatial_coordinates=missing).validate()


def test_predictor_and_later_outcome_artifacts_are_separate_and_hash_bound(
    tmp_path,
) -> None:
    dataset = _dataset(signal=True)
    manifest = ReplaySpatialManifest(
        producer_repository="IPS-Stuttgart/HippoReplayDynamics",
        producer_commit="1" * 40,
        dataset_id="PfeifferFoster-open-field-2013",
        dataset_sha256="2" * 64,
    )

    frozen = write_spatial_predictor_artifact(tmp_path / "predictors", dataset, manifest)
    loaded = load_spatial_predictor_artifact(tmp_path / "predictors")
    assert loaded.predictor_sha256 == frozen.predictor_sha256
    assert loaded.dataset.event_ids == dataset.event_ids
    np.testing.assert_allclose(loaded.dataset.log_emissions, dataset.log_emissions)
    np.testing.assert_allclose(
        loaded.dataset.spatial_coordinates,
        dataset.spatial_coordinates,
    )
    np.testing.assert_allclose(
        loaded.dataset.decoder_point_spread_cm,
        dataset.decoder_point_spread_cm,
    )
    np.testing.assert_allclose(loaded.dataset.well_masses, dataset.well_masses)

    outcomes = LaterOutcomeTable(
        event_ids=dataset.event_ids,
        outcome_time_s=dataset.event_end_s + 1.0,
        next_well_ids=tuple("well-0" for _ in dataset.event_ids),
        valid=np.ones(dataset.n_events, dtype=bool),
    )
    written_outcomes = write_later_outcome_artifact(
        tmp_path / "outcomes",
        loaded,
        outcomes,
    )
    loaded_outcomes = load_later_outcome_artifact(
        tmp_path / "outcomes",
        loaded,
    )
    assert loaded_outcomes.predictor_sha256 == loaded.predictor_sha256
    assert loaded_outcomes.outcome_sha256 == written_outcomes.outcome_sha256
    assert loaded_outcomes.outcomes.next_well_ids == outcomes.next_well_ids

    too_early = replace(
        outcomes,
        outcome_time_s=np.asarray(dataset.event_end_s).copy(),
    )
    with pytest.raises(ValueError, match="strictly after"):
        too_early.validate(dataset)


def test_predictor_hash_tampering_is_detected(tmp_path) -> None:
    dataset = _dataset()
    manifest = ReplaySpatialManifest(
        producer_repository="IPS-Stuttgart/HippoReplayDynamics",
        producer_commit="a" * 40,
        dataset_id="PfeifferFoster-open-field-2013",
        dataset_sha256="b" * 64,
    )
    directory = tmp_path / "predictors"
    write_spatial_predictor_artifact(directory, dataset, manifest)
    with (directory / "replay_spatial_predictors.npz").open("ab") as handle:
        handle.write(b"tamper")

    with pytest.raises(ValueError, match="SHA-256"):
        load_spatial_predictor_artifact(directory)
