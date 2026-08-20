"""Synthetic recovery for ACh release, sensor dynamics, and event hypotheses."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Final, TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from bayesian_ach.measurement import (
    MeasurementDataset,
    MeasurementFitConfig,
    convolve_by_session,
    double_exponential_kernel,
    fit_measurement_models,
)

MEASUREMENT_CANDIDATE_NAMES: Final[tuple[str, ...]] = (
    "predictive_surprise",
    "state_information_gain",
    "context_information_gain",
    "sensor_health_information_gain",
    "context_switch_probability",
    "visual_fault_onset_probability",
    "state_sensor_conflict",
)

MEASUREMENT_NUISANCE_NAMES: Final[tuple[str, ...]] = (
    "movement",
    "acceleration",
    "pupil",
    "theta",
    "engagement",
)


@dataclass(frozen=True, slots=True)
class MeasurementBenchmarkConfig:
    """Configuration for held-out ACh measurement-model recovery."""

    n_subjects: int = 6
    sessions_per_subject: int = 5
    train_sessions_per_subject: int = 3
    calibration_length: int = 112
    task_length: int = 144
    dt: float = 0.2
    tau_rise: float = 0.40
    tau_decay: float = 1.60
    tonic_rho: float = 0.97
    tonic_innovation_std: float = 0.25
    calibration_amplitude: float = 2.50
    phasic_amplitude: float = 0.35
    subject_amplitude_std: float = 0.16
    subject_baseline_std: float = 0.25
    session_baseline_std: float = 0.45
    nuisance_coefficients: tuple[float, ...] = (0.50, -0.25, 0.40, 0.30, 0.20)
    seed: int = 7

    def validate(self) -> None:
        if self.n_subjects < 2:
            raise ValueError("n_subjects must be at least two")
        if self.sessions_per_subject < 2:
            raise ValueError("sessions_per_subject must be at least two")
        if not 1 <= self.train_sessions_per_subject < self.sessions_per_subject:
            raise ValueError(
                "train_sessions_per_subject must be positive and below sessions_per_subject"
            )
        if self.calibration_length < 32:
            raise ValueError("calibration_length must be at least 32")
        if self.task_length < 64:
            raise ValueError("task_length must be at least 64")
        if not np.isfinite(self.dt) or self.dt <= 0.0:
            raise ValueError("dt must be finite and positive")
        if not 0.0 < self.tau_rise < self.tau_decay:
            raise ValueError("require 0 < tau_rise < tau_decay")
        if not 0.0 <= self.tonic_rho < 1.0:
            raise ValueError("tonic_rho must lie in [0, 1)")
        for name, value in (
            ("tonic_innovation_std", self.tonic_innovation_std),
            ("calibration_amplitude", self.calibration_amplitude),
            ("phasic_amplitude", self.phasic_amplitude),
            ("subject_amplitude_std", self.subject_amplitude_std),
            ("subject_baseline_std", self.subject_baseline_std),
            ("session_baseline_std", self.session_baseline_std),
        ):
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if len(self.nuisance_coefficients) != len(MEASUREMENT_NUISANCE_NAMES):
            raise ValueError("nuisance_coefficients must match the nuisance regressors")
        if not np.all(np.isfinite(self.nuisance_coefficients)):
            raise ValueError("nuisance_coefficients must be finite")


@dataclass(frozen=True, slots=True)
class MeasurementGeneratorResult:
    """Recovery summary for one generating computational event train."""

    generator: str
    winner: str
    correct: bool
    evidence_margin: float
    winner_test_innovation_r2: float
    map_tau_rise: float
    map_tau_decay: float
    map_tonic_rho: float
    map_timescales_match_truth: bool
    weighted_tau_rise: float
    weighted_tau_decay: float
    weighted_tonic_rho: float
    weighted_tonic_timescale: float
    tau_rise_absolute_error: float
    tau_decay_absolute_error: float
    tonic_rho_absolute_error: float
    nuisance_coefficient_mae: float
    subject_signal_correlation: float
    subject_signal_mae: float
    effective_grid_size: float

    def as_dict(self) -> dict[str, float | str | bool]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class MeasurementBenchmarkResult:
    """Complete synthetic measurement-model evidence package."""

    generators: tuple[MeasurementGeneratorResult, ...]
    fits: tuple[dict[str, Any], ...]
    kernel_posteriors: tuple[dict[str, Any], ...]
    samples: tuple[dict[str, Any], ...]
    summary: dict[str, Any]


if TYPE_CHECKING:
    from bayesian_ach.measurement_simulation import _LatentMeasurementDesign


def _sample_rows(
    design: _LatentMeasurementDesign,
    observed_by_generator: NDArray[np.float64],
) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    for sample_index in range(design.subject_ids.size):
        phase = "task" if design.task_mask[sample_index] else "calibration"
        row: dict[str, Any] = {
            "sample_index": sample_index,
            "subject_id": int(design.subject_ids[sample_index]),
            "session_id": int(design.session_ids[sample_index]),
            "split": "train" if design.train_mask[sample_index] else "test",
            "phase": phase,
            "is_baseline": int(design.baseline_mask[sample_index]),
            "calibration_event": float(design.calibration_event[sample_index]),
        }
        for candidate_index, name in enumerate(MEASUREMENT_CANDIDATE_NAMES):
            row[f"candidate_{name}"] = float(
                design.candidate_events[sample_index, candidate_index]
            )
            row[f"observed_if_{name}"] = float(
                observed_by_generator[candidate_index, sample_index]
            )
        for nuisance_index, name in enumerate(MEASUREMENT_NUISANCE_NAMES):
            row[f"nuisance_{name}"] = float(design.nuisance[sample_index, nuisance_index])
        rows.append(row)
    return tuple(rows)


def run_measurement_benchmark(
    config: MeasurementBenchmarkConfig,
) -> MeasurementBenchmarkResult:
    """Recover the generating Bayesian event train after realistic measurement dynamics."""

    from bayesian_ach.measurement_simulation import (
        _common_measurement_components,
        _latent_design,
        _observed_trace,
        _safe_correlation,
    )

    config.validate()
    design = _latent_design(config)
    fit_config = MeasurementFitConfig(dt=config.dt)
    generators: list[MeasurementGeneratorResult] = []
    fit_rows: list[dict[str, Any]] = []
    observed_by_generator = np.empty(
        (len(MEASUREMENT_CANDIDATE_NAMES), design.subject_ids.size),
        dtype=float,
    )
    nuisance_truth = np.asarray(config.nuisance_coefficients, dtype=float)
    common = _common_measurement_components(design, config)
    common_kernel_rows: list[dict[str, float | int]] | None = None

    for generator_index, generator in enumerate(MEASUREMENT_CANDIDATE_NAMES):
        observed, true_subject_coefficients = _observed_trace(
            design,
            generator_index,
            config,
            common,
        )
        observed_by_generator[generator_index] = observed
        dataset = MeasurementDataset(
            observed=observed,
            calibration_event=design.calibration_event,
            candidate_events=design.candidate_events,
            nuisance=design.nuisance,
            subject_ids=design.subject_ids,
            session_ids=design.session_ids,
            train_mask=design.train_mask,
            calibration_mask=design.calibration_mask,
            task_mask=design.task_mask,
            baseline_mask=design.baseline_mask,
            candidate_names=MEASUREMENT_CANDIDATE_NAMES,
            nuisance_names=MEASUREMENT_NUISANCE_NAMES,
        )
        recovery = fit_measurement_models(dataset, fit_config)
        winner = recovery.winner
        runner_up = recovery.fits[1]
        nuisance_mae = float(
            np.mean(np.abs(winner.nuisance_coefficients - nuisance_truth))
        )
        subject_correlation = _safe_correlation(
            winner.subject_signal_coefficients,
            true_subject_coefficients,
        )
        subject_mae = float(
            np.mean(
                np.abs(
                    winner.subject_signal_coefficients - true_subject_coefficients
                )
            )
        )
        calibration = recovery.calibration
        generators.append(
            MeasurementGeneratorResult(
                generator=generator,
                winner=winner.candidate,
                correct=winner.candidate == generator,
                evidence_margin=(
                    winner.marginal_test_log_likelihood
                    - runner_up.marginal_test_log_likelihood
                ),
                winner_test_innovation_r2=winner.map_test_innovation_r2,
                map_tau_rise=calibration.map_point.tau_rise,
                map_tau_decay=calibration.map_point.tau_decay,
                map_tonic_rho=calibration.map_point.tonic_rho,
                map_timescales_match_truth=bool(
                    np.isclose(calibration.map_point.tau_rise, config.tau_rise)
                    and np.isclose(calibration.map_point.tau_decay, config.tau_decay)
                    and np.isclose(calibration.map_point.tonic_rho, config.tonic_rho)
                ),
                weighted_tau_rise=calibration.weighted_tau_rise,
                weighted_tau_decay=calibration.weighted_tau_decay,
                weighted_tonic_rho=calibration.weighted_tonic_rho,
                weighted_tonic_timescale=calibration.weighted_tonic_timescale,
                tau_rise_absolute_error=abs(
                    calibration.weighted_tau_rise - config.tau_rise
                ),
                tau_decay_absolute_error=abs(
                    calibration.weighted_tau_decay - config.tau_decay
                ),
                tonic_rho_absolute_error=abs(
                    calibration.weighted_tonic_rho - config.tonic_rho
                ),
                nuisance_coefficient_mae=nuisance_mae,
                subject_signal_correlation=subject_correlation,
                subject_signal_mae=subject_mae,
                effective_grid_size=calibration.effective_grid_size,
            )
        )
        for row in recovery.fit_rows():
            fit_rows.append({"generator": generator, **row})
        if common_kernel_rows is None:
            common_kernel_rows = recovery.calibration.rows()

    kernel_rows: list[dict[str, Any]] = (
        []
        if common_kernel_rows is None
        else [dict(row) for row in common_kernel_rows]
    )
    generator_tuple = tuple(generators)
    recovery_count = sum(result.correct for result in generator_tuple)
    true_kernel = double_exponential_kernel(
        config.dt,
        config.tau_rise,
        config.tau_decay,
    )
    convolved_candidates = np.column_stack(
        [
            convolve_by_session(
                design.candidate_events[:, candidate_index],
                design.session_ids,
                true_kernel,
            )
            for candidate_index in range(len(MEASUREMENT_CANDIDATE_NAMES))
        ]
    )
    correlation_mask = design.train_mask & design.task_mask
    candidate_correlation = np.corrcoef(
        convolved_candidates[correlation_mask],
        rowvar=False,
    )
    off_diagonal = np.abs(
        candidate_correlation[~np.eye(candidate_correlation.shape[0], dtype=bool)]
    )
    summary: dict[str, Any] = {
        "experiment": "ach_measurement_model_recovery",
        "config": asdict(config),
        "candidate_names": list(MEASUREMENT_CANDIDATE_NAMES),
        "nuisance_names": list(MEASUREMENT_NUISANCE_NAMES),
        "recovery_count": recovery_count,
        "candidate_count": len(MEASUREMENT_CANDIDATE_NAMES),
        "all_generators_recovered": recovery_count == len(MEASUREMENT_CANDIDATE_NAMES),
        "median_evidence_margin": float(
            np.median([result.evidence_margin for result in generator_tuple])
        ),
        "minimum_evidence_margin": float(
            np.min([result.evidence_margin for result in generator_tuple])
        ),
        "median_winner_test_innovation_r2": float(
            np.median(
                [result.winner_test_innovation_r2 for result in generator_tuple]
            )
        ),
        "calibration_map_timescales_match_truth": all(
            result.map_timescales_match_truth for result in generator_tuple
        ),
        "calibration_map": {
            "tau_rise": generator_tuple[0].map_tau_rise,
            "tau_decay": generator_tuple[0].map_tau_decay,
            "tonic_rho": generator_tuple[0].map_tonic_rho,
        },
        "median_tau_rise_absolute_error": float(
            np.median([result.tau_rise_absolute_error for result in generator_tuple])
        ),
        "median_tau_decay_absolute_error": float(
            np.median([result.tau_decay_absolute_error for result in generator_tuple])
        ),
        "median_tonic_rho_absolute_error": float(
            np.median([result.tonic_rho_absolute_error for result in generator_tuple])
        ),
        "median_nuisance_coefficient_mae": float(
            np.median([result.nuisance_coefficient_mae for result in generator_tuple])
        ),
        "median_subject_signal_correlation": float(
            np.nanmedian([result.subject_signal_correlation for result in generator_tuple])
        ),
        "maximum_absolute_candidate_correlation_after_sensor": float(
            np.max(off_diagonal)
        ),
        "strict_separation": {
            "kernel_and_tonic_grid": "training calibration samples only",
            "candidate_coefficients": "training task samples only",
            "model_scores": "held-out task samples only",
            "test_session_baseline": "pre-task baseline samples only",
        },
        "interpretation": (
            "A forward measurement model separates a candidate phasic event drive, a latent "
            "AR(1) tonic-release process, indicator kinetics, partially pooled subject effects, "
            "baseline-only session offsets, and movement/arousal covariates before "
            "held-out model comparison."
        ),
        "scope": (
            "Kernel uncertainty is represented by a discrete calibration posterior. Regression "
            "coefficients are plug-in Gaussian-ridge estimates rather than a fully marginalized "
            "hierarchical posterior. The conditional AR(3) likelihood omits an "
            "additional independent white sensor-noise term."
        ),
    }
    return MeasurementBenchmarkResult(
        generators=generator_tuple,
        fits=tuple(fit_rows),
        kernel_posteriors=tuple(kernel_rows),
        samples=_sample_rows(design, observed_by_generator),
        summary=summary,
    )
