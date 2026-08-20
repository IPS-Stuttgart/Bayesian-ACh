"""Forward ACh measurement models with sensor and tonic-timescale uncertainty."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True, slots=True)
class MeasurementGridPoint:
    """One sensor-kernel and stationary tonic-process hypothesis."""

    tau_rise: float
    tau_decay: float
    tonic_rho: float

    def validate(self, dt: float) -> None:
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt must be finite and positive")
        if not np.isfinite(self.tau_rise) or self.tau_rise <= 0.0:
            raise ValueError("tau_rise must be finite and positive")
        if not np.isfinite(self.tau_decay) or self.tau_decay <= self.tau_rise:
            raise ValueError("tau_decay must be finite and greater than tau_rise")
        if not np.isfinite(self.tonic_rho) or not 0.0 <= self.tonic_rho < 1.0:
            raise ValueError("tonic_rho must lie in [0, 1)")

    def tonic_timescale(self, dt: float) -> float:
        """Return the AR(1) e-folding time in seconds."""

        self.validate(dt)
        if self.tonic_rho == 0.0:
            return 0.0
        return float(-dt / np.log(self.tonic_rho))

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def default_measurement_grid() -> tuple[MeasurementGridPoint, ...]:
    """Return a compact grid spanning common fast indicator dynamics."""

    return tuple(
        MeasurementGridPoint(tau_rise, tau_decay, tonic_rho)
        for tau_rise in (0.25, 0.40, 0.60)
        for tau_decay in (1.00, 1.60, 2.40)
        if tau_rise < tau_decay
        for tonic_rho in (0.92, 0.97, 0.99)
    )


@dataclass(slots=True)
class MeasurementDataset:
    """Canonical arrays for calibration-separated ACh hypothesis testing.

    Test-session baseline samples may be used to remove a session offset because
    they precede the task and contain no candidate event. Sensor-kernel and
    regression fitting use training samples only.
    """

    observed: NDArray[np.float64]
    calibration_event: NDArray[np.float64]
    candidate_events: NDArray[np.float64]
    nuisance: NDArray[np.float64]
    subject_ids: NDArray[np.int64]
    session_ids: NDArray[np.int64]
    train_mask: NDArray[np.bool_]
    calibration_mask: NDArray[np.bool_]
    task_mask: NDArray[np.bool_]
    baseline_mask: NDArray[np.bool_]
    candidate_names: tuple[str, ...]
    nuisance_names: tuple[str, ...]

    def __post_init__(self) -> None:
        self.observed = np.asarray(self.observed, dtype=float)
        self.calibration_event = np.asarray(self.calibration_event, dtype=float)
        self.candidate_events = np.asarray(self.candidate_events, dtype=float)
        self.nuisance = np.asarray(self.nuisance, dtype=float)
        self.subject_ids = np.asarray(self.subject_ids, dtype=np.int64)
        self.session_ids = np.asarray(self.session_ids, dtype=np.int64)
        self.train_mask = np.asarray(self.train_mask, dtype=bool)
        self.calibration_mask = np.asarray(self.calibration_mask, dtype=bool)
        self.task_mask = np.asarray(self.task_mask, dtype=bool)
        self.baseline_mask = np.asarray(self.baseline_mask, dtype=bool)
        self.validate()

    @property
    def n_samples(self) -> int:
        return int(np.asarray(self.observed).size)

    @property
    def n_candidates(self) -> int:
        return int(np.asarray(self.candidate_events).shape[1])

    @property
    def n_nuisance(self) -> int:
        return int(np.asarray(self.nuisance).shape[1])

    def validate(self) -> None:
        observed = np.asarray(self.observed)
        candidate_events = np.asarray(self.candidate_events)
        nuisance = np.asarray(self.nuisance)
        if observed.ndim != 1:
            raise ValueError(f"observed must be one-dimensional; got {observed.shape}")
        n_samples = observed.size
        one_dimensional = {
            "calibration_event": np.asarray(self.calibration_event),
            "subject_ids": np.asarray(self.subject_ids),
            "session_ids": np.asarray(self.session_ids),
            "train_mask": np.asarray(self.train_mask),
            "calibration_mask": np.asarray(self.calibration_mask),
            "task_mask": np.asarray(self.task_mask),
            "baseline_mask": np.asarray(self.baseline_mask),
        }
        for name, values in one_dimensional.items():
            if values.shape != (n_samples,):
                raise ValueError(f"{name} must have shape ({n_samples},); got {values.shape}")
        if candidate_events.ndim != 2 or candidate_events.shape[0] != n_samples:
            raise ValueError(
                "candidate_events must have shape (sample, candidate); "
                f"got {candidate_events.shape}"
            )
        if nuisance.ndim != 2 or nuisance.shape[0] != n_samples:
            raise ValueError(
                f"nuisance must have shape (sample, regressor); got {nuisance.shape}"
            )
        if candidate_events.shape[1] != len(self.candidate_names):
            raise ValueError("candidate_names must match candidate_events columns")
        if nuisance.shape[1] != len(self.nuisance_names):
            raise ValueError("nuisance_names must match nuisance columns")
        if len(set(self.candidate_names)) != len(self.candidate_names):
            raise ValueError("candidate_names must be unique")
        if len(set(self.nuisance_names)) != len(self.nuisance_names):
            raise ValueError("nuisance_names must be unique")
        for name, values in (
            ("observed", observed),
            ("calibration_event", np.asarray(self.calibration_event)),
            ("candidate_events", candidate_events),
            ("nuisance", nuisance),
        ):
            if not np.all(np.isfinite(values)):
                raise ValueError(f"{name} must contain only finite values")
        if not np.any(self.train_mask & self.calibration_mask):
            raise ValueError("training calibration samples are required")
        if not np.any(self.train_mask & self.task_mask):
            raise ValueError("training task samples are required")
        if not np.any((~self.train_mask) & self.task_mask):
            raise ValueError("held-out task samples are required")
        for session in np.unique(self.session_ids):
            selected = self.session_ids == session
            if not np.any(selected & self.baseline_mask):
                raise ValueError(f"session {int(session)} has no baseline samples")
        train_subjects = set(int(value) for value in self.subject_ids[self.train_mask])
        test_subjects = set(int(value) for value in self.subject_ids[~self.train_mask])
        if not test_subjects.issubset(train_subjects):
            unseen = sorted(test_subjects - train_subjects)
            raise ValueError(f"test samples contain unseen subjects: {unseen}")


@dataclass(frozen=True, slots=True)
class MeasurementFitConfig:
    """Configuration for calibration-first held-out model comparison."""

    dt: float = 0.2
    grid: tuple[MeasurementGridPoint, ...] = field(default_factory=default_measurement_grid)
    subject_penalty: float = 16.0
    variance_floor: float = 1e-8

    def validate(self) -> None:
        if not np.isfinite(self.dt) or self.dt <= 0.0:
            raise ValueError("dt must be finite and positive")
        if not self.grid:
            raise ValueError("grid must contain at least one point")
        for point in self.grid:
            point.validate(self.dt)
        if not np.isfinite(self.subject_penalty) or self.subject_penalty <= 0.0:
            raise ValueError("subject_penalty must be finite and positive")
        if not np.isfinite(self.variance_floor) or self.variance_floor <= 0.0:
            raise ValueError("variance_floor must be finite and positive")


@dataclass(frozen=True, slots=True)
class CalibrationPosterior:
    """Discrete calibration posterior over sensor and tonic timescales."""

    points: tuple[MeasurementGridPoint, ...]
    log_weights: NDArray[np.float64]
    training_log_likelihoods: NDArray[np.float64]
    map_index: int
    weighted_tau_rise: float
    weighted_tau_decay: float
    weighted_tonic_rho: float
    weighted_tonic_timescale: float
    effective_grid_size: float

    @property
    def map_point(self) -> MeasurementGridPoint:
        return self.points[self.map_index]

    def rows(self) -> list[dict[str, float | int]]:
        weights = np.exp(self.log_weights)
        return [
            {
                "grid_index": index,
                "tau_rise": point.tau_rise,
                "tau_decay": point.tau_decay,
                "tonic_rho": point.tonic_rho,
                "training_log_likelihood": float(self.training_log_likelihoods[index]),
                "posterior_weight": float(weights[index]),
            }
            for index, point in enumerate(self.points)
        ]


@dataclass(frozen=True, slots=True)
class MeasurementCandidateFit:
    """Held-out score and MAP-grid coefficients for one event hypothesis."""

    candidate: str
    marginal_test_log_likelihood: float
    test_mean_log_likelihood: float
    map_test_log_likelihood: float
    map_test_innovation_r2: float
    global_signal_coefficient: float
    nuisance_coefficients: NDArray[np.float64]
    subject_signal_coefficients: NDArray[np.float64]
    residual_innovation_std: float
    n_train: int
    n_test: int

    def as_dict(self, nuisance_names: tuple[str, ...]) -> dict[str, Any]:
        row: dict[str, Any] = {
            "candidate": self.candidate,
            "marginal_test_log_likelihood": self.marginal_test_log_likelihood,
            "test_mean_log_likelihood": self.test_mean_log_likelihood,
            "map_test_log_likelihood": self.map_test_log_likelihood,
            "map_test_innovation_r2": self.map_test_innovation_r2,
            "global_signal_coefficient": self.global_signal_coefficient,
            "residual_innovation_std": self.residual_innovation_std,
            "n_train": self.n_train,
            "n_test": self.n_test,
        }
        for name, value in zip(nuisance_names, self.nuisance_coefficients, strict=True):
            row[f"nuisance_coefficient_{name}"] = float(value)
        for index, value in enumerate(self.subject_signal_coefficients):
            row[f"subject_{index}_signal_coefficient"] = float(value)
        return row


@dataclass(frozen=True, slots=True)
class MeasurementRecoveryResult:
    """Calibration posterior and held-out candidate ranking."""

    calibration: CalibrationPosterior
    fits: tuple[MeasurementCandidateFit, ...]
    candidate_names: tuple[str, ...]
    nuisance_names: tuple[str, ...]
    grid_test_log_likelihoods: NDArray[np.float64]

    @property
    def winner(self) -> MeasurementCandidateFit:
        return self.fits[0]

    def fit_rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for rank, fit in enumerate(self.fits, start=1):
            row = {"rank": rank}
            row.update(fit.as_dict(self.nuisance_names))
            rows.append(row)
        return rows

    def grid_rows(self) -> list[dict[str, float | int | str]]:
        rows: list[dict[str, float | int | str]] = []
        for candidate_index, candidate in enumerate(self.candidate_names):
            for grid_index, point in enumerate(self.calibration.points):
                rows.append(
                    {
                        "candidate": candidate,
                        "grid_index": grid_index,
                        "tau_rise": point.tau_rise,
                        "tau_decay": point.tau_decay,
                        "tonic_rho": point.tonic_rho,
                        "calibration_posterior_weight": float(
                            np.exp(self.calibration.log_weights[grid_index])
                        ),
                        "test_log_likelihood": float(
                            self.grid_test_log_likelihoods[candidate_index, grid_index]
                        ),
                    }
                )
        return rows


def double_exponential_kernel(
    dt: float,
    tau_rise: float,
    tau_decay: float,
    *,
    duration: float | None = None,
) -> NDArray[np.float64]:
    """Return a causal, unit-peak difference-of-exponentials kernel."""

    point = MeasurementGridPoint(tau_rise, tau_decay, 0.0)
    point.validate(dt)
    if duration is None:
        duration = max(6.0 * tau_decay, dt)
    if not np.isfinite(duration) or duration <= 0.0:
        raise ValueError("duration must be finite and positive")
    times = np.arange(int(np.ceil(duration / dt)) + 1, dtype=float) * dt
    kernel = np.exp(-times / tau_decay) - np.exp(-times / tau_rise)
    peak = float(np.max(kernel))
    if peak <= 0.0:
        raise FloatingPointError("double-exponential kernel has non-positive peak")
    return np.asarray(kernel / peak, dtype=float)


def convolve_by_session(
    event_train: ArrayLike,
    session_ids: ArrayLike,
    kernel: ArrayLike,
) -> NDArray[np.float64]:
    """Apply a causal convolution independently within every session."""

    events = np.asarray(event_train, dtype=float)
    sessions = np.asarray(session_ids, dtype=np.int64)
    impulse_response = np.asarray(kernel, dtype=float)
    if events.ndim != 1 or sessions.shape != events.shape:
        raise ValueError("event_train and session_ids must be equal-length vectors")
    if impulse_response.ndim != 1 or impulse_response.size < 1:
        raise ValueError("kernel must be a non-empty vector")
    if not np.all(np.isfinite(events)) or not np.all(np.isfinite(impulse_response)):
        raise ValueError("event_train and kernel must contain only finite values")
    result = np.zeros_like(events, dtype=float)
    for session in np.unique(sessions):
        indices = np.flatnonzero(sessions == session)
        result[indices] = np.convolve(
            events[indices],
            impulse_response,
            mode="full",
        )[: indices.size]
    return result


def tonic_sensor_ar_coefficients(
    point: MeasurementGridPoint,
    dt: float,
) -> tuple[float, float, float]:
    """Return the conditional AR(3) coefficients of filtered tonic release.

    A latent AR(1) tonic release passed through rise and decay first-order
    indicator states has denominator
    ``(1-a_r L)(1-a_d L)(1-rho L)``. Consequently, after conditioning on the
    first three sensor samples, the remaining innovations are independent.
    """

    point.validate(dt)
    rise_decay = float(np.exp(-dt / point.tau_rise))
    slow_decay = float(np.exp(-dt / point.tau_decay))
    first = rise_decay + slow_decay + point.tonic_rho
    second = (
        rise_decay * slow_decay
        + rise_decay * point.tonic_rho
        + slow_decay * point.tonic_rho
    )
    third = rise_decay * slow_decay * point.tonic_rho
    return first, second, third


def fit_measurement_models(
    dataset: MeasurementDataset,
    config: MeasurementFitConfig | None = None,
) -> MeasurementRecoveryResult:
    """Fit and rank event candidates using strict calibration/train/test separation."""

    from bayesian_ach.measurement_fit import fit_measurement_models as _fit

    return _fit(dataset, config)
