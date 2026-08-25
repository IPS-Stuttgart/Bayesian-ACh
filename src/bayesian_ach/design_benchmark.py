"""Equal-budget model recovery for prospective maximin trial allocation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray

from bayesian_ach.design import (
    DESIGN_CANDIDATE_NAMES,
    TransitionDesignGridConfig,
    coupled_novelty_design,
    design_diagnostics,
    generate_transition_design_grid,
    optimize_maximin_design,
    pairwise_residual_matrix,
    profiled_gaussian_log_score_gap,
    uniform_factorial_design,
)
from bayesian_ach.design_recovery import DesignRecoveryRow, recover_design

DESIGN_NAMES: Final[tuple[str, ...]] = (
    "coupled_novelty",
    "uniform_factorial",
    "maximin_optimized",
)


@dataclass(frozen=True, slots=True)
class DesignBenchmarkConfig:
    """Configuration for equal-budget trial-design model recovery."""

    budget: int = 60
    replicates_per_generator: int = 200
    test_fraction: float = 0.35
    effect_size: float = 1.0
    noise_std: float = 1.0
    target_log_score_gap: float = 5.0
    max_point_fraction: float = 0.15
    seed: int = 7
    target_log_bf: float | None = None

    def __post_init__(self) -> None:
        if self.target_log_bf is None:
            object.__setattr__(
                self,
                "target_log_bf",
                float(self.target_log_score_gap),
            )

    @property
    def resolved_target_log_score_gap(self) -> float:
        if self.target_log_bf is None:
            return float(self.target_log_score_gap)
        if self.target_log_score_gap != 5.0 and not np.isclose(
            self.target_log_score_gap,
            self.target_log_bf,
        ):
            raise ValueError(
                "target_log_score_gap and deprecated target_log_bf disagree"
            )
        return float(self.target_log_bf)

    def validate(self) -> None:
        if self.budget < len(DESIGN_CANDIDATE_NAMES) + 1:
            raise ValueError("budget is too small for all candidate dimensions")
        if self.replicates_per_generator < 1:
            raise ValueError("replicates_per_generator must be positive")
        if not 0.0 < self.test_fraction < 1.0:
            raise ValueError("test_fraction must lie in (0, 1)")
        if self.effect_size <= 0.0 or self.noise_std <= 0.0:
            raise ValueError("effect_size and noise_std must be positive")
        target = self.resolved_target_log_score_gap
        if not np.isfinite(target) or target <= 0.0:
            raise ValueError("target_log_score_gap must be finite and positive")
        if not 0.0 < self.max_point_fraction <= 1.0:
            raise ValueError("max_point_fraction must lie in (0, 1]")


@dataclass(frozen=True, slots=True)
class DesignBenchmarkResult:
    """Complete prospective design evidence package."""

    allocations: tuple[dict[str, Any], ...]
    diagnostics: tuple[dict[str, Any], ...]
    pairwise_geometry: tuple[dict[str, Any], ...]
    recovery: tuple[DesignRecoveryRow, ...]
    optimization_trace: tuple[dict[str, Any], ...]
    grid: tuple[dict[str, Any], ...]
    summary: dict[str, Any]


def run_design_benchmark(
    config: DesignBenchmarkConfig | None = None,
    grid_config: TransitionDesignGridConfig | None = None,
) -> DesignBenchmarkResult:
    """Run design geometry and held-out recovery at a common trial budget."""

    config = DesignBenchmarkConfig() if config is None else config
    grid_config = TransitionDesignGridConfig() if grid_config is None else grid_config
    config.validate()
    rows, raw, standardized = generate_transition_design_grid(grid_config)
    optimized = optimize_maximin_design(
        standardized,
        config.budget,
        max_point_fraction=config.max_point_fraction,
        effect_size=config.effect_size,
        noise_std=config.noise_std,
        target_log_score_gap=config.resolved_target_log_score_gap,
    )
    allocations = {
        "coupled_novelty": coupled_novelty_design(rows, config.budget),
        "uniform_factorial": uniform_factorial_design(
            len(rows), config.budget, seed=config.seed
        ),
        "maximin_optimized": optimized.counts,
    }
    allocation_rows, diagnostic_rows, pairwise_rows = _design_tables(
        rows, standardized, allocations, config
    )
    recovery_rows: list[DesignRecoveryRow] = []
    for name in DESIGN_NAMES:
        recovery_rows.extend(
            recover_design(
                name,
                standardized,
                allocations[name],
                replicates=config.replicates_per_generator,
                test_fraction=config.test_fraction,
                effect_size=config.effect_size,
                noise_std=config.noise_std,
                seed=config.seed + 100_003,
            )
        )
    summary = _summary(config, grid_config, rows, diagnostic_rows, recovery_rows)
    grid_rows = _grid_rows(rows, raw, standardized)
    return DesignBenchmarkResult(
        allocations=tuple(allocation_rows),
        diagnostics=tuple(diagnostic_rows),
        pairwise_geometry=tuple(pairwise_rows),
        recovery=tuple(recovery_rows),
        optimization_trace=tuple(optimized.optimization_trace),
        grid=tuple(grid_rows),
        summary=summary,
    )


def _design_tables(
    rows: tuple[dict[str, Any], ...],
    standardized: NDArray[np.float64],
    allocations: dict[str, NDArray[np.int64]],
    config: DesignBenchmarkConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    allocation_rows: list[dict[str, Any]] = []
    diagnostics_rows: list[dict[str, Any]] = []
    pairwise_rows: list[dict[str, Any]] = []
    for name, counts in allocations.items():
        diagnostics = design_diagnostics(
            standardized,
            counts,
            effect_size=config.effect_size,
            noise_std=config.noise_std,
            target_log_score_gap=config.resolved_target_log_score_gap,
        )
        diagnostics_rows.append({"design": name, **diagnostics.as_dict()})
        geometry = pairwise_residual_matrix(standardized, counts)
        for generator_index, generator in enumerate(DESIGN_CANDIDATE_NAMES):
            for alternative_index, alternative in enumerate(DESIGN_CANDIDATE_NAMES):
                if generator_index == alternative_index:
                    continue
                residual = float(geometry[generator_index, alternative_index])
                pairwise_rows.append(
                    {
                        "design": name,
                        "generator": generator,
                        "alternative": alternative,
                        "residual_variance": residual,
                        "expected_profiled_log_score_gap_per_trial": (
                            profiled_gaussian_log_score_gap(
                                residual,
                                effect_size=config.effect_size,
                                noise_std=config.noise_std,
                            )
                        ),
                    }
                )
        for point_id, count in enumerate(counts):
            if count > 0:
                allocation_rows.append(
                    {"design": name, "count": int(count), **rows[point_id]}
                )
    return allocation_rows, diagnostics_rows, pairwise_rows


def _summary(
    config: DesignBenchmarkConfig,
    grid_config: TransitionDesignGridConfig,
    rows: tuple[dict[str, Any], ...],
    diagnostics: list[dict[str, Any]],
    recovery: list[DesignRecoveryRow],
) -> dict[str, Any]:
    diagnostics_by_name = {str(row["design"]): row for row in diagnostics}
    recovery_by_name = {
        name: [row for row in recovery if row.design == name] for name in DESIGN_NAMES
    }
    minimum = {
        name: float(min(row.recovery_rate for row in recovery_by_name[name]))
        for name in DESIGN_NAMES
    }
    mean = {
        name: float(np.mean([row.recovery_rate for row in recovery_by_name[name]]))
        for name in DESIGN_NAMES
    }
    optimized_residual = float(
        diagnostics_by_name["maximin_optimized"]["minimum_pairwise_residual_variance"]
    )
    novelty_residual = float(
        diagnostics_by_name["coupled_novelty"]["minimum_pairwise_residual_variance"]
    )
    uniform_residual = float(
        diagnostics_by_name["uniform_factorial"]["minimum_pairwise_residual_variance"]
    )
    return {
        "experiment": "prospective_maximin_trial_design",
        "config": {
            **{
                key: value
                for key, value in asdict(config).items()
                if key not in {"target_log_bf", "target_log_score_gap"}
            },
            "target_log_score_gap": config.resolved_target_log_score_gap,
        },
        "grid_config": asdict(grid_config),
        "grid_point_count": len(rows),
        "candidate_names": list(DESIGN_CANDIDATE_NAMES),
        "designs": diagnostics,
        "minimum_recovery_rate": minimum,
        "mean_recovery_rate": mean,
        "optimized_over_novelty_residual_ratio": optimized_residual
        / max(novelty_residual, 1e-15),
        "optimized_over_uniform_residual_ratio": optimized_residual
        / max(uniform_residual, 1e-15),
        "optimized_improves_worst_case_recovery_over_novelty": (
            minimum["maximin_optimized"] > minimum["coupled_novelty"]
        ),
        "scope": (
            "The finite optimizer conditions on a Gaussian candidate-amplitude/noise "
            "model. Welfare costs, carry-over, hardware constraints, and biological "
            "multiplexing must be encoded explicitly before a protocol is frozen."
        ),
    }


def _grid_rows(
    rows: tuple[dict[str, Any], ...],
    raw: NDArray[np.float64],
    standardized: NDArray[np.float64],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for point_index, row in enumerate(rows):
        enriched = dict(row)
        for candidate_index, name in enumerate(DESIGN_CANDIDATE_NAMES):
            enriched[f"raw_{name}"] = float(raw[point_index, candidate_index])
            enriched[f"standardized_{name}"] = float(
                standardized[point_index, candidate_index]
            )
        result.append(enriched)
    return result
