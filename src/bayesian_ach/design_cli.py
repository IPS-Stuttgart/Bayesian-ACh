"""Command-line interface for prospective Bayesian-ACh trial design."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from bayesian_ach.design_benchmark import DesignBenchmarkConfig, run_design_benchmark
from bayesian_ach.io import write_json, write_rows_csv


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bayesian-ach-design",
        description="Optimize and validate a finite Bayesian-ACh trial allocation.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--budget", type=int, default=60)
    parser.add_argument("--replicates", type=int, default=200)
    parser.add_argument("--effect-size", type=float, default=1.0)
    parser.add_argument("--noise-std", type=float, default=1.0)
    parser.add_argument("--target-log-bf", type=float, default=5.0)
    parser.add_argument("--max-point-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=7)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = run_design_benchmark(
        DesignBenchmarkConfig(
            budget=args.budget,
            replicates_per_generator=args.replicates,
            effect_size=args.effect_size,
            noise_std=args.noise_std,
            target_log_bf=args.target_log_bf,
            max_point_fraction=args.max_point_fraction,
            seed=args.seed,
        )
    )
    args.output.mkdir(parents=True, exist_ok=True)
    write_rows_csv(args.output / "design_allocation.csv", result.allocations)
    write_rows_csv(args.output / "design_diagnostics.csv", result.diagnostics)
    write_rows_csv(args.output / "design_pairwise_geometry.csv", result.pairwise_geometry)
    write_rows_csv(
        args.output / "design_recovery.csv",
        [row.as_dict() for row in result.recovery],
    )
    write_rows_csv(args.output / "design_optimization_trace.csv", result.optimization_trace)
    write_rows_csv(args.output / "design_grid.csv", result.grid)
    write_json(args.output / "summary.json", result.summary)
    print(f"Wrote prospective design evidence to {args.output}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
