"""Command-line evidence generation for Bayesian-ACh."""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from bayesian_ach.io import write_json, write_rows_csv
from bayesian_ach.model_recovery import fit_candidate_models, generate_synthetic_ach
from bayesian_ach.regime import RegimeRecoveryConfig, run_regime_recovery
from bayesian_ach.signals import CANDIDATE_SIGNAL_NAMES
from bayesian_ach.simulation import (
    FactorialDesignConfig,
    MatchedConfidenceConfig,
    simulate_factorial_design,
    simulate_matched_confidence,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bayesian-ach",
        description="Generate falsifiable Bayesian-ACh synthetic evidence.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    dissociate = subparsers.add_parser(
        "dissociate",
        help="run the paired matched-confidence experiment",
    )
    dissociate.add_argument("--output", type=Path, required=True)
    dissociate.add_argument("--pairs", type=int, default=512)
    dissociate.add_argument("--states", type=int, default=4)
    dissociate.add_argument("--low-concentration", type=float, default=4.0)
    dissociate.add_argument("--high-concentration", type=float, default=128.0)
    dissociate.add_argument("--seed", type=int, default=7)

    benchmark = subparsers.add_parser(
        "benchmark",
        help="run factorial simulation and held-out model recovery",
    )
    benchmark.add_argument("--output", type=Path, required=True)
    benchmark.add_argument("--trials", type=int, default=4096)
    benchmark.add_argument("--states", type=int, default=5)
    benchmark.add_argument("--noise-std", type=float, default=0.25)
    benchmark.add_argument("--seed", type=int, default=7)

    regime = subparsers.add_parser(
        "regime-benchmark",
        help="recover known context switches versus structural resets",
    )
    regime.add_argument("--output", type=Path, required=True)
    regime.add_argument("--sequences-per-class", type=int, default=48)
    regime.add_argument("--pre-length", type=int, default=64)
    regime.add_argument("--post-length", type=int, default=96)
    regime.add_argument("--states", type=int, default=4)
    regime.add_argument("--known-concentration", type=float, default=256.0)
    regime.add_argument("--novel-similarity", type=float, default=0.0)
    regime.add_argument("--reset-concentration", type=float, default=1.0)
    regime.add_argument("--context-switch-probability", type=float, default=0.02)
    regime.add_argument("--change-hazard", type=float, default=0.02)
    regime.add_argument("--seed", type=int, default=7)

    return parser


def _mean_by_condition(rows: Sequence[dict[str, Any]], field: str) -> dict[str, float]:
    values: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        values[str(row["condition"])].append(float(row[field]))
    return {condition: float(np.mean(items)) for condition, items in sorted(values.items())}


def _run_dissociate(args: argparse.Namespace) -> int:
    config = MatchedConfidenceConfig(
        n_pairs=args.pairs,
        n_states=args.states,
        low_concentration=args.low_concentration,
        high_concentration=args.high_concentration,
        seed=args.seed,
    )
    rows = simulate_matched_confidence(config)
    args.output.mkdir(parents=True, exist_ok=True)
    write_rows_csv(args.output / "matched_confidence.csv", rows)

    max_pair_mismatch: dict[str, float] = {}
    for field in ("predictive_probability", "innovation_l2", "surprise"):
        mismatches = []
        for pair_id in range(config.n_pairs):
            low, high = rows[2 * pair_id], rows[2 * pair_id + 1]
            mismatches.append(abs(float(low[field]) - float(high[field])))
        max_pair_mismatch[field] = max(mismatches, default=0.0)

    summary = {
        "experiment": "matched_confidence",
        "config": {
            "n_pairs": config.n_pairs,
            "n_states": config.n_states,
            "low_concentration": config.low_concentration,
            "high_concentration": config.high_concentration,
            "seed": config.seed,
        },
        "condition_means": {
            field: _mean_by_condition(rows, field)
            for field in CANDIDATE_SIGNAL_NAMES
        },
        "max_paired_mismatch_for_matched_quantities": max_pair_mismatch,
        "interpretation": (
            "Raw prediction and observation quantities are pair-matched; Bayesian gain and "
            "posterior-update quantities differ because concentration differs."
        ),
    }
    write_json(args.output / "summary.json", summary)
    print(f"Wrote matched-confidence evidence to {args.output}")
    return 0


def _run_benchmark(args: argparse.Namespace) -> int:
    config = FactorialDesignConfig(
        n_trials=args.trials,
        n_states=args.states,
        seed=args.seed,
    )
    rows = simulate_factorial_design(config)
    args.output.mkdir(parents=True, exist_ok=True)
    write_rows_csv(args.output / "trials.csv", rows)

    recovery_rows: list[dict[str, Any]] = []
    winners: dict[str, str] = {}
    for generator_index, generator in enumerate(CANDIDATE_SIGNAL_NAMES):
        ach = generate_synthetic_ach(
            rows,
            generator,
            noise_std=args.noise_std,
            seed=args.seed + 1009 * (generator_index + 1),
        )
        fits = fit_candidate_models(
            rows,
            ach,
            seed=args.seed + 2027 * (generator_index + 1),
        )
        winners[generator] = fits[0].candidate
        for rank, fit in enumerate(fits, start=1):
            result = {"generator": generator, "rank": rank}
            result.update(fit.as_dict())
            recovery_rows.append(result)

    write_rows_csv(args.output / "model_recovery.csv", recovery_rows)
    recovery_count = sum(generator == winner for generator, winner in winners.items())
    summary = {
        "experiment": "factorial_model_recovery",
        "config": {
            "n_trials": config.n_trials,
            "n_states": config.n_states,
            "noise_std": args.noise_std,
            "seed": config.seed,
        },
        "winners": winners,
        "self_recovery_count": recovery_count,
        "candidate_count": len(CANDIDATE_SIGNAL_NAMES),
        "all_generators_recovered": recovery_count == len(CANDIDATE_SIGNAL_NAMES),
    }
    write_json(args.output / "summary.json", summary)
    print(
        f"Wrote model-recovery evidence to {args.output}; "
        f"self-recovered {recovery_count}/{len(CANDIDATE_SIGNAL_NAMES)} generators"
    )
    return 0


def _run_regime_benchmark(args: argparse.Namespace) -> int:
    config = RegimeRecoveryConfig(
        n_sequences_per_class=args.sequences_per_class,
        pre_length=args.pre_length,
        post_length=args.post_length,
        n_states=args.states,
        known_concentration=args.known_concentration,
        novel_similarity=args.novel_similarity,
        reset_concentration=args.reset_concentration,
        context_switch_probability=args.context_switch_probability,
        change_hazard=args.change_hazard,
        seed=args.seed,
    )
    result = run_regime_recovery(config)
    args.output.mkdir(parents=True, exist_ok=True)
    write_rows_csv(
        args.output / "regime_sequences.csv",
        [sequence.as_dict() for sequence in result.sequences],
    )
    write_rows_csv(args.output / "regime_trials.csv", result.trials)
    write_json(args.output / "summary.json", result.summary)
    print(
        f"Wrote regime-recovery evidence to {args.output}; "
        f"balanced accuracy {result.summary['balanced_accuracy']:.3f}"
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "dissociate":
        return _run_dissociate(args)
    if args.command == "benchmark":
        return _run_benchmark(args)
    if args.command == "regime-benchmark":
        return _run_regime_benchmark(args)
    raise AssertionError(f"unhandled command {args.command!r}")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
