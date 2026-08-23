"""Command-line interface for exact filtering, smoothing, and replay recovery."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from bayesian_ach.io import write_json, write_rows_csv
from bayesian_ach.replay_benchmark import ReplayBenchmarkConfig, run_replay_benchmark


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bayesian-ach-replay",
        description="Run exact filtering-to-smoothing and posterior-replay recovery.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sequences", type=int, default=96)
    parser.add_argument("--sequence-length", type=int, default=64)
    parser.add_argument("--states", type=int, default=6)
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--effect-size", type=float, default=0.75)
    parser.add_argument("--noise-std", type=float, default=0.50)
    parser.add_argument("--replay-samples", type=int, default=128)
    parser.add_argument("--seed", type=int, default=7)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = run_replay_benchmark(
        ReplayBenchmarkConfig(
            n_sequences=args.sequences,
            sequence_length=args.sequence_length,
            n_states=args.states,
            train_fraction=args.train_fraction,
            effect_size=args.effect_size,
            noise_std=args.noise_std,
            replay_samples=args.replay_samples,
            seed=args.seed,
        )
    )
    args.output.mkdir(parents=True, exist_ok=True)
    write_rows_csv(
        args.output / "replay_generators.csv",
        [generator.as_dict() for generator in result.generators],
    )
    write_rows_csv(
        args.output / "replay_sequences.csv",
        [sequence.as_dict() for sequence in result.sequences],
    )
    write_rows_csv(args.output / "replay_fits.csv", result.fits)
    write_rows_csv(args.output / "replay_trials.csv", result.trials)
    write_json(args.output / "summary.json", result.summary)
    print(
        f"Wrote replay and smoothing evidence to {args.output}; "
        f"recovered {result.summary['recovery_count']}/"
        f"{result.summary['candidate_count']} generating signals"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
