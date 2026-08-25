"""Command-line export for certified finite-grid maximin allocations."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import scipy

from bayesian_ach.design_certificate import (
    CertifiedDesignConfig,
    certificate_matches_geometry,
    certify_maximin_design,
)
from bayesian_ach.design_grid import DESIGN_CANDIDATE_NAMES, generate_transition_design_grid


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty table: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _manifest(output: Path, filenames: Sequence[str]) -> None:
    rows = []
    for filename in sorted(filenames):
        path = output / filename
        rows.append(
            {
                "file": filename,
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    _write_csv(output / "SHA256SUMS.csv", rows)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--code-sha", required=True)
    parser.add_argument("--budget", type=int, default=60)
    parser.add_argument("--mode", choices=("integer", "continuous"), default="integer")
    parser.add_argument("--max-point-fraction", type=float, default=0.15)
    parser.add_argument("--absolute-gap-tolerance", type=float, default=1.0e-8)
    parser.add_argument("--relative-gap-tolerance", type=float, default=1.0e-7)
    parser.add_argument("--cut-violation-tolerance", type=float, default=1.0e-9)
    parser.add_argument("--max-iterations", type=int, default=100)
    parser.add_argument("--master-time-limit", type=float, default=120.0)
    parser.add_argument("--master-mip-relative-gap", type=float, default=1.0e-9)
    parser.add_argument("--require-certificate", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the certificate and write a checksum-locked result package."""

    args = _parser().parse_args(argv)
    config = CertifiedDesignConfig(
        budget=args.budget,
        max_point_fraction=args.max_point_fraction,
        integer=args.mode == "integer",
        absolute_gap_tolerance=args.absolute_gap_tolerance,
        relative_gap_tolerance=args.relative_gap_tolerance,
        cut_violation_tolerance=args.cut_violation_tolerance,
        max_iterations=args.max_iterations,
        master_time_limit_s=args.master_time_limit,
        master_mip_relative_gap=args.master_mip_relative_gap,
    )
    rows, _, standardized = generate_transition_design_grid()
    result = certify_maximin_design(standardized, config)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)

    allocation_rows: list[dict[str, Any]] = []
    for point_index, mass in enumerate(result.allocation):
        if mass <= 1.0e-12:
            continue
        allocation_rows.append(
            {
                **rows[point_index],
                "allocation": int(round(float(mass)))
                if result.integer
                else float(mass),
            }
        )
    trace_rows = [dict(row) for row in result.trace]
    summary = {
        "schema_version": 1,
        "experiment": "certified_finite_grid_maximin_allocation",
        "code_repository": "IPS-Stuttgart/Bayesian-ACh",
        "code_sha": args.code_sha,
        "scipy_version": scipy.__version__,
        "solver": "scipy.optimize.milp (HiGHS) with OLS cutting planes",
        "candidate_names": list(DESIGN_CANDIDATE_NAMES),
        "grid_point_count": len(rows),
        "config": asdict(config),
        "certified": result.certified,
        "lower_bound": result.lower_bound,
        "upper_bound": result.upper_bound,
        "absolute_gap": result.absolute_gap,
        "relative_gap": result.relative_gap,
        "heuristic_lower_bound": result.heuristic_lower_bound,
        "certified_improvement_over_heuristic": (
            result.lower_bound - result.heuristic_lower_bound
        ),
        "iterations": result.iterations,
        "cut_count": result.cut_count,
        "last_master_status": result.last_master_status,
        "last_master_message": result.last_master_message,
        "direct_geometry_matches_lower_bound": certificate_matches_geometry(
            standardized,
            result,
        ),
        "scope": (
            "This certificate is conditional on the independently instantiated finite "
            "grid, global candidate standardization, budget, and per-cell cap. The "
            "allocation is a count target, not a validated sequential protocol; no "
            "history constructor, washout, reset, or carry-over feasibility is implied."
        ),
    }
    _write_json(output / "certificate_summary.json", summary)
    _write_csv(output / "certified_allocation.csv", allocation_rows)
    _write_csv(output / "cut_trace.csv", trace_rows)
    _manifest(
        output,
        (
            "certificate_summary.json",
            "certified_allocation.csv",
            "cut_trace.csv",
        ),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.require_certificate and not result.certified:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
