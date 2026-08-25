#!/usr/bin/env python3
"""Read-only stratification and mixture geometry audit for the frozen N60 stress."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import subprocess
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np

from bayesian_ach.design_grid import DESIGN_CANDIDATE_NAMES, generate_transition_design_grid
from bayesian_ach.design_stress import (
    DesignStressConfig,
    _rng,
    _Scores,
    _simulate_scores,
    _Thresholds,
)

_SHA40 = re.compile(r"^[0-9a-f]{40}$")


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_provenance(repo_root: Path, expected_sha: str) -> None:
    if _SHA40.fullmatch(expected_sha) is None:
        raise ValueError("--code-sha must be a lowercase 40-character commit SHA")
    head = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "-C", str(repo_root), "status", "--porcelain", "--untracked-files=all"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if head != expected_sha or status:
        raise RuntimeError("geometry diagnostic requires the clean exact --code-sha checkout")


def _reason_counts(rows: list[dict[str, str]]) -> dict[str, int]:
    total: Counter[str] = Counter()
    for row in rows:
        total.update(json.loads(row["abstention_reasons"]))
    return dict(sorted(total.items()))


def _weighted_residual(
    response: np.ndarray,
    predictors: np.ndarray,
    weights: np.ndarray,
) -> float:
    design = np.column_stack((np.ones(response.size), predictors))
    root = np.sqrt(weights)
    coefficients, _, _, _ = np.linalg.lstsq(
        design * root[:, None],
        response * root,
        rcond=None,
    )
    residual = response - design @ coefficients
    return float(np.sum(weights * residual**2))


def _gate_counts(
    scores: _Scores,
    thresholds: _Thresholds,
) -> tuple[bool, bool, bool, bool, str]:
    pure_over_null = scores.pure_over_null > thresholds.pure_over_null
    winner_over_runner = scores.winner_over_runner > thresholds.winner_over_runner
    flexible_adequacy = scores.flexible_over_pure <= thresholds.flexible_over_pure
    all_pass = pure_over_null and winner_over_runner and flexible_adequacy
    if not pure_over_null:
        reason = "null_not_rejected"
    elif not winner_over_runner:
        reason = "pure_ambiguity"
    elif not flexible_adequacy:
        reason = "flexible_model_better"
    else:
        reason = "pure_call"
    return pure_over_null, winner_over_runner, flexible_adequacy, all_pass, reason


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--repo-root", default=Path.cwd(), type=Path)
    parser.add_argument("--code-sha", required=True)
    parser.add_argument("--recompute-only", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.recompute_only:
        if _SHA40.fullmatch(args.code_sha) is None:
            raise ValueError("--code-sha must be a lowercase 40-character commit SHA")
    else:
        _git_provenance(args.repo_root.resolve(), args.code_sha)
    artifact = args.artifact.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)

    pure = _rows(artifact / "pure_recovery.csv")
    mixtures = _rows(artifact / "mixture_evaluation.csv")
    null = _rows(artifact / "null_evaluation.csv")
    out_of_span = _rows(artifact / "out_of_span_evaluation.csv")
    threshold_rows = _rows(artifact / "thresholds.csv")
    allocation_rows = _rows(artifact / "allocations.csv")
    designs = ["coupled_novelty", "uniform_factorial", "maximin_optimized"]
    stratified: list[dict[str, Any]] = []
    for design in designs:
        design_pure = [row for row in pure if row["design"] == design]
        design_mix = [row for row in mixtures if row["design"] == design]
        design_null = next(row for row in null if row["design"] == design)
        design_oos = next(row for row in out_of_span if row["design"] == design)
        threshold = next(row for row in threshold_rows if row["design"] == design)
        weakest = min(design_pure, key=lambda row: float(row["wilson_lower"]))
        worst_mix = max(design_mix, key=lambda row: float(row["wilson_upper"]))
        stratified.append(
            {
                "design": design,
                "weakest_pure_generator": weakest["generator"],
                "weakest_pure_correct_rate": float(weakest["correct_call_rate"]),
                "weakest_pure_wilson_lower": float(weakest["wilson_lower"]),
                "weakest_pure_raw_closed_set_rate": float(
                    weakest["raw_closed_set_winner_rate"]
                ),
                "pure_reason_counts": _reason_counts(design_pure),
                "worst_mixture_pair": [
                    worst_mix["first_candidate"],
                    worst_mix["second_candidate"],
                ],
                "worst_mixture_false_pure_rate": float(
                    worst_mix["false_pure_call_rate"]
                ),
                "worst_mixture_wilson_upper": float(worst_mix["wilson_upper"]),
                "mixture_reason_counts": _reason_counts(design_mix),
                "null_false_pure_rate": float(design_null["false_pure_call_rate"]),
                "null_wilson_upper": float(design_null["wilson_upper"]),
                "null_reason_counts": json.loads(design_null["abstention_reasons"]),
                "out_of_span_false_pure_rate": float(
                    design_oos["false_pure_call_rate"]
                ),
                "out_of_span_wilson_upper": float(design_oos["wilson_upper"]),
                "out_of_span_reason_counts": json.loads(
                    design_oos["abstention_reasons"]
                ),
                "thresholds": {
                    "pure_over_null": float(threshold["pure_over_null_threshold"]),
                    "winner_over_runner": float(
                        threshold["winner_over_runner_threshold"]
                    ),
                    "flexible_over_pure": float(
                        threshold["flexible_over_pure_threshold"]
                    ),
                },
            }
        )

    _, _, signals = generate_transition_design_grid()
    counts = np.zeros(signals.shape[0], dtype=np.int64)
    for row in allocation_rows:
        if row["design"] == "maximin_optimized":
            counts[int(row["point_id"])] = int(row["count"])
    weights = counts.astype(float) / float(np.sum(counts))
    selected = counts > 0
    trial_indices = np.repeat(np.arange(counts.size), counts)
    trial_signals = signals[trial_indices]
    threshold_row = next(
        row for row in threshold_rows if row["design"] == "maximin_optimized"
    )
    thresholds = _Thresholds(
        pure_over_null=float(threshold_row["pure_over_null_threshold"]),
        winner_over_runner=float(threshold_row["winner_over_runner_threshold"]),
        flexible_over_pure=float(threshold_row["flexible_over_pure_threshold"]),
    )
    config = DesignStressConfig()
    mixture_by_pair = {
        (row["first_candidate"], row["second_candidate"]): row
        for row in mixtures
        if row["design"] == "maximin_optimized"
    }
    geometry: list[dict[str, Any]] = []
    n_test = int(round(config.test_fraction * 60))
    for first, second in combinations(range(signals.shape[1]), 2):
        first_name = DESIGN_CANDIDATE_NAMES[first]
        second_name = DESIGN_CANDIDATE_NAMES[second]
        mixture_grid = 0.5 * (signals[:, first] + signals[:, second])
        mixture_grid /= float(np.std(mixture_grid))
        constituent_residuals = {
            index: _weighted_residual(
                mixture_grid,
                signals[:, index : index + 1],
                weights,
            )
            for index in (first, second)
        }
        all_residuals = [
            _weighted_residual(
                mixture_grid,
                signals[:, index : index + 1],
                weights,
            )
            for index in range(signals.shape[1])
        ]
        pair_residual = _weighted_residual(
            mixture_grid,
            signals[:, [first, second]],
            weights,
        )
        pair_values = signals[selected][:, [first, second]]
        pair_weights = weights[selected]
        means = np.sum(pair_values * pair_weights[:, None], axis=0)
        centered = pair_values - means
        covariance = (centered * pair_weights[:, None]).T @ centered
        scales = np.sqrt(np.diag(covariance))
        correlation = float(covariance[0, 1] / (scales[0] * scales[1]))
        condition = float(np.linalg.cond(covariance))

        gate_totals: Counter[str] = Counter()
        reason_totals: Counter[str] = Counter()
        mixture = mixture_grid[trial_indices]
        for replicate in range(config.evaluation_replicates):
            scores = _simulate_scores(
                trial_signals,
                mixture,
                config=config,
                rng=_rng(
                    config.evaluation_seed,
                    2,
                    60,
                    2,
                    first,
                    second,
                    replicate,
                ),
            )
            null_pass, ambiguity_pass, adequacy_pass, all_pass, reason = _gate_counts(
                scores,
                thresholds,
            )
            gate_totals.update(
                {
                    "pure_over_null_pass": int(null_pass),
                    "winner_over_runner_pass": int(ambiguity_pass),
                    "flexible_adequacy_pass": int(adequacy_pass),
                    "all_three_pass": int(all_pass),
                }
            )
            reason_totals[reason] += 1
        frozen = mixture_by_pair[(first_name, second_name)]
        if gate_totals["all_three_pass"] != int(frozen["false_pure_calls"]):
            raise RuntimeError("diagnostic replay does not reproduce frozen false-call count")
        best_constituent = min(constituent_residuals, key=constituent_residuals.get)
        best_any = int(np.argmin(all_residuals))
        geometry.append(
            {
                "first_candidate": first_name,
                "second_candidate": second_name,
                "false_pure_call_rate": float(frozen["false_pure_call_rate"]),
                "wilson_upper": float(frozen["wilson_upper"]),
                "sequential_reason_counts": dict(sorted(reason_totals.items())),
                "independent_gate_pass_counts": dict(sorted(gate_totals.items())),
                "best_constituent": DESIGN_CANDIDATE_NAMES[best_constituent],
                "best_constituent_residual": constituent_residuals[best_constituent],
                "best_any_pure": DESIGN_CANDIDATE_NAMES[best_any],
                "best_any_pure_residual": all_residuals[best_any],
                "oracle_two_component_residual": pair_residual,
                "best_pure_profiled_gap_n_test_21": (
                    0.5 * n_test * math.log1p(all_residuals[best_any])
                ),
                "oracle_two_component_profiled_gap_n_test_21": (
                    0.5 * n_test * math.log1p(max(0.0, pair_residual))
                ),
                "weighted_candidate_correlation": correlation,
                "weighted_pair_covariance_condition_number": condition,
            }
        )

    report = {
        "schema_version": 1,
        "kind": "read_only_post_freeze_stress_diagnostic",
        "diagnostic_producer_commit": args.code_sha,
        "diagnostic_producer_clean_worktree": True,
        "diagnostic_script_sha256": _sha256(Path(__file__).resolve()),
        "source_artifact_sha256sums_sha256": _sha256(artifact / "SHA256SUMS.csv"),
        "source_producer_commit": json.loads(
            (artifact / "artifact_manifest.json").read_text(encoding="utf-8")
        )["producer_commit"],
        "no_threshold_or_evaluation_changes": True,
        "evaluation_replicates": config.evaluation_replicates,
        "calibration_replicates": config.calibration_replicates,
        "calibration_audit_replicates": config.calibration_audit_replicates,
        "threshold_seed": config.threshold_seed,
        "calibration_audit_seed": config.calibration_audit_seed,
        "evaluation_seed": config.evaluation_seed,
        "test_fraction": config.test_fraction,
        "n_test": n_test,
        "stratified": stratified,
        "maximin_mixture_geometry": geometry,
    }
    (output / "stress_diagnostic.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    with (output / "maximin_mixture_geometry.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        flat_rows = []
        for row in geometry:
            flat = dict(row)
            flat["sequential_reason_counts"] = json.dumps(
                flat["sequential_reason_counts"], sort_keys=True, separators=(",", ":")
            )
            flat["independent_gate_pass_counts"] = json.dumps(
                flat["independent_gate_pass_counts"], sort_keys=True, separators=(",", ":")
            )
            flat_rows.append(flat)
        writer = csv.DictWriter(handle, fieldnames=list(flat_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(flat_rows)
    payload_names = ("stress_diagnostic.json", "maximin_mixture_geometry.csv")
    manifest = {
        "schema_version": 1,
        "artifact": "post_freeze_stress_geometry_diagnostic",
        "producer_commit": args.code_sha,
        "producer_git_dirty": False,
        "diagnostic_script_sha256": report["diagnostic_script_sha256"],
        "source_artifact_sha256sums_sha256": report[
            "source_artifact_sha256sums_sha256"
        ],
        "files": [
            {
                "path": name,
                "bytes": (output / name).stat().st_size,
                "sha256": _sha256(output / name),
            }
            for name in payload_names
        ],
    }
    (output / "artifact_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    checksum_names = (*payload_names, "artifact_manifest.json")
    with (output / "SHA256SUMS.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["file", "bytes", "sha256"],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(
            {
                "file": name,
                "bytes": (output / name).stat().st_size,
                "sha256": _sha256(output / name),
            }
            for name in checksum_names
        )
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
