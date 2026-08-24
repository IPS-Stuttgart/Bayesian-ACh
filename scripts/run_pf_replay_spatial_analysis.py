#!/usr/bin/env python3
"""Freeze the prespecified PF replay spatial-revision comparison."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
from typing import Any, Sequence

import numpy as np

from bayesian_ach.replay_artifact import load_spatial_predictor_artifact
from bayesian_ach.replay_recovery import (
    SpatialInjectionRecoveryConfig,
    run_spatial_recovery_checks,
)
from bayesian_ach.replay_spatial import (
    SpatialComparisonConfig,
    compare_spatial_replay_candidates,
)


ANALYSIS_SCHEMA = "bayesian-ach.pf-replay-spatial-analysis.v1"
FOLD_OUTPUT = "pf_replay_spatial_candidate_folds.csv"
CONTRAST_OUTPUT = "pf_replay_spatial_target_contrasts.csv"
RAT_SCORE_OUTPUT = "pf_replay_spatial_rat_scores.csv"
RECOVERY_OUTPUT = "pf_replay_spatial_recovery.csv"
EXCLUSION_OUTPUT = "pf_replay_spatial_recovery_exclusions.csv"
GATE_OUTPUT = "pf_replay_spatial_gates.csv"
REPORT_OUTPUT = "pf_replay_spatial_report.md"
MANIFEST_OUTPUT = "pf_replay_spatial_analysis_manifest.json"
_COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")
ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _clean_commit() -> str:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=normal"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if _COMMIT_PATTERN.fullmatch(commit) is None:
        raise ValueError("analysis must run from a committed Git checkout")
    if dirty.strip():
        raise ValueError("analysis must run from a clean committed worktree")
    return commit


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot freeze empty table: {path.name}")
    fieldnames = list(rows[0])
    if any(list(row) != fieldnames for row in rows):
        raise ValueError(f"rows for {path.name} have inconsistent columns")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def run_analysis(
    predictor_directory: str | Path,
    output_directory: str | Path,
    *,
    comparison_config: SpatialComparisonConfig | None = None,
    injection_config: SpatialInjectionRecoveryConfig | None = None,
) -> dict[str, Path]:
    """Run recovery and real scoring, then freeze compact hash-bound evidence."""

    consumer_commit = _clean_commit()
    comparison_config = (
        SpatialComparisonConfig(
            minimum_rats=5,
            bootstrap_replicates=5000,
            simultaneous_confidence_level=0.95,
            seed=7,
        )
        if comparison_config is None
        else comparison_config
    )
    injection_config = (
        SpatialInjectionRecoveryConfig(
            injection_temperature=4.0,
            spatial_sigma_multipliers=(0.5, 1.0, 2.0),
            coordinate_units="cm",
            emission_noise_sd_nats=0.02,
            seed=701,
        )
        if injection_config is None
        else injection_config
    )
    comparison_config.validate()
    injection_config.validate()

    predictor_source = Path(predictor_directory)
    frozen = load_spatial_predictor_artifact(predictor_source)
    recovery = run_spatial_recovery_checks(
        frozen.dataset,
        comparison_config,
        injection_config,
    )
    comparison = compare_spatial_replay_candidates(
        frozen.dataset,
        comparison_config,
        recovery_gate=recovery,
    )

    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    fold_rows = [asdict(record) for record in comparison.folds]
    contrast_rows = [asdict(record) for record in comparison.target_contrasts]
    rat_score_rows = [
        {
            "rat": rat,
            "candidate": candidate,
            "mean_log_score_per_bin": float(comparison.rat_scores[rat_index, candidate_index]),
        }
        for rat_index, rat in enumerate(comparison.rat_ids)
        for candidate_index, candidate in enumerate(comparison.candidate_names)
    ]
    recovery_rows = [
        {"recovery_kind": kind, **asdict(record)}
        for kind, records in (
            ("pure", recovery.pure_records),
            ("mixture", recovery.mixture_records),
            ("null_negative_control", recovery.null_records),
        )
        for record in records
    ]
    exclusion_rows = [
        {"event_id": event_id, "reason": "not_in_all_candidate_complete_case_cohort"}
        for event_id in recovery.excluded_event_ids
    ]
    if not exclusion_rows:
        exclusion_rows = [{"event_id": "", "reason": "none"}]

    target_lower = np.asarray(
        [record.simultaneous_lower_bound for record in comparison.target_contrasts],
        dtype=float,
    )
    target_p = np.asarray(
        [record.exact_one_sided_sign_flip_p for record in comparison.target_contrasts],
        dtype=float,
    )
    alpha = 1.0 - comparison_config.simultaneous_confidence_level
    gate_rows = [
        {
            "gate": "minimum_independent_rats",
            "passed": len(comparison.rat_ids) >= comparison_config.minimum_rats,
            "value": len(comparison.rat_ids),
            "required": comparison_config.minimum_rats,
        },
        {
            "gate": "exact_animal_sign_flip_resolution",
            "passed": bool(np.all(np.isfinite(target_p)) and np.all(target_p <= alpha + 1e-15)),
            "value": float(np.max(target_p)) if target_p.size else float("nan"),
            "required": f"all <= {alpha}",
        },
        {
            "gate": "simultaneous_smoothing_contrasts",
            "passed": bool(np.all(np.isfinite(target_lower)) and np.all(target_lower > 0.0)),
            "value": float(np.min(target_lower)) if target_lower.size else float("nan"),
            "required": "all > 0",
        },
        {
            "gate": "post_decoder_recovery",
            "passed": recovery.passed,
            "value": len(recovery.pure_records) + len(recovery.mixture_records) + len(recovery.null_records),
            "required": "all pure pass; mixtures and null abstain",
        },
        {
            "gate": "candidate_field_collinearity",
            "passed": comparison.maximum_field_correlation
            <= comparison_config.maximum_field_correlation,
            "value": comparison.maximum_field_correlation,
            "required": comparison_config.maximum_field_correlation,
        },
        {
            "gate": "overall_identification",
            "passed": comparison.status == "identified",
            "value": comparison.status,
            "required": "identified",
        },
    ]

    paths = {
        FOLD_OUTPUT: output / FOLD_OUTPUT,
        CONTRAST_OUTPUT: output / CONTRAST_OUTPUT,
        RAT_SCORE_OUTPUT: output / RAT_SCORE_OUTPUT,
        RECOVERY_OUTPUT: output / RECOVERY_OUTPUT,
        EXCLUSION_OUTPUT: output / EXCLUSION_OUTPUT,
        GATE_OUTPUT: output / GATE_OUTPUT,
    }
    for path, rows in (
        (paths[FOLD_OUTPUT], fold_rows),
        (paths[CONTRAST_OUTPUT], contrast_rows),
        (paths[RAT_SCORE_OUTPUT], rat_score_rows),
        (paths[RECOVERY_OUTPUT], recovery_rows),
        (paths[EXCLUSION_OUTPUT], exclusion_rows),
        (paths[GATE_OUTPUT], gate_rows),
    ):
        _write_rows(path, rows)

    report = [
        "# PF replay spatial revision analysis",
        "",
        f"Status: **{comparison.status}**.",
        (
            "This is a conditional replay-content comparison, not a causal "
            "event-incidence analysis."
        ),
        (
            "Recovery is post-decoder Gaussian emission-score recovery at the "
            "empirical RUN point spread, not end-to-end spike-decoder recovery."
        ),
        "",
        f"- Predictor events: {frozen.dataset.n_events}",
        f"- Complete-case recovery events: {recovery.common_event_count}",
        f"- Excluded from recovery: {len(recovery.excluded_event_ids)}",
        f"- Independent rats scored: {len(comparison.rat_ids)}",
        f"- Descriptive winner: {comparison.winner}",
        f"- Recovery gate: {'PASS' if recovery.passed else 'FAIL'}",
        f"- Maximum field correlation: {comparison.maximum_field_correlation:.6g}",
        (
            "- Abstention reasons: "
            + (", ".join(comparison.abstention_reasons) or "none")
        ),
        "",
        "Prespecified smoothing-revision contrasts:",
    ]
    report.extend(
        (
            f"- versus {record.alternative}: margin {record.mean_margin:+.6g}, "
            f"simultaneous lower {record.simultaneous_lower_bound:+.6g}, "
            f"exact one-sided sign-flip p={record.exact_one_sided_sign_flip_p:.6g}"
        )
        for record in comparison.target_contrasts
    )
    report_path = output / REPORT_OUTPUT
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    paths[REPORT_OUTPUT] = report_path

    output_sha256 = {name: _sha256(path) for name, path in paths.items()}
    input_manifest_path = predictor_source / "replay_spatial_manifest.json"
    analysis_manifest = {
        "schema_version": ANALYSIS_SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "consumer_repository": "IPS-Stuttgart/Bayesian-ACh",
        "consumer_commit": consumer_commit,
        "consumer_clean_worktree": True,
        "predictor_sha256": frozen.predictor_sha256,
        "predictor_manifest_sha256": _sha256(input_manifest_path),
        "predictor_producer_commit": frozen.manifest.producer_commit,
        "dataset_sha256": frozen.manifest.dataset_sha256,
        "dataset_verifier_report_sha256": (
            frozen.manifest.dataset_verifier_report_sha256
        ),
        "route_manifest_file_sha256": frozen.manifest.route_manifest_file_sha256,
        "source_event_count": frozen.dataset.n_events,
        "common_event_count": recovery.common_event_count,
        "excluded_event_ids": list(recovery.excluded_event_ids),
        "comparison_config": asdict(comparison_config),
        "injection_config": asdict(injection_config),
        "recovery_scope": "post_decoder_gaussian_raw_emission_scoring",
        "event_selection_scope": frozen.manifest.event_selection_time_scope,
        "claim_scope": "conditional_replay_content_not_event_incidence",
        "recovery_passed": recovery.passed,
        "status": comparison.status,
        "abstention_reasons": list(comparison.abstention_reasons),
        "outputs_sha256": output_sha256,
    }
    manifest_path = output / MANIFEST_OUTPUT
    manifest_path.write_text(
        json.dumps(analysis_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    paths[MANIFEST_OUTPUT] = manifest_path
    return paths


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictor-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    run_analysis(args.predictor_dir, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
