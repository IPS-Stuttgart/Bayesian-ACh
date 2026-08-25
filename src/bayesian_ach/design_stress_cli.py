"""CLI for the versioned post-freeze design stress artifact."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from bayesian_ach.design_grid import generate_transition_design_grid
from bayesian_ach.design_stress import (
    STRESS_DESIGNS,
    DesignStressConfig,
    run_design_stress,
)
from bayesian_ach.io import write_json, write_rows_csv

_REPOSITORY = "IPS-Stuttgart/Bayesian-ACh"
_SHA_PATTERN = re.compile(r"^[0-9a-f]{40}$")
_PAYLOAD_FILENAMES = (
    "summary.json",
    "thresholds.csv",
    "calibration_audit.csv",
    "pure_recovery.csv",
    "null_evaluation.csv",
    "mixture_evaluation.csv",
    "out_of_span_evaluation.csv",
    "allocations.csv",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_digest(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _git_provenance(repo_root: Path, expected_sha: str) -> None:
    if _SHA_PATTERN.fullmatch(expected_sha) is None:
        raise ValueError("--code-sha must be a lowercase 40-character commit SHA")
    head = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if head != expected_sha:
        raise RuntimeError(f"checked-out HEAD {head} does not match --code-sha {expected_sha}")
    status = subprocess.run(
        ["git", "-C", str(repo_root), "status", "--porcelain", "--untracked-files=all"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status:
        raise RuntimeError("claim-bearing stress artifacts require a clean Git worktree")


def _csv_safe_rows(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    safe: list[dict[str, Any]] = []
    for row in rows:
        safe.append(
            {
                key: json.dumps(value, sort_keys=True, separators=(",", ":"))
                if isinstance(value, (dict, list, tuple))
                else value
                for key, value in row.items()
            }
        )
    return safe


def _verify_checksum_table(directory: Path) -> dict[str, str]:
    checksum_path = directory / "SHA256SUMS.csv"
    if not checksum_path.is_file():
        raise ValueError(f"missing certificate checksum table: {checksum_path}")
    locked: dict[str, str] = {}
    with checksum_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            name = str(row["file"])
            if name in locked:
                raise ValueError(f"duplicate checksum entry: {name}")
            locked[name] = str(row["sha256"])
    for name, expected in locked.items():
        candidate = directory / name
        if not candidate.is_file() or _sha256(candidate) != expected:
            raise ValueError(f"certificate package checksum mismatch: {name}")
    return locked


def _load_certified_allocation(
    allocation_path: Path,
) -> tuple[tuple[str, int], NDArray[np.int64], dict[str, Any]]:
    allocation_path = allocation_path.resolve()
    directory = allocation_path.parent
    locked = _verify_checksum_table(directory)
    summary_path = directory / "certificate_summary.json"
    for required in (allocation_path.name, summary_path.name):
        if required not in locked:
            raise ValueError(f"certificate checksum table does not bind {required}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    config = summary.get("config", {})
    if (
        summary.get("schema_version") != 1
        or summary.get("experiment") != "certified_finite_grid_maximin_allocation"
        or summary.get("certified") is not True
        or summary.get("direct_geometry_matches_lower_bound") is not True
        or config.get("integer") is not True
    ):
        raise ValueError("allocation override is not a certified integer design artifact")
    budget = int(config["budget"])
    point_count = int(summary["grid_point_count"])
    counts = np.zeros(point_count, dtype=np.int64)
    with allocation_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("certified allocation table is empty")
    for row in rows:
        point = int(row["point_id"])
        value = float(row["allocation"])
        rounded = int(round(value))
        if point < 0 or point >= point_count or abs(value - rounded) > 1.0e-9:
            raise ValueError("certified integer allocation contains an invalid row")
        if counts[point] != 0:
            raise ValueError(f"duplicate certified allocation point: {point}")
        counts[point] = rounded
    if int(np.sum(counts)) != budget:
        raise ValueError("certified allocation does not sum to its locked budget")
    provenance = {
        "certificate_code_sha": summary["code_sha"],
        "certificate_budget": budget,
        "certificate_lower_bound": summary["lower_bound"],
        "certificate_upper_bound": summary["upper_bound"],
        "certificate_absolute_gap": summary["absolute_gap"],
        "certificate_allocation_path": allocation_path.name,
        "certificate_allocation_sha256": locked[allocation_path.name],
        "certificate_summary_sha256": locked[summary_path.name],
        "certificate_checksums_sha256": _sha256(directory / "SHA256SUMS.csv"),
    }
    return ("maximin_optimized", budget), counts, provenance


def _load_locked_design_allocation(
    path: Path,
    *,
    expected_sha256: str,
    source_code_sha: str,
) -> tuple[dict[tuple[str, int], NDArray[np.int64]], dict[str, Any]]:
    path = path.resolve()
    if re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is None:
        raise ValueError("locked allocation SHA-256 must contain 64 lowercase hex characters")
    if _SHA_PATTERN.fullmatch(source_code_sha) is None:
        raise ValueError("locked allocation source code SHA must contain 40 lowercase hex characters")
    observed_sha256 = _sha256(path)
    if observed_sha256 != expected_sha256:
        raise ValueError("locked design allocation SHA-256 mismatch")
    point_count = len(generate_transition_design_grid()[0])
    counts = {
        design: np.zeros(point_count, dtype=np.int64)
        for design in STRESS_DESIGNS
    }
    seen: set[tuple[str, int]] = set()
    seeds: set[str] = set()
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("locked design allocation is empty")
    for row in rows:
        design = str(row["design"])
        if design not in counts:
            raise ValueError(f"unknown locked design name: {design}")
        point = int(row["point_id"])
        raw_count = row.get("count", row.get("allocation"))
        if raw_count is None:
            raise ValueError("locked allocation requires a count or allocation column")
        value = float(raw_count)
        rounded = int(round(value))
        key = (design, point)
        if (
            point < 0
            or point >= point_count
            or key in seen
            or rounded <= 0
            or not math.isclose(value, rounded, abs_tol=1.0e-9)
        ):
            raise ValueError("locked design allocation has an invalid row")
        counts[design][point] = rounded
        seen.add(key)
        if row.get("seed") not in {None, ""}:
            seeds.add(str(row["seed"]))
    budgets = {design: int(np.sum(value)) for design, value in counts.items()}
    if any(budget <= 0 for budget in budgets.values()):
        raise ValueError("locked allocation must contain every declared design")
    overrides = {
        (design, budgets[design]): value
        for design, value in counts.items()
    }
    provenance = {
        "kind": "chronologically_locked_primary_design_allocation",
        "source_repository": _REPOSITORY,
        "source_code_sha": source_code_sha,
        "allocation_file": path.name,
        "allocation_sha256": observed_sha256,
        "allocation_bytes": path.stat().st_size,
        "design_budgets": budgets,
        "seeds": sorted(seeds),
    }
    return overrides, provenance


def _write_artifact(
    output: Path,
    *,
    result: Any,
    config: DesignStressConfig,
    code_sha: str,
    input_provenance: Sequence[Mapping[str, Any]],
) -> None:
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "summary.json", result.summary)
    write_rows_csv(output / "thresholds.csv", _csv_safe_rows(result.thresholds))
    write_rows_csv(
        output / "calibration_audit.csv",
        _csv_safe_rows(result.calibration),
    )
    write_rows_csv(
        output / "pure_recovery.csv",
        _csv_safe_rows(result.pure_recovery),
    )
    write_rows_csv(
        output / "null_evaluation.csv",
        _csv_safe_rows(result.null_evaluation),
    )
    write_rows_csv(
        output / "mixture_evaluation.csv",
        _csv_safe_rows(result.mixture_evaluation),
    )
    write_rows_csv(
        output / "out_of_span_evaluation.csv",
        _csv_safe_rows(result.out_of_span_evaluation),
    )
    write_rows_csv(output / "allocations.csv", _csv_safe_rows(result.allocations))

    files = [
        {
            "path": name,
            "bytes": (output / name).stat().st_size,
            "sha256": _sha256(output / name),
        }
        for name in _PAYLOAD_FILENAMES
    ]
    manifest = {
        "schema_version": 1,
        "artifact": "post_freeze_design_abstention_sensitivity",
        "repository": _REPOSITORY,
        "producer_commit": code_sha,
        "producer_git_dirty": False,
        "configuration_sha256": _canonical_digest(asdict(config)),
        "inputs": list(input_provenance),
        "files": files,
    }
    write_json(output / "artifact_manifest.json", manifest)
    checksum_rows = files + [
        {
            "path": "artifact_manifest.json",
            "bytes": (output / "artifact_manifest.json").stat().st_size,
            "sha256": _sha256(output / "artifact_manifest.json"),
        }
    ]
    write_rows_csv(output / "SHA256SUMS.csv", checksum_rows)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bayesian-ach-design-stress",
        description="Freeze post-baseline pure/null/mixture abstention sensitivity.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--code-sha", required=True)
    parser.add_argument("--fixed-budgets", nargs="+", type=int, default=(60,))
    parser.add_argument("--budget-factors", nargs="*", type=float, default=())
    parser.add_argument("--calibration-replicates", type=int, default=100)
    parser.add_argument("--calibration-audit-replicates", type=int, default=100)
    parser.add_argument("--evaluation-replicates", type=int, default=200)
    parser.add_argument("--threshold-seed", type=int, default=104729)
    parser.add_argument("--calibration-audit-seed", type=int, default=130363)
    parser.add_argument("--evaluation-seed", type=int, default=155921)
    parser.add_argument("--locked-allocation", type=Path)
    parser.add_argument("--locked-allocation-sha256")
    parser.add_argument("--locked-design-code-sha")
    parser.add_argument(
        "--certified-allocation",
        action="append",
        type=Path,
        default=[],
        help="Checksum-bound certified integer allocation CSV; may be repeated.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    _git_provenance(args.repo_root.resolve(), args.code_sha)
    config = DesignStressConfig(
        fixed_budgets=tuple(args.fixed_budgets),
        budget_factors=tuple(args.budget_factors),
        calibration_replicates=args.calibration_replicates,
        calibration_audit_replicates=args.calibration_audit_replicates,
        evaluation_replicates=args.evaluation_replicates,
        threshold_seed=args.threshold_seed,
        calibration_audit_seed=args.calibration_audit_seed,
        evaluation_seed=args.evaluation_seed,
    )
    overrides: dict[tuple[str, int], NDArray[np.int64]] = {}
    provenance: list[dict[str, Any]] = []
    locked_arguments = (
        args.locked_allocation,
        args.locked_allocation_sha256,
        args.locked_design_code_sha,
    )
    if any(value is not None for value in locked_arguments):
        if not all(value is not None for value in locked_arguments):
            raise ValueError(
                "locked allocation path, SHA-256, and source code SHA are jointly required"
            )
        locked_overrides, item = _load_locked_design_allocation(
            args.locked_allocation,
            expected_sha256=args.locked_allocation_sha256,
            source_code_sha=args.locked_design_code_sha,
        )
        overrides.update(locked_overrides)
        provenance.append(item)
    for path in args.certified_allocation:
        key, counts, item = _load_certified_allocation(path)
        if key in overrides:
            raise ValueError(f"duplicate certified allocation for {key}")
        overrides[key] = counts
        provenance.append(item)
    result = run_design_stress(config, allocation_overrides=overrides)
    _write_artifact(
        args.output.resolve(),
        result=result,
        config=config,
        code_sha=args.code_sha,
        input_provenance=provenance,
    )
    print(json.dumps(result.summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
