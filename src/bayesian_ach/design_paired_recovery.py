"""Checksum-bound paired recovery for heuristic and certified maximin designs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import shutil
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from bayesian_ach.design_certificate_verify import verify_certificate_package
from bayesian_ach.design_grid import generate_transition_design_grid
from bayesian_ach.design_optimizer import optimize_maximin_design
from bayesian_ach.design_recovery import recover_design

_SHA40 = re.compile(r"^[0-9a-f]{40}$")
_SHA64 = re.compile(r"^[0-9a-f]{64}$")
_BUDGET = 60
_MAX_POINT_FRACTION = 0.15
_VARIANTS = ("heuristic_maximin", "certified_integer_maximin")


@dataclass(frozen=True, slots=True)
class PairedRecoveryConfig:
    """Frozen paired-recovery settings."""

    seeds: tuple[int, ...] = (7, 11, 19, 23, 31)
    replicates: int = 200
    test_fraction: float = 0.35
    effect_size: float = 1.0
    noise_std: float = 1.0
    recovery_seed_offset: int = 100_003

    def validate(self) -> None:
        if len(self.seeds) != 5 or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("paired recovery requires five unique seeds")
        if self.replicates < 20:
            raise ValueError("replicates must be at least 20")
        if not 0.0 < self.test_fraction < 1.0:
            raise ValueError("test_fraction must lie in (0, 1)")
        if (
            not math.isfinite(self.effect_size)
            or not math.isfinite(self.noise_std)
            or self.effect_size <= 0.0
            or self.noise_std <= 0.0
        ):
            raise ValueError("effect_size and noise_std must be finite and positive")


def sha256(path: Path) -> str:
    """Return a streaming SHA-256 digest."""

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
        raise RuntimeError("paired recovery requires the clean exact --code-sha checkout")


def _integer_count(value: str) -> int:
    number = float(value)
    rounded = int(round(number))
    if rounded <= 0 or not math.isclose(number, rounded, abs_tol=1.0e-9):
        raise ValueError("allocation contains a nonpositive or noninteger count")
    return rounded


def load_heuristic_allocation(
    path: Path,
    *,
    expected_sha256: str,
    source_code_sha: str,
) -> tuple[NDArray[np.int64], dict[str, Any]]:
    """Load and reconstruct the chronologically locked heuristic N=60 allocation."""

    path = path.resolve()
    if _SHA64.fullmatch(expected_sha256) is None or sha256(path) != expected_sha256:
        raise ValueError("heuristic allocation SHA-256 mismatch")
    if _SHA40.fullmatch(source_code_sha) is None:
        raise ValueError("heuristic source code SHA must contain 40 lowercase hex characters")
    _, _, signals = generate_transition_design_grid()
    counts = np.zeros(signals.shape[0], dtype=np.int64)
    seen: set[int] = set()
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("heuristic allocation table is empty")
    for row in rows:
        if row.get("design") != "maximin_optimized":
            continue
        point = int(row["point_id"])
        if point < 0 or point >= counts.size or point in seen:
            raise ValueError("heuristic allocation has an invalid or duplicate point")
        counts[point] = _integer_count(str(row["count"]))
        seen.add(point)
    if int(np.sum(counts)) != _BUDGET:
        raise ValueError("heuristic maximin allocation does not sum to N=60")
    reconstructed = optimize_maximin_design(
        signals,
        _BUDGET,
        max_point_fraction=_MAX_POINT_FRACTION,
        effect_size=1.0,
        noise_std=1.0,
        target_log_score_gap=5.0,
    ).counts
    if not np.array_equal(counts, reconstructed):
        raise ValueError("heuristic allocation does not match the frozen constructor")
    return counts, {
        "kind": "chronologically_locked_heuristic_maximin_n60",
        "source_repository": "IPS-Stuttgart/Bayesian-ACh",
        "source_code_sha": source_code_sha,
        "source_file": path.name,
        "source_bytes": path.stat().st_size,
        "source_sha256": expected_sha256,
        "budget": _BUDGET,
        "max_point_fraction": _MAX_POINT_FRACTION,
        "constructor_reproduced": True,
    }


def _certified_counts(path: Path, point_count: int) -> NDArray[np.int64]:
    counts = np.zeros(point_count, dtype=np.int64)
    seen: set[int] = set()
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        point = int(row["point_id"])
        if point < 0 or point >= point_count or point in seen:
            raise ValueError("certified allocation has an invalid or duplicate point")
        counts[point] = _integer_count(str(row["allocation"]))
        seen.add(point)
    if int(np.sum(counts)) != _BUDGET:
        raise ValueError("certified allocation does not sum to N=60")
    return counts


def load_certified_allocation(
    package: Path,
) -> tuple[NDArray[np.int64], dict[str, Any]]:
    """Verify and load an exact integer N=60 certificate package."""

    package = package.resolve()
    report = verify_certificate_package(package)
    if int(report["budget"]) != _BUDGET:
        raise ValueError("paired recovery requires an N=60 certificate package")
    summary_path = package / "certificate_summary.json"
    allocation_path = package / "certified_allocation.csv"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    _, _, signals = generate_transition_design_grid()
    counts = _certified_counts(allocation_path, signals.shape[0])
    return counts, {
        "kind": "certified_integer_maximin_n60",
        "source_repository": "IPS-Stuttgart/Bayesian-ACh",
        "certificate_code_sha": summary["code_sha"],
        "certificate_summary_sha256": sha256(summary_path),
        "certificate_allocation_sha256": sha256(allocation_path),
        "certificate_sha256sums_sha256": sha256(package / "SHA256SUMS.csv"),
        "budget": _BUDGET,
        "lower_bound": summary["lower_bound"],
        "upper_bound": summary["upper_bound"],
        "certified": True,
    }


def paired_recovery_rows(
    heuristic: NDArray[np.int64],
    certified: NDArray[np.int64],
    config: PairedRecoveryConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Run both schedules with identical random streams for every frozen seed."""

    config.validate()
    _, _, signals = generate_transition_design_grid()
    variants = {
        "heuristic_maximin": heuristic,
        "certified_integer_maximin": certified,
    }
    recovery_rows: list[dict[str, Any]] = []
    seed_rows: list[dict[str, Any]] = []
    for seed in config.seeds:
        by_variant: dict[str, list[float]] = {}
        for name, counts in variants.items():
            records = recover_design(
                name,
                signals,
                counts,
                replicates=config.replicates,
                test_fraction=config.test_fraction,
                effect_size=config.effect_size,
                noise_std=config.noise_std,
                seed=seed + config.recovery_seed_offset,
            )
            rates = [float(record.recovery_rate) for record in records]
            by_variant[name] = rates
            recovery_rows.extend({"seed": seed, **record.as_dict()} for record in records)
        heuristic_rates = by_variant["heuristic_maximin"]
        certified_rates = by_variant["certified_integer_maximin"]
        seed_rows.append(
            {
                "seed": seed,
                "heuristic_minimum_recovery": min(heuristic_rates),
                "certified_minimum_recovery": min(certified_rates),
                "minimum_recovery_delta_certified_minus_heuristic": (
                    min(certified_rates) - min(heuristic_rates)
                ),
                "heuristic_mean_recovery": float(np.mean(heuristic_rates)),
                "certified_mean_recovery": float(np.mean(certified_rates)),
                "mean_recovery_delta_certified_minus_heuristic": (
                    float(np.mean(certified_rates)) - float(np.mean(heuristic_rates))
                ),
            }
        )
    difference = certified - heuristic
    overlap = int(np.sum((certified > 0) & (heuristic > 0)))
    diagnostics = {
        "heuristic_support_count": int(np.count_nonzero(heuristic)),
        "certified_support_count": int(np.count_nonzero(certified)),
        "support_overlap_count": overlap,
        "support_union_count": int(np.sum((certified > 0) | (heuristic > 0))),
        "allocation_l1_distance": int(np.sum(np.abs(difference))),
        "maximum_absolute_cell_change": int(np.max(np.abs(difference))),
        "minimum_recovery_across_seeds": {
            "heuristic_maximin": min(
                float(row["heuristic_minimum_recovery"]) for row in seed_rows
            ),
            "certified_integer_maximin": min(
                float(row["certified_minimum_recovery"]) for row in seed_rows
            ),
        },
        "all_seed_minima_certified_minus_heuristic_negative": all(
            float(row["minimum_recovery_delta_certified_minus_heuristic"]) < 0.0
            for row in seed_rows
        ),
    }
    return recovery_rows, seed_rows, diagnostics


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty table: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def freeze_paired_recovery(
    *,
    output: Path,
    repo_root: Path,
    code_sha: str,
    heuristic_path: Path,
    heuristic_sha256: str,
    heuristic_source_code_sha: str,
    certificate_package: Path,
    config: PairedRecoveryConfig,
) -> dict[str, Any]:
    """Generate a self-checksummed paired-recovery evidence package."""

    _git_provenance(repo_root.resolve(), code_sha)
    heuristic, heuristic_provenance = load_heuristic_allocation(
        heuristic_path,
        expected_sha256=heuristic_sha256,
        source_code_sha=heuristic_source_code_sha,
    )
    certified, certificate_provenance = load_certified_allocation(certificate_package)
    recovery, seeds, diagnostics = paired_recovery_rows(heuristic, certified, config)
    summary = {
        "schema_version": 1,
        "experiment": "paired_heuristic_vs_certified_n60_recovery_diagnostic",
        "producer_commit": code_sha,
        "producer_clean_worktree": True,
        "config": asdict(config),
        "paired_rng": True,
        "recovery_seed_rule": (
            "seed + recovery_seed_offset; RNG reset identically for each allocation"
        ),
        "input_provenance": [heuristic_provenance, certificate_provenance],
        **diagnostics,
        "interpretation": (
            "The exact integer certificate optimizes the frozen asymptotic worst-residual "
            "objective. This paired finite-sample diagnostic does not select a replacement "
            "schedule, establish empirical superiority, or define a physical trial/animal protocol."
        ),
    }
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(heuristic_path, output / "heuristic_allocation_source.csv")
    _write_csv(output / "paired_recovery.csv", recovery)
    _write_csv(output / "seed_summary.csv", seeds)
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    payload_names = (
        "heuristic_allocation_source.csv",
        "paired_recovery.csv",
        "seed_summary.csv",
        "summary.json",
    )
    manifest = {
        "schema_version": 1,
        "artifact": summary["experiment"],
        "producer_commit": code_sha,
        "producer_git_dirty": False,
        "inputs": summary["input_provenance"],
        "files": [
            {
                "path": name,
                "bytes": (output / name).stat().st_size,
                "sha256": sha256(output / name),
            }
            for name in payload_names
        ],
    }
    (output / "artifact_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    checksum_names = (*payload_names, "artifact_manifest.json")
    _write_csv(
        output / "SHA256SUMS.csv",
        [
            {
                "file": name,
                "bytes": (output / name).stat().st_size,
                "sha256": sha256(output / name),
            }
            for name in checksum_names
        ],
    )
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bayesian-ach-design-paired-recovery",
        description="Freeze paired heuristic-versus-certified N=60 recovery diagnostics.",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--repo-root", default=Path.cwd(), type=Path)
    parser.add_argument("--code-sha", required=True)
    parser.add_argument("--heuristic-allocation", required=True, type=Path)
    parser.add_argument("--heuristic-allocation-sha256", required=True)
    parser.add_argument("--heuristic-source-code-sha", required=True)
    parser.add_argument("--certificate-package", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    summary = freeze_paired_recovery(
        output=args.output,
        repo_root=args.repo_root,
        code_sha=args.code_sha,
        heuristic_path=args.heuristic_allocation,
        heuristic_sha256=args.heuristic_allocation_sha256,
        heuristic_source_code_sha=args.heuristic_source_code_sha,
        certificate_package=args.certificate_package,
        config=PairedRecoveryConfig(),
    )
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
