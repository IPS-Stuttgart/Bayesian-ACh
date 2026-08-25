"""Independent verification for frozen integer maximin certificate packages."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from bayesian_ach.design_certificate import population_n_eff_bracket
from bayesian_ach.design_geometry import pairwise_residual_matrix
from bayesian_ach.design_grid import generate_transition_design_grid

_EXPECTED_PAYLOADS = {
    "certificate_summary.json",
    "certified_allocation.csv",
    "cut_trace.csv",
}
_SHA_PATTERN = re.compile(r"^[0-9a-f]{40}$")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _locked_payloads(directory: Path) -> dict[str, dict[str, str]]:
    checksum_path = directory / "SHA256SUMS.csv"
    if not checksum_path.is_file():
        raise ValueError(f"missing checksum table: {checksum_path}")
    with checksum_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    locked: dict[str, dict[str, str]] = {}
    for row in rows:
        name = str(row["file"])
        if name in locked:
            raise ValueError(f"duplicate checksum row: {name}")
        locked[name] = {key: str(value) for key, value in row.items()}
    if set(locked) != _EXPECTED_PAYLOADS:
        raise ValueError("checksum table does not bind the exact certificate payload set")
    for name, row in locked.items():
        path = directory / name
        if not path.is_file():
            raise ValueError(f"missing locked payload: {name}")
        if int(row["bytes"]) != path.stat().st_size:
            raise ValueError(f"locked byte count mismatch: {name}")
        if row["sha256"] != _sha256(path):
            raise ValueError(f"locked SHA-256 mismatch: {name}")
    return locked


def _allocation_counts(
    path: Path,
    *,
    point_count: int,
) -> np.ndarray:
    counts = np.zeros(point_count, dtype=np.int64)
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("certified allocation is empty")
    seen: set[int] = set()
    for row in rows:
        point = int(row["point_id"])
        value = float(row["allocation"])
        rounded = int(round(value))
        if point < 0 or point >= point_count or point in seen:
            raise ValueError("certified allocation has an invalid or duplicate point")
        if rounded <= 0 or not math.isclose(value, rounded, abs_tol=1.0e-9):
            raise ValueError("certified integer allocation contains a noninteger count")
        counts[point] = rounded
        seen.add(point)
    return counts


def verify_certificate_package(directory: Path) -> dict[str, Any]:
    """Recompute hashes, finite-grid geometry, gap, and N_eff rounding."""

    directory = directory.resolve()
    _locked_payloads(directory)
    summary = json.loads(
        (directory / "certificate_summary.json").read_text(encoding="utf-8")
    )
    if (
        summary.get("schema_version") != 1
        or summary.get("experiment") != "certified_finite_grid_maximin_allocation"
        or summary.get("certified") is not True
        or summary.get("direct_geometry_matches_lower_bound") is not True
    ):
        raise ValueError("package is not a certified finite-grid allocation")
    if _SHA_PATTERN.fullmatch(str(summary.get("code_sha", ""))) is None:
        raise ValueError("certificate code_sha is not a full lowercase commit SHA")

    config = summary["config"]
    if config.get("integer") is not True:
        raise ValueError("claim-bearing package must certify an integer allocation")
    budget = int(config["budget"])
    maximum_count = max(
        1,
        int(math.ceil(float(config["max_point_fraction"]) * budget)),
    )
    rows, _, signals = generate_transition_design_grid()
    if int(summary["grid_point_count"]) != len(rows):
        raise ValueError("certificate grid-point count does not match the canonical grid")
    counts = _allocation_counts(
        directory / "certified_allocation.csv",
        point_count=len(rows),
    )
    if int(np.sum(counts)) != budget or np.any(counts > maximum_count):
        raise ValueError("certified allocation violates its budget or per-cell cap")

    geometry = pairwise_residual_matrix(signals, counts)
    off_diagonal = ~np.eye(geometry.shape[0], dtype=bool)
    direct_lower = float(np.min(geometry[off_diagonal]))
    lower = float(summary["lower_bound"])
    upper = float(summary["upper_bound"])
    absolute_gap = float(summary["absolute_gap"])
    relative_gap = float(summary["relative_gap"])
    if (
        not np.isfinite([lower, upper, absolute_gap, relative_gap]).all()
        or lower <= 0.0
        or upper < lower
    ):
        raise ValueError("certificate bounds are not finite, positive, and ordered")
    if not math.isclose(direct_lower, lower, rel_tol=1.0e-9, abs_tol=1.0e-9):
        raise ValueError("direct finite-grid geometry does not reproduce the lower bound")
    recomputed_gap = max(0.0, upper - lower)
    if not math.isclose(absolute_gap, recomputed_gap, rel_tol=1.0e-9, abs_tol=1.0e-12):
        raise ValueError("reported absolute certificate gap is inconsistent")
    expected_relative = recomputed_gap / max(1.0, abs(lower))
    if not math.isclose(relative_gap, expected_relative, rel_tol=1.0e-9, abs_tol=1.0e-12):
        raise ValueError("reported relative certificate gap is inconsistent")
    tolerance = float(config["absolute_gap_tolerance"]) + float(
        config["relative_gap_tolerance"]
    ) * max(1.0, abs(lower))
    if absolute_gap > tolerance:
        raise ValueError("reported certified gap exceeds the frozen tolerance")

    target = summary["population_n_eff"]
    n_eff = population_n_eff_bracket(
        lower,
        upper,
        effect_size=float(target["effect_size"]),
        noise_std=float(target["noise_std"]),
        target_log_score_gap=float(target["target_log_score_gap"]),
    )
    reported_n_eff = (
        int(target["lower_index_from_residual_upper_bound"]),
        int(target["upper_index_from_residual_lower_bound"]),
    )
    if n_eff != reported_n_eff or bool(target["index_certified"]) != (n_eff[0] == n_eff[1]):
        raise ValueError("reported population N_eff bracket is inconsistent")

    with (directory / "cut_trace.csv").open(newline="", encoding="utf-8") as handle:
        trace = list(csv.DictReader(handle))
    if len(trace) != int(summary["iterations"]):
        raise ValueError("cut trace length does not match the reported iteration count")
    if int(trace[-1]["cut_count"]) != int(summary["cut_count"]):
        raise ValueError("cut trace does not end at the reported cut count")

    return {
        "package": directory.name,
        "verified": True,
        "producer_commit": summary["code_sha"],
        "budget": budget,
        "support_size": int(np.count_nonzero(counts)),
        "maximum_cell_count": int(np.max(counts)),
        "lower_bound": lower,
        "upper_bound": upper,
        "absolute_gap": absolute_gap,
        "population_n_eff_lower": n_eff[0],
        "population_n_eff_upper": n_eff[1],
        "sha256sums_sha256": _sha256(directory / "SHA256SUMS.csv"),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bayesian-ach-design-certificate-verify",
        description="Verify frozen integer maximin certificate packages.",
    )
    parser.add_argument("packages", nargs="+", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    reports = [
        verify_certificate_package(path)
        for path in _parser().parse_args(argv).packages
    ]
    print(json.dumps(reports, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
