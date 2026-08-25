#!/usr/bin/env python3
"""Independently recompute a frozen post-stress geometry diagnostic."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

_PAYLOADS = {
    "stress_diagnostic.json",
    "maximin_mixture_geometry.csv",
    "artifact_manifest.json",
}
_SHA40 = re.compile(r"^[0-9a-f]{40}$")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_hashes(package: Path) -> dict[str, dict[str, str]]:
    with (package / "SHA256SUMS.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    locked: dict[str, dict[str, str]] = {}
    for row in rows:
        name = str(row["file"])
        if name in locked:
            raise ValueError(f"duplicate checksum row: {name}")
        locked[name] = {key: str(value) for key, value in row.items()}
    if set(locked) != _PAYLOADS:
        raise ValueError("checksum table does not bind the exact diagnostic payload set")
    for name, row in locked.items():
        path = package / name
        if (
            not path.is_file()
            or int(row["bytes"]) != path.stat().st_size
            or row["sha256"] != _sha256(path)
        ):
            raise ValueError(f"locked geometry payload mismatch: {name}")
    return locked


def verify_package(
    package: Path,
    *,
    source_artifact: Path,
    repo_root: Path,
) -> dict[str, object]:
    """Verify hashes and rerun all stratification, geometry, and frozen evaluations."""

    package = package.resolve()
    source_artifact = source_artifact.resolve()
    repo_root = repo_root.resolve()
    locked = _verify_hashes(package)
    manifest = json.loads(
        (package / "artifact_manifest.json").read_text(encoding="utf-8")
    )
    producer = str(manifest.get("producer_commit", ""))
    analyzer = repo_root / "scripts/analyze_design_stress_geometry.py"
    if (
        manifest.get("schema_version") != 1
        or manifest.get("artifact") != "post_freeze_stress_geometry_diagnostic"
        or manifest.get("producer_git_dirty") is not False
        or _SHA40.fullmatch(producer) is None
        or manifest.get("diagnostic_script_sha256") != _sha256(analyzer)
        or manifest.get("source_artifact_sha256sums_sha256")
        != _sha256(source_artifact / "SHA256SUMS.csv")
    ):
        raise ValueError("invalid geometry diagnostic provenance contract")
    if {
        item["path"]: item["sha256"] for item in manifest["files"]
    } != {
        name: row["sha256"]
        for name, row in locked.items()
        if name != "artifact_manifest.json"
    }:
        raise ValueError("manifest and checksum table disagree")

    report = json.loads((package / "stress_diagnostic.json").read_text(encoding="utf-8"))
    geometry = report["maximin_mixture_geometry"]
    if (
        report.get("diagnostic_producer_commit") != producer
        or report.get("diagnostic_script_sha256") != _sha256(analyzer)
        or report.get("source_artifact_sha256sums_sha256")
        != _sha256(source_artifact / "SHA256SUMS.csv")
        or report.get("no_threshold_or_evaluation_changes") is not True
        or len(geometry) != 15
    ):
        raise ValueError("invalid geometry diagnostic report contract")
    for row in geometry:
        if (
            int(row["independent_gate_pass_counts"]["all_three_pass"])
            != int(round(float(row["false_pure_call_rate"]) * 200))
            or float(row["oracle_two_component_residual"]) > 1.0e-20
        ):
            raise ValueError("geometry diagnostic gate or oracle residual is inconsistent")

    with tempfile.TemporaryDirectory(prefix="stress-geometry-verify-") as temporary:
        recomputed = Path(temporary) / "recomputed"
        subprocess.run(
            [
                sys.executable,
                str(analyzer),
                "--artifact",
                str(source_artifact),
                "--output",
                str(recomputed),
                "--repo-root",
                str(repo_root),
                "--code-sha",
                producer,
                "--recompute-only",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        for name in _PAYLOADS:
            if (package / name).read_bytes() != (recomputed / name).read_bytes():
                raise ValueError(f"independent recomputation mismatch: {name}")
    return {
        "package": package.name,
        "verified": True,
        "producer_commit": producer,
        "pair_count": len(geometry),
        "source_artifact_sha256sums_sha256": manifest[
            "source_artifact_sha256sums_sha256"
        ],
        "sha256sums_sha256": _sha256(package / "SHA256SUMS.csv"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("package", type=Path)
    parser.add_argument("--source-artifact", required=True, type=Path)
    parser.add_argument("--repo-root", default=Path.cwd(), type=Path)
    args = parser.parse_args()
    report = verify_package(
        args.package,
        source_artifact=args.source_artifact,
        repo_root=args.repo_root,
    )
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
