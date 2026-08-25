"""CLI and independent verifier for the mixture-aware design diagnostic."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from bayesian_ach.design_mixture_diagnostic import (
    MixtureDiagnosticConfig,
    run_mixture_diagnostic,
)
from bayesian_ach.design_stress_cli import (
    _csv_safe_rows,
    _git_provenance,
    _load_locked_design_allocation,
)
from bayesian_ach.io import write_json, write_rows_csv

_REPOSITORY = "IPS-Stuttgart/Bayesian-ACh"
_BASELINE_PRODUCER_SHA = "c71695fda83ae93407599a909097962ee3fa9e0e"
_BASELINE_CHECKSUMS_SHA256 = (
    "44a5188c43bda52e6fc9dc7007cf2de44a9671e9c5477ac88c3173c06cfdbd80"
)
_BASELINE_MANIFEST_SHA256 = (
    "d840a2ec34f5a386109c7f985033b53144fdcc1b1a0e0592b3274c0e38902b64"
)
_PAYLOADS = (
    "summary.json",
    "thresholds.csv",
    "calibration_audit.csv",
    "pure_evaluation.csv",
    "null_evaluation.csv",
    "mixture_evaluation.csv",
    "out_of_span_evaluation.csv",
    "geometry.csv",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_digest(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _verify_checksum_rows(directory: Path) -> tuple[dict[str, Any], ...]:
    path = directory / "SHA256SUMS.csv"
    if not path.is_file():
        raise ValueError(f"missing checksum table: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = tuple(dict(row) for row in csv.DictReader(handle))
    if not rows:
        raise ValueError("checksum table is empty")
    seen: set[str] = set()
    for row in rows:
        name = str(row.get("path", row.get("file", "")))
        if not name or name in seen:
            raise ValueError("checksum table contains a missing or duplicate path")
        seen.add(name)
        candidate = directory / name
        if not candidate.is_file():
            raise ValueError(f"checksum table payload is missing: {name}")
        if int(row["bytes"]) != candidate.stat().st_size:
            raise ValueError(f"locked byte count mismatch: {name}")
        if str(row["sha256"]) != _sha256(candidate):
            raise ValueError(f"SHA-256 mismatch: {name}")
    return rows


def verify_baseline_artifact(directory: Path) -> dict[str, Any]:
    """Verify and bind the immutable original failure artifact."""

    directory = directory.resolve()
    manifest_path = directory / "artifact_manifest.json"
    checksums_path = directory / "SHA256SUMS.csv"
    if _sha256(checksums_path) != _BASELINE_CHECKSUMS_SHA256:
        raise ValueError("baseline checksum-table SHA-256 mismatch")
    if _sha256(manifest_path) != _BASELINE_MANIFEST_SHA256:
        raise ValueError("baseline manifest SHA-256 mismatch")
    rows = _verify_checksum_rows(directory)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("artifact") != "post_freeze_design_abstention_sensitivity"
        or manifest.get("producer_commit") != _BASELINE_PRODUCER_SHA
        or manifest.get("producer_git_dirty") is not False
    ):
        raise ValueError("baseline artifact provenance does not match the lock")
    return {
        "kind": "immutable_original_design_stress_failure",
        "producer_commit": _BASELINE_PRODUCER_SHA,
        "checksum_table_sha256": _BASELINE_CHECKSUMS_SHA256,
        "manifest_sha256": _BASELINE_MANIFEST_SHA256,
        "verified_payload_count": len(rows),
    }


def _write_artifact(
    output: Path,
    *,
    result: Any,
    config: MixtureDiagnosticConfig,
    code_sha: str,
    inputs: Sequence[Mapping[str, Any]],
) -> None:
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "summary.json", result.summary)
    write_rows_csv(output / "thresholds.csv", _csv_safe_rows(result.thresholds))
    write_rows_csv(
        output / "calibration_audit.csv",
        _csv_safe_rows(result.calibration_audit),
    )
    write_rows_csv(
        output / "pure_evaluation.csv",
        _csv_safe_rows(result.pure_evaluation),
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
    write_rows_csv(output / "geometry.csv", _csv_safe_rows(result.geometry))
    files = [
        {
            "path": name,
            "bytes": (output / name).stat().st_size,
            "sha256": _sha256(output / name),
        }
        for name in _PAYLOADS
    ]
    manifest = {
        "schema_version": 1,
        "artifact": "post_failure_mixture_aware_design_diagnostic",
        "repository": _REPOSITORY,
        "producer_commit": code_sha,
        "producer_git_dirty": False,
        "configuration": asdict(config),
        "configuration_sha256": _canonical_digest(asdict(config)),
        "inputs": list(inputs),
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


def verify_mixture_diagnostic_artifact(directory: Path) -> dict[str, Any]:
    """Independently verify payload hashes and the locked scientific config."""

    directory = directory.resolve()
    rows = _verify_checksum_rows(directory)
    locked_names = {
        str(row.get("path", row.get("file", "")))
        for row in rows
    }
    if locked_names != {*_PAYLOADS, "artifact_manifest.json"}:
        raise ValueError("diagnostic checksum table has an unexpected payload set")
    manifest = json.loads(
        (directory / "artifact_manifest.json").read_text(encoding="utf-8")
    )
    config = manifest.get("configuration", {})
    if (
        manifest.get("schema_version") != 1
        or manifest.get("artifact")
        != "post_failure_mixture_aware_design_diagnostic"
        or manifest.get("producer_git_dirty") is not False
        or config.get("threshold_seed") != 196613
        or config.get("calibration_audit_seed") != 262147
        or config.get("evaluation_seed") != 324949
        or config.get("minimum_pure_retention_wilson_lower") != 0.70
        or config.get("minimum_rejection_power_wilson_lower") != 0.70
        or config.get("folds") != 3
        or config.get("budget") != 60
    ):
        raise ValueError("diagnostic manifest does not match the locked config")
    if manifest.get("configuration_sha256") != _canonical_digest(config):
        raise ValueError("diagnostic configuration digest mismatch")
    inputs = manifest.get("inputs", [])
    baseline = [
        item
        for item in inputs
        if item.get("kind") == "immutable_original_design_stress_failure"
    ]
    if (
        len(baseline) != 1
        or baseline[0].get("manifest_sha256") != _BASELINE_MANIFEST_SHA256
        or baseline[0].get("checksum_table_sha256")
        != _BASELINE_CHECKSUMS_SHA256
    ):
        raise ValueError("diagnostic does not bind the immutable baseline")
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    gates = summary.get("technical_gates", {})
    if not gates or not all(bool(value) for value in gates.values()):
        raise ValueError("diagnostic technical gates did not all pass")
    return {
        "producer_commit": manifest["producer_commit"],
        "configuration_sha256": manifest["configuration_sha256"],
        "verified_payload_count": len(rows),
        "baseline_manifest_sha256": _BASELINE_MANIFEST_SHA256,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bayesian-ach-mixture-diagnostic",
        description="Freeze the prospectively configured post-failure diagnostic.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--code-sha", required=True)
    parser.add_argument("--baseline-artifact", type=Path, required=True)
    parser.add_argument("--locked-allocation", type=Path, required=True)
    parser.add_argument("--locked-allocation-sha256", required=True)
    parser.add_argument("--locked-design-code-sha", required=True)
    parser.add_argument("--locked-allocation-seed", type=int, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    repo_root = args.repo_root.resolve()
    _git_provenance(repo_root, args.code_sha)
    baseline = verify_baseline_artifact(args.baseline_artifact)
    overrides, allocation = _load_locked_design_allocation(
        args.locked_allocation,
        expected_sha256=args.locked_allocation_sha256,
        source_code_sha=args.locked_design_code_sha,
        allocation_seed=args.locked_allocation_seed,
    )
    key = ("maximin_optimized", 60)
    if key not in overrides:
        raise ValueError("locked allocation does not contain maximin N=60")
    config = MixtureDiagnosticConfig()
    result = run_mixture_diagnostic(overrides[key], config)
    result.summary["chronology"] = (
        "Method, thresholds, power gate, and fresh streams were committed "
        "after immutable baseline failure d1251ddd and before this evaluation."
    )
    result.summary["input_provenance"] = {
        "baseline": baseline,
        "allocation": allocation,
    }
    _write_artifact(
        args.output.resolve(),
        result=result,
        config=config,
        code_sha=args.code_sha,
        inputs=(baseline, allocation),
    )
    verified = verify_mixture_diagnostic_artifact(args.output)
    print(json.dumps({"summary": result.summary, "verification": verified}, indent=2))
    return 0


def verify_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="bayesian-ach-verify-mixture-diagnostic"
    )
    parser.add_argument("artifact", type=Path)
    args = parser.parse_args(argv)
    print(
        json.dumps(
            verify_mixture_diagnostic_artifact(args.artifact),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
