"""Independent verification for frozen paired maximin recovery evidence."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from bayesian_ach.design_paired_recovery import (
    PairedRecoveryConfig,
    load_certified_allocation,
    load_heuristic_allocation,
    paired_recovery_rows,
    sha256,
)

_PAYLOADS = {
    "heuristic_allocation_source.csv",
    "paired_recovery.csv",
    "seed_summary.csv",
    "summary.json",
    "artifact_manifest.json",
}
_SHA40 = re.compile(r"^[0-9a-f]{40}$")


def _locked_payloads(directory: Path) -> dict[str, dict[str, str]]:
    checksum = directory / "SHA256SUMS.csv"
    with checksum.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    locked: dict[str, dict[str, str]] = {}
    for row in rows:
        name = str(row["file"])
        if name in locked:
            raise ValueError(f"duplicate checksum row: {name}")
        locked[name] = {key: str(value) for key, value in row.items()}
    if set(locked) != _PAYLOADS:
        raise ValueError("checksum table does not bind the exact paired-recovery payload set")
    for name, row in locked.items():
        path = directory / name
        if not path.is_file():
            raise ValueError(f"missing locked payload: {name}")
        if int(row["bytes"]) != path.stat().st_size or row["sha256"] != sha256(path):
            raise ValueError(f"locked paired-recovery payload mismatch: {name}")
    return locked


def _provenance_item(summary: Mapping[str, Any], kind: str) -> Mapping[str, Any]:
    items = [item for item in summary["input_provenance"] if item.get("kind") == kind]
    if len(items) != 1:
        raise ValueError(f"expected one provenance item of kind {kind}")
    return items[0]


def _compare_rows(
    path: Path,
    expected: Sequence[Mapping[str, Any]],
) -> None:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != len(expected):
        raise ValueError(f"row count mismatch: {path.name}")
    for observed, target in zip(rows, expected, strict=True):
        if set(observed) != set(target):
            raise ValueError(f"column mismatch: {path.name}")
        for key, value in target.items():
            raw = str(observed[key])
            if isinstance(value, bool):
                equal = raw == str(value)
            elif isinstance(value, int):
                equal = int(raw) == value
            elif isinstance(value, float):
                equal = math.isclose(
                    float(raw),
                    value,
                    rel_tol=1.0e-12,
                    abs_tol=1.0e-12,
                )
            else:
                equal = raw == str(value)
            if not equal:
                raise ValueError(f"value mismatch in {path.name}: {key}")


def _compare_summary_value(observed: Any, expected: Any, name: str) -> None:
    if isinstance(expected, bool):
        equal = observed is expected
    elif isinstance(expected, int):
        equal = int(observed) == expected
    elif isinstance(expected, float):
        equal = math.isclose(
            float(observed),
            expected,
            rel_tol=1.0e-12,
            abs_tol=1.0e-12,
        )
    elif isinstance(expected, Mapping):
        equal = set(observed) == set(expected)
        if equal:
            for key, value in expected.items():
                _compare_summary_value(observed[key], value, f"{name}.{key}")
            return
    else:
        equal = observed == expected
    if not equal:
        raise ValueError(f"summary mismatch: {name}")


def verify_paired_recovery_package(
    directory: Path,
    *,
    certificate_package: Path,
) -> dict[str, Any]:
    """Recompute checksums, inputs, simulations, and all headline diagnostics."""

    directory = directory.resolve()
    locked = _locked_payloads(directory)
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    manifest = json.loads(
        (directory / "artifact_manifest.json").read_text(encoding="utf-8")
    )
    producer = str(summary.get("producer_commit", ""))
    if (
        summary.get("schema_version") != 1
        or summary.get("experiment")
        != "paired_heuristic_vs_certified_n60_recovery_diagnostic"
        or summary.get("producer_clean_worktree") is not True
        or _SHA40.fullmatch(producer) is None
    ):
        raise ValueError("invalid paired-recovery summary contract")
    if (
        manifest.get("schema_version") != 1
        or manifest.get("producer_commit") != producer
        or manifest.get("producer_git_dirty") is not False
        or set(item["path"] for item in manifest["files"])
        != _PAYLOADS - {"artifact_manifest.json"}
    ):
        raise ValueError("invalid paired-recovery manifest contract")
    for item in manifest["files"]:
        row = locked[item["path"]]
        if int(item["bytes"]) != int(row["bytes"]) or item["sha256"] != row["sha256"]:
            raise ValueError("manifest and checksum table disagree")

    config_values = dict(summary["config"])
    config_values["seeds"] = tuple(config_values["seeds"])
    config = PairedRecoveryConfig(**config_values)
    config.validate()
    if config != PairedRecoveryConfig():
        raise ValueError("paired-recovery config differs from the frozen default")
    heuristic_input = _provenance_item(
        summary,
        "chronologically_locked_heuristic_maximin_n60",
    )
    heuristic, heuristic_report = load_heuristic_allocation(
        directory / "heuristic_allocation_source.csv",
        expected_sha256=str(heuristic_input["source_sha256"]),
        source_code_sha=str(heuristic_input["source_code_sha"]),
    )
    certified, certificate_report = load_certified_allocation(certificate_package)
    for key in (
        "source_code_sha",
        "source_sha256",
        "budget",
        "max_point_fraction",
        "constructor_reproduced",
    ):
        _compare_summary_value(heuristic_input[key], heuristic_report[key], f"heuristic.{key}")
    certificate_input = _provenance_item(summary, "certified_integer_maximin_n60")
    for key in (
        "certificate_code_sha",
        "certificate_summary_sha256",
        "certificate_allocation_sha256",
        "certificate_sha256sums_sha256",
        "budget",
        "lower_bound",
        "upper_bound",
        "certified",
    ):
        _compare_summary_value(
            certificate_input[key],
            certificate_report[key],
            f"certificate.{key}",
        )

    recovery_rows, seed_rows, diagnostics = paired_recovery_rows(
        heuristic,
        certified,
        config,
    )
    _compare_rows(directory / "paired_recovery.csv", recovery_rows)
    _compare_rows(directory / "seed_summary.csv", seed_rows)
    for key, value in diagnostics.items():
        _compare_summary_value(summary[key], value, key)
    expected_count = len(config.seeds) * 2 * 6
    if len(recovery_rows) != expected_count:
        raise ValueError("paired-recovery table does not cover every seed/design/generator")
    return {
        "package": directory.name,
        "verified": True,
        "producer_commit": producer,
        "seed_count": len(config.seeds),
        "replicates_per_generator_per_seed": config.replicates,
        "recovery_row_count": expected_count,
        "allocation_l1_distance": diagnostics["allocation_l1_distance"],
        "heuristic_minimum_recovery": diagnostics["minimum_recovery_across_seeds"][
            "heuristic_maximin"
        ],
        "certified_minimum_recovery": diagnostics["minimum_recovery_across_seeds"][
            "certified_integer_maximin"
        ],
        "sha256sums_sha256": sha256(directory / "SHA256SUMS.csv"),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bayesian-ach-design-paired-recovery-verify",
        description="Verify frozen paired maximin recovery evidence.",
    )
    parser.add_argument("package", type=Path)
    parser.add_argument("--certificate-package", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = verify_paired_recovery_package(
        args.package,
        certificate_package=args.certificate_package,
    )
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
