from __future__ import annotations

import csv
import itertools
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from bayesian_ach.design_certificate import (
    CertifiedDesignConfig,
    MaximinCertificate,
    _master_upper_bound,
    certificate_matches_geometry,
    certify_maximin_design,
)
from bayesian_ach.design_certificate_cli import main
from bayesian_ach.design_geometry import pairwise_residual_matrix


def _objective(signals: np.ndarray, counts: np.ndarray) -> float:
    residuals = pairwise_residual_matrix(signals, counts)
    off_diagonal = ~np.eye(residuals.shape[0], dtype=bool)
    return float(np.min(residuals[off_diagonal]))


def test_integer_cutting_plane_matches_brute_force() -> None:
    signals = np.asarray(
        [
            [-1.2, -0.7],
            [-0.4, 0.9],
            [0.1, -0.2],
            [0.8, 0.3],
            [1.3, 1.1],
        ],
        dtype=float,
    )
    config = CertifiedDesignConfig(
        budget=4,
        max_point_fraction=0.5,
        integer=True,
        absolute_gap_tolerance=1.0e-9,
        relative_gap_tolerance=1.0e-9,
        max_iterations=100,
        master_time_limit_s=30.0,
        master_mip_relative_gap=1.0e-10,
    )
    certificate = certify_maximin_design(signals, config)

    feasible = (
        np.asarray(values, dtype=np.int64)
        for values in itertools.product(range(3), repeat=signals.shape[0])
        if sum(values) == config.budget
    )
    brute_value = max(_objective(signals, counts) for counts in feasible)

    assert certificate.certified
    assert certificate.lower_bound == pytest.approx(brute_value, abs=1.0e-8)
    assert certificate.upper_bound - certificate.lower_bound <= 1.0e-8
    assert np.allclose(certificate.allocation, np.rint(certificate.allocation))
    assert int(np.sum(certificate.allocation)) == config.budget
    assert certificate_matches_geometry(signals, certificate)


def test_continuous_certificate_bounds_integer_optimum() -> None:
    signals = np.asarray(
        [
            [-1.0, -0.2],
            [-0.5, 1.0],
            [0.0, -0.8],
            [0.5, 0.7],
            [1.0, 0.1],
        ],
        dtype=float,
    )
    integer = certify_maximin_design(
        signals,
        CertifiedDesignConfig(
            budget=4,
            max_point_fraction=0.5,
            integer=True,
            max_iterations=100,
            master_time_limit_s=30.0,
        ),
    )
    continuous = certify_maximin_design(
        signals,
        CertifiedDesignConfig(
            budget=4,
            max_point_fraction=0.5,
            integer=False,
            max_iterations=100,
            master_time_limit_s=30.0,
        ),
    )

    assert integer.certified
    assert continuous.certified
    assert continuous.lower_bound + 1.0e-8 >= integer.lower_bound
    assert continuous.upper_bound - continuous.lower_bound <= 2.0e-7


def test_mip_dual_bound_is_used_even_for_success_status() -> None:
    result = SimpleNamespace(success=True, mip_dual_bound=-0.31)
    assert _master_upper_bound(result, integer=True, incumbent=0.30) == pytest.approx(
        0.31
    )


def test_invalid_initial_integer_allocation_is_rejected() -> None:
    signals = np.asarray([[-1.0, 0.0], [0.0, 1.0], [1.0, -1.0]])
    with pytest.raises(ValueError, match="integer initial_allocation"):
        certify_maximin_design(
            signals,
            CertifiedDesignConfig(budget=3, max_point_fraction=1.0),
            initial_allocation=np.asarray([0.5, 1.0, 1.5]),
        )


def test_certificate_cli_writes_hash_bound_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import bayesian_ach.design_certificate_cli as cli

    rows = (
        {"point_id": 0, "condition": "a"},
        {"point_id": 1, "condition": "b"},
    )
    signals = np.asarray([[-1.0, 0.5], [1.0, -0.5]])
    certificate = MaximinCertificate(
        allocation=np.asarray([1.0, 1.0]),
        integer=True,
        certified=True,
        lower_bound=0.2,
        upper_bound=0.2,
        absolute_gap=0.0,
        relative_gap=0.0,
        heuristic_lower_bound=0.1,
        iterations=1,
        cut_count=2,
        last_master_status=0,
        last_master_message="optimal",
        trace=(
            {
                "iteration": 1,
                "status": 0,
                "message": "optimal",
                "cut_count": 2,
            },
        ),
    )
    monkeypatch.setattr(
        cli,
        "generate_transition_design_grid",
        lambda: (rows, signals, signals),
    )
    monkeypatch.setattr(cli, "certify_maximin_design", lambda *_args, **_kwargs: certificate)
    monkeypatch.setattr(cli, "certificate_matches_geometry", lambda *_args: True)

    output = tmp_path / "certificate"
    assert (
        main(
            [
                "--output",
                str(output),
                "--code-sha",
                "deadbeef",
                "--budget",
                "2",
                "--max-point-fraction",
                "1.0",
                "--require-certificate",
            ]
        )
        == 0
    )
    summary = json.loads((output / "certificate_summary.json").read_text())
    assert summary["certified"]
    assert summary["code_sha"] == "deadbeef"
    with (output / "SHA256SUMS.csv").open(newline="", encoding="utf-8") as handle:
        manifest = list(csv.DictReader(handle))
    assert {row["file"] for row in manifest} == {
        "certificate_summary.json",
        "certified_allocation.csv",
        "cut_trace.csv",
    }
