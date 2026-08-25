from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from bayesian_ach import design_paired_recovery as paired
from bayesian_ach.design_grid import generate_transition_design_grid
from bayesian_ach.design_optimizer import optimize_maximin_design
from bayesian_ach.design_paired_recovery_verify import verify_paired_recovery_package


def _heuristic_source(path: Path) -> Path:
    _, _, signals = generate_transition_design_grid()
    counts = optimize_maximin_design(
        signals,
        60,
        max_point_fraction=0.15,
        effect_size=1.0,
        noise_std=1.0,
        target_log_score_gap=5.0,
    ).counts
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["design", "count", "point_id"])
        writer.writeheader()
        for point in np.flatnonzero(counts):
            writer.writerow(
                {
                    "design": "maximin_optimized",
                    "count": int(counts[point]),
                    "point_id": int(point),
                }
            )
    return path


def _certificate_package() -> Path:
    return Path(__file__).resolve().parents[1] / "results/certified-maximin-design/n60"


def test_locked_heuristic_is_hash_bound_and_reconstructed(tmp_path: Path) -> None:
    path = _heuristic_source(tmp_path / "heuristic.csv")
    digest = paired.sha256(path)
    counts, report = paired.load_heuristic_allocation(
        path,
        expected_sha256=digest,
        source_code_sha="1" * 40,
    )
    assert int(np.sum(counts)) == 60
    assert report["constructor_reproduced"] is True
    path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256"):
        paired.load_heuristic_allocation(
            path,
            expected_sha256=digest,
            source_code_sha="1" * 40,
        )


def test_paired_package_recomputes_and_detects_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    heuristic = _heuristic_source(tmp_path / "heuristic.csv")
    monkeypatch.setattr(paired, "_git_provenance", lambda *_args, **_kwargs: None)
    output = tmp_path / "evidence"
    summary = paired.freeze_paired_recovery(
        output=output,
        repo_root=tmp_path,
        code_sha="2" * 40,
        heuristic_path=heuristic,
        heuristic_sha256=paired.sha256(heuristic),
        heuristic_source_code_sha="1" * 40,
        certificate_package=_certificate_package(),
        config=paired.PairedRecoveryConfig(),
    )
    assert summary["paired_rng"] is True
    report = verify_paired_recovery_package(
        output,
        certificate_package=_certificate_package(),
    )
    assert report["verified"] is True
    assert report["recovery_row_count"] == 60

    seed_summary = output / "seed_summary.csv"
    seed_summary.write_text(
        seed_summary.read_text(encoding="utf-8").replace("0.83", "0.84", 1),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="payload mismatch"):
        verify_paired_recovery_package(
            output,
            certificate_package=_certificate_package(),
        )
